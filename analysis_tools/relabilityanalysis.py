# reliabilityanalysis.py
# Purpose: Comprehensive Reliability & Robustness Analysis Suite.
# Implements tactics to verify trust in the optimization results.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import copy
import time
import contextlib
from collections import Counter
from dataclasses import dataclass
import io
import os
import sys
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import csv
from pathlib import Path
from datetime import datetime
import argparse
import platform
from matplotlib.ticker import StrMethodFormatter
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Project Imports
from orbit_launch.config import StarshipBlock2, EARTH_CONFIG, RELIABILITY_ANALYSIS_TOGGLES
from orbit_launch import debug, guidance
from orbit_launch.vehicle import Vehicle
from orbit_launch.environment import Environment
from orbit_launch.main import (
    OPTIMIZATION_G_GUARD_BAND,
    OPTIMIZATION_Q_GUARD_BAND_PA,
    solve_optimal_trajectory,
)
from orbit_launch.simulation import run_simulation
from orbit_launch.trajectory_metrics import (
    circular_target_speed_m_s,
    ellipsoidal_altitude_m,
    spherical_altitude_m,
    target_orbit_radius_m,
)

# --- Helper to suppress output during batch runs ---
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            yield

# Centralized strict terminal tolerances used by all reliability analyses.
# Chosen to be tighter than before while still allowing realistic numerical/model scatter.
TERMINAL_ALT_TOL_M = 5_000.0
TERMINAL_VEL_TOL_M_S = 10.0
TERMINAL_RADIAL_VEL_TOL_M_S = 5.0
TERMINAL_INC_TOL_DEG = 0.25


def resolve_target_inclination_deg(cfg, env_cfg):
    """Return the effective target inclination used by the solver/checks."""
    if cfg.target_inclination is None:
        return abs(float(env_cfg.launch_latitude))
    return float(cfg.target_inclination)


def apply_analysis_profile(toggles, profile):
    """Apply a named analysis profile to a toggle copy."""
    if profile == "course_core":
        toggles.randomized_multistart = False
        toggles.grid_independence = True
        toggles.interval_replay_audit = True
        toggles.theoretical_efficiency = True
        toggles.integrator_tolerance = True
        toggles.smooth_integrator_benchmark = True
        toggles.drift = True
    return toggles




class TeeStream:
    def __init__(self, *targets):
        self.targets = targets

    def write(self, data):
        for target in self.targets:
            target.write(data)
        return len(data)

    def flush(self):
        for target in self.targets:
            target.flush()

    def isatty(self):
        return any(getattr(target, "isatty", lambda: False)() for target in self.targets)


class BinaryLogTextStream:
    def __init__(self, binary_target, write_lock):
        self.binary_target = binary_target
        self.write_lock = write_lock

    def write(self, data):
        if not data:
            return 0
        if isinstance(data, bytes):
            text = data.decode("utf-8", errors="replace")
            chunk = data
        else:
            text = str(data)
            chunk = text.encode("utf-8", errors="replace")
        with self.write_lock:
            self.binary_target.write(chunk)
            self.binary_target.flush()
        return len(text)

    def flush(self):
        with self.write_lock:
            self.binary_target.flush()


def _stream_fileno(stream):
    fileno = getattr(stream, "fileno", None)
    if fileno is None:
        return None
    try:
        return int(fileno())
    except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
        return None


@dataclass(frozen=True)
class SeedParams:
    time_scale: float
    t2_scale: float
    pos_sigma: float
    vel_sigma: float
    mass_sigma: float
    throttle_sigma: float
    direction_sigma_deg: float
    direction_component_sigma: float = np.nan

    def as_dict(self) -> dict[str, float]:
        return {
            "time_scale": float(self.time_scale),
            "t2_scale": float(self.t2_scale),
            "pos_sigma": float(self.pos_sigma),
            "vel_sigma": float(self.vel_sigma),
            "mass_sigma": float(self.mass_sigma),
            "throttle_sigma": float(self.throttle_sigma),
            "direction_sigma_deg": float(self.direction_sigma_deg),
            "direction_component_sigma": float(self.direction_component_sigma),
        }


@dataclass(frozen=True)
class StructuredSeedSpec:
    label: str
    time_scale: float
    t2_scale: float
    pos_sigma: float
    vel_sigma: float
    mass_sigma: float
    throttle_sigma: float
    direction_sigma_deg: float
    direction_axis: tuple[float, float, float]


@dataclass
class SeedVariant:
    label: str
    guess: dict[str, Any]
    params: SeedParams


MIN_WARM_START_TRIALS = 5
DEFAULT_WARM_START_TRIALS = 10


# Keep the deterministic replay-compatible seeds orthogonal:
# time_* vary phase timing, state_* vary lateral thrust direction.
DEFAULT_STRUCTURED_SEED_SPECS = (
    StructuredSeedSpec(
        label="time_minus",
        time_scale=0.9925,
        t2_scale=0.9950,
        pos_sigma=-0.0025,
        vel_sigma=-0.0025,
        mass_sigma=-0.0015,
        throttle_sigma=0.0,
        direction_sigma_deg=0.35,
        direction_axis=(0.0, 0.0, 1.0),
    ),
    StructuredSeedSpec(
        label="time_plus",
        time_scale=1.0075,
        t2_scale=1.0050,
        pos_sigma=0.0025,
        vel_sigma=0.0025,
        mass_sigma=0.0015,
        throttle_sigma=0.0,
        direction_sigma_deg=-0.35,
        direction_axis=(0.0, 0.0, 1.0),
    ),
    StructuredSeedSpec(
        label="state_minus",
        time_scale=1.0000,
        t2_scale=1.0000,
        pos_sigma=-0.0050,
        vel_sigma=-0.0050,
        mass_sigma=-0.0030,
        throttle_sigma=0.0,
        direction_sigma_deg=0.35,
        direction_axis=(0.0, 0.0, 1.0),
    ),
    StructuredSeedSpec(
        label="state_plus",
        time_scale=1.0000,
        t2_scale=1.0000,
        pos_sigma=0.0050,
        vel_sigma=0.0050,
        mass_sigma=0.0030,
        throttle_sigma=0.0,
        direction_sigma_deg=-0.35,
        direction_axis=(0.0, 0.0, 1.0),
    ),
)


class WarmStartSeedFactory:
    def __init__(self, structured_specs=DEFAULT_STRUCTURED_SEED_SPECS):
        self.structured_specs = tuple(structured_specs)

    @staticmethod
    def normalize_direction_history(td):
        arr = np.array(td, dtype=float, copy=True)
        if arr.ndim != 2 or arr.shape[0] != 3:
            return arr
        norms = np.linalg.norm(arr, axis=0)
        good = norms > 1.0e-12
        if np.any(good):
            arr[:, good] = arr[:, good] / norms[good]
        if np.any(~good):
            arr[:, ~good] = np.array([[1.0], [0.0], [0.0]])
        return arr

    @staticmethod
    def perturb_state_history(X, rng, pos_sigma, vel_sigma, mass_sigma):
        arr = np.array(X, dtype=float, copy=True)
        if arr.ndim != 2 or arr.shape[0] < 7 or arr.shape[1] < 1:
            return arr

        def apply_rowwise_relative_jitter(row_slice, sigma):
            for j in range(row_slice.start, row_slice.stop):
                row = np.array(arr[j, :], dtype=float, copy=True)
                noise = rng.uniform(-sigma, sigma, size=row.shape)
                pert = row * (1.0 + noise)
                near_zero = np.abs(row) <= 1.0e-12
                if np.any(near_zero):
                    ref_scale = float(np.nanmedian(np.abs(row)))
                    if not np.isfinite(ref_scale) or ref_scale <= 0.0:
                        ref_scale = float(np.nanmax(np.abs(row)))
                    if not np.isfinite(ref_scale) or ref_scale <= 0.0:
                        ref_scale = 1.0
                    pert[near_zero] = row[near_zero] + noise[near_zero] * ref_scale
                arr[j, :] = pert

        apply_rowwise_relative_jitter(slice(0, 3), pos_sigma)
        apply_rowwise_relative_jitter(slice(3, 6), vel_sigma)
        apply_rowwise_relative_jitter(slice(6, 7), mass_sigma)
        arr[:, 0] = X[:, 0]
        arr[6, :] = np.minimum.accumulate(arr[6, :])
        arr[6, :] = np.clip(arr[6, :], 1.0e3, None)
        return arr

    @staticmethod
    def apply_relative_state_bias(X, pos_bias, vel_bias, mass_bias):
        arr = np.array(X, dtype=float, copy=True)
        if arr.ndim != 2 or arr.shape[0] < 7 or arr.shape[1] < 1:
            return arr

        weights = np.linspace(0.0, 1.0, arr.shape[1])

        def apply_rowwise_bias(row_slice, rel_bias):
            if abs(float(rel_bias)) <= 1.0e-12:
                return
            for j in range(row_slice.start, row_slice.stop):
                row = np.array(arr[j, :], dtype=float, copy=True)
                pert = row * (1.0 + rel_bias * weights)
                near_zero = np.abs(row) <= 1.0e-12
                if np.any(near_zero):
                    ref_scale = float(np.nanmedian(np.abs(row)))
                    if not np.isfinite(ref_scale) or ref_scale <= 0.0:
                        ref_scale = float(np.nanmax(np.abs(row)))
                    if not np.isfinite(ref_scale) or ref_scale <= 0.0:
                        ref_scale = 1.0
                    pert[near_zero] = row[near_zero] + rel_bias * weights[near_zero] * ref_scale
                arr[j, :] = pert

        apply_rowwise_bias(slice(0, 3), pos_bias)
        apply_rowwise_bias(slice(3, 6), vel_bias)
        apply_rowwise_bias(slice(6, 7), mass_bias)
        arr[:, 0] = X[:, 0]
        arr[6, :] = np.minimum.accumulate(arr[6, :])
        arr[6, :] = np.clip(arr[6, :], 1.0e3, None)
        return arr

    @staticmethod
    def apply_throttle_bias(throttle_history, min_throttle, rel_bias):
        arr = np.array(throttle_history, dtype=float, copy=True)
        if arr.ndim != 1 or arr.shape[0] < 1 or abs(float(rel_bias)) <= 1.0e-12:
            return np.clip(arr, min_throttle, 1.0)
        weights = np.linspace(0.0, 1.0, arr.shape[0])
        if rel_bias > 0.0:
            # Apply positive bias against the remaining headroom instead of
            # multiplying and clipping, which otherwise collapses the "plus"
            # seeds into no-ops once nominal throttle is already near 1.0.
            arr += rel_bias * weights * np.maximum(0.0, 1.0 - arr)
        else:
            arr += rel_bias * weights * np.maximum(0.0, arr - min_throttle)
        return np.clip(arr, min_throttle, 1.0)

    @staticmethod
    def orthogonal_unit_vector(direction, axis_seed):
        d = np.array(direction, dtype=float, copy=True).reshape(-1)
        if d.shape[0] != 3:
            raise ValueError("Direction vector must have three components.")
        d_norm = np.linalg.norm(d)
        if d_norm <= 1.0e-12:
            return np.array([0.0, 1.0, 0.0])
        d = d / d_norm

        axis = np.array(axis_seed, dtype=float, copy=True).reshape(-1)
        if axis.shape[0] != 3 or np.linalg.norm(axis) <= 1.0e-12:
            axis = np.array([0.0, 0.0, 1.0])
        axis = axis / np.linalg.norm(axis)

        ortho = axis - np.dot(axis, d) * d
        ortho_norm = np.linalg.norm(ortho)
        if ortho_norm <= 1.0e-12:
            fallback = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(fallback, d)) > 0.9:
                fallback = np.array([0.0, 1.0, 0.0])
            ortho = fallback - np.dot(fallback, d) * d
            ortho_norm = np.linalg.norm(ortho)
        if ortho_norm <= 1.0e-12:
            return np.array([0.0, 1.0, 0.0])
        return ortho / ortho_norm

    def apply_direction_bias(self, direction_history, bias_deg, axis_seed):
        arr = self.normalize_direction_history(direction_history)
        if arr.ndim != 2 or arr.shape[0] != 3 or arr.shape[1] < 1 or abs(float(bias_deg)) <= 1.0e-12:
            return arr

        angles = np.radians(bias_deg) * np.linspace(0.0, 1.0, arr.shape[1])
        for k in range(arr.shape[1]):
            ang = float(angles[k])
            if abs(ang) <= 1.0e-12:
                continue
            d = arr[:, k]
            tilt = self.orthogonal_unit_vector(d, axis_seed)
            arr[:, k] = np.cos(ang) * d + np.sin(ang) * tilt
        return self.normalize_direction_history(arr)

    @staticmethod
    def _rebuild_state_histories(guess, state_rebuilder):
        if state_rebuilder is None:
            return guess
        try:
            rebuilt = state_rebuilder(copy.deepcopy(guess))
        except Exception:
            return guess
        if not isinstance(rebuilt, dict):
            return guess
        for key in ("X1", "X2", "X3"):
            if key in rebuilt:
                guess[key] = rebuilt[key]
        return guess

    def build_structured_variants(self, base_guess, cfg, state_rebuilder=None):
        if not isinstance(base_guess, dict):
            raise ValueError("Initial guess must be a dict.")

        variants = []
        for spec in self.structured_specs:
            guess = copy.deepcopy(base_guess)

            t1 = float(guess.get("T1", cfg.sequence.min_stage_1_burn))
            t3 = float(guess.get("T3", cfg.sequence.min_stage_2_burn))
            guess["T1"] = max(cfg.sequence.min_stage_1_burn, t1 * float(spec.time_scale))
            guess["T3"] = max(cfg.sequence.min_stage_2_burn, t3 * float(spec.time_scale))
            t2_scale = 1.0
            if guess.get("T2", None) is not None:
                guess["T2"] = max(0.0, float(guess["T2"]) * float(spec.t2_scale))
                t2_scale = float(spec.t2_scale)

            if state_rebuilder is None:
                for key in ("X1", "X2", "X3"):
                    val = guess.get(key, None)
                    if val is None:
                        continue
                    guess[key] = self.apply_relative_state_bias(
                        val,
                        pos_bias=spec.pos_sigma,
                        vel_bias=spec.vel_sigma,
                        mass_bias=spec.mass_sigma,
                    )

            for key in ("TH1", "TH3"):
                val = guess.get(key, None)
                if val is None:
                    continue
                guess[key] = self.apply_throttle_bias(val, cfg.sequence.min_throttle, spec.throttle_sigma)

            for key in ("TD1", "TD3"):
                val = guess.get(key, None)
                if val is None:
                    continue
                guess[key] = self.apply_direction_bias(val, spec.direction_sigma_deg, spec.direction_axis)

            guess = self._rebuild_state_histories(guess, state_rebuilder)

            variants.append(
                SeedVariant(
                    label=str(spec.label),
                    guess=guess,
                    params=SeedParams(
                        time_scale=spec.time_scale,
                        t2_scale=t2_scale,
                        pos_sigma=spec.pos_sigma,
                        vel_sigma=spec.vel_sigma,
                        mass_sigma=spec.mass_sigma,
                        throttle_sigma=spec.throttle_sigma,
                        direction_sigma_deg=spec.direction_sigma_deg,
                        direction_component_sigma=np.nan,
                    ),
                )
            )
        return variants

    def build_randomized_variant(self, base_guess, cfg, rng, label, state_rebuilder=None):
        if not isinstance(base_guess, dict):
            raise ValueError("Initial guess must be a dict.")

        guess = copy.deepcopy(base_guess)

        frac = 0.01
        time_scale = float(1.0 + rng.uniform(-frac, frac))
        pos_sigma = float(frac)
        vel_sigma = float(frac)
        mass_sigma = float(frac)
        throttle_sigma = float(frac)
        direction_sigma = float(frac)

        t1 = float(guess.get("T1", cfg.sequence.min_stage_1_burn))
        t3 = float(guess.get("T3", cfg.sequence.min_stage_2_burn))
        guess["T1"] = max(cfg.sequence.min_stage_1_burn, t1 * time_scale)
        guess["T3"] = max(cfg.sequence.min_stage_2_burn, t3 * time_scale)
        t2_scale = 1.0
        if guess.get("T2", None) is not None:
            t2_scale = float(1.0 + float(rng.uniform(-frac, frac)))
            guess["T2"] = max(0.0, float(guess["T2"]) * t2_scale)

        if state_rebuilder is None:
            for key in ("X1", "X2", "X3"):
                val = guess.get(key, None)
                if val is not None:
                    guess[key] = self.perturb_state_history(val, rng, pos_sigma, vel_sigma, mass_sigma)

        for key in ("TH1", "TH3"):
            val = guess.get(key, None)
            if val is None:
                continue
            th = np.array(val, dtype=float, copy=True)
            th *= 1.0 + rng.uniform(-throttle_sigma, throttle_sigma, size=th.shape)
            guess[key] = np.clip(th, cfg.sequence.min_throttle, 1.0)

        for key in ("TD1", "TD3"):
            val = guess.get(key, None)
            if val is None:
                continue
            td = np.array(val, dtype=float, copy=True)
            td += rng.normal(0.0, direction_sigma, size=td.shape)
            guess[key] = self.normalize_direction_history(td)

        guess = self._rebuild_state_histories(guess, state_rebuilder)

        return SeedVariant(
            label=label,
            guess=guess,
            params=SeedParams(
                time_scale=time_scale,
                t2_scale=t2_scale,
                pos_sigma=pos_sigma,
                vel_sigma=vel_sigma,
                mass_sigma=mass_sigma,
                throttle_sigma=throttle_sigma,
                direction_sigma_deg=np.nan,
                direction_component_sigma=direction_sigma,
            ),
        )

    def build_trial_variants(self, base_guess, cfg, rng, n_trials, state_rebuilder=None):
        n_use = max(MIN_WARM_START_TRIALS, int(n_trials))
        variants = [
            SeedVariant(
                label="nominal",
                guess=copy.deepcopy(base_guess),
                params=SeedParams(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            )
        ]
        structured = self.build_structured_variants(base_guess, cfg, state_rebuilder=state_rebuilder)
        variants.extend(structured[: max(0, n_use - 1)])

        next_index = len(variants)
        while len(variants) < n_use:
            label = f"rand_{next_index:02d}"
            variants.append(self.build_randomized_variant(base_guess, cfg, rng, label, state_rebuilder=state_rebuilder))
            next_index += 1
        return variants[:n_use]


class WarmStartMultiseedAnalyzer:
    def __init__(self, suite, evaluate_terminal_state_fn, seed_factory=None):
        self.suite = suite
        self.evaluate_terminal_state = evaluate_terminal_state_fn
        self.seed_factory = seed_factory or WarmStartSeedFactory()
        self.logs_dir = Path(self.suite.output_dir) / "logs"

    @staticmethod
    def _resample_series(x_raw, y_raw, x_new):
        x = np.asarray(x_raw, dtype=float).reshape(-1)
        y = np.asarray(y_raw, dtype=float).reshape(-1)
        x_new_arr = np.asarray(x_new, dtype=float).reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(mask) <= 0:
            return np.full_like(x_new_arr, np.nan, dtype=float)
        x = x[mask]
        y = y[mask]
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        x_unique, unique_idx = np.unique(x, return_index=True)
        y_unique = y[unique_idx]
        if len(x_unique) <= 1 or np.allclose(x_unique, x_unique[0]):
            return np.full_like(x_new_arr, float(y_unique[0]), dtype=float)
        return np.interp(x_new_arr, x_unique, y_unique)

    @staticmethod
    def _resample_state_history(t_raw, y_raw, t_grid):
        t = np.asarray(t_raw, dtype=float).reshape(-1)
        y = np.asarray(y_raw, dtype=float)
        t_new = np.asarray(t_grid, dtype=float).reshape(-1)
        if t.ndim != 1 or y.ndim != 2 or y.shape[1] != len(t) or len(t) < 1:
            return None
        mask = np.isfinite(t) & np.all(np.isfinite(y), axis=0)
        if np.count_nonzero(mask) <= 0:
            return None
        t = t[mask]
        y = y[:, mask]
        order = np.argsort(t)
        t = t[order]
        y = y[:, order]
        t_unique, unique_idx = np.unique(t, return_index=True)
        y_unique = y[:, unique_idx]
        if len(t_unique) <= 1 or np.allclose(t_unique, t_unique[0]):
            return np.repeat(y_unique[:, -1][:, None], len(t_new), axis=1)
        t_eval = np.clip(t_new, float(t_unique[0]), float(t_unique[-1]))
        return np.vstack([np.interp(t_eval, t_unique, y_unique[row, :]) for row in range(y_unique.shape[0])])

    @staticmethod
    def _compute_downrange_km(r_hist, env_cfg, t_hist):
        omega_e = env_cfg.earth_omega_vector[2]
        r_ecef_0 = None
        out = []
        for idx, t_val in enumerate(t_hist):
            r_eci = r_hist[:, idx]
            theta = omega_e * t_val + env_cfg.initial_rotation
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            x_ecef = cos_t * r_eci[0] + sin_t * r_eci[1]
            y_ecef = -sin_t * r_eci[0] + cos_t * r_eci[1]
            z_ecef = r_eci[2]
            r_ecef = np.array([x_ecef, y_ecef, z_ecef], dtype=float)
            if r_ecef_0 is None:
                r_ecef_0 = r_ecef
            denom = np.linalg.norm(r_ecef_0) * np.linalg.norm(r_ecef)
            dot_val = np.dot(r_ecef_0, r_ecef) / max(denom, 1.0e-12)
            angle = np.arccos(np.clip(dot_val, -1.0, 1.0))
            out.append(angle * env_cfg.earth_radius_equator / 1000.0)
        return np.array(out, dtype=float)

    def _build_trajectory_profile(self, sim_res, env_cfg, n_samples=301):
        if not isinstance(sim_res, dict):
            return None
        t = np.asarray(sim_res.get("t", None), dtype=float)
        y = np.asarray(sim_res.get("y", None), dtype=float)
        if t.ndim != 1 or y.ndim != 2 or len(t) < 1 or y.shape[0] < 7 or y.shape[1] != len(t):
            return None
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
            return None

        progress_grid = np.linspace(0.0, 1.0, max(2, int(n_samples)))
        if len(t) == 1 or float(t[-1]) <= float(t[0]) + 1.0e-12:
            progress_raw = np.zeros_like(t)
        else:
            progress_raw = (t - float(t[0])) / max(float(t[-1] - t[0]), 1.0e-12)

        r_hist = y[0:3, :]
        speed_raw = np.linalg.norm(y[3:6, :], axis=0)
        mass_raw = y[6, :]
        altitude_raw = np.array(
            [ellipsoidal_altitude_m(r_hist[:, i], env_cfg) / 1000.0 for i in range(r_hist.shape[1])],
            dtype=float,
        )
        downrange_raw = self._compute_downrange_km(r_hist, env_cfg, t)

        return {
            "progress": progress_grid,
            "time_s": self._resample_series(progress_raw, t, progress_grid),
            "downrange_km": self._resample_series(progress_raw, downrange_raw, progress_grid),
            "ellipsoidal_altitude_km": self._resample_series(progress_raw, altitude_raw, progress_grid),
            "velocity_m_s": self._resample_series(progress_raw, speed_raw, progress_grid),
            "mass_kg": self._resample_series(progress_raw, mass_raw, progress_grid),
            "x_m": self._resample_series(progress_raw, r_hist[0, :], progress_grid),
            "y_m": self._resample_series(progress_raw, r_hist[1, :], progress_grid),
            "z_m": self._resample_series(progress_raw, r_hist[2, :], progress_grid),
        }

    def _rebuild_guess_states(self, guess, cfg, env_cfg):
        try:
            with suppress_stdout():
                env = Environment(copy.deepcopy(env_cfg))
                veh = Vehicle(cfg, env)
                sim_input = {
                    "T1": float(guess["T1"]),
                    "T2": float(guess.get("T2", 0.0)),
                    "T3": float(guess["T3"]),
                    "U1": np.vstack([np.atleast_2d(np.asarray(guess["TH1"], dtype=float)), np.asarray(guess["TD1"], dtype=float)]),
                    "U3": np.vstack([np.atleast_2d(np.asarray(guess["TH3"], dtype=float)), np.asarray(guess["TD3"], dtype=float)]),
                }
                sim_res = run_simulation(sim_input, veh, cfg, rtol=1e-6, atol=1e-8, return_phase_results=True)
            phase_results = sim_res.get("phase_results", None)
            if not isinstance(phase_results, dict):
                return None

            rebuilt = copy.deepcopy(guess)
            n1 = int(np.asarray(guess["TH1"]).shape[0])
            boost = phase_results.get("boost", None)
            if not isinstance(boost, dict):
                return None
            x1 = self._resample_state_history(boost.get("t", []), boost.get("y", []), np.linspace(0.0, float(guess["T1"]), n1 + 1))
            if x1 is None:
                return None
            rebuilt["X1"] = x1

            use_coast = float(guess.get("T2", 0.0)) > 1.0e-4 and guess.get("X2", None) is not None
            t_start_ship = float(guess["T1"])
            if use_coast:
                coast = phase_results.get("coast", None)
                if not isinstance(coast, dict):
                    return None
                n2 = int(np.asarray(guess["X2"]).shape[1] - 1)
                t_end_coast = float(guess["T1"] + guess["T2"])
                x2 = self._resample_state_history(coast.get("t", []), coast.get("y", []), np.linspace(float(guess["T1"]), t_end_coast, n2 + 1))
                if x2 is None:
                    return None
                rebuilt["X2"] = x2
                t_start_ship = t_end_coast

            ship = phase_results.get("ship", None)
            if not isinstance(ship, dict):
                return None
            n3 = int(np.asarray(guess["TH3"]).shape[0])
            x3 = self._resample_state_history(ship.get("t", []), ship.get("y", []), np.linspace(t_start_ship, t_start_ship + float(guess["T3"]), n3 + 1))
            if x3 is None:
                return None
            rebuilt["X3"] = x3
            return rebuilt
        except Exception:
            return None

    @staticmethod
    def _empty_result(trial_index, label, params, status, runtime_s=np.nan):
        return {
            "trial_index": int(trial_index),
            "label": str(label),
            "params": params.as_dict() if isinstance(params, SeedParams) else dict(params),
            "solver_success": 0,
            "mission_success": 0,
            "mission_success_slack": 0,
            "terminal_strict_success": 0,
            "path_ok": 0,
            "q_ok": 0,
            "g_ok": 0,
            "raw_path_ok": 0,
            "raw_q_ok": 0,
            "raw_g_ok": 0,
            "status": str(status),
            "final_mass_kg": np.nan,
            "runtime_s": float(runtime_s) if np.isfinite(runtime_s) else np.nan,
            "solver_status_class": str(status),
            "solver_return_status": "",
            "solver_iter_count": np.nan,
            "log_path": "",
            "log_available": 0,
            "trajectory_available": 0,
            "terminal_status": "",
            "terminal_valid": 0,
            "alt_err_m": np.nan,
            "vel_err_m_s": np.nan,
            "radial_velocity_m_s": np.nan,
            "inclination_err_deg": np.nan,
            "fuel_margin_kg": np.nan,
            "terminal_criteria_ratio": np.nan,
            "max_q_pa": np.nan,
            "max_g": np.nan,
            "q_margin_pa": np.nan,
            "g_margin": np.nan,
            "q_utilization": np.nan,
            "g_utilization": np.nan,
            "path_utilization_ratio": np.nan,
            "trajectory_profile": None,
        }

    def _build_nominal_guidance_guess(self):
        try:
            cfg = copy.deepcopy(self.suite.base_config)
            env_cfg = copy.deepcopy(self.suite.base_env_config)
            env = Environment(env_cfg)
            veh = Vehicle(cfg, env)
            with suppress_stdout():
                return guidance.get_initial_guess(cfg, veh, env, num_nodes=cfg.num_nodes)
        except Exception:
            return None

    @staticmethod
    def _print_progress(label, done, total, prefix="  "):
        total_i = max(1, int(total))
        done_i = min(max(0, int(done)), total_i)
        width = 22
        filled = int(round(width * done_i / total_i))
        bar = "#" * filled + "-" * (width - filled)
        print(f"{prefix}{label}: [{bar}] {done_i}/{total_i}")

    @staticmethod
    def _compute_insertion_zoom_window(trajectory_profiles, progress_threshold=0.85):
        x_segments = []
        y_segments = []
        for profile in trajectory_profiles:
            if not isinstance(profile, dict):
                continue
            progress = np.asarray(profile.get("progress", []), dtype=float)
            downrange_km = np.asarray(profile.get("downrange_km", []), dtype=float)
            altitude_km = np.asarray(profile.get("ellipsoidal_altitude_km", []), dtype=float)
            if progress.ndim != 1 or downrange_km.ndim != 1 or altitude_km.ndim != 1:
                continue
            if len(progress) != len(downrange_km) or len(progress) != len(altitude_km):
                continue
            mask = (
                np.isfinite(progress)
                & np.isfinite(downrange_km)
                & np.isfinite(altitude_km)
                & (progress >= float(progress_threshold))
            )
            if np.count_nonzero(mask) < 2:
                continue
            x_segments.append(downrange_km[mask])
            y_segments.append(altitude_km[mask])

        if not x_segments or not y_segments:
            return None

        x_all = np.concatenate(x_segments)
        y_all = np.concatenate(y_segments)
        if x_all.size < 2 or y_all.size < 2:
            return None

        x_min = float(np.nanmin(x_all))
        x_max = float(np.nanmax(x_all))
        y_min = float(np.nanmin(y_all))
        y_max = float(np.nanmax(y_all))
        if not all(np.isfinite(v) for v in (x_min, x_max, y_min, y_max)):
            return None

        x_pad = max(2.0, 0.08 * max(x_max - x_min, 1.0))
        y_pad = max(2.0, 0.12 * max(y_max - y_min, 1.0))
        return (
            (x_min - x_pad, x_max + x_pad),
            (max(0.0, y_min - y_pad), y_max + y_pad),
        )

    @staticmethod
    def _classify_solver_return_status(return_status):
        status = str(return_status or "").strip()
        if not status:
            return "SOLVE_FAIL"
        mapping = {
            "Solve_Succeeded": "SOLVE_OK",
            "Solved_To_Acceptable_Level": "SOLVE_OK",
            "Restoration_Failed": "RESTORATION_FAIL",
            "Maximum_Iterations_Exceeded": "MAX_ITER",
            "Maximum_CpuTime_Exceeded": "MAX_CPU",
            "Infeasible_Problem_Detected": "INFEASIBLE",
            "Search_Direction_Becomes_Too_Small": "SEARCH_SMALL",
            "Diverging_Iterates": "DIVERGING",
            "Invalid_Number_Detected": "INVALID_NUM",
            "NonIpopt_Exception_Thrown": "INTERRUPTED",
            "User_Requested_Stop": "INTERRUPTED",
            "Feasible_Point_Found": "FEASIBLE_POINT",
        }
        if status in mapping:
            return mapping[status]
        safe = "".join(ch if ch.isalnum() else "_" for ch in status.upper())
        safe = safe.strip("_")
        return safe or "SOLVE_FAIL"

    @staticmethod
    def _terminal_criteria_ratio(terminal_metrics):
        if not isinstance(terminal_metrics, dict):
            return np.nan
        ratios = []
        checks = (
            ("alt_err_m", 1.0),
            ("vel_err_m_s", 1.0),
            ("radial_velocity_m_s", 1.0),
            ("inclination_err_deg", 1.0),
        )
        tolerances = {
            "alt_err_m": 5.0e3,
            "vel_err_m_s": 100.0,
            "radial_velocity_m_s": 10.0,
            "inclination_err_deg": 0.5,
        }
        for key, scale in checks:
            val = float(terminal_metrics.get(key, np.nan))
            tol = tolerances[key] * scale
            if np.isfinite(val) and tol > 0.0:
                ratios.append(abs(val) / tol)
        return float(np.max(ratios)) if ratios else np.nan

    @staticmethod
    def _outcome_color(status):
        palette = {
            "MISSION_OK": "#2a9d8f",
            "PASS_RAW": "#2a9d8f",
            "PASS_SLACK": "#f4a261",
            "PATH": "#e9c46a",
            "PATH_FAIL": "#e76f51",
            "MISSION_FAIL": "#f4a261",
            "BAD_SOL": "#b56576",
            "SIM_ERROR": "#7f5539",
            "RESTORATION_FAIL": "#d62828",
            "MAX_ITER": "#f77f00",
            "MAX_CPU": "#bc6c25",
            "INTERRUPTED": "#6c757d",
            "INFEASIBLE": "#8e44ad",
            "DIVERGING": "#577590",
            "INVALID_NUM": "#6d597a",
            "SEARCH_SMALL": "#adb5bd",
            "FEASIBLE_POINT": "#90be6d",
            "SOLVE_FAIL": "#9d4edd",
            "SOLVE_ERR": "#495057",
            "GUESS_FAIL": "#495057",
            "INTERNAL_ERR": "#495057",
        }
        return palette.get(str(status), "#577590")

    @staticmethod
    def _status_display(status):
        return str(status).replace("_", " ")

    @staticmethod
    def _format_summary_value(value, unit=""):
        if not np.isfinite(value):
            return f"n/a{unit}"
        value_f = float(value)
        abs_val = abs(value_f)
        if abs_val == 0.0:
            text = "0"
        elif abs_val < 1.0e-2 or abs_val >= 1.0e4:
            text = f"{value_f:.3e}"
        else:
            text = f"{value_f:.3f}"
        return f"{text}{unit}"

    @staticmethod
    def _sanitize_label(label):
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(label))
        safe = safe.strip("_")
        return safe or "trial"

    def _trial_log_path(self, trial_index, label):
        return self.logs_dir / f"{int(trial_index) + 1:02d}_{self._sanitize_label(label)}.log"

    def _prepare_logs_dir(self):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        for old_log in self.logs_dir.glob("*.log"):
            try:
                old_log.unlink()
            except OSError:
                pass

    @staticmethod
    def _write_text_file(path, text, mode="w"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode, encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass

    @staticmethod
    def _seed_family_for_label(label):
        label_str = str(label)
        if label_str == "nominal":
            return "nominal"
        if label_str.startswith("rand_"):
            return "randomized"
        return "structured"

    def _seed_family(self, label):
        return self._seed_family_for_label(label)

    @classmethod
    def _seed_display_labels(cls, labels):
        structured_map = {
            "nominal": "Nominal",
            "time_minus": "Timing -",
            "time_plus": "Timing +",
            "state_minus": "Lateral -",
            "state_plus": "Lateral +",
        }
        display = []
        random_count = 0
        for label in labels:
            label_str = str(label)
            if label_str in structured_map:
                display.append(structured_map[label_str])
                continue
            if cls._seed_family_for_label(label_str) == "randomized":
                random_count += 1
                display.append(f"Random {random_count}")
                continue
            display.append(label_str.replace("_", " ").title())
        return display

    @classmethod
    def _seed_group_break_indices(cls, labels):
        families = [cls._seed_family_for_label(label) for label in labels]
        return [
            idx
            for idx in range(1, len(families))
            if families[idx] != families[idx - 1]
        ]

    def _perturbation_semantics(self, label, rebuild_states=False):
        family = self._seed_family(label)
        if family == "nominal":
            return "none"
        if family == "randomized":
            if rebuild_states:
                return "random_time_control_perturbation_with_state_rebuild"
            return "random_state_control_perturbation"
        if rebuild_states:
            return "deterministic_time_direction_bias_with_state_rebuild"
        return "deterministic_state_control_bias"

    @staticmethod
    def _direction_component_sigma_to_rms_deg(component_sigma):
        sigma = float(component_sigma)
        if not np.isfinite(sigma) or sigma < 0.0:
            return np.nan
        return float(np.degrees(np.sqrt(2.0) * sigma))

    def _configured_seed_metadata(self, label, params, rebuild_states=False):
        family = self._seed_family(label)
        pos_sigma = float(params.get("pos_sigma", np.nan))
        vel_sigma = float(params.get("vel_sigma", np.nan))
        mass_sigma = float(params.get("mass_sigma", np.nan))
        throttle_sigma = float(params.get("throttle_sigma", np.nan))
        direction_bias_deg = float(params.get("direction_sigma_deg", np.nan))
        direction_component_sigma = float(params.get("direction_component_sigma", np.nan))

        if rebuild_states:
            pos_sigma = 0.0
            vel_sigma = 0.0
            mass_sigma = 0.0

        direction_rms_deg_est = np.nan
        if family == "randomized":
            direction_rms_deg_est = self._direction_component_sigma_to_rms_deg(direction_component_sigma)
            direction_bias_deg = np.nan
        elif np.isfinite(direction_bias_deg):
            direction_rms_deg_est = abs(direction_bias_deg)

        if family == "nominal":
            direction_bias_deg = 0.0
            direction_component_sigma = 0.0
            if not np.isfinite(direction_rms_deg_est):
                direction_rms_deg_est = 0.0

        return {
            "seed_family": family,
            "perturbation_semantics": self._perturbation_semantics(label, rebuild_states=rebuild_states),
            "state_rebuild_applied": int(bool(rebuild_states)),
            "configured_position_perturbation": pos_sigma,
            "configured_velocity_perturbation": vel_sigma,
            "configured_mass_perturbation": mass_sigma,
            "configured_throttle_perturbation": throttle_sigma,
            "configured_direction_perturbation_deg": direction_bias_deg,
            "configured_direction_component_sigma": direction_component_sigma,
            "configured_direction_rms_deg_est": direction_rms_deg_est,
        }

    @staticmethod
    def _spread_candidate_indices(reference_idx, raw_successful_profile_indices):
        raw_idx = {int(idx) for idx in raw_successful_profile_indices}
        if reference_idx is not None:
            raw_idx.discard(int(reference_idx))
        return raw_idx

    @staticmethod
    def _runtime_summary_medians(rows):
        raw_pass = np.array(
            [float(row["runtime_s"]) for row in rows if bool(row.get("mission_success", 0)) and np.isfinite(row.get("runtime_s", np.nan))],
            dtype=float,
        )
        non_raw = np.array(
            [float(row["runtime_s"]) for row in rows if (not bool(row.get("mission_success", 0))) and np.isfinite(row.get("runtime_s", np.nan))],
            dtype=float,
        )
        failure = np.array(
            [float(row["runtime_s"]) for row in rows if (not bool(row.get("mission_success_slack", 0))) and np.isfinite(row.get("runtime_s", np.nan))],
            dtype=float,
        )
        return {
            "median_success_runtime_s": float(np.nanmedian(raw_pass)) if raw_pass.size > 0 else np.nan,
            "median_non_raw_runtime_s": float(np.nanmedian(non_raw)) if non_raw.size > 0 else np.nan,
            "median_failure_runtime_s": float(np.nanmedian(failure)) if failure.size > 0 else np.nan,
        }

    def _write_trial_log_fallback(self, log_path, variant, result):
        lines = [
            "[Trial Log Recovery]",
            f"trial={int(result.get('trial_index', -1)) + 1}",
            f"label={variant.label}",
            f"status={result.get('status', '')}",
            f"solver_status_class={result.get('solver_status_class', '')}",
            f"solver_return_status={result.get('solver_return_status', '')}",
            f"solver_iter_count={result.get('solver_iter_count', np.nan)}",
            f"runtime_s={result.get('runtime_s', np.nan)}",
            f"trajectory_available={result.get('trajectory_available', 0)}",
            "note=Full stdout/stderr trial log was unavailable; this recovery summary was written explicitly.",
            "",
        ]
        self._write_text_file(log_path, "\n".join(lines), mode="w")

    def _append_trial_log_footer(self, log_path, result):
        lines = [
            "",
            "[Trial Summary]",
            f"status={result.get('status', '')}",
            f"solver_status_class={result.get('solver_status_class', '')}",
            f"solver_return_status={result.get('solver_return_status', '')}",
            f"solver_iter_count={result.get('solver_iter_count', np.nan)}",
            f"runtime_s={result.get('runtime_s', np.nan)}",
            f"mission_success_raw={result.get('mission_success', 0)}",
            f"mission_success_slack={result.get('mission_success_slack', 0)}",
            "",
        ]
        self._write_text_file(log_path, "\n".join(lines), mode="a")

    @staticmethod
    def _rms(values):
        arr = np.asarray(values, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size <= 0:
            return np.nan
        return float(np.sqrt(np.mean(np.square(arr))))

    def _compute_realized_seed_metrics(self, base_guess, variant_guess):
        out = {
            "realized_t1_scale": np.nan,
            "realized_t2_scale": np.nan,
            "realized_t3_scale": np.nan,
            "realized_position_rms_m": np.nan,
            "realized_velocity_rms_m_s": np.nan,
            "realized_mass_rms_kg": np.nan,
            "realized_throttle_rms_abs": np.nan,
            "realized_direction_rms_deg": np.nan,
        }
        if not isinstance(base_guess, dict) or not isinstance(variant_guess, dict):
            return out

        for key, out_key in (("T1", "realized_t1_scale"), ("T2", "realized_t2_scale"), ("T3", "realized_t3_scale")):
            base_val = float(base_guess.get(key, np.nan))
            var_val = float(variant_guess.get(key, np.nan))
            if np.isfinite(base_val) and abs(base_val) > 1.0e-12 and np.isfinite(var_val):
                out[out_key] = var_val / base_val

        pos_norms = []
        vel_norms = []
        mass_deltas = []
        for key in ("X1", "X2", "X3"):
            base_hist_raw = base_guess.get(key, None)
            var_hist_raw = variant_guess.get(key, None)
            if base_hist_raw is None or var_hist_raw is None:
                continue
            base_hist = np.asarray(base_hist_raw, dtype=float)
            var_hist = np.asarray(var_hist_raw, dtype=float)
            if base_hist.ndim != 2 or var_hist.ndim != 2 or base_hist.shape != var_hist.shape or base_hist.shape[0] < 7:
                continue
            pos_norms.extend(np.linalg.norm(var_hist[0:3, :] - base_hist[0:3, :], axis=0).tolist())
            vel_norms.extend(np.linalg.norm(var_hist[3:6, :] - base_hist[3:6, :], axis=0).tolist())
            mass_deltas.extend(np.abs(var_hist[6, :] - base_hist[6, :]).tolist())

        throttle_deltas = []
        for key in ("TH1", "TH3"):
            base_th_raw = base_guess.get(key, None)
            var_th_raw = variant_guess.get(key, None)
            if base_th_raw is None or var_th_raw is None:
                continue
            base_th = np.asarray(base_th_raw, dtype=float).reshape(-1)
            var_th = np.asarray(var_th_raw, dtype=float).reshape(-1)
            if base_th.ndim != 1 or var_th.ndim != 1 or base_th.shape != var_th.shape:
                continue
            throttle_deltas.extend(np.abs(var_th - base_th).tolist())

        direction_angles_deg = []
        for key in ("TD1", "TD3"):
            base_td_raw = base_guess.get(key, None)
            var_td_raw = variant_guess.get(key, None)
            if base_td_raw is None or var_td_raw is None:
                continue
            base_td = np.asarray(base_td_raw, dtype=float)
            var_td = np.asarray(var_td_raw, dtype=float)
            if base_td.ndim != 2 or var_td.ndim != 2 or base_td.shape != var_td.shape or base_td.shape[0] != 3:
                continue
            base_norm = np.linalg.norm(base_td, axis=0)
            var_norm = np.linalg.norm(var_td, axis=0)
            good = (base_norm > 1.0e-12) & (var_norm > 1.0e-12)
            if not np.any(good):
                continue
            dots = np.sum(base_td[:, good] * var_td[:, good], axis=0) / (base_norm[good] * var_norm[good])
            angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
            direction_angles_deg.extend(angles.tolist())

        out["realized_position_rms_m"] = self._rms(pos_norms)
        out["realized_velocity_rms_m_s"] = self._rms(vel_norms)
        out["realized_mass_rms_kg"] = self._rms(mass_deltas)
        out["realized_throttle_rms_abs"] = self._rms(throttle_deltas)
        out["realized_direction_rms_deg"] = self._rms(direction_angles_deg)
        return out

    def _planned_labels(self, n_trials):
        n_use = max(MIN_WARM_START_TRIALS, int(n_trials))
        labels = ["nominal"]
        labels.extend(spec.label for spec in self.seed_factory.structured_specs[: max(0, n_use - 1)])
        next_index = len(labels)
        while len(labels) < n_use:
            labels.append(f"rand_{next_index:02d}")
            next_index += 1
        return labels

    @contextlib.contextmanager
    def _tee_trial_output(self, log_path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        current_stdout = sys.stdout
        current_stderr = sys.stderr
        base_stdout = sys.__stdout__
        base_stderr = sys.__stderr__
        stdout_fd = _stream_fileno(base_stdout)
        stderr_fd = _stream_fileno(base_stderr)
        needs_python_tee = current_stdout is not base_stdout or current_stderr is not base_stderr

        # If native file descriptors are unavailable, fall back to Python-level redirection only.
        if stdout_fd is None or stderr_fd is None:
            with open(log_path, "w", encoding="utf-8", buffering=1) as log_file:
                tee_out = TeeStream(current_stdout, log_file)
                tee_err = TeeStream(current_stderr, log_file)
                with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                    yield
            return

        with open(log_path, "wb", buffering=0) as log_file:
            write_lock = threading.Lock()
            log_text = BinaryLogTextStream(log_file, write_lock)

            def fanout(read_fd, mirror_fd):
                try:
                    while True:
                        chunk = os.read(read_fd, 8192)
                        if not chunk:
                            break
                        try:
                            os.write(mirror_fd, chunk)
                        except OSError:
                            pass
                        with write_lock:
                            log_file.write(chunk)
                            log_file.flush()
                finally:
                    try:
                        os.close(read_fd)
                    except OSError:
                        pass

            for stream in (current_stdout, current_stderr, base_stdout, base_stderr):
                flush = getattr(stream, "flush", None)
                if flush is None:
                    continue
                try:
                    flush()
                except Exception:
                    pass
            saved_stdout_fd = os.dup(stdout_fd)
            saved_stderr_fd = os.dup(stderr_fd)
            stdout_read_fd, stdout_write_fd = os.pipe()
            stderr_read_fd, stderr_write_fd = os.pipe()
            stdout_thread = threading.Thread(
                target=fanout,
                args=(stdout_read_fd, saved_stdout_fd),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=fanout,
                args=(stderr_read_fd, saved_stderr_fd),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()
            os.dup2(stdout_write_fd, stdout_fd)
            os.dup2(stderr_write_fd, stderr_fd)
            os.close(stdout_write_fd)
            os.close(stderr_write_fd)

            try:
                if needs_python_tee:
                    tee_out = TeeStream(current_stdout, log_text)
                    tee_err = TeeStream(current_stderr, log_text)
                    with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                        yield
                else:
                    yield
            finally:
                try:
                    for stream in (current_stdout, current_stderr, sys.stdout, sys.stderr):
                        flush = getattr(stream, "flush", None)
                        if flush is None:
                            continue
                        try:
                            flush()
                        except Exception:
                            pass
                finally:
                    os.dup2(saved_stdout_fd, stdout_fd)
                    os.dup2(saved_stderr_fd, stderr_fd)
                    os.close(saved_stdout_fd)
                    os.close(saved_stderr_fd)
                    stdout_thread.join(timeout=2.0)
                    stderr_thread.join(timeout=2.0)

    def _run_trial(self, trial_index, variant):
        cfg = copy.deepcopy(self.suite.base_config)
        env_cfg = copy.deepcopy(self.suite.base_env_config)
        t0 = time.time()
        log_path = self._trial_log_path(trial_index, variant.label)

        solver_ok = False
        mission_ok = False
        mission_ok_slack = False
        terminal_strict_ok = False
        path_ok = False
        q_ok = False
        g_ok = False
        raw_path_ok = False
        raw_q_ok = False
        raw_g_ok = False
        m_final = np.nan
        status = "FAILED"
        trajectory_profile = None
        solver_status_class = "SOLVE_FAIL"
        solver_return_status = ""
        solver_iter_count = np.nan
        terminal_status = ""
        terminal_valid = False
        alt_err_m = np.nan
        vel_err_m_s = np.nan
        radial_velocity_m_s = np.nan
        inclination_err_deg = np.nan
        fuel_margin_kg = np.nan
        terminal_criteria_ratio = np.nan
        max_q_pa = np.nan
        max_g = np.nan
        q_margin_pa = np.nan
        g_margin = np.nan
        q_utilization = np.nan
        g_utilization = np.nan
        path_utilization_ratio = np.nan

        try:
            self._write_text_file(
                log_path,
                "\n".join(
                    [
                        "[Trial Log Placeholder]",
                        f"trial={int(trial_index) + 1}",
                        f"label={variant.label}",
                        "",
                    ]
                ),
                mode="w",
            )
            with self._tee_trial_output(log_path):
                print(f"[Trial Log] trial={int(trial_index) + 1} label={variant.label}")
                print(f"[Trial Log] path={log_path}")
                try:
                    env = Environment(env_cfg)
                    veh = Vehicle(cfg, env)
                    opt_res = solve_optimal_trajectory(
                        cfg,
                        veh,
                        env,
                        initial_guess_override=variant.guess,
                    )
                    solver_ok = bool(opt_res.get("success", False))
                    solver_return_status = str(opt_res.get("solver_return_status", ""))
                    solver_status_class = self._classify_solver_return_status(solver_return_status)
                    iter_count = opt_res.get("solver_iter_count", np.nan)
                    try:
                        solver_iter_count = int(iter_count)
                    except Exception:
                        solver_iter_count = np.nan
                    X3 = opt_res.get("X3", None)
                    if solver_ok and isinstance(X3, np.ndarray) and X3.ndim == 2 and X3.shape[0] >= 7 and X3.shape[1] >= 1:
                        m_try = float(X3[6, -1])
                        if np.isfinite(m_try) and m_try > 0.0:
                            m_final = m_try

                    if solver_ok and np.isfinite(m_final):
                        try:
                            sim_res = run_simulation(opt_res, veh, cfg, rtol=1e-9, atol=1e-12)
                            trajectory_profile = self._build_trajectory_profile(sim_res, env_cfg)
                            terminal = self.evaluate_terminal_state(sim_res, cfg, env_cfg)
                            traj = self.suite._trajectory_diagnostics(sim_res, veh, cfg)
                            path = self.suite._evaluate_path_compliance(
                                traj,
                                cfg,
                                q_slack_pa=self.suite.path_q_slack_pa,
                                g_slack=self.suite.path_g_slack,
                            )
                            path_raw = self.suite._evaluate_path_compliance(
                                traj,
                                cfg,
                                q_slack_pa=0.0,
                                g_slack=0.0,
                            )
                            terminal_status = str(terminal.get("status", ""))
                            terminal_valid = bool(terminal.get("terminal_valid", False))
                            alt_err_m = float(terminal.get("alt_err_m", np.nan))
                            vel_err_m_s = float(terminal.get("vel_err_m_s", np.nan))
                            radial_velocity_m_s = float(terminal.get("radial_velocity_m_s", np.nan))
                            inclination_err_deg = float(terminal.get("inclination_err_deg", np.nan))
                            fuel_margin_kg = float(terminal.get("fuel_margin_kg", np.nan))
                            terminal_criteria_ratio = self._terminal_criteria_ratio(terminal)
                            max_q_pa = float(path.get("max_q_pa", np.nan))
                            max_g = float(path.get("max_g", np.nan))
                            q_limit_pa = float(path.get("q_limit_pa", cfg.max_q_limit))
                            g_limit = float(path.get("g_limit", cfg.max_g_load))
                            if np.isfinite(max_q_pa):
                                q_margin_pa = q_limit_pa - max_q_pa
                            if np.isfinite(max_g):
                                g_margin = g_limit - max_g
                            if np.isfinite(max_q_pa) and np.isfinite(q_limit_pa) and q_limit_pa > 0.0:
                                q_utilization = max_q_pa / q_limit_pa
                            if np.isfinite(max_g) and np.isfinite(g_limit) and g_limit > 0.0:
                                g_utilization = max_g / g_limit
                            valid_path_util = [val for val in (q_utilization, g_utilization) if np.isfinite(val)]
                            if valid_path_util:
                                path_utilization_ratio = float(max(valid_path_util))
                            terminal_strict_ok = bool(terminal.get("strict_ok", False))
                            path_ok = bool(path.get("path_ok", False))
                            q_ok = bool(path.get("q_ok", False))
                            g_ok = bool(path.get("g_ok", False))
                            raw_path_ok = bool(path_raw.get("path_ok", False))
                            raw_q_ok = bool(path_raw.get("q_ok", False))
                            raw_g_ok = bool(path_raw.get("g_ok", False))
                            mission_ok = bool(terminal_strict_ok and raw_path_ok)
                            mission_ok_slack = bool(terminal_strict_ok and path_ok)
                            if mission_ok:
                                status = "PASS_RAW"
                            elif mission_ok_slack:
                                status = "PASS_SLACK"
                            elif terminal_strict_ok and not path_ok:
                                status = "PATH_FAIL"
                            elif not terminal_strict_ok:
                                status = terminal.get("status", "MISS")
                            else:
                                status = "MISSION_FAIL"
                        except Exception:
                            traceback.print_exc()
                            status = "SIM_ERROR"
                    elif solver_ok:
                        solver_ok = False
                        status = "BAD_SOL"
                    else:
                        status = solver_status_class
                except Exception:
                    traceback.print_exc()
                    status = "SOLVE_ERR"
                    solver_status_class = "SOLVE_ERR"
                print(f"[Trial Log] status={status}")
        except Exception:
            status = "SOLVE_ERR"
            solver_status_class = "SOLVE_ERR"

        result = {
            "trial_index": int(trial_index),
            "label": str(variant.label),
            "params": variant.params.as_dict(),
            "solver_success": int(solver_ok),
            "mission_success": int(mission_ok),
            "mission_success_slack": int(mission_ok_slack),
            "terminal_strict_success": int(terminal_strict_ok),
            "path_ok": int(path_ok),
            "q_ok": int(q_ok),
            "g_ok": int(g_ok),
            "raw_path_ok": int(raw_path_ok),
            "raw_q_ok": int(raw_q_ok),
            "raw_g_ok": int(raw_g_ok),
            "status": str(status),
            "final_mass_kg": float(m_final) if np.isfinite(m_final) else np.nan,
            "runtime_s": float(time.time() - t0),
            "solver_status_class": str(solver_status_class),
            "solver_return_status": solver_return_status,
            "solver_iter_count": solver_iter_count,
            "log_path": str(log_path),
            "log_available": 0,
            "trajectory_available": int(isinstance(trajectory_profile, dict)),
            "terminal_status": str(terminal_status),
            "terminal_valid": int(terminal_valid),
            "alt_err_m": alt_err_m,
            "vel_err_m_s": vel_err_m_s,
            "radial_velocity_m_s": radial_velocity_m_s,
            "inclination_err_deg": inclination_err_deg,
            "fuel_margin_kg": fuel_margin_kg,
            "terminal_criteria_ratio": terminal_criteria_ratio,
            "max_q_pa": max_q_pa,
            "max_g": max_g,
            "q_margin_pa": q_margin_pa,
            "g_margin": g_margin,
            "q_utilization": q_utilization,
            "g_utilization": g_utilization,
            "path_utilization_ratio": path_utilization_ratio,
            "trajectory_profile": trajectory_profile,
        }
        if not log_path.exists() or log_path.stat().st_size <= 0:
            self._write_trial_log_fallback(log_path, variant, result)
        self._append_trial_log_footer(log_path, result)
        result["log_available"] = int(log_path.exists() and log_path.stat().st_size > 0)
        return result

    def analyze(self, n_trials=DEFAULT_WARM_START_TRIALS):
        debug._print_sub_header("Warm-Start Multi-Seed Robustness")
        n_use = max(MIN_WARM_START_TRIALS, int(n_trials))
        rng = np.random.default_rng(self.suite.seed_sequence.spawn(1)[0])
        planned_labels = self._planned_labels(n_use)
        self._prepare_logs_dir()

        rows = []
        solver_success_flags = []
        mission_success_flags = []
        mission_success_slack_flags = []
        masses_scored = []
        runtimes = []
        labels = []
        trajectory_profiles = []
        trial_results = {}
        variants_by_index = {}

        print(
            f"{'Trial':<7} | {'Guess':<11} | {'Time x':<8} | {'Pos %':<7} | {'Vel %':<7} | "
            f"{'Mass %':<8} | {'Throt %':<8} | {'Dir est':<10} | {'Solve':<5} | {'Mission':<7} | {'Status':<17} | "
            f"{'Final Mass (kg)':<15} | {'Runtime (s)':<10}"
        )
        print("-" * 174)

        self._print_progress("Warm-start setup", 0, 3)
        print("  Building nominal guidance warm start...")
        guidance_guess = self._build_nominal_guidance_guess()
        self._print_progress("Warm-start setup", 1, 3)

        structured_count = 0
        variants = []
        state_rebuilder_active = False
        if guidance_guess is not None:
            state_rebuilder = lambda guess: self._rebuild_guess_states(guess, self.suite.base_config, self.suite.base_env_config)
            state_rebuilder_active = True
            structured_count = min(len(self.seed_factory.structured_specs), max(0, n_use - 1))
            variants = self.seed_factory.build_trial_variants(
                guidance_guess,
                self.suite.base_config,
                rng,
                n_use,
                state_rebuilder=state_rebuilder,
            )
        print(f"  Prepared {structured_count + int(guidance_guess is not None)} deterministic seed(s).")
        self._print_progress("Warm-start setup", 2, 3)

        if guidance_guess is None:
            for i in range(n_use):
                label = planned_labels[i]
                params = SeedParams(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                trial_results[i] = self._empty_result(i, label, params, "GUESS_FAIL", runtime_s=0.0)
        else:
            print(f"  Queued {len(variants)} trial(s) for evaluation.")
        self._print_progress("Warm-start setup", 3, 3)

        total_trials = len(variants)
        for done_trials, variant in enumerate(variants, start=1):
            variants_by_index[done_trials - 1] = variant
            print(f"  [Run {done_trials}/{total_trials}] Starting: {variant.label}")
            print(f"  [Run {done_trials}/{total_trials}] Log: {self._trial_log_path(done_trials - 1, variant.label)}")
            result = self._run_trial(done_trials - 1, variant)
            trial_results[done_trials - 1] = result
            dt_trial = float(result["runtime_s"]) if np.isfinite(result["runtime_s"]) else np.nan
            status_trial = str(result["status"])
            if np.isfinite(dt_trial):
                print(f"  [Run {done_trials}/{total_trials}] Finished: {variant.label} ({status_trial}, {dt_trial:.2f}s)")
            else:
                print(f"  [Run {done_trials}/{total_trials}] Finished: {variant.label} ({status_trial})")
            self._print_progress("Multistart trials", done_trials, total_trials)

        for i in range(n_use):
            r = trial_results.get(i)
            if r is None:
                r = self._empty_result(
                    i,
                    planned_labels[i],
                    SeedParams(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
                    "INTERNAL_ERR",
                )

            params = r["params"]
            m_final = float(r["final_mass_kg"]) if np.isfinite(r["final_mass_kg"]) else np.nan
            solver_ok = int(r["solver_success"])
            mission_ok = int(r["mission_success"])
            mission_ok_slack = int(r.get("mission_success_slack", 0))
            terminal_strict_ok = int(r["terminal_strict_success"])
            path_ok = int(r["path_ok"])
            q_ok = int(r["q_ok"])
            g_ok = int(r["g_ok"])
            raw_path_ok = int(r.get("raw_path_ok", 0))
            raw_q_ok = int(r.get("raw_q_ok", 0))
            raw_g_ok = int(r.get("raw_g_ok", 0))
            status = str(r["status"])
            dt = float(r["runtime_s"]) if np.isfinite(r["runtime_s"]) else np.nan
            label = str(r["label"])
            trajectory_profiles.append(r.get("trajectory_profile", None))
            variant = variants_by_index.get(i)
            realized_metrics = self._compute_realized_seed_metrics(guidance_guess, variant.guess) if guidance_guess is not None and variant is not None else {}
            variant_uses_state_rebuild = bool(state_rebuilder_active and label != "nominal")
            configured_metrics = self._configured_seed_metadata(
                label,
                params,
                rebuild_states=variant_uses_state_rebuild,
            )
            direction_display = configured_metrics["configured_direction_rms_deg_est"]

            labels.append(label)
            masses_scored.append(m_final if mission_ok and np.isfinite(m_final) else np.nan)
            runtimes.append(dt)
            solver_success_flags.append(solver_ok)
            mission_success_flags.append(mission_ok)
            mission_success_slack_flags.append(mission_ok_slack)

            rows.append(
                {
                    "trial": i + 1,
                    "label": label,
                    "seed_family": configured_metrics["seed_family"],
                    "perturbation_semantics": configured_metrics["perturbation_semantics"],
                    "state_rebuild_applied": configured_metrics["state_rebuild_applied"],
                    "time_scale": params["time_scale"],
                    "t2_scale": params["t2_scale"],
                    "configured_position_perturbation": configured_metrics["configured_position_perturbation"],
                    "configured_velocity_perturbation": configured_metrics["configured_velocity_perturbation"],
                    "configured_mass_perturbation": configured_metrics["configured_mass_perturbation"],
                    "configured_throttle_perturbation": configured_metrics["configured_throttle_perturbation"],
                    "configured_direction_perturbation_deg": configured_metrics["configured_direction_perturbation_deg"],
                    "configured_direction_component_sigma": configured_metrics["configured_direction_component_sigma"],
                    "configured_direction_rms_deg_est": configured_metrics["configured_direction_rms_deg_est"],
                    "realized_t1_scale": realized_metrics.get("realized_t1_scale", np.nan),
                    "realized_t2_scale": realized_metrics.get("realized_t2_scale", np.nan),
                    "realized_t3_scale": realized_metrics.get("realized_t3_scale", np.nan),
                    "realized_position_rms_m": realized_metrics.get("realized_position_rms_m", np.nan),
                    "realized_velocity_rms_m_s": realized_metrics.get("realized_velocity_rms_m_s", np.nan),
                    "realized_mass_rms_kg": realized_metrics.get("realized_mass_rms_kg", np.nan),
                    "realized_throttle_rms_abs": realized_metrics.get("realized_throttle_rms_abs", np.nan),
                    "realized_direction_rms_deg": realized_metrics.get("realized_direction_rms_deg", np.nan),
                    "solver_success": solver_ok,
                    "mission_success": mission_ok,
                    "mission_success_slack": mission_ok_slack,
                    "terminal_strict_success": terminal_strict_ok,
                    "path_ok": path_ok,
                    "q_ok": q_ok,
                    "g_ok": g_ok,
                    "raw_path_ok": raw_path_ok,
                    "raw_q_ok": raw_q_ok,
                    "raw_g_ok": raw_g_ok,
                    "status": status,
                    "solver_status_class": str(r.get("solver_status_class", "")),
                    "final_mass_kg": m_final,
                    "runtime_s": dt,
                    "log_path": str(r.get("log_path", "")),
                    "log_available": int(r.get("log_available", 0)),
                    "solver_return_status": str(r.get("solver_return_status", "")),
                    "solver_iter_count": r.get("solver_iter_count", np.nan),
                    "trajectory_available": int(r.get("trajectory_available", 0)),
                    "terminal_status": str(r.get("terminal_status", "")),
                    "terminal_valid": int(r.get("terminal_valid", 0)),
                    "alt_err_m": r.get("alt_err_m", np.nan),
                    "vel_err_m_s": r.get("vel_err_m_s", np.nan),
                    "radial_velocity_m_s": r.get("radial_velocity_m_s", np.nan),
                    "inclination_err_deg": r.get("inclination_err_deg", np.nan),
                    "fuel_margin_kg": r.get("fuel_margin_kg", np.nan),
                    "terminal_criteria_ratio": r.get("terminal_criteria_ratio", np.nan),
                    "max_q_pa": r.get("max_q_pa", np.nan),
                    "max_g": r.get("max_g", np.nan),
                    "q_margin_pa": r.get("q_margin_pa", np.nan),
                    "g_margin": r.get("g_margin", np.nan),
                    "q_utilization": r.get("q_utilization", np.nan),
                    "g_utilization": r.get("g_utilization", np.nan),
                    "path_utilization_ratio": r.get("path_utilization_ratio", np.nan),
                }
            )
            print(
                f"{i+1:<7} | {label:<11} | {params['time_scale']:<8.3f} | "
                f"{100.0 * abs(configured_metrics['configured_position_perturbation']):<7.2f} | {100.0 * abs(configured_metrics['configured_velocity_perturbation']):<7.2f} | "
                f"{100.0 * abs(configured_metrics['configured_mass_perturbation']):<8.2f} | {100.0 * abs(configured_metrics['configured_throttle_perturbation']):<8.2f} | "
                f"{direction_display:<10.2f} | {solver_ok:<5d} | {mission_ok:<7d} | "
                f"{status:<17} | {m_final:<15.2f} | {dt:<10.2f}"
            )

        solver_rate = float(np.mean(solver_success_flags)) if solver_success_flags else 0.0
        success_rate = float(np.mean(mission_success_flags)) if mission_success_flags else 0.0
        success_rate_slack = float(np.mean(mission_success_slack_flags)) if mission_success_slack_flags else 0.0
        valid_masses = np.array([m for m in masses_scored if np.isfinite(m)], dtype=float)
        if len(valid_masses) > 0:
            best_mass = float(np.max(valid_masses))
            mass_gap_kg = [(best_mass - m) if np.isfinite(m) else np.nan for m in masses_scored]
        else:
            best_mass = np.nan
            mass_gap_kg = [np.nan for _ in masses_scored]
        mass_spread = float(np.max(valid_masses) - np.min(valid_masses)) if len(valid_masses) > 1 else np.nan

        print(f"\nSolver convergence rate: {solver_rate:.1%}")
        print(f"Raw mission success rate: {success_rate:.1%}")
        print(f"Slack-accepted mission success rate: {success_rate_slack:.1%}")
        if np.isfinite(best_mass):
            print(f"Best final mass among mission-valid runs: {best_mass:.3f} kg")
        if np.isfinite(mass_spread):
            print(f"Final-mass spread across mission-valid runs: {mass_spread:.3f} kg")
        if success_rate >= 0.8 and (not np.isfinite(mass_spread) or mass_spread < 300.0):
            print(f">>> {debug.Style.GREEN}PASS: Multi-start test supports robust local optimum.{debug.Style.RESET}")
        else:
            print(f">>> {debug.Style.YELLOW}WARN: Multi-start reveals convergence/objective sensitivity.{debug.Style.RESET}")

        self.suite._save_table(
            "randomized_multistart",
            [
                "trial",
                "label",
                "seed_family",
                "perturbation_semantics",
                "state_rebuild_applied",
                "time_scale",
                "t2_scale",
                "configured_position_perturbation",
                "configured_velocity_perturbation",
                "configured_mass_perturbation",
                "configured_throttle_perturbation",
                "configured_direction_perturbation_deg",
                "configured_direction_component_sigma",
                "configured_direction_rms_deg_est",
                "realized_t1_scale",
                "realized_t2_scale",
                "realized_t3_scale",
                "realized_position_rms_m",
                "realized_velocity_rms_m_s",
                "realized_mass_rms_kg",
                "realized_throttle_rms_abs",
                "realized_direction_rms_deg",
                "solver_success",
                "mission_success",
                "mission_success_slack",
                "terminal_strict_success",
                "path_ok",
                "q_ok",
                "g_ok",
                "raw_path_ok",
                "raw_q_ok",
                "raw_g_ok",
                "status",
                "solver_status_class",
                "trajectory_available",
                "terminal_status",
                "terminal_valid",
                "alt_err_m",
                "vel_err_m_s",
                "radial_velocity_m_s",
                "inclination_err_deg",
                "fuel_margin_kg",
                "terminal_criteria_ratio",
                "final_mass_kg",
                "final_mass_scored_kg",
                "final_mass_gap_to_best_kg",
                "max_q_pa",
                "max_g",
                "q_margin_pa",
                "g_margin",
                "q_utilization",
                "g_utilization",
                "path_utilization_ratio",
                "runtime_s",
                "log_path",
                "log_available",
                "solver_return_status",
                "solver_iter_count",
            ],
            [
                (
                    rows[i]["trial"],
                    rows[i]["label"],
                    rows[i]["seed_family"],
                    rows[i]["perturbation_semantics"],
                    rows[i]["state_rebuild_applied"],
                    rows[i]["time_scale"],
                    rows[i]["t2_scale"],
                    rows[i]["configured_position_perturbation"],
                    rows[i]["configured_velocity_perturbation"],
                    rows[i]["configured_mass_perturbation"],
                    rows[i]["configured_throttle_perturbation"],
                    rows[i]["configured_direction_perturbation_deg"],
                    rows[i]["configured_direction_component_sigma"],
                    rows[i]["configured_direction_rms_deg_est"],
                    rows[i]["realized_t1_scale"],
                    rows[i]["realized_t2_scale"],
                    rows[i]["realized_t3_scale"],
                    rows[i]["realized_position_rms_m"],
                    rows[i]["realized_velocity_rms_m_s"],
                    rows[i]["realized_mass_rms_kg"],
                    rows[i]["realized_throttle_rms_abs"],
                    rows[i]["realized_direction_rms_deg"],
                    rows[i]["solver_success"],
                    rows[i]["mission_success"],
                    rows[i]["mission_success_slack"],
                    rows[i]["terminal_strict_success"],
                    rows[i]["path_ok"],
                    rows[i]["q_ok"],
                    rows[i]["g_ok"],
                    rows[i]["raw_path_ok"],
                    rows[i]["raw_q_ok"],
                    rows[i]["raw_g_ok"],
                    rows[i]["status"],
                    rows[i]["solver_status_class"],
                    rows[i]["trajectory_available"],
                    rows[i]["terminal_status"],
                    rows[i]["terminal_valid"],
                    rows[i]["alt_err_m"],
                    rows[i]["vel_err_m_s"],
                    rows[i]["radial_velocity_m_s"],
                    rows[i]["inclination_err_deg"],
                    rows[i]["fuel_margin_kg"],
                    rows[i]["terminal_criteria_ratio"],
                    rows[i]["final_mass_kg"],
                    masses_scored[i],
                    mass_gap_kg[i],
                    rows[i]["max_q_pa"],
                    rows[i]["max_g"],
                    rows[i]["q_margin_pa"],
                    rows[i]["g_margin"],
                    rows[i]["q_utilization"],
                    rows[i]["g_utilization"],
                    rows[i]["path_utilization_ratio"],
                    rows[i]["runtime_s"],
                    rows[i]["log_path"],
                    rows[i]["log_available"],
                    rows[i]["solver_return_status"],
                    rows[i]["solver_iter_count"],
                )
                for i in range(len(rows))
            ],
        )

        status_counts = Counter(str(row["status"]) for row in rows)
        solver_status_counts = Counter(str(row["solver_status_class"]) for row in rows)
        available_profile_indices = [i for i, profile in enumerate(trajectory_profiles) if isinstance(profile, dict)]
        raw_successful_profile_indices = [i for i in available_profile_indices if bool(rows[i]["mission_success"])]
        slack_successful_profile_indices = [i for i in available_profile_indices if bool(rows[i]["mission_success_slack"])]
        slack_only_profile_indices = [i for i in slack_successful_profile_indices if not bool(rows[i]["mission_success"])]
        omitted_replay_labels = [rows[i]["label"] for i in range(len(rows)) if i not in available_profile_indices]
        omitted_success_envelope_labels = [rows[i]["label"] for i in range(len(rows)) if i not in raw_successful_profile_indices]

        reference_idx = None
        for candidate_idx in [0]:
            if candidate_idx in raw_successful_profile_indices:
                reference_idx = candidate_idx
                break
        if reference_idx is None and raw_successful_profile_indices:
            reference_idx = raw_successful_profile_indices[0]
        if reference_idx is None and 0 in available_profile_indices:
            reference_idx = 0
        if reference_idx is None and available_profile_indices:
            reference_idx = available_profile_indices[0]

        trajectory_rows = []
        delta_rows = []
        max_position_sep_km = np.nan
        max_alt_delta_km = np.nan
        max_vel_delta_m_s = np.nan
        max_terminal_criteria_ratio = np.nan
        max_path_utilization_ratio = np.nan
        median_success_runtime_s = np.nan
        median_non_raw_runtime_s = np.nan
        median_failure_runtime_s = np.nan
        reference_label = labels[reference_idx] if reference_idx is not None else "n/a"

        if reference_idx is not None:
            ref_profile = trajectory_profiles[reference_idx]
            ref_alt = np.asarray(ref_profile["ellipsoidal_altitude_km"], dtype=float)
            ref_downrange = np.asarray(ref_profile["downrange_km"], dtype=float)
            ref_vel = np.asarray(ref_profile["velocity_m_s"], dtype=float)
            ref_mass = np.asarray(ref_profile["mass_kg"], dtype=float)
            ref_pos = np.vstack(
                [
                    np.asarray(ref_profile["x_m"], dtype=float),
                    np.asarray(ref_profile["y_m"], dtype=float),
                    np.asarray(ref_profile["z_m"], dtype=float),
                ]
            )
        else:
            ref_profile = None
            ref_alt = np.array([], dtype=float)
            ref_downrange = np.array([], dtype=float)
            ref_vel = np.array([], dtype=float)
            ref_mass = np.array([], dtype=float)
            ref_pos = np.empty((3, 0), dtype=float)

        max_position_candidates = []
        max_alt_candidates = []
        max_vel_candidates = []
        raw_spread_candidate_indices = self._spread_candidate_indices(reference_idx, raw_successful_profile_indices)

        for i, profile in enumerate(trajectory_profiles):
            if not isinstance(profile, dict):
                continue
            progress = np.asarray(profile["progress"], dtype=float)
            time_s = np.asarray(profile["time_s"], dtype=float)
            downrange_km = np.asarray(profile["downrange_km"], dtype=float)
            altitude_km = np.asarray(profile["ellipsoidal_altitude_km"], dtype=float)
            velocity_m_s = np.asarray(profile["velocity_m_s"], dtype=float)
            mass_kg = np.asarray(profile["mass_kg"], dtype=float)
            x_m = np.asarray(profile["x_m"], dtype=float)
            y_m = np.asarray(profile["y_m"], dtype=float)
            z_m = np.asarray(profile["z_m"], dtype=float)

            for k in range(len(progress)):
                trajectory_rows.append(
                    (
                        rows[i]["trial"],
                        rows[i]["label"],
                        rows[i]["status"],
                        rows[i]["solver_success"],
                        rows[i]["mission_success"],
                        rows[i]["mission_success_slack"],
                        rows[i]["raw_path_ok"],
                        rows[i]["path_ok"],
                        progress[k],
                        time_s[k],
                        downrange_km[k],
                        altitude_km[k],
                        velocity_m_s[k],
                        mass_kg[k],
                        x_m[k],
                        y_m[k],
                        z_m[k],
                    )
                )

            if ref_profile is None:
                continue

            delta_alt_km = altitude_km - ref_alt
            delta_downrange_km = downrange_km - ref_downrange
            delta_vel_m_s = velocity_m_s - ref_vel
            delta_mass_kg = mass_kg - ref_mass
            pos_sep_km = np.linalg.norm(np.vstack([x_m, y_m, z_m]) - ref_pos, axis=0) / 1000.0

            if i in raw_spread_candidate_indices:
                if np.any(np.isfinite(pos_sep_km)):
                    max_position_candidates.append(float(np.nanmax(pos_sep_km)))
                if np.any(np.isfinite(delta_alt_km)):
                    max_alt_candidates.append(float(np.nanmax(np.abs(delta_alt_km))))
                if np.any(np.isfinite(delta_vel_m_s)):
                    max_vel_candidates.append(float(np.nanmax(np.abs(delta_vel_m_s))))

            for k in range(len(progress)):
                delta_rows.append(
                    (
                        rows[i]["trial"],
                        rows[i]["label"],
                        rows[i]["status"],
                        rows[i]["solver_success"],
                        rows[i]["mission_success"],
                        rows[i]["mission_success_slack"],
                        rows[i]["raw_path_ok"],
                        rows[i]["path_ok"],
                        progress[k],
                        delta_downrange_km[k],
                        delta_alt_km[k],
                        delta_vel_m_s[k],
                        delta_mass_kg[k],
                        pos_sep_km[k],
                    )
                )

        for i, profile in enumerate(trajectory_profiles):
            if isinstance(profile, dict):
                continue
            trajectory_rows.append(
                (
                    rows[i]["trial"],
                    rows[i]["label"],
                    rows[i]["status"],
                    rows[i]["solver_success"],
                    rows[i]["mission_success"],
                    rows[i]["mission_success_slack"],
                    rows[i]["raw_path_ok"],
                    rows[i]["path_ok"],
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                )
            )
            delta_rows.append(
                (
                    rows[i]["trial"],
                    rows[i]["label"],
                    rows[i]["status"],
                    rows[i]["solver_success"],
                    rows[i]["mission_success"],
                    rows[i]["mission_success_slack"],
                    rows[i]["raw_path_ok"],
                    rows[i]["path_ok"],
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                )
            )

        if max_position_candidates:
            max_position_sep_km = float(np.nanmax(np.array(max_position_candidates, dtype=float)))
        if max_alt_candidates:
            max_alt_delta_km = float(np.nanmax(np.array(max_alt_candidates, dtype=float)))
        if max_vel_candidates:
            max_vel_delta_m_s = float(np.nanmax(np.array(max_vel_candidates, dtype=float)))

        terminal_ratio_values = np.array(
            [float(row.get("terminal_criteria_ratio", np.nan)) for row in rows if np.isfinite(row.get("terminal_criteria_ratio", np.nan))],
            dtype=float,
        )
        if terminal_ratio_values.size > 0:
            max_terminal_criteria_ratio = float(np.nanmax(terminal_ratio_values))

        path_util_values = np.array(
            [float(row.get("path_utilization_ratio", np.nan)) for row in rows if np.isfinite(row.get("path_utilization_ratio", np.nan))],
            dtype=float,
        )
        if path_util_values.size > 0:
            max_path_utilization_ratio = float(np.nanmax(path_util_values))

        runtime_medians = self._runtime_summary_medians(rows)
        median_success_runtime_s = float(runtime_medians["median_success_runtime_s"])
        median_non_raw_runtime_s = float(runtime_medians["median_non_raw_runtime_s"])
        median_failure_runtime_s = float(runtime_medians["median_failure_runtime_s"])

        summary_rows = [
            ("reference_label", reference_label),
            ("solver_rate", solver_rate),
            ("mission_success_rate_raw", success_rate),
            ("mission_success_rate_slack", success_rate_slack),
            ("log_available_count", int(sum(int(row.get("log_available", 0)) for row in rows))),
            ("trajectory_available_count", len(available_profile_indices)),
            ("successful_replay_count_raw", len(raw_successful_profile_indices)),
            ("successful_replay_count_slack", len(slack_successful_profile_indices)),
            ("slack_only_replay_count", len(slack_only_profile_indices)),
            ("omitted_replay_labels", ",".join(omitted_replay_labels)),
            ("omitted_success_envelope_labels", ",".join(omitted_success_envelope_labels)),
            ("mass_spread_kg", mass_spread),
            ("max_position_separation_km", max_position_sep_km),
            ("max_altitude_delta_km", max_alt_delta_km),
            ("max_velocity_delta_m_s", max_vel_delta_m_s),
            ("max_terminal_criteria_ratio", max_terminal_criteria_ratio),
            ("max_path_utilization_ratio", max_path_utilization_ratio),
            ("median_success_runtime_s", median_success_runtime_s),
            ("median_non_raw_runtime_s", median_non_raw_runtime_s),
            ("median_failure_runtime_s", median_failure_runtime_s),
        ]
        summary_rows.extend(
            (f"status_count_{status_key.lower()}", count)
            for status_key, count in sorted(status_counts.items())
        )
        summary_rows.extend(
            (f"solver_status_count_{status_key.lower()}", count)
            for status_key, count in sorted(solver_status_counts.items())
        )

        if trajectory_rows:
            self.suite._save_table(
                "randomized_multistart_trajectories",
                [
                    "trial",
                    "label",
                    "status",
                    "solver_success",
                    "mission_success",
                    "mission_success_slack",
                    "raw_path_ok",
                    "path_ok_slack",
                    "progress",
                    "time_s",
                    "downrange_km",
                    "ellipsoidal_altitude_km",
                    "velocity_m_s",
                    "mass_kg",
                    "x_m",
                    "y_m",
                    "z_m",
                ],
                trajectory_rows,
            )
        if delta_rows:
            self.suite._save_table(
                "randomized_multistart_deltas",
                [
                    "trial",
                    "label",
                    "status",
                    "solver_success",
                    "mission_success",
                    "mission_success_slack",
                    "raw_path_ok",
                    "path_ok_slack",
                    "progress",
                    "delta_downrange_km",
                    "delta_altitude_km",
                    "delta_velocity_m_s",
                    "delta_mass_kg",
                    "position_separation_km",
                ],
                delta_rows,
            )
        self.suite._save_table(
            "randomized_multistart_trajectory_summary",
            ["metric", "value"],
            summary_rows,
        )

        if reference_idx is not None:
            print(f"Reference trajectory for comparisons: {reference_label}")
            if np.isfinite(max_position_sep_km):
                print(f"Max position separation vs {reference_label}: {max_position_sep_km:.3f} km")
            if np.isfinite(max_alt_delta_km):
                print(f"Max altitude delta vs {reference_label}: {max_alt_delta_km:.3f} km")
            if np.isfinite(max_vel_delta_m_s):
                print(f"Max velocity delta vs {reference_label}: {max_vel_delta_m_s:.3f} m/s")

        fig = plt.figure(figsize=(12.0, 8.4))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.35, 0.9, 1.0])
        ax_status = fig.add_subplot(gs[0, 0])
        ax_path = fig.add_subplot(gs[1, 0])
        ax_terminal = fig.add_subplot(gs[2, 0], sharex=ax_path)

        seed_positions = np.arange(len(rows), dtype=float)
        raw_seed_labels = [str(row["label"]) for row in rows]
        seed_labels = self._seed_display_labels(raw_seed_labels)
        group_breaks = self._seed_group_break_indices(raw_seed_labels)
        runtime_vals = np.array([float(row["runtime_s"]) if np.isfinite(row["runtime_s"]) else 0.0 for row in rows], dtype=float)
        finite_runtime_vals = runtime_vals[np.isfinite(runtime_vals)]
        max_runtime = float(np.nanmax(finite_runtime_vals)) if finite_runtime_vals.size > 0 else 1.0
        if not np.isfinite(max_runtime) or max_runtime <= 0.0:
            max_runtime = 1.0
        runtime_pad = 0.06 * max_runtime
        status_colors = [self._outcome_color(row["status"]) for row in rows]

        ax_status.barh(seed_positions, runtime_vals, color=status_colors, edgecolor="0.25", linewidth=0.7, alpha=0.9)
        ax_status.set_yticks(seed_positions)
        ax_status.set_yticklabels(seed_labels)
        ax_status.invert_yaxis()
        ax_status.set_xlim(0.0, max_runtime * 1.35)
        ax_status.set_xlabel("Trial runtime (s)")
        ax_status.grid(True, axis="x", alpha=0.25)
        ax_status.text(0.01, 0.98, "a)", transform=ax_status.transAxes, ha="left", va="top", fontsize=10, fontweight="bold", color="0.2")
        for break_idx in group_breaks:
            ax_status.axhline(break_idx - 0.5, color="0.65", linestyle=":", linewidth=1.0, zorder=0)

        for i, row in enumerate(rows):
            iter_info = row["solver_iter_count"]
            status_label = self._status_display(row["status"])
            if status_label == "PASS RAW" and np.isfinite(iter_info):
                annotation = f"{int(iter_info)} it"
            elif np.isfinite(iter_info):
                annotation = f"{status_label} | {int(iter_info)} it"
            else:
                solver_note = str(row["solver_return_status"]).replace("_", " ").strip()
                annotation = status_label if status_label else "n/a"
                if solver_note:
                    annotation += f" | {solver_note}"
            x_text = runtime_vals[i] + runtime_pad
            if not np.isfinite(x_text):
                x_text = runtime_pad
            ax_status.text(x_text, seed_positions[i], annotation, va="center", fontsize=8)

        slack_only_count = int(sum(1 for row in rows if bool(row["mission_success_slack"]) and not bool(row["mission_success"])))
        solver_fail_count = int(len(rows) - sum(solver_success_flags))
        non_raw_labels = [str(row["label"]) for row in rows if str(row["status"]) != "PASS_RAW"]
        non_raw_preview = ", ".join(non_raw_labels[:4]) if non_raw_labels else "none"
        if len(non_raw_labels) > 4:
            non_raw_preview += f", +{len(non_raw_labels) - 4} more"
        q_max_kpa = np.array([float(row.get("max_q_pa", np.nan)) / 1000.0 for row in rows], dtype=float)
        g_max_series = np.array([float(row.get("max_g", np.nan)) for row in rows], dtype=float)
        alt_err_abs_m = np.abs(np.array([float(row.get("alt_err_m", np.nan)) for row in rows], dtype=float))
        vel_err_abs_m_s = np.abs(np.array([float(row.get("vel_err_m_s", np.nan)) for row in rows], dtype=float))
        q_max_summary = float(np.nanmax(q_max_kpa)) if np.any(np.isfinite(q_max_kpa)) else np.nan
        g_max_summary = float(np.nanmax(g_max_series)) if np.any(np.isfinite(g_max_series)) else np.nan
        alt_err_summary = float(np.nanmax(alt_err_abs_m)) if np.any(np.isfinite(alt_err_abs_m)) else np.nan
        vel_err_summary = float(np.nanmax(vel_err_abs_m_s)) if np.any(np.isfinite(vel_err_abs_m_s)) else np.nan
        count_lines = [
            f"{self._status_display(status_key)}: {count}"
            for status_key, count in sorted(status_counts.items(), key=lambda item: (-item[1], item[0]))
        ]
        count_lines.extend(
            [
                f"Raw pass: {sum(mission_success_flags)}/{n_use}",
                f"Solver success: {sum(solver_success_flags)}/{n_use}",
                f"Median raw-pass runtime: {self._format_summary_value(median_success_runtime_s, ' s')}",
                f"Max Q: {self._format_summary_value(q_max_summary, ' kPa')}",
                f"Max g: {self._format_summary_value(g_max_summary, ' g')}",
                f"Max |alt err|: {self._format_summary_value(alt_err_summary, ' m')}",
                f"Max |vel err|: {self._format_summary_value(vel_err_summary, ' m/s')}",
                f"Mass spread: {self._format_summary_value(1000.0 * mass_spread if np.isfinite(mass_spread) else np.nan, ' g')}",
            ]
        )
        ax_status.text(
            0.98,
            0.02,
            "\n".join(count_lines),
            transform=ax_status.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            bbox=dict(facecolor="white", alpha=0.88, edgecolor="0.7"),
        )

        q_mask = np.isfinite(q_max_kpa)
        g_mask = np.isfinite(g_max_series)

        non_raw_indices = [i for i, row in enumerate(rows) if str(row["status"]) != "PASS_RAW"]
        for idx in non_raw_indices:
            ax_path.axvspan(idx - 0.45, idx + 0.45, color=status_colors[idx], alpha=0.08, lw=0)
        for break_idx in group_breaks:
            ax_path.axvline(break_idx - 0.5, color="0.65", linestyle=":", linewidth=1.0, zorder=0)
        ax_path_g = ax_path.twinx()
        ax_path_g.patch.set_alpha(0.0)
        q_limit_line = ax_path.axhline(self.suite.base_config.max_q_limit / 1000.0, color="tab:blue", linestyle="--", linewidth=1.0, label="Max-Q limit")
        g_limit_line = ax_path_g.axhline(self.suite.base_config.max_g_load, color="tab:orange", linestyle="--", linewidth=1.0, label="Max g limit")
        q_handle = None
        g_handle = None
        if np.any(q_mask):
            ax_path.plot(seed_positions[q_mask], q_max_kpa[q_mask], color="tab:blue", linewidth=1.0, zorder=1)
            q_handle = ax_path.scatter(
                seed_positions[q_mask],
                q_max_kpa[q_mask],
                color="tab:blue",
                s=54,
                edgecolors="0.2",
                linewidths=0.6,
                zorder=3,
                label="Max Q",
            )
            q_candidates = [self.suite.base_config.max_q_limit / 1000.0]
            q_candidates.extend(q_max_kpa[q_mask].tolist())
            q_min = min(q_candidates)
            q_max_val = max(q_candidates)
            q_span = max(q_max_val - q_min, 0.15)
            q_pad = max(0.08, 0.12 * q_span)
            ax_path.set_ylim(max(0.0, q_min - q_pad), q_max_val + q_pad)
        else:
            ax_path.text(0.5, 0.5, "No replayed path metrics available.", transform=ax_path.transAxes, ha="center", va="center")
        if np.any(g_mask):
            ax_path_g.plot(seed_positions[g_mask], g_max_series[g_mask], color="tab:orange", linewidth=1.0, zorder=1)
            g_handle = ax_path_g.scatter(
                seed_positions[g_mask],
                g_max_series[g_mask],
                color="tab:orange",
                marker="s",
                s=50,
                edgecolors="0.2",
                linewidths=0.6,
                zorder=3,
                label="Max g",
            )
            g_candidates = [self.suite.base_config.max_g_load]
            g_candidates.extend(g_max_series[g_mask].tolist())
            g_min = min(g_candidates)
            g_max_val = max(g_candidates)
            g_span = max(g_max_val - g_min, 0.02)
            g_pad = max(0.01, 0.12 * g_span)
            ax_path_g.set_ylim(max(0.0, g_min - g_pad), g_max_val + g_pad)
        else:
            ax_path_g.set_ylim(0.0, max(1.0, self.suite.base_config.max_g_load * 1.1))

        ax_path.set_ylabel("Max dynamic pressure (kPa)")
        ax_path_g.set_ylabel("Max g-load (g)")
        ax_path.grid(True, axis="y", alpha=0.3)
        ax_path.text(0.01, 0.98, "b)", transform=ax_path.transAxes, ha="left", va="top", fontsize=10, fontweight="bold", color="0.2")
        ax_path.tick_params(axis="x", labelbottom=False)
        path_handles = []
        path_labels = []
        if q_handle is not None:
            path_handles.extend([q_handle, q_limit_line])
            path_labels.extend(["Max Q", "Max-Q limit"])
        if g_handle is not None:
            path_handles.extend([g_handle, g_limit_line])
            path_labels.extend(["Max g", "Max g limit"])
        if path_handles:
            ax_path.legend(path_handles, path_labels, loc="upper left", fontsize=8)

        alt_mask = np.isfinite(alt_err_abs_m)
        vel_mask = np.isfinite(vel_err_abs_m_s)
        ax_vel = ax_terminal.twinx()
        ax_vel.patch.set_alpha(0.0)
        for idx in non_raw_indices:
            ax_terminal.axvspan(idx - 0.45, idx + 0.45, color=status_colors[idx], alpha=0.08, lw=0)
        for break_idx in group_breaks:
            ax_terminal.axvline(break_idx - 0.5, color="0.65", linestyle=":", linewidth=1.0, zorder=0)
        alt_handle = None
        vel_handle = None
        alt_tol_line = None
        vel_tol_line = None
        if np.any(alt_mask):
            ax_terminal.plot(seed_positions[alt_mask], alt_err_abs_m[alt_mask], color="tab:green", linewidth=1.0, zorder=1)
            alt_handle = ax_terminal.scatter(
                seed_positions[alt_mask],
                alt_err_abs_m[alt_mask],
                color="tab:green",
                s=56,
                edgecolors="0.2",
                linewidths=0.6,
                zorder=3,
                label="|Altitude error|",
            )
            alt_candidates = alt_err_abs_m[alt_mask].tolist()
            alt_max = max(alt_candidates)
            alt_scale = max(alt_max, 1.0)
            alt_span = max(alt_max - min(alt_candidates), 0.25 * alt_scale, 0.5)
            alt_pad = max(0.05 * alt_scale, 0.12 * alt_span, 0.2)
            alt_upper = alt_max + alt_pad
            ax_terminal.set_ylim(0.0, alt_upper)
            if TERMINAL_ALT_TOL_M <= alt_upper:
                alt_tol_line = ax_terminal.axhline(
                    TERMINAL_ALT_TOL_M,
                    color="tab:green",
                    linestyle="--",
                    linewidth=1.0,
                    label="Altitude tolerance",
                )
            else:
                ax_terminal.text(
                    0.98,
                    0.92,
                    f"Altitude tolerance: {TERMINAL_ALT_TOL_M:.0f} m (off-scale)",
                    transform=ax_terminal.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    color="tab:green",
                )
        else:
            ax_terminal.text(0.5, 0.5, "No replayed terminal metrics available.", transform=ax_terminal.transAxes, ha="center", va="center")
        if np.any(vel_mask):
            ax_vel.plot(seed_positions[vel_mask], vel_err_abs_m_s[vel_mask], color="tab:red", linewidth=1.0, zorder=1)
            vel_handle = ax_vel.scatter(
                seed_positions[vel_mask],
                vel_err_abs_m_s[vel_mask],
                color="tab:red",
                marker="^",
                s=56,
                edgecolors="0.2",
                linewidths=0.6,
                zorder=3,
                label="|Velocity error|",
            )
            vel_candidates = vel_err_abs_m_s[vel_mask].tolist()
            vel_max = max(vel_candidates)
            vel_scale = max(vel_max, 0.1)
            vel_span = max(vel_max - min(vel_candidates), 0.25 * vel_scale, 0.02)
            vel_pad = max(0.05 * vel_scale, 0.12 * vel_span, 0.01)
            vel_upper = vel_max + vel_pad
            ax_vel.set_ylim(0.0, vel_upper)
            if TERMINAL_VEL_TOL_M_S <= vel_upper:
                vel_tol_line = ax_vel.axhline(
                    TERMINAL_VEL_TOL_M_S,
                    color="tab:red",
                    linestyle="--",
                    linewidth=1.0,
                    label="Velocity tolerance",
                )
            else:
                ax_vel.text(
                    0.98,
                    0.82,
                    f"Velocity tolerance: {TERMINAL_VEL_TOL_M_S:.1f} m/s (off-scale)",
                    transform=ax_vel.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    color="tab:red",
                )
        else:
            ax_vel.set_ylim(0.0, max(1.0, TERMINAL_VEL_TOL_M_S * 1.1))

        ax_terminal.set_ylabel("|Altitude error| (m)")
        ax_vel.set_ylabel("|Velocity error| (m/s)")
        ax_terminal.set_xlabel("Seed")
        ax_terminal.set_xticks(seed_positions)
        ax_terminal.set_xticklabels(seed_labels, rotation=35, ha="right")
        ax_terminal.grid(True, axis="y", alpha=0.3)
        ax_terminal.text(0.01, 0.98, "c)", transform=ax_terminal.transAxes, ha="left", va="top", fontsize=10, fontweight="bold", color="0.2")
        handles = []
        labels = []
        if alt_handle is not None:
            handles.append(alt_handle)
            labels.append("|Altitude error|")
        if alt_tol_line is not None:
            handles.append(alt_tol_line)
            labels.append("Altitude tolerance")
        if vel_handle is not None:
            handles.append(vel_handle)
            labels.append("|Velocity error|")
        if vel_tol_line is not None:
            handles.append(vel_tol_line)
            labels.append("Velocity tolerance")
        if handles:
            ax_terminal.legend(handles, labels, loc="upper left", fontsize=8)

        self.suite._finalize_figure(fig, "randomized_multistart")

def evaluate_terminal_state(
    sim_res,
    cfg,
    env_cfg,
    alt_tol_m=TERMINAL_ALT_TOL_M,
    vel_tol_m_s=TERMINAL_VEL_TOL_M_S,
    radial_vel_tol_m_s=TERMINAL_RADIAL_VEL_TOL_M_S,
    inc_tol_deg=TERMINAL_INC_TOL_DEG
):
    """
    Centralized terminal-state checker used across reliability analyses.
    Returns a dict with validity flags, errors, and pass/fail status labels.
    """
    metrics = {
        "terminal_valid": False,
        "trajectory_valid": False,
        "orbit_ok": False,
        "fuel_ok": False,
        "strict_ok": False,
        "status": "SIM_ERROR",
        "r_f_m": np.nan,
        "spherical_height_m": np.nan,
        "ellipsoidal_altitude_m": np.nan,
        "v_f_m_s": np.nan,
        "target_alt_m": float(cfg.target_altitude),
        "target_orbit_radius_m": np.nan,
        "target_vel_m_s": np.nan,
        "target_inclination_deg": np.nan,
        "alt_err_m": np.nan,
        "spherical_height_err_m": np.nan,
        "vel_err_m_s": np.nan,
        "radial_velocity_m_s": np.nan,
        "fpa_deg": np.nan,
        "inclination_deg": np.nan,
        "inclination_err_deg": np.nan,
        "m_final_kg": np.nan,
        "m_dry_limit_kg": float(cfg.stage_2.dry_mass + cfg.payload_mass),
        "fuel_margin_kg": np.nan
    }

    if not isinstance(sim_res, dict):
        return metrics

    y = sim_res.get("y", None)
    t = sim_res.get("t", None)
    u = sim_res.get("u", None)
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[0] < 7 or y.shape[1] < 1:
        return metrics
    if not isinstance(t, np.ndarray) or t.ndim != 1 or len(t) < 1:
        return metrics
    if not isinstance(u, np.ndarray) or u.ndim != 2 or u.shape[0] < 4:
        return metrics
    if y.shape[1] != len(t) or u.shape[1] != len(t):
        return metrics
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(u)):
        return metrics
    if np.any(np.diff(t) < -1e-9):
        return metrics
    if np.any(y[6, :] <= 0.0):
        return metrics

    final_state = y[:, -1]
    if not np.all(np.isfinite(final_state)) or final_state[6] <= 0.0:
        return metrics
    r0 = np.linalg.norm(y[0:3, 0])
    rf = np.linalg.norm(final_state[0:3])
    if r0 < 0.5 * env_cfg.earth_radius_equator or rf < 0.25 * env_cfg.earth_radius_equator:
        # Catches dummy fallback arrays while still allowing physically crashed trajectories.
        return metrics

    # Trajectory-level physical validity checks.
    R_eq = env_cfg.earth_radius_equator
    R_pol = R_eq * (1.0 - env_cfg.earth_flattening)
    ellipsoid_metric = (y[0, :] / R_eq) ** 2 + (y[1, :] / R_eq) ** 2 + (y[2, :] / R_pol) ** 2
    ground_impact = bool(np.nanmin(ellipsoid_metric) <= 1.0 + 1e-8)
    termination = sim_res.get("termination", {})
    if isinstance(termination, dict) and termination.get("reason") == "ground_impact":
        ground_impact = True
    if ground_impact:
        metrics["status"] = "CRASH"
        return metrics

    spherical_height = spherical_altitude_m(final_state[0:3], env_cfg)
    ellipsoidal_alt = ellipsoidal_altitude_m(final_state[0:3], env_cfg)
    v_f = np.linalg.norm(final_state[3:6])
    m_final = float(final_state[6])
    target_alt = float(cfg.target_altitude)
    target_radius = target_orbit_radius_m(target_alt, env_cfg)
    target_vel = circular_target_speed_m_s(target_alt, env_cfg)
    target_inc_deg = float(cfg.target_inclination) if cfg.target_inclination is not None else abs(float(env_cfg.launch_latitude))

    alt_err = spherical_height - target_alt
    vel_err = v_f - target_vel
    radial_speed = float(np.dot(final_state[0:3], final_state[3:6]) / (rf + 1e-12))
    sin_fpa = np.clip(radial_speed / (v_f + 1e-12), -1.0, 1.0)
    fpa_deg = float(np.degrees(np.arcsin(sin_fpa)))
    h_vec = np.cross(final_state[0:3], final_state[3:6])
    h_mag = np.linalg.norm(h_vec)
    if h_mag > 1e-12:
        inclination_deg = float(np.degrees(np.arccos(np.clip(h_vec[2] / h_mag, -1.0, 1.0))))
        inc_err_deg = inclination_deg - target_inc_deg
    else:
        inclination_deg = np.nan
        inc_err_deg = np.nan
    fuel_margin = m_final - metrics["m_dry_limit_kg"]

    orbit_ok = (
        abs(alt_err) < alt_tol_m
        and abs(vel_err) < vel_tol_m_s
        and abs(radial_speed) < radial_vel_tol_m_s
        and np.isfinite(inc_err_deg)
        and abs(inc_err_deg) < inc_tol_deg
    )
    fuel_ok = fuel_margin > 0.0
    strict_ok = orbit_ok and fuel_ok

    metrics.update({
        "terminal_valid": True,
        "trajectory_valid": True,
        "orbit_ok": orbit_ok,
        "fuel_ok": fuel_ok,
        "strict_ok": strict_ok,
        "r_f_m": spherical_height,
        "spherical_height_m": spherical_height,
        "ellipsoidal_altitude_m": ellipsoidal_alt,
        "v_f_m_s": v_f,
        "target_orbit_radius_m": target_radius,
        "target_vel_m_s": target_vel,
        "target_inclination_deg": target_inc_deg,
        "alt_err_m": alt_err,
        "spherical_height_err_m": alt_err,
        "vel_err_m_s": vel_err,
        "radial_velocity_m_s": radial_speed,
        "fpa_deg": fpa_deg,
        "inclination_deg": inclination_deg,
        "inclination_err_deg": inc_err_deg,
        "m_final_kg": m_final,
        "fuel_margin_kg": fuel_margin
    })

    if strict_ok:
        metrics["status"] = "PASS"
    elif (not orbit_ok) and (not fuel_ok):
        metrics["status"] = "MISS+FUEL"
    elif not orbit_ok:
        metrics["status"] = "MISS"
    else:
        metrics["status"] = "FUEL"

    return metrics

def _ms_refine_peak_time(t_series, y_series, idx_peak):
    """
    Peak refinement helper used by reliability diagnostics.
    The refined value is bounded by the local sampled maximum so the
    quadratic fit cannot invent a path-constraint violation between nodes.
    """
    t = np.asarray(t_series, dtype=float)
    y = np.asarray(y_series, dtype=float)
    i = int(idx_peak)
    if len(t) < 3 or i <= 0 or i >= len(t) - 1:
        return float(t[i]), float(y[i])

    t_w = t[i - 1:i + 2]
    y_w = y[i - 1:i + 2]
    if not (np.all(np.isfinite(t_w)) and np.all(np.isfinite(y_w))):
        return float(t[i]), float(y[i])

    x = t_w - t[i]
    A = np.column_stack([x * x, x, np.ones_like(x)])
    try:
        a, b, c = np.linalg.lstsq(A, y_w, rcond=None)[0]
    except np.linalg.LinAlgError:
        return float(t[i]), float(y[i])
    if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(c):
        return float(t[i]), float(y[i])
    if a >= 0.0 or abs(a) < 1e-18:
        return float(t[i]), float(y[i])

    x_peak = -b / (2.0 * a)
    if x_peak < x[0] or x_peak > x[-1]:
        return float(t[i]), float(y[i])

    y_peak = a * x_peak * x_peak + b * x_peak + c
    if not np.isfinite(y_peak):
        return float(t[i]), float(y[i])
    sampled_peak = float(np.nanmax(y_w))
    tol = max(1.0e-12, 1.0e-9 * abs(sampled_peak))
    if y_peak > sampled_peak + tol:
        return float(t[i]), float(y[i])
    return float(t[i] + x_peak), float(y_peak)


def _ms_trajectory_diagnostics(sim_res, veh, cfg):
    """
    Standalone trajectory diagnostics for process workers.
    Mirrors ReliabilitySuite._trajectory_diagnostics.
    """
    out = {
        "valid": False,
        "max_q_pa": np.nan,
        "max_q_time_s": np.nan,
        "max_g": np.nan,
        "max_g_time_s": np.nan,
        "staging_time_s": np.nan,
        "depletion_time_s": np.nan,
        "q_violation": False,
        "g_violation": False
    }

    if not isinstance(sim_res, dict):
        return out
    t = sim_res.get("t", None)
    y = sim_res.get("y", None)
    u = sim_res.get("u", None)
    if not isinstance(t, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(u, np.ndarray):
        return out
    if y.ndim != 2 or u.ndim != 2 or len(t) < 1 or y.shape[0] < 7 or u.shape[0] < 4:
        return out
    if y.shape[1] != len(t) or u.shape[1] != len(t):
        return out
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(u)):
        return out
    if np.any(y[6, :] <= 0.0):
        return out

    termination = sim_res.get("termination", {})
    if isinstance(termination, dict) and termination.get("reason") == "ground_impact":
        return out

    R_eq = veh.env.config.earth_radius_equator
    R_pol = R_eq * (1.0 - veh.env.config.earth_flattening)
    ellipsoid_metric = (y[0, :] / R_eq) ** 2 + (y[1, :] / R_eq) ** 2 + (y[2, :] / R_pol) ** 2
    if np.min(ellipsoid_metric) <= 1.0 + 1e-8:
        return out

    m_stage2_wet = cfg.stage_2.dry_mass + cfg.stage_2.propellant_mass + cfg.payload_mass
    g0 = veh.env.config.g0
    q_hist = np.full(len(t), np.nan, dtype=float)
    g_hist = np.full(len(t), np.nan, dtype=float)

    for i in range(len(t)):
        r_i = y[0:3, i]
        v_i = y[3:6, i]
        m_i = y[6, i]
        th_i = u[0, i]
        dir_i = u[1:, i]
        ti = t[i]

        env_state = veh.env.get_state_sim(r_i, ti)
        v_rel = v_i - env_state["wind_velocity"]
        q = 0.5 * env_state["density"] * np.dot(v_rel, v_rel)
        q_hist[i] = float(q)

        if th_i < 0.01:
            stage_mode = "coast" if m_i > m_stage2_wet + 1000.0 else "coast_2"
        else:
            stage_mode = "boost" if m_i > m_stage2_wet + 1000.0 else "ship"
        dyn = veh.get_dynamics(y[:, i], th_i, dir_i, ti, stage_mode=stage_mode, scaling=None)
        sensed_acc = dyn[3:6] - env_state["gravity"]
        g_load = np.linalg.norm(sensed_acc) / g0
        g_hist[i] = float(g_load)

    if np.all(~np.isfinite(q_hist)) or np.all(~np.isfinite(g_hist)):
        return out
    idx_q = int(np.nanargmax(q_hist))
    idx_g = int(np.nanargmax(g_hist))
    max_q_time, max_q = _ms_refine_peak_time(t, q_hist, idx_q)
    max_g_time, max_g = _ms_refine_peak_time(t, g_hist, idx_g)

    dm = np.diff(y[6, :])
    if len(dm) > 0:
        idx_stage = int(np.argmin(dm))
        if dm[idx_stage] < -1000.0:
            out["staging_time_s"] = float(t[idx_stage])
    m_dry = cfg.stage_2.dry_mass + cfg.payload_mass
    dep_idx = np.where(y[6, :] <= m_dry + 1e-6)[0]
    if len(dep_idx) > 0:
        out["depletion_time_s"] = float(t[int(dep_idx[0])])

    out["valid"] = True
    out["max_q_pa"] = max_q if np.isfinite(max_q) else np.nan
    out["max_q_time_s"] = max_q_time
    out["max_g"] = max_g if np.isfinite(max_g) else np.nan
    out["max_g_time_s"] = max_g_time
    out["q_violation"] = bool(np.isfinite(out["max_q_pa"]) and out["max_q_pa"] > cfg.max_q_limit)
    out["g_violation"] = bool(np.isfinite(out["max_g"]) and out["max_g"] > cfg.max_g_load)
    return out


def _ms_evaluate_path_compliance(traj_diag, cfg, q_slack_pa, g_slack):
    out = {
        "path_ok": False,
        "q_ok": False,
        "g_ok": False,
        "max_q_pa": np.nan,
        "max_g": np.nan,
        "q_limit_pa": float(cfg.max_q_limit),
        "g_limit": float(cfg.max_g_load),
        "q_slack_pa": float(q_slack_pa),
        "g_slack": float(g_slack),
    }
    if not isinstance(traj_diag, dict) or not bool(traj_diag.get("valid", False)):
        return out
    max_q = float(traj_diag.get("max_q_pa", np.nan))
    max_g = float(traj_diag.get("max_g", np.nan))
    q_ok = bool(np.isfinite(max_q) and max_q <= cfg.max_q_limit + q_slack_pa)
    g_ok = bool(np.isfinite(max_g) and max_g <= cfg.max_g_load + g_slack)
    out.update({
        "path_ok": bool(q_ok and g_ok),
        "q_ok": q_ok,
        "g_ok": g_ok,
        "max_q_pa": max_q,
        "max_g": max_g,
    })
    return out

def _run_grid_independence_worker(payload):
    """
    Solve one grid-independence point (single node count).
    """
    node_count = int(payload["num_nodes"])
    base_cfg = payload["base_cfg"]
    base_env_cfg = payload["base_env_cfg"]
    q_slack_pa = float(payload["q_slack_pa"])
    g_slack = float(payload["g_slack"])

    cfg = copy.deepcopy(base_cfg)
    cfg.num_nodes = node_count

    t0 = time.time()
    m_final = np.nan
    status = "Failed"
    solver_ok = False
    terminal_valid = 0
    strict_terminal_ok = 0
    strict_path_ok = 0
    raw_path_ok = 0
    raw_q_ok = 0
    raw_g_ok = 0
    q_ok = 0
    g_ok = 0
    max_q_pa = np.nan
    max_g = np.nan
    try:
        with suppress_stdout():
            env_i = Environment(copy.deepcopy(base_env_cfg))
            veh_i = Vehicle(cfg, env_i)
            res = solve_optimal_trajectory(cfg, veh_i, env_i, print_level=0)
        if res.get("success", False):
            m_final = float(res["X3"][6, -1])
            solver_ok = True
            with suppress_stdout():
                sim_res = run_simulation(res, veh_i, cfg, rtol=1e-9, atol=1e-12)
            terminal = evaluate_terminal_state(sim_res, cfg, env_i.config)
            traj = _ms_trajectory_diagnostics(sim_res, veh_i, cfg)
            path_slack = _ms_evaluate_path_compliance(traj, cfg, q_slack_pa, g_slack)
            path_raw = _ms_evaluate_path_compliance(traj, cfg, 0.0, 0.0)
            terminal_valid = int(terminal.get("terminal_valid", False))
            strict_terminal_ok = int(terminal.get("strict_ok", False))
            strict_path_ok = int(bool(terminal.get("strict_ok", False) and path_slack.get("path_ok", False)))
            raw_path_ok = int(bool(terminal.get("strict_ok", False) and path_raw.get("path_ok", False)))
            raw_q_ok = int(path_raw.get("q_ok", False))
            raw_g_ok = int(path_raw.get("g_ok", False))
            q_ok = int(path_slack.get("q_ok", False))
            g_ok = int(path_slack.get("g_ok", False))
            max_q_pa = float(path_slack.get("max_q_pa", np.nan))
            max_g = float(path_slack.get("max_g", np.nan))
            if raw_path_ok:
                status = "PASS_RAW"
            elif strict_path_ok:
                status = "PASS_SLACK"
            elif strict_terminal_ok:
                status = "PATH_FAIL"
            else:
                status = str(terminal.get("status", "Failed"))
    except Exception:
        m_final = np.nan
        status = "Failed"
        solver_ok = False

    return {
        "nodes": node_count,
        "solver_success": int(solver_ok),
        "status": status,
        "final_mass_kg": float(m_final) if np.isfinite(m_final) else np.nan,
        "terminal_valid": int(terminal_valid),
        "strict_terminal_ok": int(strict_terminal_ok),
        "strict_path_ok": int(strict_path_ok),
        "raw_path_ok": int(raw_path_ok),
        "raw_q_ok": int(raw_q_ok),
        "raw_g_ok": int(raw_g_ok),
        "q_ok": int(q_ok),
        "g_ok": int(g_ok),
        "max_q_pa": float(max_q_pa) if np.isfinite(max_q_pa) else np.nan,
        "max_g": float(max_g) if np.isfinite(max_g) else np.nan,
        "runtime_s": float(time.time() - t0),
    }


def _run_interval_replay_phase_worker(payload):
    """
    Replay one transcription phase interval-by-interval with a different integrator.
    """
    phase_name = str(payload["phase_name"])
    X = np.asarray(payload["X"], dtype=float)
    U = payload.get("U", None)
    if U is not None:
        U = np.asarray(U, dtype=float)
    T = float(payload["T"])
    t_start = float(payload["t_start"])
    stage_mode = str(payload["stage_mode"])
    cfg = payload["cfg"]
    env_cfg = payload["env_cfg"]

    env_i = Environment(copy.deepcopy(env_cfg))
    veh_i = Vehicle(copy.deepcopy(cfg), env_i)

    if U is None:
        N_loc = X.shape[1] - 1
    else:
        N_loc = U.shape[1]
    dt = T / max(N_loc, 1)

    pos_err = np.full(N_loc, np.nan, dtype=float)
    vel_err = np.full(N_loc, np.nan, dtype=float)
    mass_err = np.full(N_loc, np.nan, dtype=float)
    rel_state_max = np.full(N_loc, np.nan, dtype=float)
    q_max = np.full(N_loc, np.nan, dtype=float)
    g_max = np.full(N_loc, np.nan, dtype=float)
    solver_ok = np.ones(N_loc, dtype=int)

    for k in range(N_loc):
        xk = X[:, k]
        if U is None:
            throttle = 0.0
            u_dir = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            uk = U[:, k]
            throttle = float(uk[0])
            u_dir = np.array(uk[1:], dtype=float)
        tk = float(t_start + k * dt)

        def dyn(t, x):
            return veh_i.get_dynamics(x, throttle, u_dir, t, stage_mode=stage_mode, scaling=None)

        try:
            sol = solve_ivp(
                dyn,
                (tk, tk + dt),
                xk,
                method="DOP853",
                rtol=1e-10,
                atol=1e-12,
                dense_output=True,
            )
        except Exception:
            solver_ok[k] = 0
            continue

        if not bool(sol.success) or sol.y.shape[1] < 1:
            solver_ok[k] = 0
            continue

        x_replay = np.asarray(sol.y[:, -1], dtype=float)
        mismatch = x_replay - X[:, k + 1]
        denom = np.maximum(np.abs(X[:, k + 1]), 1.0)
        pos_err[k] = float(np.linalg.norm(mismatch[0:3]))
        vel_err[k] = float(np.linalg.norm(mismatch[3:6]))
        mass_err[k] = float(abs(mismatch[6]))
        rel_state_max[k] = float(np.max(np.abs(mismatch) / denom))

        sample_t = np.linspace(tk, tk + dt, 17)
        if sol.sol is not None:
            x_samples = np.asarray(sol.sol(sample_t), dtype=float)
        else:
            x_samples = np.asarray(sol.y, dtype=float)
            sample_t = np.asarray(sol.t, dtype=float)

        q_peak = -np.inf
        g_peak = -np.inf
        for ti, xi in zip(sample_t, x_samples.T):
            env_state = env_i.get_state_sim(xi[0:3], float(ti))
            v_rel = xi[3:6] - env_state["wind_velocity"]
            q_here = 0.5 * env_state["density"] * float(np.dot(v_rel, v_rel))
            dyn_here = veh_i.get_dynamics(xi, throttle, u_dir, float(ti), stage_mode=stage_mode, scaling=None)
            sensed_acc = dyn_here[3:6] - env_state["gravity"]
            g_here = float(np.linalg.norm(sensed_acc) / env_i.config.g0)
            q_peak = max(q_peak, q_here)
            g_peak = max(g_peak, g_here)
        q_max[k] = float(q_peak)
        g_max[k] = float(g_peak)

    return {
        "phase_name": phase_name,
        "position_error": pos_err,
        "velocity_error": vel_err,
        "mass_error": mass_err,
        "relative_error": rel_state_max,
        "max_q_pa": q_max,
        "max_g": g_max,
        "solver_ok": solver_ok,
    }


def _export_csv(path, headers, rows):
    def _format_cell(value):
        if value is None:
            return "nan"
        if isinstance(value, (np.floating, float)):
            if np.isnan(value):
                return "nan"
            if np.isinf(value):
                return "inf" if value > 0 else "-inf"
            return f"{float(value):.12g}"
        if isinstance(value, (np.integer, int)):
            return str(int(value))
        if isinstance(value, (np.bool_, bool)):
            return "1" if bool(value) else "0"
        return str(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([_format_cell(v) for v in row])


class ReliabilitySuite:
    def __init__(self, output_dir=None, save_figures=True, show_plots=True, random_seed=1337, analysis_profile="default"):
        self.base_config = copy.deepcopy(StarshipBlock2)
        self.base_env_config = copy.deepcopy(EARTH_CONFIG)
        self.analysis_profile = str(analysis_profile)
        self.test_toggles = apply_analysis_profile(copy.deepcopy(RELIABILITY_ANALYSIS_TOGGLES), self.analysis_profile)
        self.random_seed = int(random_seed)
        self.seed_sequence = np.random.SeedSequence(self.random_seed)
        self.rng = np.random.default_rng(self.random_seed)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            self.output_dir = Path("reliability_outputs") / stamp
        else:
            self.output_dir = Path(output_dir)
        self.fig_dir = self.output_dir / "figures"
        self.data_dir = self.output_dir / "data"
        # Keep one switch for all persisted artifacts (figures + CSV tables).
        self.save_outputs = bool(save_figures)
        self.save_figures = self.save_outputs
        self.show_plots = bool(show_plots)
        self._analysis_execution = []
        self._summary = {}
        # Path-constraint compliance tolerances for reliability classification.
        # Small slack avoids false negatives from interpolation/integration noise.
        self.path_q_slack_pa = 500.0
        self.path_g_slack = 0.05

        # Cache one baseline optimization to keep all analyses comparable.
        self._baseline_opt_res = None
        
        # Initialize Baseline Physics
        self.env = Environment(self.base_env_config)
        self.veh = Vehicle(self.base_config, self.env)

        # Report-oriented plotting defaults.
        plt.rcParams.update({
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "-",
            "lines.linewidth": 1.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight"
        })
        
        print(f"\n{debug.Style.BOLD}=== RELIABILITY ANALYSIS SUITE ==={debug.Style.RESET}")
        print(f"Target: {self.base_config.name}")
        print(f"Analysis profile: {self.analysis_profile}")
        print(f"Output directory: {self.output_dir}")
        print(f"Reproducible seed: {self.random_seed}")
        self._save_run_metadata()

    def _save_run_metadata(self):
        self._save_table(
            "run_metadata",
            ["key", "value"],
            [
                ("timestamp_utc", datetime.utcnow().isoformat(timespec="seconds")),
                ("analysis_output_dir", str(self.output_dir)),
                ("analysis_profile", self.analysis_profile),
                ("vehicle_name", self.base_config.name),
                ("target_altitude_m", self.base_config.target_altitude),
                ("requested_target_inclination_deg", self.base_config.target_inclination),
                ("effective_target_inclination_deg", resolve_target_inclination_deg(self.base_config, self.base_env_config)),
                ("launch_latitude_deg", self.base_env_config.launch_latitude),
                ("num_nodes", self.base_config.num_nodes),
                ("max_iter", self.base_config.max_iter),
                ("max_q_limit_pa", self.base_config.max_q_limit),
                ("max_g_limit", self.base_config.max_g_load),
                ("optimization_q_guard_band_pa", OPTIMIZATION_Q_GUARD_BAND_PA),
                ("optimization_g_guard_band", OPTIMIZATION_G_GUARD_BAND),
                ("terminal_alt_tol_m", TERMINAL_ALT_TOL_M),
                ("terminal_vel_tol_m_s", TERMINAL_VEL_TOL_M_S),
                ("terminal_radial_vel_tol_m_s", TERMINAL_RADIAL_VEL_TOL_M_S),
                ("terminal_inc_tol_deg", TERMINAL_INC_TOL_DEG),
                ("seed", self.random_seed),
                ("python_version", platform.python_version()),
                ("numpy_version", np.__version__),
            ],
        )
        # Preserve dataclass field order so the CSV matches execution/toggle order.
        toggle_rows = [(k, int(v)) for k, v in self.test_toggles.__dict__.items()]
        self._save_table("enabled_test_toggles", ["toggle", "enabled"], toggle_rows)

    def _get_baseline_opt_res(self):
        if self._baseline_opt_res is None:
            with suppress_stdout():
                self._baseline_opt_res = solve_optimal_trajectory(
                    self.base_config, self.veh, self.env, print_level=0
                )
            if not self._baseline_opt_res.get("success", False):
                raise RuntimeError(
                    "Baseline optimization failed. Reliability analyses that depend on baseline controls "
                    "cannot proceed until optimization converges."
                )
        return self._baseline_opt_res

    def _save_table(self, name, headers, rows):
        if not self.save_outputs:
            return
        out = self.data_dir / f"{name}.csv"
        _export_csv(out, headers, rows)
        print(f"  [Saved] {out}")

    def _finalize_figure(self, fig, stem):
        layout_engine_getter = getattr(fig, "get_layout_engine", None)
        layout_engine = layout_engine_getter() if callable(layout_engine_getter) else None
        if layout_engine is None:
            fig.tight_layout()
        if self.save_outputs:
            png_path = self.fig_dir / f"{stem}.png"
            pdf_path = self.fig_dir / f"{stem}.pdf"
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            fig.savefig(pdf_path, bbox_inches="tight")
            print(f"  [Saved] {png_path}")
            print(f"  [Saved] {pdf_path}")
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)

    def _run_if_enabled(self, flag_name, label, fn, *args, progress=None, **kwargs):
        enabled = bool(getattr(self.test_toggles, flag_name, True))
        t0 = time.time()
        prog = f"[{progress}] " if progress is not None else ""
        if enabled:
            print(f"\n{prog}Starting: {label}")
            try:
                fn(*args, **kwargs)
                status = "completed"
                err = ""
            except BaseException as exc:
                status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
                err = str(exc)
                dt = time.time() - t0
                print(f"{prog}{'Interrupted' if status == 'interrupted' else 'Failed'}: {label} ({dt:.2f}s)")
                self._analysis_execution.append((flag_name, label, int(enabled), status, dt, err[:240]))
                raise
            dt = time.time() - t0
            print(f"{prog}Completed: {label} ({dt:.2f}s)")
            self._analysis_execution.append((flag_name, label, int(enabled), status, dt, err))
            return True
        print(f"\n{prog}[Skip] {label} disabled in RELIABILITY_ANALYSIS_TOGGLES.{flag_name}")
        self._analysis_execution.append((flag_name, label, int(enabled), "skipped", time.time() - t0, ""))
        return False

    @staticmethod
    def _print_parallel_progress(label, done, total, prefix="  "):
        total_i = max(1, int(total))
        done_i = max(0, min(int(done), total_i))
        pct = 100.0 * done_i / total_i
        print(f"{prefix}[Progress] {label}: {done_i}/{total_i} ({pct:5.1f}%)", flush=True)

    @staticmethod
    def _refine_peak_time(t_series, y_series, idx_peak):
        """
        Refine a sampled peak time/value with a local quadratic fit.
        Falls back to sampled values if the parabola is ill-posed or if the
        fitted extremum overshoots the local sampled envelope.
        """
        t = np.asarray(t_series, dtype=float)
        y = np.asarray(y_series, dtype=float)
        i = int(idx_peak)
        if len(t) < 3 or i <= 0 or i >= len(t) - 1:
            return float(t[i]), float(y[i])

        t_w = t[i - 1:i + 2]
        y_w = y[i - 1:i + 2]
        if not (np.all(np.isfinite(t_w)) and np.all(np.isfinite(y_w))):
            return float(t[i]), float(y[i])

        # Center time to improve conditioning.
        x = t_w - t[i]
        A = np.column_stack([x * x, x, np.ones_like(x)])
        try:
            a, b, c = np.linalg.lstsq(A, y_w, rcond=None)[0]
        except np.linalg.LinAlgError:
            return float(t[i]), float(y[i])
        if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(c):
            return float(t[i]), float(y[i])
        if a >= 0.0 or abs(a) < 1e-18:
            # Not a local maximum (or too flat to resolve robustly).
            return float(t[i]), float(y[i])

        x_peak = -b / (2.0 * a)
        if x_peak < x[0] or x_peak > x[-1]:
            return float(t[i]), float(y[i])

        y_peak = a * x_peak * x_peak + b * x_peak + c
        if not np.isfinite(y_peak):
            return float(t[i]), float(y[i])
        sampled_peak = float(np.nanmax(y_w))
        tol = max(1.0e-12, 1.0e-9 * abs(sampled_peak))
        if y_peak > sampled_peak + tol:
            return float(t[i]), float(y[i])
        return float(t[i] + x_peak), float(y_peak)

    def _trajectory_diagnostics(self, sim_res, veh, cfg):
        """
        Extract trajectory-level metrics used by reliability tests.
        """
        out = {
            "valid": False,
            "max_q_pa": np.nan,
            "max_q_time_s": np.nan,
            "max_g": np.nan,
            "max_g_time_s": np.nan,
            "staging_time_s": np.nan,
            "depletion_time_s": np.nan,
            "q_violation": False,
            "g_violation": False
        }

        if not isinstance(sim_res, dict):
            return out
        t = sim_res.get("t", None)
        y = sim_res.get("y", None)
        u = sim_res.get("u", None)
        if not isinstance(t, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(u, np.ndarray):
            return out
        if y.ndim != 2 or u.ndim != 2 or len(t) < 1 or y.shape[0] < 7 or u.shape[0] < 4:
            return out
        if y.shape[1] != len(t) or u.shape[1] != len(t):
            return out
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(u)):
            return out
        if np.any(y[6, :] <= 0.0):
            return out

        termination = sim_res.get("termination", {})
        if isinstance(termination, dict) and termination.get("reason") == "ground_impact":
            return out

        R_eq = veh.env.config.earth_radius_equator
        R_pol = R_eq * (1.0 - veh.env.config.earth_flattening)
        ellipsoid_metric = (y[0, :] / R_eq) ** 2 + (y[1, :] / R_eq) ** 2 + (y[2, :] / R_pol) ** 2
        if np.min(ellipsoid_metric) <= 1.0 + 1e-8:
            return out

        m_stage2_wet = cfg.stage_2.dry_mass + cfg.stage_2.propellant_mass + cfg.payload_mass
        g0 = veh.env.config.g0
        q_hist = np.full(len(t), np.nan, dtype=float)
        g_hist = np.full(len(t), np.nan, dtype=float)

        for i in range(len(t)):
            r_i = y[0:3, i]
            v_i = y[3:6, i]
            m_i = y[6, i]
            th_i = u[0, i]
            dir_i = u[1:, i]
            ti = t[i]

            env_state = veh.env.get_state_sim(r_i, ti)
            v_rel = v_i - env_state["wind_velocity"]
            q = 0.5 * env_state["density"] * np.dot(v_rel, v_rel)
            q_hist[i] = float(q)

            if th_i < 0.01:
                stage_mode = "coast" if m_i > m_stage2_wet + 1000.0 else "coast_2"
            else:
                stage_mode = "boost" if m_i > m_stage2_wet + 1000.0 else "ship"
            dyn = veh.get_dynamics(y[:, i], th_i, dir_i, ti, stage_mode=stage_mode, scaling=None)
            sensed_acc = dyn[3:6] - env_state["gravity"]
            g_load = np.linalg.norm(sensed_acc) / g0
            g_hist[i] = float(g_load)

        if np.all(~np.isfinite(q_hist)) or np.all(~np.isfinite(g_hist)):
            return out
        idx_q = int(np.nanargmax(q_hist))
        idx_g = int(np.nanargmax(g_hist))
        max_q_time, max_q = self._refine_peak_time(t, q_hist, idx_q)
        max_g_time, max_g = self._refine_peak_time(t, g_hist, idx_g)

        dm = np.diff(y[6, :])
        if len(dm) > 0:
            idx_stage = int(np.argmin(dm))
            if dm[idx_stage] < -1000.0:
                out["staging_time_s"] = float(t[idx_stage])
        m_dry = cfg.stage_2.dry_mass + cfg.payload_mass
        dep_idx = np.where(y[6, :] <= m_dry + 1e-6)[0]
        if len(dep_idx) > 0:
            out["depletion_time_s"] = float(t[int(dep_idx[0])])

        out["valid"] = True
        out["max_q_pa"] = max_q if np.isfinite(max_q) else np.nan
        out["max_q_time_s"] = max_q_time
        out["max_g"] = max_g if np.isfinite(max_g) else np.nan
        out["max_g_time_s"] = max_g_time
        out["q_violation"] = bool(np.isfinite(out["max_q_pa"]) and out["max_q_pa"] > cfg.max_q_limit)
        out["g_violation"] = bool(np.isfinite(out["max_g"]) and out["max_g"] > cfg.max_g_load)
        return out

    def _evaluate_path_compliance(self, traj_diag, cfg, q_slack_pa=None, g_slack=None):
        """
        Evaluate path-constraint compliance with small numerical slack.
        """
        q_slack_use = self.path_q_slack_pa if q_slack_pa is None else float(q_slack_pa)
        g_slack_use = self.path_g_slack if g_slack is None else float(g_slack)
        out = {
            "path_ok": False,
            "q_ok": False,
            "g_ok": False,
            "max_q_pa": np.nan,
            "max_g": np.nan,
            "q_limit_pa": float(cfg.max_q_limit),
            "g_limit": float(cfg.max_g_load),
            "q_slack_pa": q_slack_use,
            "g_slack": g_slack_use,
        }
        if not isinstance(traj_diag, dict) or not bool(traj_diag.get("valid", False)):
            return out
        max_q = float(traj_diag.get("max_q_pa", np.nan))
        max_g = float(traj_diag.get("max_g", np.nan))
        q_ok = bool(np.isfinite(max_q) and max_q <= cfg.max_q_limit + q_slack_use)
        g_ok = bool(np.isfinite(max_g) and max_g <= cfg.max_g_load + g_slack_use)
        out.update({
            "path_ok": bool(q_ok and g_ok),
            "q_ok": q_ok,
            "g_ok": g_ok,
            "max_q_pa": max_q,
            "max_g": max_g,
        })
        return out

    def _is_strict_path_success(self, terminal_metrics, traj_diag, cfg):
        """
        Mission success for uncertainty/reliability stats:
        terminal strict success + path constraints within reliability slack.
        """
        if not isinstance(terminal_metrics, dict):
            return False
        if not bool(terminal_metrics.get("strict_ok", False)):
            return False
        path = self._evaluate_path_compliance(traj_diag, cfg)
        return bool(path["path_ok"])

    def run_all(self, max_workers=None):
        """Runs enabled analyses from shortest expected runtime to longest."""
        self._analysis_execution = []
        ran_any = False
        if max_workers is None:
            worker_cap = min(4, max(1, os.cpu_count() or 1))
        else:
            worker_cap = max(1, int(max_workers))

        def _run_drift_with_cached_baseline():
            opt_res = self._get_baseline_opt_res()
            sim_res = run_simulation(opt_res, self.veh, self.base_config)
            self.analyze_drift(opt_res, sim_res)

        # Baseline-dependent analyses share one cached optimization result.
        baseline_flags = (
            "drift",
            "theoretical_efficiency",
            "integrator_tolerance",
            "interval_replay_audit",
        )
        try:
            if any(bool(getattr(self.test_toggles, name, True)) for name in baseline_flags):
                print(f"\n{debug.Style.BOLD}--- Precomputing Shared Baseline Solution ---{debug.Style.RESET}")
                self._get_baseline_opt_res()

            # Run grid independence first so the main discretization check is available early.
            ordered_tests = [
                ("grid_independence", "Grid independence", self.analyze_grid_independence, (), {"max_workers": worker_cap}),
                ("smooth_integrator_benchmark", "Smooth ODE integrator benchmark", self.analyze_smooth_integrator_benchmark, (), {}),
                ("drift", "Drift analysis", _run_drift_with_cached_baseline, (), {}),
                ("theoretical_efficiency", "Theoretical efficiency", self.analyze_theoretical_efficiency, (), {}),
                ("integrator_tolerance", "Integrator tolerance", self.analyze_integrator_tolerance, (), {}),
                ("interval_replay_audit", "Interval replay audit", self.analyze_interval_replay_audit, (), {"max_workers": worker_cap}),
                ("randomized_multistart", "Warm-start multi-seed robustness", self.analyze_randomized_multistart, (), {}),
            ]
            total_steps = len(ordered_tests)
            for i, (flag, label, fn, args, kwargs) in enumerate(ordered_tests, start=1):
                progress = f"{i}/{total_steps}"
                ran_any |= self._run_if_enabled(flag, label, fn, *args, progress=progress, **kwargs)

            if not ran_any:
                print("\nNo analyses executed. Enable tests in RELIABILITY_ANALYSIS_TOGGLES in config.py.")
        finally:
            self._save_table(
                "analysis_execution_log",
                ["toggle", "label", "enabled", "status", "runtime_s", "error_message"],
                self._analysis_execution
            )

    # 1. GRID INDEPENDENCE STUDY
    def analyze_grid_independence(self, max_workers=None):
        debug._print_sub_header("1. Grid Independence Study")
        # Broad convergence sweep for the final grid study.
        node_counts = list(range(20, 141, 20))
        masses = []
        runtimes = []
        success_flags = []
        strict_path_flags = []
        raw_path_flags = []
        # Keep the grid study serial. Parallel CasADi/IPOPT solves become memory-heavy,
        # and the denser midpoint path checks can trigger native crashes on smaller machines.
        max_workers = 1
        baseline_node = int(self.base_config.num_nodes)
        baseline_opt = self._get_baseline_opt_res()
        if max_workers > len(node_counts) - 1:
            max_workers = max(1, len(node_counts) - 1)
        jobs = [
            {
                "num_nodes": int(N),
                "base_cfg": self.base_config,
                "base_env_cfg": self.base_env_config,
                "q_slack_pa": float(self.path_q_slack_pa),
                "g_slack": float(self.path_g_slack),
            }
            for N in node_counts
            if int(N) != baseline_node
        ]
        job_results = {}

        print(
            f"{'Nodes':<6} | {'Final Mass (kg)':<15} | {'Runtime (s)':<10} | "
            f"{'RawPath':<7} | {'SlackPath':<9} | {'Status':<11}"
        )
        print("-" * 73)

        with suppress_stdout():
            baseline_sim = run_simulation(baseline_opt, self.veh, self.base_config, rtol=1e-9, atol=1e-12)
        baseline_terminal = evaluate_terminal_state(baseline_sim, self.base_config, self.base_env_config)
        baseline_traj = self._trajectory_diagnostics(baseline_sim, self.veh, self.base_config)
        baseline_path_slack = self._evaluate_path_compliance(baseline_traj, self.base_config)
        baseline_path_raw = self._evaluate_path_compliance(
            baseline_traj, self.base_config, q_slack_pa=0.0, g_slack=0.0
        )
        baseline_status = (
            "PASS_RAW" if baseline_terminal.get("strict_ok", False) and baseline_path_raw["path_ok"]
            else "PASS_SLACK" if baseline_terminal.get("strict_ok", False) and baseline_path_slack["path_ok"]
            else "PATH_FAIL" if baseline_terminal.get("strict_ok", False)
            else str(baseline_terminal.get("status", "Failed"))
        )
        job_results[baseline_node] = {
            "nodes": baseline_node,
            "solver_success": int(baseline_opt.get("success", False)),
            "status": baseline_status,
            "final_mass_kg": float(baseline_opt["X3"][6, -1]),
            "terminal_valid": int(baseline_terminal.get("terminal_valid", False)),
            "strict_terminal_ok": int(baseline_terminal.get("strict_ok", False)),
            "strict_path_ok": int(bool(baseline_terminal.get("strict_ok", False) and baseline_path_slack["path_ok"])),
            "raw_path_ok": int(bool(baseline_terminal.get("strict_ok", False) and baseline_path_raw["path_ok"])),
            "raw_q_ok": int(baseline_path_raw["q_ok"]),
            "raw_g_ok": int(baseline_path_raw["g_ok"]),
            "q_ok": int(baseline_path_slack["q_ok"]),
            "g_ok": int(baseline_path_slack["g_ok"]),
            "max_q_pa": float(baseline_path_slack["max_q_pa"]),
            "max_g": float(baseline_path_slack["max_g"]),
            "runtime_s": 0.0,
        }

        parallel_active = bool(max_workers > 1 and len(jobs) > 1)
        if parallel_active:
            print(f"  Parallel execution enabled (workers={max_workers}).")
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    fut_to_n = {executor.submit(_run_grid_independence_worker, payload): int(payload["num_nodes"]) for payload in jobs}
                    parallel_ok = True
                    total_jobs = len(fut_to_n)
                    done_jobs = 0
                    for fut in as_completed(fut_to_n):
                        done_jobs += 1
                        self._print_parallel_progress("Grid jobs", done_jobs, total_jobs)
                        n_key = fut_to_n[fut]
                        try:
                            job_results[n_key] = fut.result()
                        except Exception:
                            parallel_ok = False
                            break
                if not parallel_ok:
                    print("  Worker failure during grid study; switching to serial mode.")
                    parallel_active = False
                    job_results = {}
            except Exception:
                print("  Parallel execution unavailable; switching to serial mode.")
                parallel_active = False
                job_results = {}

        if not parallel_active:
            print("  Serial execution mode.")
            for payload in jobs:
                n_key = int(payload["num_nodes"])
                try:
                    job_results[n_key] = _run_grid_independence_worker(payload)
                except Exception:
                    job_results[n_key] = {
                        "nodes": n_key,
                        "solver_success": 0,
                        "status": "Failed",
                        "final_mass_kg": np.nan,
                        "runtime_s": np.nan,
                    }

        for N in node_counts:
            r = job_results.get(int(N), None)
            if r is None:
                r = {
                    "nodes": int(N),
                    "solver_success": 0,
                    "status": "Failed",
                    "final_mass_kg": np.nan,
                    "runtime_s": np.nan,
                }
            m_final = float(r.get("final_mass_kg", np.nan))
            dt = float(r.get("runtime_s", np.nan))
            ok = int(r.get("solver_success", 0))
            strict_path_ok = int(r.get("strict_path_ok", 0))
            raw_path_ok = int(r.get("raw_path_ok", 0))
            status = str(r.get("status", "Failed"))

            masses.append(m_final)
            runtimes.append(dt)
            success_flags.append(ok)
            strict_path_flags.append(strict_path_ok)
            raw_path_flags.append(raw_path_ok)
            print(f"{N:<6} | {m_final:<15.1f} | {dt:<10.2f} | {raw_path_ok:<7d} | {strict_path_ok:<9d} | {status:<11}")

        self._save_table(
            "grid_independence",
            [
                "nodes", "final_mass_kg", "runtime_s", "solver_success",
                "terminal_valid", "strict_terminal_ok", "strict_path_ok",
                "raw_path_ok", "raw_q_ok", "raw_g_ok", "q_ok", "g_ok",
                "max_q_pa", "max_g", "status"
            ],
            [
                (
                    int(node_counts[i]),
                    masses[i],
                    runtimes[i],
                    success_flags[i],
                    int(job_results[int(node_counts[i])].get("terminal_valid", 0)),
                    int(job_results[int(node_counts[i])].get("strict_terminal_ok", 0)),
                    int(job_results[int(node_counts[i])].get("strict_path_ok", 0)),
                    int(job_results[int(node_counts[i])].get("raw_path_ok", 0)),
                    int(job_results[int(node_counts[i])].get("raw_q_ok", 0)),
                    int(job_results[int(node_counts[i])].get("raw_g_ok", 0)),
                    int(job_results[int(node_counts[i])].get("q_ok", 0)),
                    int(job_results[int(node_counts[i])].get("g_ok", 0)),
                    float(job_results[int(node_counts[i])].get("max_q_pa", np.nan)),
                    float(job_results[int(node_counts[i])].get("max_g", np.nan)),
                    str(job_results[int(node_counts[i])].get("status", "Failed")),
                )
                for i in range(len(node_counts))
            ]
        )
            
        # Convergence Check
        mass_by_nodes = {node_counts[i]: masses[i] for i in range(len(node_counts))}
        m_100 = mass_by_nodes.get(100, np.nan)
        m_140 = mass_by_nodes.get(140, np.nan)
        if np.isfinite(m_100) and np.isfinite(m_140):
            diff = abs(m_140 - m_100)
            print(f"\nMass Delta (140 vs 100 nodes): {diff:.1f} kg")
            if diff < 100.0 and bool(job_results[100].get("strict_path_ok", 0)) and bool(job_results[140].get("strict_path_ok", 0)):
                print(f">>> {debug.Style.GREEN}PASS: Grid Independent (<100kg change) with replay-valid endpoints.{debug.Style.RESET}")
            else:
                print(f">>> {debug.Style.YELLOW}WARN: Grid study did not show both convergence and replay-valid endpoints.{debug.Style.RESET}")
        else:
            print(f"\n>>> {debug.Style.YELLOW}WARN: Cannot assess grid independence because node 100 or 140 failed.{debug.Style.RESET}")
        
        # Visualization: convergence and replay validity. Runtime is omitted from the
        # paper figure because the reference node reuses the cached baseline solve,
        # so its marginal runtime is not directly comparable to the other sweeps.
        valid_indices = [i for i, m in enumerate(masses) if np.isfinite(m)]
        if len(valid_indices) == 0:
            print("  No converged grid points available for plotting.")
            return

        valid_nodes = np.array([node_counts[i] for i in valid_indices], dtype=float)
        valid_masses = np.array([masses[i] for i in valid_indices], dtype=float)
        valid_runtimes = np.array([runtimes[i] for i in valid_indices], dtype=float)
        ref_node = int(valid_nodes[-1])
        ref_mass = float(valid_masses[-1])
        mass_err = np.abs(valid_masses - ref_mass)

        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(10, 8.5), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.0, 0.8]}
        )

        ax1.plot(valid_nodes, valid_masses, "o-", color="tab:blue", label="Final mass")
        ax1.axhline(ref_mass, color="tab:blue", linestyle="--", alpha=0.5, label=f"Reference (N={ref_node})")
        if np.isfinite(m_100) and np.isfinite(m_140):
            ax1.scatter([100, 140], [m_100, m_140], color="k", s=40, zorder=3)
        ax1.set_ylabel("Final Mass (kg)")
        ax1.set_title("Grid Independence: Objective Convergence and Replay Validity")
        ax1.ticklabel_format(style="plain", axis="y", useOffset=False)
        ax1.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best")

        # Omit the reference node itself from error plotting (its error is exactly zero by construction).
        err_mask = np.arange(len(valid_nodes)) != (len(valid_nodes) - 1)
        if np.any(err_mask):
            ax2.semilogy(
                valid_nodes[err_mask],
                np.maximum(mass_err[err_mask], 1e-12),
                "o-",
                color="tab:purple",
                label=f"|m(N)-m({ref_node})|",
            )
        else:
            ax2.text(0.5, 0.5, "Insufficient points for error curve", transform=ax2.transAxes, ha="center", va="center")
        ax2.axhline(100.0, color="tab:orange", linestyle="--", alpha=0.8, label="Criterion (100 kg)")
        ax2.set_ylabel("Mass Error (kg)")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(loc="best")

        slack_vals = np.array([strict_path_flags[i] for i in valid_indices], dtype=float)
        raw_vals = np.array([raw_path_flags[i] for i in valid_indices], dtype=float)
        ax3.scatter(valid_nodes, slack_vals + 0.02, color="tab:green", marker="o", label="Replay-valid (slack)")
        ax3.scatter(valid_nodes, raw_vals - 0.02, color="tab:purple", marker="x", label="Replay-valid (raw)")
        ax3.set_xlabel("Number of Nodes")
        ax3.set_ylabel("Replay Validity")
        ax3.set_ylim(-0.1, 1.1)
        ax3.set_yticks([0.0, 1.0])
        ax3.set_yticklabels(["Invalid", "Valid"])
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="best")
        self._finalize_figure(fig, "grid_independence_convergence_cost")

    # 2. INTEGRATOR TOLERANCE SWEEP
    def analyze_integrator_tolerance(self):
        debug._print_sub_header("2. Integrator Tolerance Sweep")
        
        # Get a solution first
        opt_res = self._get_baseline_opt_res()
            
        # Readable refined sweep: 1 / 5 / 1 / 5 pattern, with ATOL kept three
        # orders tighter than RTOL throughout.
        tols = [
            (1e-6, 1e-9),
            (5e-7, 5e-10),
            (1e-7, 1e-10),
            (5e-8, 5e-11),
            (1e-8, 1e-11),
            (5e-9, 5e-12),
            (1e-9, 1e-12),
            (5e-10, 5e-13),
            (1e-10, 1e-13),
            (5e-11, 5e-14),
            (1e-11, 1e-14),
            (5e-12, 5e-15),
            (1e-12, 1e-15),
        ]
        rows = []
        
        print(
            f"{'RTOL':<10} | {'ATOL':<10} | {'Height err (m)':<14} | {'Vel err (m/s)':<14} | "
            f"{'Max-Q (kPa)':<12} | {'Max-g':<8} | {'Path':<6} | {'Status':<10}"
        )
        print("-" * 109)
        
        for rtol, atol in tols:
            with suppress_stdout():
                sim_res = run_simulation(opt_res, self.veh, self.base_config, rtol=rtol, atol=atol)

            terminal = evaluate_terminal_state(sim_res, self.base_config, self.base_env_config)
            traj = self._trajectory_diagnostics(sim_res, self.veh, self.base_config)
            path = self._evaluate_path_compliance(traj, self.base_config)
            path_raw = self._evaluate_path_compliance(traj, self.base_config, q_slack_pa=0.0, g_slack=0.0)
            strict_path_ok = int(self._is_strict_path_success(terminal, traj, self.base_config))

            status = terminal.get("status", "SIM_ERROR")
            if bool(terminal.get("strict_ok", False)) and bool(path_raw.get("path_ok", False)):
                status = "PASS_RAW"
            elif bool(terminal.get("strict_ok", False)) and bool(strict_path_ok):
                status = "PASS_SLACK"
            elif bool(terminal.get("strict_ok", False)) and not bool(strict_path_ok):
                status = "PATH"

            rows.append((
                rtol,
                atol,
                terminal.get("spherical_height_m", np.nan),
                terminal.get("v_f_m_s", np.nan),
                terminal.get("spherical_height_err_m", np.nan),
                terminal.get("vel_err_m_s", np.nan),
                terminal.get("radial_velocity_m_s", np.nan),
                int(terminal.get("terminal_valid", False)),
                int(terminal.get("strict_ok", False)),
                strict_path_ok,
                int(path_raw["path_ok"]),
                int(path_raw["q_ok"]),
                int(path_raw["g_ok"]),
                int(path["q_ok"]),
                int(path["g_ok"]),
                path["max_q_pa"],
                path["max_g"],
                status,
            ))
            print(
                f"{rtol:<10.1e} | {atol:<10.1e} | {terminal.get('spherical_height_err_m', np.nan):<14.2f} | "
                f"{terminal.get('vel_err_m_s', np.nan):<14.3f} | {path['max_q_pa']/1000.0:<12.3f} | "
                f"{path['max_g']:<8.3f} | {strict_path_ok:<6d} | {status:<10}"
            )

        self._save_table(
            "integrator_tolerance",
            [
                "rtol", "atol",
                "final_spherical_height_m", "final_velocity_m_s",
                "spherical_height_error_m", "velocity_error_m_s", "radial_velocity_m_s",
                "terminal_valid", "strict_terminal_ok", "strict_path_ok",
                "raw_path_ok", "raw_q_ok", "raw_g_ok",
                "q_ok", "g_ok", "max_q_pa", "max_g", "status"
            ],
            rows
        )

        # Compare the operating tolerance against the tightest result using multiple metrics.
        idx_std = next(i for i, (rtol_i, atol_i) in enumerate(tols) if np.isclose(rtol_i, 1e-9) and np.isclose(atol_i, 1e-12))
        ref = rows[-1]
        op = rows[idx_std]
        loose = rows[0]
        loose_rtol, loose_atol = tols[0]
        op_rtol, op_atol = tols[idx_std]
        ref_rtol, ref_atol = tols[-1]

        def _drift(a, b):
            if np.isfinite(a) and np.isfinite(b):
                return abs(a - b)
            return np.nan

        drift_alt_loose = _drift(loose[4], ref[4])
        drift_alt_op = _drift(op[4], ref[4])
        drift_vel_op = _drift(op[5], ref[5])
        drift_radial_op = _drift(op[6], ref[6])
        drift_q_op = _drift(op[15], ref[15])
        drift_g_op = _drift(op[16], ref[16])

        print(f"\nDrift vs tightest ({ref_rtol:.0e}/{ref_atol:.0e}):")
        print(f"  Altitude error drift ({loose_rtol:.0e} -> ref): {drift_alt_loose:.2f} m")
        print(f"  Altitude error drift ({op_rtol:.0e} -> ref): {drift_alt_op:.2f} m")
        print(f"  Velocity error drift ({op_rtol:.0e} -> ref): {drift_vel_op:.4f} m/s")
        print(f"  Radial-velocity drift ({op_rtol:.0e} -> ref): {drift_radial_op:.4f} m/s")
        print(f"  Max-Q drift ({op_rtol:.0e} -> ref): {drift_q_op:.3f} Pa")
        print(f"  Max-g drift ({op_rtol:.0e} -> ref): {drift_g_op:.6f} g")

        op_path_ok = bool(op[9])
        ref_path_ok = bool(ref[9])
        converged_op = (
            op_path_ok
            and ref_path_ok
            and np.isfinite(drift_alt_op) and drift_alt_op < 50.0
            and np.isfinite(drift_vel_op) and drift_vel_op < 0.5
            and np.isfinite(drift_radial_op) and drift_radial_op < 0.5
            and np.isfinite(drift_q_op) and drift_q_op < 50.0
            and np.isfinite(drift_g_op) and drift_g_op < 0.02
        )
        if converged_op:
            print(
                f">>> {debug.Style.GREEN}PASS: Operating tolerance ({op_rtol:.0e}/{op_atol:.0e}) is converged and "
                f"path-compliant.{debug.Style.RESET}"
            )
        else:
            print(
                f">>> {debug.Style.RED}FAIL: Operating tolerance check failed (needs both convergence "
                f"and strict path compliance).{debug.Style.RESET}"
            )

        rtols = np.array([r[0] for r in rows[:-1]], dtype=float)
        d_alt = np.array([_drift(r[4], ref[4]) for r in rows[:-1]], dtype=float)
        d_vel = np.array([_drift(r[5], ref[5]) for r in rows[:-1]], dtype=float)
        d_rad = np.array([_drift(r[6], ref[6]) for r in rows[:-1]], dtype=float)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        baseline_rtol = tols[-1][0]

        if np.any(np.isfinite(d_alt) & (d_alt > 0.0)):
            mask = np.isfinite(d_alt) & (d_alt > 0.0)
            ax1.loglog(rtols[mask], d_alt[mask], "o-", color="tab:blue", label="Altitude drift")
        ax1.axhline(50.0, color="tab:orange", linestyle="--", alpha=0.8, label="Convergence criterion")
        ax1.set_ylabel("Altitude Drift (m)")
        ax1.set_title(f"Integrator Convergence (Baseline: rtol={baseline_rtol:.0e})")
        ax1.grid(True, which="both", ls="-", alpha=0.3)
        ax1.legend(loc="best")

        if np.any(np.isfinite(d_vel) & (d_vel > 0.0)):
            mask = np.isfinite(d_vel) & (d_vel > 0.0)
            ax2.loglog(rtols[mask], d_vel[mask], "s-", color="tab:orange", label="Velocity drift")
        ax2.axhline(0.5, color="tab:red", linestyle="--", alpha=0.8, label="Convergence criterion")
        ax2.set_ylabel("Velocity Drift (m/s)")
        ax2.grid(True, which="both", ls="-", alpha=0.3)
        ax2.legend(loc="best")

        if np.any(np.isfinite(d_rad) & (d_rad > 0.0)):
            mask = np.isfinite(d_rad) & (d_rad > 0.0)
            ax3.loglog(rtols[mask], d_rad[mask], "^-", color="tab:green", label="Radial-velocity drift")
        ax3.axhline(0.5, color="tab:red", linestyle="--", alpha=0.8, label="Convergence criterion")
        ax3.set_xlabel("Integrator Relative Tolerance (rtol)")
        ax3.set_ylabel("Radial-Velocity Drift (m/s)")
        ax3.grid(True, which="both", ls="-", alpha=0.3)
        ax3.legend(loc="best")

        for ax in (ax1, ax2, ax3):
            ax.invert_xaxis()  # Higher precision (lower tol) to the right.
        self._finalize_figure(fig, "integrator_tolerance_convergence")

    def analyze_randomized_multistart(self, n_trials=10):
        WarmStartMultiseedAnalyzer(self, evaluate_terminal_state).analyze(
            n_trials=n_trials,
        )

    def analyze_global_launch_cost(self, lat_step=5.0, solver_max_cpu_time=120.0):
        debug._print_sub_header("Global Launch-Cost Latitude Sweep")
        rows = self._run_global_launch_cost_sweep(
            lat_step=lat_step,
            solver_max_cpu_time=solver_max_cpu_time,
        )
        self._save_table(
            "global_launch_cost_latitude_sweep",
            [
                "latitude_deg",
                "fuel_consumed_kg",
                "final_mass_kg",
                "earth_rotation_speed_m_s",
                "solver_success",
            ],
            [
                (
                    float(row["latitude_deg"]),
                    float(row["fuel_consumed_kg"]),
                    float(row["final_mass_kg"]),
                    float(row["earth_rotation_speed_m_s"]),
                    int(bool(row["solver_success"])),
                )
                for row in rows
            ],
        )
        figure_available = self._render_global_launch_cost_heatmap(rows, "global_launch_cost_heatmap")
        return {
            "rows": rows,
            "figure_available": bool(figure_available),
        }

    def _run_global_launch_cost_sweep(self, lat_step=5.0, solver_max_cpu_time=120.0):
        lat_step = float(lat_step)
        if lat_step <= 0.0:
            raise ValueError("lat_step must be positive.")
        solver_max_cpu_time = float(solver_max_cpu_time)
        lat_values = np.arange(-90.0, 90.0 + 1.0e-9, lat_step, dtype=float)
        rows = []

        for lat in lat_values:
            cfg_i = copy.deepcopy(self.base_config)
            env_cfg_i = copy.deepcopy(self.base_env_config)
            env_cfg_i.launch_latitude = float(lat)
            env_cfg_i.launch_longitude = 0.0
            cfg_i.target_inclination = None

            try:
                with suppress_stdout():
                    env_i = Environment(env_cfg_i)
                    veh_i = Vehicle(cfg_i, env_i)
                    _, v_launch = env_i.get_launch_site_state()
                    res = solve_optimal_trajectory(
                        cfg_i,
                        veh_i,
                        env_i,
                        print_level=0,
                        solver_max_cpu_time=solver_max_cpu_time,
                    )
                success = bool(res.get("success", False))
                m_final = float(res["X3"][6, -1]) if success else np.nan
                fuel_used = float(cfg_i.launch_mass - m_final) if success else np.nan
                v_rot = float(np.linalg.norm(v_launch))
            except Exception:
                success = False
                m_final = np.nan
                fuel_used = np.nan
                v_rot = np.nan

            rows.append(
                {
                    "latitude_deg": float(lat),
                    "fuel_consumed_kg": float(fuel_used),
                    "final_mass_kg": float(m_final),
                    "earth_rotation_speed_m_s": float(v_rot),
                    "solver_success": int(success),
                }
            )
            print(
                f"  [Heatmap] lat={lat:6.1f} deg | success={int(success)} | "
                f"fuel={fuel_used if np.isfinite(fuel_used) else np.nan:.2f} kg"
            )

        return rows

    def _render_global_launch_cost_heatmap(self, rows, stem):
        ordered = sorted(rows, key=lambda row: float(row["latitude_deg"]))
        lat_values = np.array([float(row["latitude_deg"]) for row in ordered], dtype=float)
        fuel_values = np.array([float(row["fuel_consumed_kg"]) for row in ordered], dtype=float)
        success_values = np.array([int(row.get("solver_success", 0)) for row in ordered], dtype=int)
        lon_values = np.linspace(-180.0, 180.0, 73, dtype=float)

        if lat_values.size == 0 or fuel_values.size == 0:
            return False

        fuel_lat = fuel_values.copy()
        valid_success = np.isfinite(fuel_lat) & (success_values == 1)
        if not np.any(valid_success):
            return False
        if np.any(~valid_success):
            fuel_lat[~valid_success] = np.nan

        lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
        fuel_grid = np.repeat(fuel_lat[:, None], len(lon_values), axis=1)
        rad_lat = np.radians(lat_grid)
        rad_lon = np.radians(lon_grid)
        x = np.cos(rad_lat) * np.cos(rad_lon)
        y = np.cos(rad_lat) * np.sin(rad_lon)
        z = np.sin(rad_lat)

        norm = Normalize(vmin=float(np.nanmin(fuel_grid)), vmax=float(np.nanmax(fuel_grid)))
        cmap = plt.get_cmap("cividis")
        filled = np.nan_to_num(fuel_grid, nan=float(np.nanmax(fuel_grid)))
        colors = cmap(norm(filled))
        if np.any(~np.isfinite(fuel_grid)):
            colors[~np.isfinite(fuel_grid)] = np.array([0.72, 0.72, 0.72, 1.0], dtype=float)

        fig = plt.figure(figsize=(7.0, 6.2), layout="constrained")
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            x,
            y,
            z,
            facecolors=colors,
            rstride=1,
            cstride=1,
            linewidth=0.0,
            antialiased=True,
            shade=False,
        )

        phi = np.linspace(0.0, 2.0 * np.pi, 240)
        ax.plot(np.cos(phi), np.sin(phi), np.zeros_like(phi), color="white", alpha=0.35, linewidth=0.7)
        mer = np.linspace(-np.pi / 2.0, np.pi / 2.0, 160)
        ax.plot(np.cos(mer), np.zeros_like(mer), np.sin(mer), color="white", alpha=0.25, linewidth=0.6)

        best_idx = int(np.nanargmin(fuel_values[valid_success]))
        best_lat = float(lat_values[valid_success][best_idx])
        best_lat_rad = np.radians(best_lat)
        ax.scatter(
            [np.cos(best_lat_rad)],
            [0.0],
            [np.sin(best_lat_rad)],
            color="black",
            s=18,
            depthshade=False,
        )

        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.045, pad=0.04)
        cbar.set_label("Fuel consumed (kg)")
        ax.view_init(elev=18.0, azim=-48.0)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()

        self._finalize_figure(fig, stem)
        return True

    def analyze_interval_replay_audit(self, max_workers=None):
        debug._print_sub_header("Interval Replay Audit")
        opt_res = self._get_baseline_opt_res()
        if max_workers is None:
            max_workers = min(3, max(1, os.cpu_count() or 1))
        max_workers = max(1, int(max_workers))

        phase_jobs = [
            {
                "phase_name": "boost",
                "X": opt_res["X1"],
                "U": opt_res["U1"],
                "T": float(opt_res["T1"]),
                "t_start": 0.0,
                "stage_mode": "boost",
                "cfg": self.base_config,
                "env_cfg": self.base_env_config,
            }
        ]
        t_phase3 = float(opt_res["T1"] + opt_res.get("T2", 0.0))
        if opt_res.get("T2", 0.0) > 1e-8 and "X2" in opt_res:
            phase_jobs.append(
                {
                    "phase_name": "coast",
                    "X": opt_res["X2"],
                    "U": None,
                    "T": float(opt_res["T2"]),
                    "t_start": float(opt_res["T1"]),
                    "stage_mode": "coast",
                    "cfg": self.base_config,
                    "env_cfg": self.base_env_config,
                }
            )
        phase_jobs.append(
            {
                "phase_name": "ship",
                "X": opt_res["X3"],
                "U": opt_res["U3"],
                "T": float(opt_res["T3"]),
                "t_start": t_phase3,
                "stage_mode": "ship",
                "cfg": self.base_config,
                "env_cfg": self.base_env_config,
            }
        )

        phase_results = {}
        parallel_active = bool(max_workers > 1 and len(phase_jobs) > 1)
        if parallel_active:
            print(f"  Parallel execution enabled (workers={max_workers}).")
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    fut_to_name = {
                        executor.submit(_run_interval_replay_phase_worker, payload): str(payload["phase_name"])
                        for payload in phase_jobs
                    }
                    parallel_ok = True
                    total_phases = len(fut_to_name)
                    done_phases = 0
                    for fut in as_completed(fut_to_name):
                        done_phases += 1
                        self._print_parallel_progress("Replay phases", done_phases, total_phases)
                        key = fut_to_name[fut]
                        try:
                            phase_results[key] = fut.result()
                        except Exception:
                            parallel_ok = False
                            break
                if not parallel_ok:
                    print("  Worker failure during interval replay audit; switching to serial mode.")
                    parallel_active = False
                    phase_results = {}
            except Exception:
                print("  Parallel execution unavailable; switching to serial mode.")
                parallel_active = False
                phase_results = {}

        if not parallel_active:
            print("  Serial execution mode.")
            for payload in phase_jobs:
                key = str(payload["phase_name"])
                phase_results[key] = _run_interval_replay_phase_worker(payload)

        phase_series = []
        for payload in phase_jobs:
            key = str(payload["phase_name"])
            r = phase_results.get(key, None)
            if r is None:
                continue
            pos_s = np.asarray(r.get("position_error", []), dtype=float)
            vel_s = np.asarray(r.get("velocity_error", []), dtype=float)
            mass_s = np.asarray(r.get("mass_error", []), dtype=float)
            rel_s = np.asarray(r.get("relative_error", []), dtype=float)
            q_s = np.asarray(r.get("max_q_pa", []), dtype=float)
            g_s = np.asarray(r.get("max_g", []), dtype=float)
            ok_s = np.asarray(r.get("solver_ok", []), dtype=int)
            phase_series.append((key, pos_s, vel_s, mass_s, rel_s, q_s, g_s, ok_s))

        def _series_max(values):
            finite_max = []
            for arr in values:
                arr = np.asarray(arr, dtype=float)
                if arr.size > 0 and np.any(np.isfinite(arr)):
                    finite_max.append(float(np.nanmax(arr)))
            return float(max(finite_max)) if finite_max else np.nan

        total_intervals = int(sum(len(s[1]) for s in phase_series))

        def _series_limit_counts(values, limit):
            exceedances = 0
            finite_count = 0
            for arr in values:
                arr = np.asarray(arr, dtype=float)
                if arr.size == 0:
                    continue
                finite_mask = np.isfinite(arr)
                finite_count += int(np.count_nonzero(finite_mask))
                if np.any(finite_mask):
                    exceedances += int(np.count_nonzero(arr[finite_mask] > limit))
            missing_count = max(total_intervals - finite_count, 0)
            return exceedances, missing_count

        max_pos = _series_max([s[1] for s in phase_series])
        max_vel = _series_max([s[2] for s in phase_series])
        max_mass = _series_max([s[3] for s in phase_series])
        max_rel = _series_max([s[4] for s in phase_series])
        max_q_dense = _series_max([s[5] for s in phase_series])
        max_g_dense = _series_max([s[6] for s in phase_series])
        interval_failures = int(sum(int(np.size(s[7]) - np.count_nonzero(s[7])) for s in phase_series))
        q_violations, q_missing = _series_limit_counts([s[5] for s in phase_series], float(self.base_config.max_q_limit))
        g_violations, g_missing = _series_limit_counts([s[6] for s in phase_series], float(self.base_config.max_g_load))

        print(f"Max interval replay position error: {max_pos:.3e} m")
        print(f"Max interval replay velocity error: {max_vel:.3e} m/s")
        print(f"Max interval replay mass error:     {max_mass:.3e} kg")
        print(f"Max state-relative mismatch:        {max_rel:.3e}")
        print(f"Max dense-sampled Q during audit:   {max_q_dense/1000.0:.3f} kPa")
        print(f"Max dense-sampled g during audit:   {max_g_dense:.3f} g")
        print(f"Interval replay solver failures:    {interval_failures}")
        print(f"Dense replay Q violations:          {q_violations}")
        print(f"Dense replay g violations:          {g_violations}")
        if q_missing > 0 or g_missing > 0:
            print(f"Dense replay samples missing:       q={q_missing}, g={g_missing}")
        if (
            interval_failures == 0
            and q_violations == 0
            and g_violations == 0
            and q_missing == 0
            and g_missing == 0
            and np.isfinite(max_rel)
            and max_rel < 1e-6
        ):
            print(f">>> {debug.Style.GREEN}PASS: Interval replay mismatches stay small under an independent solver.{debug.Style.RESET}")
        else:
            print(
                f">>> {debug.Style.YELLOW}WARN: Interval replay audit found non-negligible mismatch, "
                f"solver failures, or dense replay path-limit violations.{debug.Style.RESET}"
            )

        rows = []
        for phase_name, pos_s, vel_s, m_s, rel_s, q_s, g_s, ok_s in phase_series:
            for k in range(len(pos_s)):
                q_margin = q_s[k] - self.base_config.max_q_limit if np.isfinite(q_s[k]) else np.nan
                g_margin = g_s[k] - self.base_config.max_g_load if np.isfinite(g_s[k]) else np.nan
                rows.append((phase_name, k, int(ok_s[k]), pos_s[k], vel_s[k], m_s[k], rel_s[k], q_s[k], g_s[k], q_margin, g_margin))
        self._save_table(
            "interval_replay_audit",
            [
                "phase",
                "interval_index",
                "solver_ok",
                "end_position_error_m",
                "end_velocity_error_m_s",
                "end_mass_error_kg",
                "max_relative_error",
                "sampled_max_q_pa",
                "sampled_max_g",
                "q_margin_pa",
                "g_margin",
            ],
            rows
        )

        eps = 1e-18
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        offset = 0
        for phase_name, pos_s, vel_s, m_s, rel_s, q_s, g_s, ok_s in phase_series:
            idx = np.arange(offset, offset + len(pos_s))
            ax1.semilogy(idx, np.maximum(pos_s, eps), label=f"{phase_name.title()} end-position error")
            ax2.semilogy(idx, np.maximum(rel_s, eps), label=f"{phase_name.title()} max relative error")
            offset += len(pos_s)
            if offset < total_intervals:
                ax1.axvline(offset - 0.5, color="0.5", linestyle="--", linewidth=0.8, alpha=0.6)
                ax2.axvline(offset - 0.5, color="0.5", linestyle="--", linewidth=0.8, alpha=0.6)

        ax1.set_ylabel("End-position error (m)")
        ax1.set_title("Interval Replay Audit")
        ax1.legend(loc="best")
        ax1.grid(True, which="both", ls="-", alpha=0.3)

        ax2.axhline(1e-6, color="tab:orange", linestyle="--", alpha=0.8, label="Reference threshold (1e-6)")
        ax2.set_xlabel("Node interval index")
        ax2.set_ylabel("Max relative error (-)")
        ax2.legend(loc="best")
        ax2.grid(True, which="both", ls="-", alpha=0.3)
        self._finalize_figure(fig, "interval_replay_audit")


    def _compute_drift_metrics(self, optimization_data, simulation_data):
        """
        Compute phase-wise optimizer-vs-simulator drift metrics.
        Mirrors debug.analyze_trajectory_drift logic but returns structured values.
        """
        if not isinstance(optimization_data, dict) or not isinstance(simulation_data, dict):
            return {"valid": False, "reason": "invalid_input"}

        phases_opt = []
        try:
            N1 = optimization_data["X1"].shape[1]
            t1_end = float(optimization_data["T1"])
            phases_opt.append({"t": np.linspace(0.0, t1_end, N1), "x": optimization_data["X1"], "label": "Phase 1"})

            current_t = t1_end
            if "X2" in optimization_data and float(optimization_data.get("T2", 0.0)) > 1e-4:
                N2 = optimization_data["X2"].shape[1]
                t2_end = current_t + float(optimization_data["T2"])
                phases_opt.append({"t": np.linspace(current_t, t2_end, N2), "x": optimization_data["X2"], "label": "Phase 2"})
                current_t = t2_end

            N3 = optimization_data["X3"].shape[1]
            t3_end = current_t + float(optimization_data["T3"])
            phases_opt.append({"t": np.linspace(current_t, t3_end, N3), "x": optimization_data["X3"], "label": "Phase 3"})
        except Exception:
            return {"valid": False, "reason": "bad_opt_data"}

        t_sim = simulation_data.get("t", None)
        y_sim = simulation_data.get("y", None)
        if not isinstance(t_sim, np.ndarray) or not isinstance(y_sim, np.ndarray):
            return {"valid": False, "reason": "bad_sim_data"}
        if y_sim.ndim != 2 or y_sim.shape[0] < 7 or len(t_sim) < 2 or y_sim.shape[1] != len(t_sim):
            return {"valid": False, "reason": "bad_sim_shape"}

        split_indices = [0] + (np.where(np.diff(t_sim) <= 1e-9)[0] + 1).tolist() + [len(t_sim)]
        sim_segments = []
        for i in range(len(split_indices) - 1):
            a = int(split_indices[i])
            b = int(split_indices[i + 1])
            if b - a >= 2:
                sim_segments.append({"t": t_sim[a:b], "y": y_sim[:, a:b]})

        if len(phases_opt) != len(sim_segments):
            return {"valid": False, "reason": "phase_count_mismatch"}

        phase_rows = []
        max_pos_err = 0.0
        max_vel_err = 0.0
        max_mass_err = 0.0
        sum_sq_pos = 0.0
        sum_sq_vel = 0.0
        sum_sq_mass = 0.0
        sample_count = 0
        final_position_drift = np.nan
        final_velocity_drift = np.nan
        final_mass_drift = np.nan

        for p_opt, p_sim in zip(phases_opt, sim_segments):
            f_sim = interp1d(
                p_sim["t"], p_sim["y"], axis=1,
                kind="linear", bounds_error=False, fill_value="extrapolate"
            )
            y_interp = f_sim(p_opt["t"])
            pos_err = np.linalg.norm(p_opt["x"][0:3, :] - y_interp[0:3, :], axis=0)
            vel_err = np.linalg.norm(p_opt["x"][3:6, :] - y_interp[3:6, :], axis=0)
            mass_err = np.abs(p_opt["x"][6, :] - y_interp[6, :])
            p_max = float(np.max(pos_err))
            p_rms = float(np.sqrt(np.mean(pos_err * pos_err)))
            v_max = float(np.max(vel_err))
            v_rms = float(np.sqrt(np.mean(vel_err * vel_err)))
            m_max = float(np.max(mass_err))
            m_rms = float(np.sqrt(np.mean(mass_err * mass_err)))
            phase_rows.append((p_opt["label"], p_max, p_rms, v_max, v_rms, m_max, m_rms))
            max_pos_err = max(max_pos_err, p_max)
            max_vel_err = max(max_vel_err, v_max)
            max_mass_err = max(max_mass_err, m_max)
            sum_sq_pos += float(np.sum(pos_err * pos_err))
            sum_sq_vel += float(np.sum(vel_err * vel_err))
            sum_sq_mass += float(np.sum(mass_err * mass_err))
            sample_count += int(len(pos_err))
            final_position_drift = float(pos_err[-1])
            final_velocity_drift = float(vel_err[-1])
            final_mass_drift = float(mass_err[-1])

        if sample_count <= 0:
            return {"valid": False, "reason": "no_overlap_samples"}

        pass_flag = bool(max_pos_err <= 2000.0 and max_vel_err <= 30.0 and max_mass_err <= 500.0)
        return {
            "valid": True,
            "phase_rows": phase_rows,
            "max_pos_m": float(max_pos_err),
            "max_vel_m_s": float(max_vel_err),
            "max_mass_kg": float(max_mass_err),
            "rms_pos_m": float(np.sqrt(sum_sq_pos / sample_count)),
            "rms_vel_m_s": float(np.sqrt(sum_sq_vel / sample_count)),
            "rms_mass_kg": float(np.sqrt(sum_sq_mass / sample_count)),
            "final_position_drift_m": final_position_drift,
            "final_velocity_drift_m_s": final_velocity_drift,
            "final_mass_drift_kg": final_mass_drift,
            "pass_flag": int(pass_flag)
        }

    # 3. DRIFT ANALYSIS
    def analyze_drift(self, opt_res, sim_res):
        debug._print_sub_header("3. Drift Analysis (Opt vs Sim)")
        drift = self._compute_drift_metrics(opt_res, sim_res)
        if not drift.get("valid", False):
            reason = drift.get("reason", "unknown")
            print(f"  Drift analysis could not be completed ({reason}).")
            self._save_table(
                "drift_summary",
                ["metric", "value"],
                [("valid", 0), ("reason", reason)]
            )
            return

        print(
            f"{'Phase':<10} | {'Max Pos (m)':<12} | {'RMS Pos (m)':<12} | "
            f"{'Max Vel (m/s)':<14} | {'RMS Vel (m/s)':<14} | "
            f"{'Max Mass (kg)':<14} | {'RMS Mass (kg)':<14}"
        )
        print("-" * 108)
        for phase, p_max, p_rms, v_max, v_rms, m_max, m_rms in drift["phase_rows"]:
            print(
                f"{phase:<10} | {p_max:<12.3f} | {p_rms:<12.3f} | "
                f"{v_max:<14.4f} | {v_rms:<14.4f} | "
                f"{m_max:<14.4f} | {m_rms:<14.4f}"
            )
        print(f"  Max Position Drift: {drift['max_pos_m']:.3f} m")
        print(f"  Max Velocity Drift: {drift['max_vel_m_s']:.4f} m/s")
        print(f"  Max Mass Drift:     {drift['max_mass_kg']:.4f} kg")
        print(f"  RMS Position Drift: {drift['rms_pos_m']:.3f} m")
        print(f"  RMS Velocity Drift: {drift['rms_vel_m_s']:.4f} m/s")
        print(f"  RMS Mass Drift:     {drift['rms_mass_kg']:.4f} kg")
        print(f"  Final Position Drift: {drift['final_position_drift_m']:.3f} m")
        print(f"  Final Velocity Drift: {drift['final_velocity_drift_m_s']:.4f} m/s")
        print(f"  Final Mass Drift:     {drift['final_mass_drift_kg']:.4f} kg")
        if drift["pass_flag"]:
            print(f">>> {debug.Style.GREEN}PASS: Simulation concurs with optimizer trajectory.{debug.Style.RESET}")
        else:
            print(f">>> {debug.Style.YELLOW}WARN: Drift exceeds reliability thresholds.{debug.Style.RESET}")

        self._save_table(
            "drift_phase_metrics",
            [
                "phase",
                "max_position_drift_m",
                "rms_position_drift_m",
                "max_velocity_drift_m_s",
                "rms_velocity_drift_m_s",
                "max_mass_drift_kg",
                "rms_mass_drift_kg",
            ],
            drift["phase_rows"]
        )
        self._save_table(
            "drift_summary",
            ["metric", "value"],
            [
                ("valid", 1),
                ("max_position_drift_m", drift["max_pos_m"]),
                ("max_velocity_drift_m_s", drift["max_vel_m_s"]),
                ("max_mass_drift_kg", drift["max_mass_kg"]),
                ("rms_position_drift_m", drift["rms_pos_m"]),
                ("rms_velocity_drift_m_s", drift["rms_vel_m_s"]),
                ("rms_mass_drift_kg", drift["rms_mass_kg"]),
                ("final_position_drift_m", drift["final_position_drift_m"]),
                ("final_velocity_drift_m_s", drift["final_velocity_drift_m_s"]),
                ("final_mass_drift_kg", drift["final_mass_drift_kg"]),
                ("pass_flag", drift["pass_flag"]),
                ("threshold_position_m", 2000.0),
                ("threshold_velocity_m_s", 30.0),
                ("threshold_mass_kg", 500.0),
            ]
        )

        phases = [r[0] for r in drift["phase_rows"]]
        pos_vals = [r[1] for r in drift["phase_rows"]]
        vel_vals = [r[3] for r in drift["phase_rows"]]
        mass_vals = [r[5] for r in drift["phase_rows"]]
        x = np.arange(len(phases))

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        # Use log-scale bars so tiny drifts remain visible against large thresholds.
        def _plot_drift_panel(ax, vals, threshold, color, ylabel):
            vals_arr = np.asarray(vals, dtype=float)
            vals_plot = np.maximum(vals_arr, 1e-9)
            finite_vals = vals_plot[np.isfinite(vals_plot)]
            if finite_vals.size == 0:
                finite_vals = np.array([1e-9], dtype=float)
            ax.bar(x, vals_plot, color=color, alpha=0.8)
            ax.axhline(threshold, color="tab:orange", linestyle="--", alpha=0.8, label="Threshold")
            ax.set_yscale("log")
            y_min = min(np.min(finite_vals) * 0.6, threshold * 1e-4)
            y_max = max(np.max(finite_vals) * 2.0, threshold * 2.0)
            ax.set_ylim(max(y_min, 1e-10), y_max)
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", axis="y", alpha=0.3)

        _plot_drift_panel(ax1, pos_vals, 2000.0, "tab:blue", "Position Drift (m)")
        ax1.legend(loc="best")
        _plot_drift_panel(ax2, vel_vals, 30.0, "tab:green", "Velocity Drift (m/s)")
        _plot_drift_panel(ax3, mass_vals, 500.0, "tab:purple", "Mass Drift (kg)")
        ax3.set_xticks(x)
        ax3.set_xticklabels(phases)

        fig.suptitle("Optimizer vs Simulator Drift by Phase")
        self._finalize_figure(fig, "drift_summary")

        self._summary["q3_drift"] = {
            "max_position_drift_m": drift["max_pos_m"],
            "max_velocity_drift_m_s": drift["max_vel_m_s"],
            "max_mass_drift_kg": drift["max_mass_kg"],
            "rms_position_drift_m": drift["rms_pos_m"],
            "rms_velocity_drift_m_s": drift["rms_vel_m_s"],
            "rms_mass_drift_kg": drift["rms_mass_kg"],
            "final_position_drift_m": drift["final_position_drift_m"],
            "final_velocity_drift_m_s": drift["final_velocity_drift_m_s"],
            "final_mass_drift_kg": drift["final_mass_drift_kg"],
            "pass_flag": drift["pass_flag"],
        }

    # 14. THEORETICAL EFFICIENCY
    def analyze_theoretical_efficiency(self):
        debug._print_sub_header("14. Idealized Delta-V Reference (Hohmann-like Comparison)")

        # 1. Get Simulation Data
        opt_res = self._get_baseline_opt_res()
        with suppress_stdout():
            sim_res = run_simulation(opt_res, self.veh, self.base_config)

        terminal_metrics = evaluate_terminal_state(sim_res, self.base_config, self.base_env_config)
        traj_diag = self._trajectory_diagnostics(sim_res, self.veh, self.base_config)
        path_diag = self._evaluate_path_compliance(traj_diag, self.base_config)
        terminal_strict_ok = bool(terminal_metrics.get("strict_ok", False))
        mission_success = bool(self._is_strict_path_success(terminal_metrics, traj_diag, self.base_config))
        if not terminal_metrics["terminal_valid"]:
            print("  Cannot evaluate efficiency: simulation produced invalid terminal state.")
            return

        # 2. Calculate Actual Delta-V (Integrated)
        t, y, u = sim_res["t"], sim_res["y"], sim_res["u"]
        dv_actual = 0.0
        for i in range(len(t) - 1):
            dt = t[i + 1] - t[i]
            m, th = y[6, i], u[0, i]
            if th > 0.01:
                m_stg2_wet = (
                    self.base_config.stage_2.dry_mass
                    + self.base_config.stage_2.propellant_mass
                    + self.base_config.payload_mass
                )
                stg = self.base_config.stage_1 if m > m_stg2_wet + 1000 else self.base_config.stage_2
                r = y[0:3, i]
                env_state = self.env.get_state_sim(r, t[i])
                isp_eff = stg.isp_vac + (env_state["pressure"] / stg.p_sl) * (stg.isp_sl - stg.isp_vac)
                thrust = th * stg.thrust_vac * (isp_eff / stg.isp_vac)
                dv_actual += (thrust / m) * dt

        if dv_actual < 1.0:
            print("  Simulation failed to generate Delta-V (Crash or no burn).")
            return

        # 3. Calculate an idealized lower-bound delta-v reference (Hohmann-like).
        # This is a diagnostic reference for ascent losses, not a strict fuel lower bound.
        mu = self.env.config.earth_mu
        R1 = self.env.config.earth_radius_equator
        R2 = R1 + self.base_config.target_altitude
        lat = self.base_env_config.launch_latitude
        v_surf = self.env.config.earth_omega_vector[2] * R1 * np.cos(np.radians(lat))

        a_tx = (R1 + R2) / 2.0
        v_peri_tx = np.sqrt(mu * (2 / R1 - 1 / a_tx))
        v_apo_tx = np.sqrt(mu * (2 / R2 - 1 / a_tx))
        v_circ_2 = np.sqrt(mu / R2)

        dv_burn1 = v_peri_tx - v_surf
        dv_burn2 = v_circ_2 - v_apo_tx
        dv_theoretical = dv_burn1 + dv_burn2

        efficiency = (dv_theoretical / dv_actual) * 100.0

        print(f"  Idealized Lower-Bound Delta-V Reference (Hohmann-like): {dv_theoretical:.1f} m/s")
        print(f"    - Burn 1 (Surf -> Transfer):     {dv_burn1:.1f} m/s")
        print(f"    - Burn 2 (Circularization):      {dv_burn2:.1f} m/s")
        print(f"  Actual Delta-V (Simulation):       {dv_actual:.1f} m/s")
        print(f"  Gravity/Drag/Steering Losses:      {dv_actual - dv_theoretical:.1f} m/s")
        print(f"  Mission Efficiency:                {efficiency:.1f}%")
        print("  Note: This is an idealized delta-v reference, not a strict lower bound on fuel burned.")
        if terminal_strict_ok and not mission_success:
            print(
                "  Terminal target met but path constraints were violated "
                f"(q_ok={int(path_diag['q_ok'])}, g_ok={int(path_diag['g_ok'])})."
            )
        if not mission_success:
            print(
                f">>> {debug.Style.YELLOW}NOTE: Strict mission criteria (terminal + path) were not met; "
                f"efficiency is diagnostic only.{debug.Style.RESET}"
            )

        self._save_table(
            "theoretical_efficiency",
            [
                "mission_success_strict_path",
                "terminal_strict_success",
                "q_ok",
                "g_ok",
                "dv_theoretical_m_s",
                "dv_actual_m_s",
                "losses_m_s",
                "efficiency_percent",
            ],
            [
                (
                    int(mission_success),
                    int(terminal_strict_ok),
                    int(path_diag["q_ok"]),
                    int(path_diag["g_ok"]),
                    dv_theoretical,
                    dv_actual,
                    dv_actual - dv_theoretical,
                    efficiency,
                )
            ],
        )
        losses = dv_actual - dv_theoretical
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left: direct Delta-V comparison.
        bar_labels = ["Idealized Reference", "Actual Trajectory"]
        bar_vals = [dv_theoretical, dv_actual]
        bar_colors = ["tab:blue", "tab:green"]
        bars = ax1.bar(bar_labels, bar_vals, color=bar_colors, alpha=0.8)
        for bar, val in zip(bars, bar_vals):
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + 0.015 * max(bar_vals),
                f"{val:.0f} m/s",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        delta_sign = "+" if losses >= 0.0 else "-"
        ax1.annotate(
            f"{delta_sign}{abs(losses):.0f} m/s",
            xy=(1, dv_actual),
            xytext=(0.48, 0.92),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="0.25"),
            ha="center",
            va="center",
            fontsize=10,
        )
        ax1.set_ylabel("Delta-V (m/s)")
        ax1.set_title("Delta-V Budget Comparison")
        ax1.grid(True, axis="y", alpha=0.3)

        # Right: efficiency context, intentionally without a pass/fail threshold line.
        eff_color = "tab:orange"
        x_max = max(100.0, efficiency + 8.0)
        ax2.barh(["Mission Efficiency"], [efficiency], color=eff_color, alpha=0.85)
        ax2.set_xlim(0.0, x_max)
        ax2.set_xlabel("Efficiency (%)")
        ax2.set_title("Efficiency Context")
        ax2.grid(True, axis="x", alpha=0.3)
        ax2.text(
            efficiency + 1.0,
            0,
            f"{efficiency:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
            color=eff_color,
        )
        ax2.text(
            0.03,
            0.95,
            (
                f"Losses: {losses:.0f} m/s\n"
                "Diagnostic only\n"
                "Not a fuel lower bound"
            ),
            transform=ax2.transAxes,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.7"),
        )

        fig.suptitle("Idealized Delta-V Reference (Hohmann-like vs Simulated Trajectory)")
        self._finalize_figure(fig, "theoretical_efficiency")

        if mission_success:
            print(
                f">>> {debug.Style.YELLOW}NOTE: Efficiency is a contextual delta-v benchmark only; "
                f"it is not used as a pass/fail criterion.{debug.Style.RESET}"
            )
        else:
            print(f">>> {debug.Style.YELLOW}NOTE: Skip pass/fail efficiency grading because mission was off-target.{debug.Style.RESET}")

    # 15. SMOOTH ODE BENCHMARK (FORMAL ORDER CHECK)
    def analyze_smooth_integrator_benchmark(self):
        debug._print_sub_header("15. Smooth ODE Benchmark (Formal Integrator Order)")
        print("Benchmarking Euler and RK4 on a smooth scalar decay ODE with exact solution.")

        k = 1.3
        T = 4.0
        y0 = 1.0
        dt_values = np.array([0.40, 0.20, 0.10, 0.05, 0.025, 0.0125], dtype=float)

        def f(y):
            return -k * y

        def exact(t):
            return np.exp(-k * t)

        euler_errors = []
        rk4_errors = []

        print(f"{'dt (s)':<10} | {'Euler Err':<12} | {'RK4 Err':<12}")
        print("-" * 42)
        for dt in dt_values:
            # Euler
            t = 0.0
            y = float(y0)
            while t < T - 1e-14:
                h = min(dt, T - t)
                y = y + h * f(y)
                t += h
            err_eu = abs(y - exact(T))
            euler_errors.append(err_eu)

            # RK4
            t = 0.0
            y = float(y0)
            while t < T - 1e-14:
                h = min(dt, T - t)
                k1 = f(y)
                k2 = f(y + 0.5 * h * k1)
                k3 = f(y + 0.5 * h * k2)
                k4 = f(y + h * k3)
                y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                t += h
            err_rk = abs(y - exact(T))
            rk4_errors.append(err_rk)

            print(f"{dt:<10.4f} | {err_eu:<12.4e} | {err_rk:<12.4e}")

        # Fit in the asymptotic region (smaller dt) to avoid coarse-step effects.
        fit_idx = np.where(dt_values <= 0.1)[0]
        if len(fit_idx) < 3:
            fit_idx = np.arange(len(dt_values))
        log_dt = np.log10(dt_values[fit_idx])
        log_eu = np.log10(np.maximum(np.array(euler_errors)[fit_idx], 1e-16))
        log_rk = np.log10(np.maximum(np.array(rk4_errors)[fit_idx], 1e-16))
        coeff_eu = np.polyfit(log_dt, log_eu, 1)
        coeff_rk = np.polyfit(log_dt, log_rk, 1)

        slope_eu = float(coeff_eu[0])
        slope_rk = float(coeff_rk[0])
        print(f"  Estimated Euler order (dt<=0.1s): {slope_eu:.2f} (expected ~1)")
        print(f"  Estimated RK4 order (dt<=0.1s):   {slope_rk:.2f} (expected ~4)")
        euler_pass = int(abs(slope_eu - 1.0) <= 0.25)
        rk4_pass = int(abs(slope_rk - 4.0) <= 0.35)
        overall_pass = int(bool(euler_pass and rk4_pass))
        if overall_pass:
            print(f">>> {debug.Style.GREEN}PASS: Formal integrator orders match theory.{debug.Style.RESET}")
        else:
            print(f">>> {debug.Style.YELLOW}WARN: Measured formal order deviates from theory.{debug.Style.RESET}")

        self._save_table(
            "smooth_integrator_benchmark",
            ["dt_s", "euler_state_error", "rk4_state_error"],
            [(float(dt_values[i]), float(euler_errors[i]), float(rk4_errors[i])) for i in range(len(dt_values))]
        )
        self._save_table(
            "smooth_integrator_benchmark_fit",
            ["method", "slope"],
            [("Euler", slope_eu), ("RK4", slope_rk)]
        )
        self._save_table(
            "smooth_integrator_benchmark_summary",
            ["metric", "value"],
            [
                ("euler_slope", slope_eu),
                ("rk4_slope", slope_rk),
                ("euler_pass", euler_pass),
                ("rk4_pass", rk4_pass),
                ("overall_pass", overall_pass),
            ]
        )

        fig = plt.figure(figsize=(10, 6))
        plt.loglog(dt_values, euler_errors, "o-r", label=f"Euler (slope={slope_eu:.2f})")
        plt.loglog(dt_values, rk4_errors, "s-b", label=f"RK4 (slope={slope_rk:.2f})")
        plt.xlabel("Time Step dt (s)")
        plt.ylabel("State Error at T")
        plt.title("Smooth ODE Benchmark: Formal Integrator Order")
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend()
        self._finalize_figure(fig, "smooth_integrator_benchmark")
        self._summary["q3_integrator_order"] = {
            "euler_slope": slope_eu,
            "rk4_slope": slope_rk,
            "pass_flag": overall_pass,
        }

def build_cli_parser():
    parser = argparse.ArgumentParser(description="Reliability and robustness analysis suite.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for exported figures and CSV tables.")
    parser.add_argument("--seed", type=int, default=1337, help="Global random seed for reproducible stochastic analyses.")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum parallel workers (set 1 for fully serial mode).")
    parser.add_argument("--course-core", action="store_true", help="Run only the recommended paper-safe course-core evidence set.")
    parser.add_argument("--analysis", choices=("all", "warm-start-multiseed", "global-launch-cost"), default="all", help="Select a single analysis entry point instead of the full suite.")
    parser.add_argument("--trials", type=int, default=DEFAULT_WARM_START_TRIALS, help="Warm-start multi-seed trial count when --analysis=warm-start-multiseed.")
    parser.add_argument("--heatmap-lat-step", type=float, default=5.0, help="Latitude step in degrees when --analysis=global-launch-cost.")
    parser.add_argument("--heatmap-solver-cpu-time", type=float, default=120.0, help="Per-latitude CPU time cap for --analysis=global-launch-cost.")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive plot windows (save figures only).")
    parser.add_argument("--no-save", action="store_true", help="Do not save figures/CSV outputs.")
    return parser


def main(argv=None):
    args = build_cli_parser().parse_args(argv)

    suite = ReliabilitySuite(
        output_dir=args.output_dir,
        save_figures=not args.no_save,
        show_plots=not args.no_show,
        random_seed=args.seed,
        analysis_profile="course_core" if args.course_core else "default",
    )
    if args.analysis == "warm-start-multiseed":
        suite.analyze_randomized_multistart(n_trials=args.trials)
        return
    if args.analysis == "global-launch-cost":
        suite.analyze_global_launch_cost(
            lat_step=args.heatmap_lat_step,
            solver_max_cpu_time=args.heatmap_solver_cpu_time,
        )
        return
    suite.run_all(max_workers=args.max_workers)


if __name__ == "__main__":
    main()
