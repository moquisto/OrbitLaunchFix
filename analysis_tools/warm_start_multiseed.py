from __future__ import annotations

import argparse
import contextlib
import copy
import io
import os
from pathlib import Path
import sys
import time
import threading
import traceback
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    module_dir = Path(__file__).resolve().parent
    repo_root = module_dir.parent
    for path_entry in (str(module_dir), str(repo_root)):
        if path_entry not in sys.path:
            sys.path.insert(0, path_entry)

from orbit_launch import debug, guidance
from orbit_launch.environment import Environment
from orbit_launch.main import solve_optimal_trajectory
from orbit_launch.simulation import run_simulation
from orbit_launch.trajectory_metrics import ellipsoidal_altitude_m
from orbit_launch.vehicle import Vehicle


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as fnull:
        with contextlib.redirect_stdout(fnull):
            yield


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
    pos_sigma: float
    vel_sigma: float
    mass_sigma: float
    throttle_sigma: float
    direction_sigma_deg: float

    def as_dict(self) -> dict[str, float]:
        return {
            "time_scale": float(self.time_scale),
            "pos_sigma": float(self.pos_sigma),
            "vel_sigma": float(self.vel_sigma),
            "mass_sigma": float(self.mass_sigma),
            "throttle_sigma": float(self.throttle_sigma),
            "direction_sigma_deg": float(self.direction_sigma_deg),
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


DEFAULT_STRUCTURED_SEED_SPECS = (
    StructuredSeedSpec(
        label="time_minus",
        time_scale=0.9925,
        t2_scale=0.9950,
        pos_sigma=-0.0025,
        vel_sigma=-0.0025,
        mass_sigma=-0.0015,
        throttle_sigma=0.0030,
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
        throttle_sigma=-0.0030,
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
        throttle_sigma=-0.0020,
        direction_sigma_deg=0.50,
        direction_axis=(0.0, 1.0, 0.0),
    ),
    StructuredSeedSpec(
        label="state_plus",
        time_scale=1.0000,
        t2_scale=1.0000,
        pos_sigma=0.0050,
        vel_sigma=0.0050,
        mass_sigma=0.0030,
        throttle_sigma=0.0020,
        direction_sigma_deg=-0.50,
        direction_axis=(0.0, 1.0, 0.0),
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
        arr *= 1.0 + rel_bias * weights
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

    def build_structured_variants(self, base_guess, cfg):
        if not isinstance(base_guess, dict):
            raise ValueError("Initial guess must be a dict.")

        variants = []
        for spec in self.structured_specs:
            guess = copy.deepcopy(base_guess)

            t1 = float(guess.get("T1", cfg.sequence.min_stage_1_burn))
            t3 = float(guess.get("T3", cfg.sequence.min_stage_2_burn))
            guess["T1"] = max(cfg.sequence.min_stage_1_burn, t1 * float(spec.time_scale))
            guess["T3"] = max(cfg.sequence.min_stage_2_burn, t3 * float(spec.time_scale))
            if guess.get("T2", None) is not None:
                guess["T2"] = max(0.0, float(guess["T2"]) * float(spec.t2_scale))

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

            variants.append(
                SeedVariant(
                    label=str(spec.label),
                    guess=guess,
                    params=SeedParams(
                        time_scale=spec.time_scale,
                        pos_sigma=spec.pos_sigma,
                        vel_sigma=spec.vel_sigma,
                        mass_sigma=spec.mass_sigma,
                        throttle_sigma=spec.throttle_sigma,
                        direction_sigma_deg=spec.direction_sigma_deg,
                    ),
                )
            )
        return variants

    def build_randomized_variant(self, base_guess, cfg, rng, label):
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
        direction_sigma_deg = float(np.degrees(np.arcsin(direction_sigma)))

        t1 = float(guess.get("T1", cfg.sequence.min_stage_1_burn))
        t3 = float(guess.get("T3", cfg.sequence.min_stage_2_burn))
        guess["T1"] = max(cfg.sequence.min_stage_1_burn, t1 * time_scale)
        guess["T3"] = max(cfg.sequence.min_stage_2_burn, t3 * time_scale)
        if guess.get("T2", None) is not None:
            guess["T2"] = max(0.0, float(guess["T2"]) * (1.0 + float(rng.uniform(-frac, frac))))

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

        return SeedVariant(
            label=label,
            guess=guess,
            params=SeedParams(
                time_scale=time_scale,
                pos_sigma=pos_sigma,
                vel_sigma=vel_sigma,
                mass_sigma=mass_sigma,
                throttle_sigma=throttle_sigma,
                direction_sigma_deg=direction_sigma_deg,
            ),
        )

    def build_trial_variants(self, base_guess, cfg, rng, n_trials):
        n_use = max(MIN_WARM_START_TRIALS, int(n_trials))
        variants = [
            SeedVariant(
                label="nominal",
                guess=copy.deepcopy(base_guess),
                params=SeedParams(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            )
        ]
        structured = self.build_structured_variants(base_guess, cfg)
        variants.extend(structured[: max(0, n_use - 1)])

        next_index = len(variants)
        while len(variants) < n_use:
            label = f"rand_{next_index:02d}"
            variants.append(self.build_randomized_variant(base_guess, cfg, rng, label))
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

    @staticmethod
    def _empty_result(trial_index, label, params, status, runtime_s=np.nan):
        return {
            "trial_index": int(trial_index),
            "label": str(label),
            "params": params.as_dict() if isinstance(params, SeedParams) else dict(params),
            "solver_success": 0,
            "mission_success": 0,
            "terminal_strict_success": 0,
            "path_ok": 0,
            "q_ok": 0,
            "g_ok": 0,
            "status": str(status),
            "final_mass_kg": np.nan,
            "runtime_s": float(runtime_s) if np.isfinite(runtime_s) else np.nan,
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
    def _sanitize_label(label):
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(label))
        safe = safe.strip("_")
        return safe or "trial"

    def _trial_log_path(self, trial_index, label):
        return self.logs_dir / f"{int(trial_index) + 1:02d}_{self._sanitize_label(label)}.log"

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
        terminal_strict_ok = False
        path_ok = False
        q_ok = False
        g_ok = False
        m_final = np.nan
        status = "FAILED"
        trajectory_profile = None

        try:
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
                            terminal_strict_ok = bool(terminal.get("strict_ok", False))
                            path_ok = bool(path.get("path_ok", False))
                            q_ok = bool(path.get("q_ok", False))
                            g_ok = bool(path.get("g_ok", False))
                            mission_ok = bool(terminal_strict_ok and path_ok)
                            if mission_ok:
                                status = "MISSION_OK"
                            elif terminal_strict_ok and not path_ok:
                                status = "PATH"
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
                        status = "SOLVE_FAIL"
                except Exception:
                    traceback.print_exc()
                    status = "SOLVE_ERR"
                print(f"[Trial Log] status={status}")
        except Exception:
            status = "SOLVE_ERR"

        return {
            "trial_index": int(trial_index),
            "label": str(variant.label),
            "params": variant.params.as_dict(),
            "solver_success": int(solver_ok),
            "mission_success": int(mission_ok),
            "terminal_strict_success": int(terminal_strict_ok),
            "path_ok": int(path_ok),
            "q_ok": int(q_ok),
            "g_ok": int(g_ok),
            "status": str(status),
            "final_mass_kg": float(m_final) if np.isfinite(m_final) else np.nan,
            "runtime_s": float(time.time() - t0),
            "trajectory_profile": trajectory_profile,
        }

    def analyze(self, n_trials=DEFAULT_WARM_START_TRIALS):
        debug._print_sub_header("Warm-Start Multi-Seed Robustness")
        n_use = max(MIN_WARM_START_TRIALS, int(n_trials))
        rng = np.random.default_rng(self.suite.seed_sequence.spawn(1)[0])
        planned_labels = self._planned_labels(n_use)

        rows = []
        solver_success_flags = []
        mission_success_flags = []
        masses_scored = []
        runtimes = []
        labels = []
        trajectory_profiles = []
        trial_results = {}

        print(
            f"{'Trial':<7} | {'Guess':<11} | {'Time x':<8} | {'Pos %':<7} | {'Vel %':<7} | "
            f"{'Mass %':<8} | {'Throt %':<8} | {'Dir (deg)':<10} | {'Solve':<5} | {'Mission':<7} | {'Status':<12} | "
            f"{'Final Mass (kg)':<15} | {'Runtime (s)':<10}"
        )
        print("-" * 169)

        self._print_progress("Warm-start setup", 0, 3)
        print("  Building nominal guidance warm start...")
        guidance_guess = self._build_nominal_guidance_guess()
        self._print_progress("Warm-start setup", 1, 3)

        structured_count = 0
        variants = []
        if guidance_guess is not None:
            structured_count = min(len(self.seed_factory.structured_specs), max(0, n_use - 1))
            variants = self.seed_factory.build_trial_variants(guidance_guess, self.suite.base_config, rng, n_use)
        print(f"  Prepared {structured_count + int(guidance_guess is not None)} deterministic seed(s).")
        self._print_progress("Warm-start setup", 2, 3)

        if guidance_guess is None:
            for i in range(n_use):
                label = planned_labels[i]
                params = SeedParams(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                trial_results[i] = self._empty_result(i, label, params, "GUESS_FAIL", runtime_s=0.0)
        else:
            print(f"  Queued {len(variants)} trial(s) for evaluation.")
        self._print_progress("Warm-start setup", 3, 3)

        total_trials = len(variants)
        for done_trials, variant in enumerate(variants, start=1):
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
                r = self._empty_result(i, planned_labels[i], SeedParams(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan), "INTERNAL_ERR")

            params = r["params"]
            m_final = float(r["final_mass_kg"]) if np.isfinite(r["final_mass_kg"]) else np.nan
            solver_ok = int(r["solver_success"])
            mission_ok = int(r["mission_success"])
            terminal_strict_ok = int(r["terminal_strict_success"])
            path_ok = int(r["path_ok"])
            q_ok = int(r["q_ok"])
            g_ok = int(r["g_ok"])
            status = str(r["status"])
            dt = float(r["runtime_s"]) if np.isfinite(r["runtime_s"]) else np.nan
            label = str(r["label"])
            trajectory_profiles.append(r.get("trajectory_profile", None))

            labels.append(label)
            masses_scored.append(m_final if mission_ok and np.isfinite(m_final) else np.nan)
            runtimes.append(dt)
            solver_success_flags.append(solver_ok)
            mission_success_flags.append(mission_ok)

            rows.append(
                {
                    "trial": i + 1,
                    "label": label,
                    "time_scale": params["time_scale"],
                    "position_noise_sigma": params["pos_sigma"],
                    "velocity_noise_sigma": params["vel_sigma"],
                    "mass_noise_sigma": params["mass_sigma"],
                    "throttle_noise_sigma": params["throttle_sigma"],
                    "direction_noise_sigma_deg": params["direction_sigma_deg"],
                    "solver_success": solver_ok,
                    "mission_success": mission_ok,
                    "terminal_strict_success": terminal_strict_ok,
                    "path_ok": path_ok,
                    "q_ok": q_ok,
                    "g_ok": g_ok,
                    "status": status,
                    "final_mass_kg": m_final,
                    "runtime_s": dt,
                }
            )
            print(
                f"{i+1:<7} | {label:<11} | {params['time_scale']:<8.3f} | "
                f"{100.0 * params['pos_sigma']:<7.2f} | {100.0 * params['vel_sigma']:<7.2f} | "
                f"{100.0 * params['mass_sigma']:<8.2f} | {100.0 * params['throttle_sigma']:<8.2f} | "
                f"{params['direction_sigma_deg']:<10.2f} | {solver_ok:<5d} | {mission_ok:<7d} | "
                f"{status:<12} | {m_final:<15.2f} | {dt:<10.2f}"
            )

        solver_rate = float(np.mean(solver_success_flags)) if solver_success_flags else 0.0
        success_rate = float(np.mean(mission_success_flags)) if mission_success_flags else 0.0
        valid_masses = np.array([m for m in masses_scored if np.isfinite(m)], dtype=float)
        if len(valid_masses) > 0:
            best_mass = float(np.max(valid_masses))
            mass_gap_kg = [(best_mass - m) if np.isfinite(m) else np.nan for m in masses_scored]
        else:
            best_mass = np.nan
            mass_gap_kg = [np.nan for _ in masses_scored]
        mass_spread = float(np.max(valid_masses) - np.min(valid_masses)) if len(valid_masses) > 1 else np.nan

        print(f"\nSolver convergence rate: {solver_rate:.1%}")
        print(f"Strict mission success rate: {success_rate:.1%}")
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
                "time_scale",
                "position_noise_sigma",
                "velocity_noise_sigma",
                "mass_noise_sigma",
                "throttle_noise_sigma",
                "direction_noise_sigma_deg",
                "solver_success",
                "mission_success",
                "terminal_strict_success",
                "path_ok",
                "q_ok",
                "g_ok",
                "status",
                "final_mass_kg",
                "final_mass_scored_kg",
                "final_mass_gap_to_best_kg",
                "runtime_s",
            ],
            [
                (
                    rows[i]["trial"],
                    rows[i]["label"],
                    rows[i]["time_scale"],
                    rows[i]["position_noise_sigma"],
                    rows[i]["velocity_noise_sigma"],
                    rows[i]["mass_noise_sigma"],
                    rows[i]["throttle_noise_sigma"],
                    rows[i]["direction_noise_sigma_deg"],
                    rows[i]["solver_success"],
                    rows[i]["mission_success"],
                    rows[i]["terminal_strict_success"],
                    rows[i]["path_ok"],
                    rows[i]["q_ok"],
                    rows[i]["g_ok"],
                    rows[i]["status"],
                    rows[i]["final_mass_kg"],
                    masses_scored[i],
                    mass_gap_kg[i],
                    rows[i]["runtime_s"],
                )
                for i in range(len(rows))
            ],
        )

        reference_idx = None
        if trajectory_profiles and isinstance(trajectory_profiles[0], dict):
            reference_idx = 0
        else:
            for i, profile in enumerate(trajectory_profiles):
                if isinstance(profile, dict):
                    reference_idx = i
                    break

        trajectory_rows = []
        delta_rows = []
        max_position_sep_km = np.nan
        max_alt_delta_km = np.nan
        max_vel_delta_m_s = np.nan
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

            if i != reference_idx:
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
                        progress[k],
                        delta_downrange_km[k],
                        delta_alt_km[k],
                        delta_vel_m_s[k],
                        delta_mass_kg[k],
                        pos_sep_km[k],
                    )
                )

        if max_position_candidates:
            max_position_sep_km = float(np.nanmax(np.array(max_position_candidates, dtype=float)))
        if max_alt_candidates:
            max_alt_delta_km = float(np.nanmax(np.array(max_alt_candidates, dtype=float)))
        if max_vel_candidates:
            max_vel_delta_m_s = float(np.nanmax(np.array(max_vel_candidates, dtype=float)))

        if trajectory_rows:
            self.suite._save_table(
                "randomized_multistart_trajectories",
                [
                    "trial",
                    "label",
                    "status",
                    "solver_success",
                    "mission_success",
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
                [
                    ("reference_label", reference_label),
                    ("solver_rate", solver_rate),
                    ("mission_success_rate", success_rate),
                    ("mass_spread_kg", mass_spread),
                    ("max_position_separation_km", max_position_sep_km),
                    ("max_altitude_delta_km", max_alt_delta_km),
                    ("max_velocity_delta_m_s", max_vel_delta_m_s),
                ],
            )

        if reference_idx is not None:
            print(f"Reference trajectory for comparisons: {reference_label}")
            if np.isfinite(max_position_sep_km):
                print(f"Max position separation vs {reference_label}: {max_position_sep_km:.3f} km")
            if np.isfinite(max_alt_delta_km):
                print(f"Max altitude delta vs {reference_label}: {max_alt_delta_km:.3f} km")
            if np.isfinite(max_vel_delta_m_s):
                print(f"Max velocity delta vs {reference_label}: {max_vel_delta_m_s:.3f} m/s")

        fig = plt.figure(figsize=(13.5, 10.0))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
        ax1 = fig.add_subplot(gs[0, 0])
        ax_zoom = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = ax3.twinx()
        colors = plt.get_cmap("tab10")

        if reference_idx is None:
            ax1.axis("off")
            ax_zoom.axis("off")
            ax2.axis("off")
            ax3.axis("off")
            ax4.axis("off")
            ax1.text(
                0.5,
                0.5,
                "No forward-replayed trajectories were available for comparison.",
                ha="center",
                va="center",
                fontsize=12,
            )
        else:
            for i, profile in enumerate(trajectory_profiles):
                if not isinstance(profile, dict):
                    continue
                color = "black" if i == reference_idx else colors(i % 10)
                mission_ok = bool(rows[i]["mission_success"])
                line_style = "-" if mission_ok else "--"
                line_width = 2.3 if i == reference_idx else 1.6
                alpha = 0.95 if mission_ok else 0.70
                label = rows[i]["label"]
                if i == reference_idx:
                    label = f"{label} (reference)"
                elif not mission_ok:
                    label = f"{label} (path/terminal miss)"

                downrange_km = np.asarray(profile["downrange_km"], dtype=float)
                altitude_km = np.asarray(profile["ellipsoidal_altitude_km"], dtype=float)
                progress_pct = 100.0 * np.asarray(profile["progress"], dtype=float)
                velocity_m_s = np.asarray(profile["velocity_m_s"], dtype=float)
                pos_sep_km = np.linalg.norm(
                    np.vstack(
                        [
                            np.asarray(profile["x_m"], dtype=float),
                            np.asarray(profile["y_m"], dtype=float),
                            np.asarray(profile["z_m"], dtype=float),
                        ]
                    )
                    - ref_pos,
                    axis=0,
                ) / 1000.0

                ax1.plot(downrange_km, altitude_km, color=color, linestyle=line_style, linewidth=line_width, alpha=alpha, label=label)
                ax1.scatter(downrange_km[-1], altitude_km[-1], color=color, s=18, alpha=alpha)
                ax_zoom.plot(downrange_km, altitude_km, color=color, linestyle=line_style, linewidth=line_width, alpha=alpha)
                ax_zoom.scatter(downrange_km[-1], altitude_km[-1], color=color, s=18, alpha=alpha)

                delta_alt_km = altitude_km - ref_alt
                delta_vel_m_s = velocity_m_s - ref_vel
                ax2.plot(progress_pct, delta_alt_km, color=color, linestyle=line_style, linewidth=line_width, alpha=alpha)
                ax3.plot(progress_pct, pos_sep_km, color=color, linestyle=line_style, linewidth=line_width, alpha=alpha)
                ax4.plot(progress_pct, delta_vel_m_s, color=color, linestyle=":", linewidth=max(1.1, line_width - 0.3), alpha=min(1.0, alpha + 0.05))

            ax1.set_title("Forward-Replayed Trajectories from Warm-Start Seeds")
            ax1.set_xlabel("Downrange (km)")
            ax1.set_ylabel("Ellipsoidal altitude (km)")
            ax1.legend(loc="best", fontsize=8)

            zoom_window = self._compute_insertion_zoom_window(trajectory_profiles)
            ax_zoom.set_title("Orbital Insertion Zoom")
            ax_zoom.set_xlabel("Downrange (km)")
            ax_zoom.set_ylabel("Ellipsoidal altitude (km)")
            if zoom_window is not None:
                (x_lo, x_hi), (y_lo, y_hi) = zoom_window
                ax_zoom.set_xlim(x_lo, x_hi)
                ax_zoom.set_ylim(y_lo, y_hi)
            ax_zoom.grid(True, alpha=0.35)

            ax2.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
            ax2.set_title(f"Altitude delta vs {reference_label}")
            ax2.set_xlabel("Mission progress (%)")
            ax2.set_ylabel("Delta altitude (km)")

            ax3.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
            ax3.set_title(f"Position / speed difference vs {reference_label}")
            ax3.set_xlabel("Mission progress (%)")
            ax3.set_ylabel("Position separation (km)")
            ax4.axhline(0.0, color="0.55", linestyle=":", linewidth=0.9)
            ax4.set_ylabel("Delta speed (m/s)")

            summary = (
                f"Trials: {n_use}\n"
                f"Solver: {solver_rate:.1%}\n"
                f"Mission: {success_rate:.1%}\n"
                + (f"Mass spread: {mass_spread:.3f} kg\n" if np.isfinite(mass_spread) else "Mass spread: n/a\n")
                + (f"Max pos sep: {max_position_sep_km:.3f} km\n" if np.isfinite(max_position_sep_km) else "Max pos sep: n/a\n")
                + (f"Max |dv|: {max_vel_delta_m_s:.3f} m/s" if np.isfinite(max_vel_delta_m_s) else "Max |dv|: n/a")
            )
            ax_zoom.text(
                0.015,
                0.985,
                summary,
                transform=ax_zoom.transAxes,
                va="top",
                ha="left",
                bbox=dict(facecolor="white", alpha=0.82, edgecolor="0.7"),
            )

        self.suite._finalize_figure(fig, "randomized_multistart")


def build_cli_parser():
    parser = argparse.ArgumentParser(
        description="Run only the warm-start multi-seed reliability test.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for exported figures and CSV tables.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Global random seed for reproducible runs.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_WARM_START_TRIALS,
        help="Number of warm-start trials to evaluate. Runs are nominal, four structured variants, then randomized variants up to the requested count.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive plot windows.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save figures or CSV outputs.",
    )
    return parser


def main(argv=None):
    args = build_cli_parser().parse_args(argv)

    sys.modules.setdefault("analysis_tools.warm_start_multiseed", sys.modules[__name__])
    sys.modules.setdefault("warm_start_multiseed", sys.modules[__name__])
    if __package__ in (None, ""):
        from relabilityanalysis import ReliabilitySuite
    else:
        from .relabilityanalysis import ReliabilitySuite

    t0 = time.time()
    print(
        f"[Runner] Warm-start multi-seed test: trials={max(MIN_WARM_START_TRIALS, int(args.trials))}, "
        f"seed={int(args.seed)}"
    )
    suite = ReliabilitySuite(
        output_dir=args.output_dir,
        save_figures=not args.no_save,
        show_plots=not args.no_show,
        random_seed=args.seed,
    )
    print("[Runner] Starting analysis...")
    suite.analyze_randomized_multistart(n_trials=args.trials)
    print(f"[Runner] Finished in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
