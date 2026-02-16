# reliabilityanalysis.py
# Purpose: Comprehensive Reliability & Robustness Analysis Suite.
# Implements tactics to verify trust in the optimization results.

import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import contextlib
import os
import sys
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import csv
from pathlib import Path
from datetime import datetime
import argparse
import platform

# Project Imports
from config import StarshipBlock2, EARTH_CONFIG, RELIABILITY_ANALYSIS_TOGGLES
from vehicle import Vehicle
from environment import Environment
from main import solve_optimal_trajectory
from simulation import run_simulation
import debug

# --- Helper to suppress output during batch runs ---
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            yield



def wilson_interval(successes, n, z=1.96):
    """Wilson score confidence interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 0.0
    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    half = z * np.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)




def evaluate_terminal_state(
    sim_res,
    cfg,
    env_cfg,
    alt_tol_m=10000.0,
    vel_tol_m_s=20.0,
    radial_vel_tol_m_s=25.0,
    inc_tol_deg=0.75
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
        "v_f_m_s": np.nan,
        "target_alt_m": float(cfg.target_altitude),
        "target_vel_m_s": np.nan,
        "target_inclination_deg": np.nan,
        "alt_err_m": np.nan,
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

    r_f = rf - env_cfg.earth_radius_equator
    v_f = np.linalg.norm(final_state[3:6])
    m_final = float(final_state[6])
    target_alt = float(cfg.target_altitude)
    target_vel = np.sqrt(env_cfg.earth_mu / (env_cfg.earth_radius_equator + target_alt))
    target_inc_deg = float(cfg.target_inclination) if cfg.target_inclination is not None else abs(float(env_cfg.launch_latitude))

    alt_err = r_f - target_alt
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
        "r_f_m": r_f,
        "v_f_m_s": v_f,
        "target_vel_m_s": target_vel,
        "target_inclination_deg": target_inc_deg,
        "alt_err_m": alt_err,
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


def normalized_orbit_error(
    metrics,
    alt_tol_m=10000.0,
    vel_tol_m_s=20.0,
    radial_vel_tol_m_s=25.0,
    inc_tol_deg=0.75,
):
    """
    Return a unitless terminal-orbit error norm.
    Values <= 1.0 indicate orbit criteria are satisfied (fuel excluded).
    """
    if not isinstance(metrics, dict) or not bool(metrics.get("terminal_valid", False)):
        return np.nan
    terms = [
        float(metrics.get("alt_err_m", np.nan)) / alt_tol_m,
        float(metrics.get("vel_err_m_s", np.nan)) / vel_tol_m_s,
        float(metrics.get("radial_velocity_m_s", np.nan)) / radial_vel_tol_m_s,
        float(metrics.get("inclination_err_deg", np.nan)) / inc_tol_deg,
    ]
    if not np.all(np.isfinite(terms)):
        return np.nan
    return float(np.sqrt(np.sum(np.square(terms))))


def _export_csv(path, headers, rows):
    def _format_cell(value):
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
    def __init__(self, output_dir=None, save_figures=True, show_plots=True, random_seed=1337):
        self.base_config = copy.deepcopy(StarshipBlock2)
        self.base_env_config = copy.deepcopy(EARTH_CONFIG)
        self.test_toggles = copy.deepcopy(RELIABILITY_ANALYSIS_TOGGLES)
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

        # Cache one baseline optimization to keep all analyses comparable.
        self._baseline_opt_res = None
        
        # Initialize Baseline Physics
        self.env = Environment(self.base_env_config)
        self.veh = Vehicle(self.base_config, self.env)

        # Report-oriented plotting defaults.
        plt.rcParams.update({
            "figure.dpi": 120,
            "savefig.dpi": 300,
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
                ("vehicle_name", self.base_config.name),
                ("target_altitude_m", self.base_config.target_altitude),
                ("target_inclination_deg", self.base_config.target_inclination),
                ("num_nodes", self.base_config.num_nodes),
                ("max_iter", self.base_config.max_iter),
                ("max_q_limit_pa", self.base_config.max_q_limit),
                ("max_g_limit", self.base_config.max_g_load),
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

    def _run_if_enabled(self, flag_name, label, fn, *args, **kwargs):
        enabled = bool(getattr(self.test_toggles, flag_name, True))
        t0 = time.time()
        if enabled:
            try:
                fn(*args, **kwargs)
                status = "completed"
                err = ""
            except Exception as exc:
                status = "error"
                err = str(exc)
                self._analysis_execution.append((flag_name, label, int(enabled), status, time.time() - t0, err[:240]))
                raise
            self._analysis_execution.append((flag_name, label, int(enabled), status, time.time() - t0, err))
            return True
        print(f"\n[Skip] {label} disabled in RELIABILITY_ANALYSIS_TOGGLES.{flag_name}")
        self._analysis_execution.append((flag_name, label, int(enabled), "skipped", time.time() - t0, ""))
        return False

    @staticmethod
    def _rankdata(values):
        """Simple rank transform with average-tie handling."""
        x = np.asarray(values, dtype=float)
        n = len(x)
        if n == 0:
            return np.array([], dtype=float)
        order = np.argsort(x, kind="mergesort")
        ranks_sorted = np.arange(1, n + 1, dtype=float)
        xs = x[order]
        i = 0
        while i < n:
            j = i + 1
            while j < n and xs[j] == xs[i]:
                j += 1
            if j - i > 1:
                avg_rank = 0.5 * (ranks_sorted[i] + ranks_sorted[j - 1])
                ranks_sorted[i:j] = avg_rank
            i = j
        ranks = np.empty(n, dtype=float)
        ranks[order] = ranks_sorted
        return ranks

    @staticmethod
    def _corr(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if len(a) < 2 or len(b) < 2:
            return np.nan
        a_std = np.std(a)
        b_std = np.std(b)
        if a_std < 1e-16 or b_std < 1e-16:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    @staticmethod
    def _refine_peak_time(t_series, y_series, idx_peak):
        """
        Refine a sampled peak time/value with a local quadratic fit.
        Falls back to sampled values if the parabola is ill-posed.
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
        return float(t[i] + x_peak), float(y_peak)

    def _draw_uncertainty_multipliers(self, rng, model="gaussian_independent"):
        """
        Draw uncertainty multipliers for thrust, ISP, and atmosphere density.
        """
        if model == "gaussian_independent":
            thrust_mult = rng.normal(1.0, 0.02)
            isp_mult = rng.normal(1.0, 0.01)
            dens_mult = rng.normal(1.0, 0.10)
        elif model == "uniform_bounded":
            thrust_mult = rng.uniform(0.94, 1.06)
            isp_mult = rng.uniform(0.97, 1.03)
            dens_mult = rng.uniform(0.70, 1.30)
        elif model == "gaussian_correlated":
            sig = np.array([0.02, 0.01, 0.10], dtype=float)
            corr = np.array([
                [1.0,  0.55, -0.20],
                [0.55, 1.0, -0.15],
                [-0.20, -0.15, 1.0]
            ], dtype=float)
            cov = np.outer(sig, sig) * corr
            draw = rng.multivariate_normal(np.ones(3), cov)
            thrust_mult, isp_mult, dens_mult = draw[0], draw[1], draw[2]
        else:
            raise ValueError(f"Unknown uncertainty model '{model}'")

        # Guardrails for physically meaningful environment and engine factors.
        thrust_mult = float(np.clip(thrust_mult, 0.80, 1.20))
        isp_mult = float(np.clip(isp_mult, 0.85, 1.15))
        dens_mult = float(np.clip(dens_mult, 0.20, 2.00))
        return thrust_mult, isp_mult, dens_mult

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

    def _simulate_uncertainty_case(self, opt_res, thrust_mult, isp_mult, dens_mult):
        """
        Run one perturbed open-loop simulation and return terminal/constraint diagnostics.
        """
        cfg = copy.deepcopy(self.base_config)
        env_cfg = copy.deepcopy(self.base_env_config)
        cfg.stage_1.thrust_vac *= thrust_mult
        cfg.stage_2.thrust_vac *= thrust_mult
        cfg.stage_1.isp_vac *= isp_mult
        cfg.stage_2.isp_vac *= isp_mult
        env_cfg.density_multiplier = dens_mult

        sim_res = {"y": np.zeros((7, 1)), "t": np.array([0.0]), "u": np.zeros((4, 1))}
        try:
            with suppress_stdout():
                env_mc = Environment(env_cfg)
                veh_mc = Vehicle(cfg, env_mc)
                sim_res = run_simulation(opt_res, veh_mc, cfg)
            terminal = evaluate_terminal_state(sim_res, cfg, env_cfg)
            traj = self._trajectory_diagnostics(sim_res, veh_mc, cfg)
        except Exception:
            terminal = evaluate_terminal_state({"y": np.zeros((7, 1))}, cfg, env_cfg)
            traj = {
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

        return {
            "thrust_multiplier": float(thrust_mult),
            "isp_multiplier": float(isp_mult),
            "density_multiplier": float(dens_mult),
            "terminal": terminal,
            "trajectory": traj
        }

    def run_all(self, monte_carlo_samples=200):
        """Runs all analysis modules sequentially."""
        self._analysis_execution = []
        ran_any = False
        mc_n = max(1, int(monte_carlo_samples))
        mc_max_precision = max(mc_n, int(np.ceil(1.30 * mc_n)))
        # Keep a conservative minimum before CI-based stopping to avoid over-optimistic early stops.
        mc_min_precision = min(mc_max_precision, max(80, mc_n // 4))
        mc_batch = max(5, min(50, max(1, mc_n // 10)))

        # Q1: Credible minimum-fuel solution (local optimum robustness).
        ran_any |= self._run_if_enabled(
            "initial_guess_robustness",
            "Initial-guess robustness",
            self.analyze_initial_guess_robustness
        )
        ran_any |= self._run_if_enabled("grid_independence", "Grid independence", self.analyze_grid_independence)
        ran_any |= self._run_if_enabled(
            "collocation_defect_audit",
            "Collocation defect audit",
            self.analyze_collocation_defect_audit
        )
        ran_any |= self._run_if_enabled(
            "theoretical_efficiency",
            "Theoretical efficiency",
            self.analyze_theoretical_efficiency
        )

        # Q2: Numerical/statistical accuracy and uncertainty.
        ran_any |= self._run_if_enabled("integrator_tolerance", "Integrator tolerance", self.analyze_integrator_tolerance)
        ran_any |= self._run_if_enabled(
            "event_time_convergence",
            "Event-time convergence",
            self.analyze_event_time_convergence
        )
        ran_any |= self._run_if_enabled(
            "monte_carlo_precision_target",
            "Monte Carlo precision target",
            self.analyze_monte_carlo_precision_target,
            min_samples=mc_min_precision,
            max_samples=mc_max_precision,
            batch_size=mc_batch
        )
        ran_any |= self._run_if_enabled(
            "global_sensitivity",
            "Global sensitivity ranking",
            self.analyze_global_sensitivity,
            N_samples=mc_n
        )

        # Q3: Code correctness/reliability sanity checks.
        ran_any |= self._run_if_enabled(
            "smooth_integrator_benchmark",
            "Smooth ODE integrator benchmark",
            self.analyze_smooth_integrator_benchmark
        )
        ran_any |= self._run_if_enabled(
            "conservative_invariants",
            "Conservative-invariant audit",
            self.analyze_conservative_invariants
        )

        # Q5/Q7: Cliff-edge behavior and conclusion evidence.
        ran_any |= self._run_if_enabled(
            "finite_time_sensitivity",
            "Finite-time sensitivity",
            self.analyze_chaos_lyapunov
        )
        ran_any |= self._run_if_enabled("bifurcation", "Bifurcation", self.analyze_bifurcation)
        ran_any |= self._run_if_enabled(
            "bifurcation_2d_map",
            "Bifurcation 2D map",
            self.analyze_bifurcation_2d_map
        )

        # Q3: End-to-end optimizer vs simulator consistency.
        deep_dive_flags = [
            "drift",
        ]
        if any(bool(getattr(self.test_toggles, name, True)) for name in deep_dive_flags):
            print(f"\n{debug.Style.BOLD}--- Generating Baseline Solution for Deep Dive ---{debug.Style.RESET}")
            opt_res = self._get_baseline_opt_res()
            sim_res = None
            if bool(getattr(self.test_toggles, "drift", True)):
                sim_res = run_simulation(opt_res, self.veh, self.base_config)
            ran_any |= self._run_if_enabled("drift", "Drift analysis", self.analyze_drift, opt_res, sim_res)
        else:
            print("\n[Skip] All deep-dive tests are disabled in RELIABILITY_ANALYSIS_TOGGLES.")

        if not ran_any:
            print("\nNo analyses executed. Enable tests in RELIABILITY_ANALYSIS_TOGGLES in config.py.")

        self._save_table(
            "analysis_execution_log",
            ["toggle", "label", "enabled", "status", "runtime_s", "error_message"],
            self._analysis_execution
        )

    # 1. GRID INDEPENDENCE STUDY
    def analyze_grid_independence(self):
        debug._print_sub_header("1. Grid Independence Study")
        # User limit: Max 140 nodes
        # Increased resolution for smoother convergence graph
        node_counts = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
        masses = []
        runtimes = []
        success_flags = []
        
        print(f"{'Nodes':<6} | {'Final Mass (kg)':<15} | {'Runtime (s)':<10} | {'Status':<10}")
        print("-" * 50)
        
        for N in node_counts:
            cfg = copy.deepcopy(self.base_config)
            cfg.num_nodes = N
            
            t0 = time.time()
            try:
                with suppress_stdout():
                    res = solve_optimal_trajectory(cfg, self.veh, self.env, print_level=0)
                
                if res.get("success", False):
                    m_final = res['X3'][6, -1]
                    status = "Converged"
                    ok = True
                else:
                    m_final = np.nan
                    status = "Failed"
                    ok = False
            except Exception:
                m_final = np.nan
                status = "Failed"
                ok = False
            
            dt = time.time() - t0
            masses.append(m_final)
            runtimes.append(dt)
            success_flags.append(int(ok))
            print(f"{N:<6} | {m_final:<15.1f} | {dt:<10.2f} | {status:<10}")

        self._save_table(
            "grid_independence",
            ["nodes", "final_mass_kg", "runtime_s", "solver_success"],
            [(node_counts[i], masses[i], runtimes[i], success_flags[i]) for i in range(len(node_counts))]
        )
            
        # Convergence Check
        mass_by_nodes = {node_counts[i]: masses[i] for i in range(len(node_counts))}
        m_100 = mass_by_nodes.get(100, np.nan)
        m_140 = mass_by_nodes.get(140, np.nan)
        if np.isfinite(m_100) and np.isfinite(m_140):
            diff = abs(m_140 - m_100)
            print(f"\nMass Delta (140 vs 100 nodes): {diff:.1f} kg")
            if diff < 100.0:
                print(f">>> {debug.Style.GREEN}PASS: Grid Independent (<100kg change).{debug.Style.RESET}")
            else:
                print(f">>> {debug.Style.YELLOW}WARN: Grid Dependent (Solution still changing).{debug.Style.RESET}")
        else:
            print(f"\n>>> {debug.Style.YELLOW}WARN: Cannot assess grid independence because node 100 or 140 failed.{debug.Style.RESET}")
        
        # Visualization: Convergence vs Cost
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Filter valid runs for plotting
        valid_indices = [i for i, m in enumerate(masses) if np.isfinite(m)]
        valid_nodes = [node_counts[i] for i in valid_indices]
        valid_masses = [masses[i] for i in valid_indices]
        valid_runtimes = [runtimes[i] for i in valid_indices]
        
        color = 'tab:blue'
        ax1.set_xlabel('Number of Nodes')
        ax1.set_ylabel('Final Mass (kg)', color=color)
        ax1.plot(valid_nodes, valid_masses, 'o-', color=color, label='Final Mass')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Runtime (s)', color=color)
        ax2.plot(valid_nodes, valid_runtimes, 'x--', color=color, label='Runtime')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Grid Independence Study: Convergence & Cost')
        self._finalize_figure(fig, "grid_independence_convergence_cost")

    # 2. INTEGRATOR TOLERANCE SWEEP
    def analyze_integrator_tolerance(self):
        debug._print_sub_header("2. Integrator Tolerance Sweep")
        
        # Get a solution first
        opt_res = self._get_baseline_opt_res()
            
        # Increased resolution: Logarithmic sweep from 1e-6 to 1e-12
        tols = [
            (1e-6, 1e-9), (5e-7, 5e-10), (1e-7, 1e-10), (5e-8, 5e-11), (1e-8, 1e-11),
            (5e-9, 5e-12), (1e-9, 1e-12), (1e-10, 1e-13), (1e-11, 1e-14), (1e-12, 1e-14)
        ]
        results = []
        
        print(f"{'RTOL':<8} | {'ATOL':<8} | {'Final Alt (km)':<15} | {'Final Vel (m/s)':<15}")
        print("-" * 55)
        
        for rtol, atol in tols:
            with suppress_stdout():
                sim_res = run_simulation(opt_res, self.veh, self.base_config, rtol=rtol, atol=atol)
            
            r_f = np.linalg.norm(sim_res['y'][0:3, -1]) - self.env.config.earth_radius_equator
            v_f = np.linalg.norm(sim_res['y'][3:6, -1])
            results.append((r_f, v_f))
            print(f"{rtol:<8.0e} | {atol:<8.0e} | {r_f/1000:<15.3f} | {v_f:<15.3f}")

        self._save_table(
            "integrator_tolerance",
            ["rtol", "atol", "final_altitude_m", "final_velocity_m_s"],
            [(tols[i][0], tols[i][1], results[i][0], results[i][1]) for i in range(len(tols))]
        )
            
        # Check consistency
        # Compare the Standard (1e-9) vs Tightest (1e-12) for Pass/Fail
        # Find index of 1e-9 (approx) and last one
        idx_std = 6 # (1e-9, 1e-12) is at index 6
        drift_standard = abs(results[idx_std][0] - results[-1][0])
        drift_loose = abs(results[0][0] - results[-1][0])
        
        print(f"\nDrift (1e-6 vs 1e-12): {drift_loose:.1f} m")
        print(f"Drift (1e-9 vs 1e-12): {drift_standard:.1f} m")
        
        if drift_standard < 50.0:
            print(f">>> {debug.Style.GREEN}PASS: Simulation is converged at operating tolerance (1e-9).{debug.Style.RESET}")
        else:
            print(f">>> {debug.Style.RED}FAIL: Numerical Instability at 1e-9 (Drift {drift_standard:.1f}m).{debug.Style.RESET}")
            
        # Visualization: Convergence Plot
        # Calculate drifts relative to tightest tolerance (last one)
        ref_r = results[-1][0]
        drifts = [abs(r - ref_r) for r, v in results[:-1]]
        rtols = [t[0] for t in tols[:-1]]
        
        fig = plt.figure(figsize=(10, 6))
        plt.loglog(rtols, drifts, 'bo-', label='Altitude Drift')
        plt.xlabel('Integrator Relative Tolerance (rtol)')
        plt.ylabel('Position Drift vs Baseline (m)')
        baseline_rtol = tols[-1][0]
        plt.title(f'Integrator Convergence (Baseline: rtol={baseline_rtol:.0e})')
        plt.grid(True, which="both", ls="-")
        plt.gca().invert_xaxis() # Standard convention: higher precision (lower tol) to the right
        self._finalize_figure(fig, "integrator_tolerance_convergence")

    def analyze_initial_guess_robustness(self, n_trials=5):
        debug._print_sub_header("Initial-Guess Robustness (Warm-Start Variants)")
        variants = [
            {"label": "nominal", "start_shift": 0.0, "end_shift": 0.0, "gain_scale": 1.00},
            {"label": "early_pitch", "start_shift": -3.0, "end_shift": -3.0, "gain_scale": 1.00},
            {"label": "late_pitch", "start_shift": 3.0, "end_shift": 3.0, "gain_scale": 1.00},
            {"label": "low_gain", "start_shift": 0.0, "end_shift": 0.0, "gain_scale": 0.70},
            {"label": "high_gain", "start_shift": 0.0, "end_shift": 0.0, "gain_scale": 1.30},
        ]
        n_use = max(1, min(int(n_trials), len(variants)))
        rows = []
        masses = []
        runtimes = []
        successes = 0

        print(f"{'Case':<12} | {'Status':<10} | {'Final Mass (kg)':<15} | {'Runtime (s)':<10}")
        print("-" * 58)

        for i in range(n_use):
            cfg = copy.deepcopy(self.base_config)
            env_cfg = copy.deepcopy(self.base_env_config)
            v = variants[i]

            base_start = cfg.sequence.pitch_start_time
            base_end = cfg.sequence.pitch_end_time
            cfg.sequence.pitch_start_time = max(0.0, base_start + v["start_shift"])
            cfg.sequence.pitch_end_time = max(cfg.sequence.pitch_start_time + 1.0, base_end + v["end_shift"])
            cfg.sequence.pitch_gain = max(0.0, cfg.sequence.pitch_gain * v["gain_scale"])

            env_i = Environment(env_cfg)
            veh_i = Vehicle(cfg, env_i)
            t0 = time.time()
            try:
                with suppress_stdout():
                    res = solve_optimal_trajectory(cfg, veh_i, env_i, print_level=0)
                ok = bool(res.get("success", False))
                m_final = float(res["X3"][6, -1]) if ok else np.nan
            except Exception:
                ok = False
                m_final = np.nan
            dt = time.time() - t0

            successes += int(ok)
            masses.append(m_final)
            runtimes.append(dt)
            rows.append((v["label"], int(ok), m_final, dt))
            print(f"{v['label']:<12} | {('Converged' if ok else 'Failed'):<10} | {m_final:<15.2f} | {dt:<10.2f}")

        valid_masses = np.array([m for m in masses if np.isfinite(m)], dtype=float)
        mass_spread = float(np.max(valid_masses) - np.min(valid_masses)) if len(valid_masses) > 1 else np.nan
        if len(valid_masses) >= 1:
            mass_ref = float(np.median(valid_masses))
            mass_delta_g = [((m - mass_ref) * 1000.0) if np.isfinite(m) else np.nan for m in masses]
        else:
            mass_ref = np.nan
            mass_delta_g = [np.nan for _ in masses]
        success_rate = successes / n_use
        print(f"\nSuccess rate: {success_rate:.1%}")
        if np.isfinite(mass_spread):
            print(f"Final-mass spread across successful runs: {mass_spread:.6f} kg ({mass_spread*1000.0:.2f} g)")
        if success_rate >= 0.8 and (not np.isfinite(mass_spread) or mass_spread < 300.0):
            print(f">>> {debug.Style.GREEN}PASS: Optimization is robust to warm-start variants.{debug.Style.RESET}")
        else:
            print(f">>> {debug.Style.YELLOW}WARN: Warm-start sensitivity detected (or low convergence rate).{debug.Style.RESET}")

        self._save_table(
            "initial_guess_robustness",
            ["case", "solver_success", "final_mass_kg", "final_mass_delta_g", "runtime_s"],
            [
                (rows[i][0], rows[i][1], rows[i][2], mass_delta_g[i], rows[i][3])
                for i in range(len(rows))
            ]
        )

        fig, ax1 = plt.subplots(figsize=(10, 6))
        x = np.arange(n_use)
        ax1.bar(x, runtimes, color="tab:blue", alpha=0.35, label="Runtime (s)")
        ax2 = ax1.twinx()
        ax2.plot(x, mass_delta_g, "o-", color="tab:green", label="Final mass delta (g)")
        labels = [variants[i]["label"] for i in range(n_use)]
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=20)
        ax1.set_ylabel("Runtime (s)")
        ax2.set_ylabel("Final Mass Delta (g)")
        if np.isfinite(mass_ref):
            ax2.axhline(0.0, color="tab:green", linestyle="--", alpha=0.4)
        ax1.set_title("Warm-Start Variant Robustness")
        lines, labels_l = ax1.get_legend_handles_labels()
        lines2, labels_r = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels_l + labels_r, loc="best")
        self._finalize_figure(fig, "initial_guess_robustness")

    def analyze_collocation_defect_audit(self):
        debug._print_sub_header("Collocation Defect Audit")
        opt_res = self._get_baseline_opt_res()

        def phase_defects(X, U, T, t_start, stage_mode):
            if U is None:
                N_loc = X.shape[1] - 1
            else:
                N_loc = U.shape[1]
            dt = T / max(N_loc, 1)
            pos_def = np.zeros(N_loc)
            vel_def = np.zeros(N_loc)
            mass_def = np.zeros(N_loc)
            rel_state_max = np.zeros(N_loc)
            for k in range(N_loc):
                xk = X[:, k]
                if U is None:
                    throttle = 0.0
                    u_dir = np.array([1.0, 0.0, 0.0])
                else:
                    uk = U[:, k]
                    throttle = float(uk[0])
                    u_dir = uk[1:]
                tk = float(t_start + k * dt)

                def dyn(x, t):
                    return self.veh.get_dynamics(x, throttle, u_dir, t, stage_mode=stage_mode, scaling=None)

                k1 = dyn(xk, tk)
                k2 = dyn(xk + 0.5 * dt * k1, tk + 0.5 * dt)
                k3 = dyn(xk + 0.5 * dt * k2, tk + 0.5 * dt)
                k4 = dyn(xk + dt * k3, tk + dt)
                x_pred = xk + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                defect = X[:, k + 1] - x_pred
                denom = np.maximum(np.abs(X[:, k + 1]), 1.0)
                pos_def[k] = float(np.linalg.norm(defect[0:3]))
                vel_def[k] = float(np.linalg.norm(defect[3:6]))
                mass_def[k] = float(abs(defect[6]))
                rel_state_max[k] = float(np.max(np.abs(defect) / denom))
            return pos_def, vel_def, mass_def, rel_state_max

        phase_series = []
        pos1, vel1, m1, rel1 = phase_defects(opt_res["X1"], opt_res["U1"], opt_res["T1"], 0.0, "boost")
        phase_series.append(("boost", pos1, vel1, m1, rel1))

        t_phase3 = float(opt_res["T1"] + opt_res.get("T2", 0.0))
        if opt_res.get("T2", 0.0) > 1e-8 and "X2" in opt_res:
            pos2, vel2, m2, rel2 = phase_defects(opt_res["X2"], None, opt_res["T2"], float(opt_res["T1"]), "coast")
            phase_series.append(("coast", pos2, vel2, m2, rel2))

        pos3, vel3, m3, rel3 = phase_defects(opt_res["X3"], opt_res["U3"], opt_res["T3"], t_phase3, "ship")
        phase_series.append(("ship", pos3, vel3, m3, rel3))

        max_pos = float(max(np.max(s[1]) for s in phase_series))
        max_vel = float(max(np.max(s[2]) for s in phase_series))
        max_mass = float(max(np.max(s[3]) for s in phase_series))
        max_rel = float(max(np.max(s[4]) for s in phase_series))

        print(f"Max collocation position defect: {max_pos:.3e} m")
        print(f"Max collocation velocity defect: {max_vel:.3e} m/s")
        print(f"Max collocation mass defect:     {max_mass:.3e} kg")
        print(f"Max state-relative defect:       {max_rel:.3e}")
        if max_rel < 1e-4:
            print(f">>> {debug.Style.GREEN}PASS: Collocation defects are small relative to state scales.{debug.Style.RESET}")
        else:
            print(f">>> {debug.Style.YELLOW}WARN: Non-negligible collocation defects detected.{debug.Style.RESET}")

        rows = []
        for phase_name, pos_s, vel_s, m_s, rel_s in phase_series:
            for k in range(len(pos_s)):
                rows.append((phase_name, k, pos_s[k], vel_s[k], m_s[k], rel_s[k]))
        self._save_table(
            "collocation_defect_audit",
            ["phase", "interval_index", "position_defect_m", "velocity_defect_m_s", "mass_defect_kg", "max_relative_defect"],
            rows
        )

        eps = 1e-18
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
        for phase_name, pos_s, vel_s, m_s, _ in phase_series:
            ax1.semilogy(np.maximum(pos_s, eps), label=f"{phase_name.title()} position defect")
        ax1.set_ylabel("Position Defect (m)")
        ax1.set_title("Collocation Defect Audit")
        ax1.legend()
        ax1.grid(True, which="both", ls="-")
        for phase_name, _, _, _, rel_s in phase_series:
            ax2.semilogy(np.maximum(rel_s, eps), label=f"{phase_name.title()} max relative defect")
        ax2.set_xlabel("Interval Index")
        ax2.set_ylabel("Max Relative Defect (-)")
        ax2.legend()
        ax2.grid(True, which="both", ls="-")
        self._finalize_figure(fig, "collocation_defect_audit")


    # 3. DRIFT ANALYSIS
    def analyze_drift(self, opt_res, sim_res):
        debug._print_sub_header("3. Drift Analysis (Opt vs Sim)")
        debug.analyze_trajectory_drift(opt_res, sim_res)

    # 11. CHAOS / LYAPUNOV ANALYSIS (Upgrade 2)
    def analyze_chaos_lyapunov(self):
        debug._print_sub_header("11. Finite-Time Sensitivity Analysis (Thrust Perturbation)")
        
        # 1. Nominal Run
        print("Generating Nominal Trajectory...")
        opt_res = self._get_baseline_opt_res()
        with suppress_stdout():
            sim_nom = run_simulation(opt_res, self.veh, self.base_config)
            
        # 2. Perturbed Run (Perturb Thrust by 0.1%)
        # Changing thrust is more sensitive than density for open-loop divergence
        print("Generating Perturbed Trajectory (Thrust + 0.1%)...")
        pert_config = copy.deepcopy(self.base_config)
        pert_config.stage_1.thrust_vac *= 1.001
        pert_veh = Vehicle(pert_config, self.env)
        
        with suppress_stdout():
            sim_pert = run_simulation(opt_res, pert_veh, pert_config)
            
        # 3. Analysis
        # Interpolate Perturbed to Nominal Time Grid
        t_nom = sim_nom['t']
        y_nom = sim_nom['y']
        
        t_pert = sim_pert['t']
        y_pert = sim_pert['y']
        
        if len(t_nom) < 4 or len(t_pert) < 4:
             print("  Trajectory too short for Lyapunov analysis.")
             return

        # FIX: Remove duplicates from time arrays (caused by phase concatenation)
        # Cubic spline interpolation requires strictly increasing x values.
        _, idx_nom = np.unique(t_nom, return_index=True)
        idx_nom = np.sort(idx_nom)
        t_nom = t_nom[idx_nom]
        y_nom = y_nom[:, idx_nom]
        
        _, idx_pert = np.unique(t_pert, return_index=True)
        idx_pert = np.sort(idx_pert)
        t_pert = t_pert[idx_pert]
        y_pert = y_pert[:, idx_pert]
        
        # Resample to dense grid for high-resolution plotting (2000 points)
        t_dense = np.linspace(t_nom[0], t_nom[-1], 2000)
        
        f_nom = interp1d(t_nom, y_nom, axis=1, kind='cubic', fill_value="extrapolate")
        f_pert = interp1d(t_pert, y_pert, axis=1, kind='cubic', fill_value="extrapolate")
        
        y_nom_interp = f_nom(t_dense)
        y_pert_interp = f_pert(t_dense)
        
        # Calculate divergence (Euclidean distance in Position).
        delta_r_raw = np.linalg.norm(y_nom_interp[0:3, :] - y_pert_interp[0:3, :], axis=0)
        # Use a physical floor (1 mm) for log-scaling to avoid artificial -inf spikes.
        delta_r_floor = 1e-3
        delta_r = np.maximum(delta_r_raw, delta_r_floor)
        log_delta = np.log(delta_r)
        
        # Quantitative Analysis
        # Finite-time divergence growth-rate estimate from log-distance slope.
        # Fit to the middle 50% of the trajectory to avoid transient and saturation
        idx_start = int(len(t_dense) * 0.2)
        idx_end = int(len(t_dense) * 0.8)
        # Skip the initial floor-dominated segment when possible.
        floor_release = np.where(delta_r_raw > 10.0 * delta_r_floor)[0]
        if len(floor_release) > 0:
            idx_start = max(idx_start, int(floor_release[0]))
        if idx_end <= idx_start + 5:
            idx_start = int(len(t_dense) * 0.2)
            idx_end = int(len(t_dense) * 0.8)
        coeffs = np.polyfit(t_dense[idx_start:idx_end], log_delta[idx_start:idx_end], 1)
        growth_rate = coeffs[0]
        final_div = delta_r_raw[-1] / 1000.0 # km
        fit_x = t_dense[idx_start:idx_end]
        fit_y = log_delta[idx_start:idx_end]
        fit_pred = coeffs[0] * fit_x + coeffs[1]
        ss_res = np.sum((fit_y - fit_pred)**2)
        ss_tot = np.sum((fit_y - np.mean(fit_y))**2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-16)

        self._save_table(
            "finite_time_divergence",
            ["time_s", "delta_r_raw_m", "delta_r_for_log_m", "log_delta_r"],
            [(t_dense[i], delta_r_raw[i], delta_r[i], log_delta[i]) for i in range(len(t_dense))]
        )
        
        print("  Note: this compares nominal dynamics to a perturbed-parameter run.")
        print("  It is a finite-time sensitivity metric, not a formal Lyapunov exponent.")
        print(f"  Estimated growth rate: ~{growth_rate:.4f} s^-1")
        print(f"  Linear-fit R^2 (mid-window): {r2:.4f}")
        print(f"  Divergence at T={t_dense[-1]:.1f}s: {final_div:.2f} km")
        
        # Plot
        fig = plt.figure(figsize=(10, 6))
        plt.plot(t_dense, log_delta, 'b-', linewidth=1.5, label='Log Divergence ln(|Δr|)')
        
        # Plot Fit Line
        y_fit = coeffs[0] * t_dense[idx_start:idx_end] + coeffs[1]
        plt.plot(t_dense[idx_start:idx_end], y_fit, 'r--', linewidth=2, label=f'Growth-rate fit ({growth_rate:.4f} s^-1)')
        
        plt.xlabel('Time (s)')
        plt.ylabel('ln(|delta_r|) [Log Divergence]')
        plt.title(
            f'Finite-Time Sensitivity to Parameter Perturbation (Thrust +0.1%)\n'
            f'Growth Rate = {growth_rate:.4f} s^-1, R² = {r2:.3f}, Final Div = {final_div:.1f} km'
        )
        plt.grid(True)
        plt.legend()
        self._finalize_figure(fig, "chaos_lyapunov")
        print(f">>> {debug.Style.GREEN}PASS: Finite-time sensitivity analysis complete.{debug.Style.RESET}")

    # 13. BIFURCATION ANALYSIS (Upgrade 4)
    def analyze_bifurcation(self):
        debug._print_sub_header("13. Bifurcation Analysis (Open-Loop Feasibility Sweep)")
        
        # 1. Get Nominal Controls
        opt_res = self._get_baseline_opt_res()
            
        if not opt_res.get("success", False):
            print("  Baseline optimization failed. Skipping Bifurcation.")
            return

        # 2. Parameter Sweep (Thrust Multiplier)
        multipliers = np.linspace(0.90, 1.10, 21) # +/- 10%
        final_alts = []
        final_vels = []
        margins = []
        orbit_errors = []
        orbit_ok_flags = []
        strict_ok_flags = []
        statuses = []
        
        print(f"{'Thrust Mult':<12} | {'Final Alt (km)':<15} | {'Orbit Err':<10} | {'Status':<10}")
        print("-" * 62)
        
        for mult in multipliers:
            # Perturb
            cfg = copy.deepcopy(self.base_config)
            cfg.stage_1.thrust_vac *= mult
            cfg.stage_2.thrust_vac *= mult
            
            # Run Sim (Open Loop)
            with suppress_stdout():
                veh_bif = Vehicle(cfg, self.env)
                sim_res = run_simulation(opt_res, veh_bif, cfg)
            
            metrics = evaluate_terminal_state(sim_res, cfg, self.base_env_config)
            final_alts.append(metrics["r_f_m"] / 1000.0 if metrics["terminal_valid"] else np.nan)
            final_vels.append(metrics["v_f_m_s"] if metrics["terminal_valid"] else np.nan)
            margins.append(metrics["fuel_margin_kg"])
            orbit_err = normalized_orbit_error(metrics)
            orbit_errors.append(orbit_err)
            orbit_ok_flags.append(int(bool(metrics.get("orbit_ok", False))))
            strict_ok_flags.append(int(bool(metrics.get("strict_ok", False))))

            if metrics["strict_ok"]:
                status = "Orbit"
            elif metrics["terminal_valid"]:
                status = metrics["status"]
            else:
                status = "SIM_ERROR"
            statuses.append(status)
            orbit_err_str = f"{orbit_err:8.3f}" if np.isfinite(orbit_err) else "   nan   "
            print(f"{mult:<12.2f} | {final_alts[-1]:<15.1f} | {orbit_err_str:<10} | {status:<10}")

        self._save_table(
            "bifurcation_thrust_sweep",
            [
                "thrust_multiplier", "final_altitude_km", "final_velocity_m_s",
                "fuel_margin_kg", "normalized_orbit_error", "orbit_ok", "strict_ok", "status"
            ],
            [
                (
                    multipliers[i], final_alts[i], final_vels[i], margins[i],
                    orbit_errors[i], orbit_ok_flags[i], strict_ok_flags[i], statuses[i]
                )
                for i in range(len(multipliers))
            ]
        )

        # Feasibility boundary is defined by normalized orbit error = 1.
        critical_mult = None
        for i in range(1, len(multipliers)):
            e0 = orbit_errors[i - 1]
            e1 = orbit_errors[i]
            if not (np.isfinite(e0) and np.isfinite(e1)):
                continue
            f0 = e0 - 1.0
            f1 = e1 - 1.0
            if abs(f0) < 1e-12:
                critical_mult = float(multipliers[i - 1])
                break
            if f0 * f1 < 0.0:
                critical_mult = float(np.interp(0.0, [f0, f1], [multipliers[i - 1], multipliers[i]]))
                break

        fuel_boundary = None
        for i in range(1, len(multipliers)):
            if margins[i-1] <= 0.0 <= margins[i]:
                fuel_boundary = float(np.interp(0.0, [margins[i-1], margins[i]], [multipliers[i-1], multipliers[i]]))
                break
            if margins[i-1] >= 0.0 >= margins[i]:
                fuel_boundary = float(np.interp(0.0, [margins[i-1], margins[i]], [multipliers[i-1], multipliers[i]]))
                break

        # 3. Plotting
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Thrust Multiplier')
        ax1.set_ylabel('Final Altitude (km)', color=color)
        ax1.plot(multipliers, final_alts, 'o-', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.axhline(self.base_config.target_altitude/1000.0, color='k', linestyle='--', alpha=0.3, label='Target')
        ax1.grid(True)
        
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Normalized Orbit Error (-)', color=color)
        ax2.plot(multipliers, orbit_errors, 'x--', color=color)
        ax2.axhline(1.0, color='tab:orange', linestyle=':', alpha=0.7, label='Orbit threshold (=1)')
        ax2.tick_params(axis='y', labelcolor=color)

        if critical_mult is not None:
            ax1.axvline(critical_mult, color='purple', linestyle='-.', alpha=0.7, label=f'Feasibility boundary ~ {critical_mult:.3f}')
            print(f"  Estimated orbit-feasibility boundary near thrust multiplier {critical_mult:.3f}")
        else:
            print("  No orbit-feasibility crossing (error=1) found within scanned thrust range.")
        if fuel_boundary is not None:
            print(f"  Fuel-margin boundary near thrust multiplier {fuel_boundary:.3f}")
        else:
            print("  No fuel-margin sign change in scanned thrust range (not fuel-limited here).")
        
        plt.title('Bifurcation Analysis: Sensitivity to Thrust\n(Open-Loop Control Feasibility)')
        self._finalize_figure(fig, "bifurcation_thrust")
        
        print(f">>> {debug.Style.GREEN}PASS: Bifurcation diagram generated.{debug.Style.RESET}")

    # 14. THEORETICAL EFFICIENCY (Hohmann Comparison)
    def analyze_theoretical_efficiency(self):
        debug._print_sub_header("14. Theoretical Efficiency (Hohmann Comparison)")
        
        # 1. Get Simulation Data
        opt_res = self._get_baseline_opt_res()
        with suppress_stdout():
            sim_res = run_simulation(opt_res, self.veh, self.base_config)

        terminal_metrics = evaluate_terminal_state(sim_res, self.base_config, self.base_env_config)
        mission_success = terminal_metrics["strict_ok"]
        if not terminal_metrics["terminal_valid"]:
            print("  Cannot evaluate efficiency: simulation produced invalid terminal state.")
            return
            
        # 2. Calculate Actual Delta-V (Integrated)
        t, y, u = sim_res['t'], sim_res['y'], sim_res['u']
        dv_actual = 0.0
        for i in range(len(t)-1):
            dt = t[i+1] - t[i]
            m, th = y[6, i], u[0, i]
            if th > 0.01:
                m_stg2_wet = self.base_config.stage_2.dry_mass + self.base_config.stage_2.propellant_mass + self.base_config.payload_mass
                stg = self.base_config.stage_1 if m > m_stg2_wet + 1000 else self.base_config.stage_2
                r = y[0:3, i]
                env_state = self.env.get_state_sim(r, t[i])
                isp_eff = stg.isp_vac + (env_state['pressure']/stg.p_sl)*(stg.isp_sl - stg.isp_vac)
                thrust = th * stg.thrust_vac * (isp_eff/stg.isp_vac)
                dv_actual += (thrust / m) * dt

        if dv_actual < 1.0:
             print("  Simulation failed to generate Delta-V (Crash or no burn).")
             return

        # 3. Calculate Theoretical Minimum (Hohmann-like)
        mu = self.env.config.earth_mu
        R1 = self.env.config.earth_radius_equator
        R2 = R1 + self.base_config.target_altitude
        lat = self.base_env_config.launch_latitude
        v_surf = self.env.config.earth_omega_vector[2] * R1 * np.cos(np.radians(lat))
        
        a_tx = (R1 + R2) / 2.0
        v_peri_tx = np.sqrt(mu * (2/R1 - 1/a_tx))
        v_apo_tx  = np.sqrt(mu * (2/R2 - 1/a_tx))
        v_circ_2  = np.sqrt(mu / R2)
        
        dv_burn1 = v_peri_tx - v_surf
        dv_burn2 = v_circ_2 - v_apo_tx
        dv_theoretical = dv_burn1 + dv_burn2
        
        efficiency = (dv_theoretical / dv_actual) * 100.0
        
        print(f"  Theoretical Min Delta-V (Hohmann): {dv_theoretical:.1f} m/s")
        print(f"    - Burn 1 (Surf -> Transfer):     {dv_burn1:.1f} m/s")
        print(f"    - Burn 2 (Circularization):      {dv_burn2:.1f} m/s")
        print(f"  Actual Delta-V (Simulation):       {dv_actual:.1f} m/s")
        print(f"  Gravity/Drag/Steering Losses:      {dv_actual - dv_theoretical:.1f} m/s")
        print(f"  Mission Efficiency:                {efficiency:.1f}%")
        if not mission_success:
            print(f">>> {debug.Style.YELLOW}NOTE: Terminal orbit criteria were not met; efficiency is diagnostic only.{debug.Style.RESET}")

        self._save_table(
            "theoretical_efficiency",
            ["mission_success", "dv_theoretical_m_s", "dv_actual_m_s", "losses_m_s", "efficiency_percent"],
            [(int(mission_success), dv_theoretical, dv_actual, dv_actual - dv_theoretical, efficiency)]
        )
        
        if mission_success and efficiency > 85.0:
            print(f">>> {debug.Style.GREEN}PASS: High efficiency (>85%). Trajectory is near-optimal.{debug.Style.RESET}")
        elif mission_success:
            print(f">>> {debug.Style.YELLOW}NOTE: Efficiency is {efficiency:.1f}%. Losses are significant.{debug.Style.RESET}")
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

        fig = plt.figure(figsize=(10, 6))
        plt.loglog(dt_values, euler_errors, "o-r", label=f"Euler (slope={slope_eu:.2f})")
        plt.loglog(dt_values, rk4_errors, "s-b", label=f"RK4 (slope={slope_rk:.2f})")
        plt.xlabel("Time Step dt (s)")
        plt.ylabel("State Error at T")
        plt.title("Smooth ODE Benchmark: Formal Integrator Order")
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend()
        self._finalize_figure(fig, "smooth_integrator_benchmark")

    # 16. CONSERVATIVE INVARIANT TEST
    def analyze_conservative_invariants(self):
        debug._print_sub_header("16. Conservative Invariants (Two-Body, No Thrust/Drag)")
        print("Checking energy and angular-momentum conservation in pure two-body dynamics.")

        opt_res = self._get_baseline_opt_res()
        with suppress_stdout():
            sim_res = run_simulation(opt_res, self.veh, self.base_config)
        if "y" not in sim_res or sim_res["y"].shape[1] < 1:
            print("  Simulation did not return a valid state history. Skipping.")
            return

        y_init = sim_res["y"][0:6, -1].copy()
        mu = self.env.config.earth_mu

        def two_body(t, y):
            r = y[0:3]
            v = y[3:6]
            r_norm = max(np.linalg.norm(r), 1.0)
            a = -mu * r / (r_norm ** 3)
            return np.concatenate([v, a])

        T_prop = 5400.0
        with suppress_stdout():
            res = solve_ivp(
                fun=two_body,
                t_span=(0.0, T_prop),
                y0=y_init,
                method="RK45",
                rtol=1e-10,
                atol=1e-12
            )
        if not res.success or res.y.shape[1] < 2:
            print("  Two-body propagation failed. Skipping.")
            return

        r = res.y[0:3, :]
        v = res.y[3:6, :]
        r_norm = np.linalg.norm(r, axis=0)
        v_norm_sq = np.sum(v * v, axis=0)
        energy = 0.5 * v_norm_sq - mu / np.maximum(r_norm, 1.0)
        h_vec = np.cross(r.T, v.T).T
        h_mag = np.linalg.norm(h_vec, axis=0)
        e0 = energy[0]
        h0 = h_mag[0]
        rel_e = np.abs((energy - e0) / (abs(e0) + 1e-16))
        rel_h = np.abs((h_mag - h0) / (abs(h0) + 1e-16))

        print(f"  Max relative energy drift:         {np.max(rel_e):.3e}")
        print(f"  Max relative angular momentum drift:{np.max(rel_h):.3e}")

        self._save_table(
            "conservative_invariants",
            ["time_s", "specific_energy_J_per_kg", "angular_momentum_m2_s", "rel_energy_drift", "rel_h_drift"],
            [
                (float(res.t[i]), float(energy[i]), float(h_mag[i]), float(rel_e[i]), float(rel_h[i]))
                for i in range(len(res.t))
            ]
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.semilogy(res.t, rel_e, "b-")
        ax1.set_ylabel("Relative Energy Drift")
        ax1.grid(True, which="both", alpha=0.3)
        ax2.semilogy(res.t, rel_h, "r-")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Relative |h| Drift")
        ax2.grid(True, which="both", alpha=0.3)
        fig.suptitle("Conservative Invariant Check (Two-Body Dynamics)")
        self._finalize_figure(fig, "conservative_invariants")

    # 17. MONTE CARLO PRECISION TARGET
    def analyze_monte_carlo_precision_target(
        self,
        target_half_width=0.04,
        target_relative_half_width=0.60,
        min_samples=80,
        max_samples=300,
        batch_size=20
    ):
        debug._print_sub_header("17. Monte Carlo Precision Target (CI-Based Stop)")
        print("Running Monte Carlo until Wilson 95% CI half-width reaches target precision.")

        if min_samples < 1 or max_samples < min_samples or batch_size < 1:
            print("  Invalid parameters for precision-target Monte Carlo. Skipping.")
            return

        opt_res = self._get_baseline_opt_res()
        rng = np.random.default_rng(self.seed_sequence.spawn(1)[0])
        successes = 0
        n = 0
        batch_rows = []
        stop_reason = "max_samples"
        t0 = time.time()

        while n < max_samples:
            n_batch = min(batch_size, max_samples - n)
            for _ in range(n_batch):
                thrust_mult, isp_mult, dens_mult = self._draw_uncertainty_multipliers(rng, model="gaussian_independent")
                result = self._simulate_uncertainty_case(opt_res, thrust_mult, isp_mult, dens_mult)
                successes += int(result["terminal"]["strict_ok"])
            n += n_batch

            lo, hi = wilson_interval(successes, n, z=1.96)
            p = successes / n
            half_width = 0.5 * (hi - lo)
            rel_half_width = half_width / max(p, 1.0 / max(n, 1))
            batch_rows.append((n, p, lo, hi, half_width, rel_half_width))
            print(
                f"  N={n:4d} | Success={p*100:6.2f}% | CI=[{lo*100:5.2f}, {hi*100:5.2f}] | "
                f"Half-width={half_width*100:5.2f}% | Relative={rel_half_width*100:5.1f}%"
            )

            abs_ok = half_width <= target_half_width
            rel_ok = True
            if target_relative_half_width is not None:
                rel_ok = rel_half_width <= float(target_relative_half_width)
            if n >= min_samples and abs_ok and rel_ok:
                stop_reason = "target_precision"
                break

        elapsed = time.time() - t0
        final_n, final_p, final_lo, final_hi, final_half, final_rel_half = batch_rows[-1]
        print(f"  Stop reason: {stop_reason}")
        print(
            f"  Final: N={final_n}, Success={final_p*100:.2f}%, "
            f"CI=[{final_lo*100:.2f}, {final_hi*100:.2f}], "
            f"Half-width={final_half*100:.2f}% ({final_rel_half*100:.1f}% relative)"
        )
        print(f"  Runtime: {elapsed:.1f}s")
        if stop_reason != "target_precision":
            print(
                f">>> {debug.Style.YELLOW}WARN: Precision targets not met before max samples "
                f"(N={max_samples}).{debug.Style.RESET}"
            )

        self._save_table(
            "monte_carlo_precision_target",
            [
                "n", "success_rate", "wilson_low_95", "wilson_high_95",
                "ci_half_width_95", "relative_half_width", "target_half_width",
                "target_relative_half_width"
            ],
            [
                (
                    row[0], row[1], row[2], row[3], row[4], row[5],
                    target_half_width,
                    np.nan if target_relative_half_width is None else target_relative_half_width
                )
                for row in batch_rows
            ]
        )

        fig, ax1 = plt.subplots(figsize=(10, 6))
        n_vals = np.array([r[0] for r in batch_rows], dtype=int)
        p_vals = np.array([r[1] for r in batch_rows], dtype=float) * 100.0
        lo_vals = np.array([r[2] for r in batch_rows], dtype=float) * 100.0
        hi_vals = np.array([r[3] for r in batch_rows], dtype=float) * 100.0
        hw_vals = np.array([r[4] for r in batch_rows], dtype=float) * 100.0
        rel_hw_vals = np.array([r[5] for r in batch_rows], dtype=float) * 100.0

        ax1.plot(n_vals, p_vals, "b-o", label="Success rate")
        ax1.fill_between(n_vals, lo_vals, hi_vals, color="b", alpha=0.15, label="Wilson 95% CI")
        ax1.set_xlabel("Samples N")
        ax1.set_ylabel("Success Rate (%)", color="b")
        ax1.tick_params(axis="y", labelcolor="b")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(n_vals, hw_vals, "r-s", label="CI half-width")
        ax2.axhline(target_half_width * 100.0, color="r", linestyle="--", alpha=0.7, label="Target half-width")
        if target_relative_half_width is not None:
            ax2.plot(n_vals, rel_hw_vals, color="tab:purple", marker="^", linestyle="-.", label="Relative half-width")
            ax2.axhline(
                target_relative_half_width * 100.0,
                color="tab:purple",
                linestyle=":",
                alpha=0.7,
                label="Target relative half-width"
            )
        ax2.set_ylabel("CI Half-Width (%)", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
        plt.title("Monte Carlo Precision Target (Adaptive Sample Size)")
        self._finalize_figure(fig, "monte_carlo_precision_target")

    # 18. GLOBAL SENSITIVITY RANKING (PRCC / SPEARMAN)
    def analyze_global_sensitivity(self, N_samples=200):
        debug._print_sub_header(f"18. Global Sensitivity Ranking (N={N_samples})")
        print("Estimating rank-based sensitivities of terminal metrics to uncertain inputs.")

        if N_samples < 20:
            print("  Need at least 20 samples for stable rank correlations. Skipping.")
            return

        opt_res = self._get_baseline_opt_res()
        rng = np.random.default_rng(self.seed_sequence.spawn(1)[0])
        rows = []
        X = []
        y_alt = []
        y_vel = []
        y_fuel = []
        t_progress0 = time.time()
        bar_width = 32
        update_every = max(1, N_samples // 100)
        is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        last_reported_pct = -1

        def _print_progress(done, force=False):
            nonlocal last_reported_pct
            frac = done / max(N_samples, 1)
            filled = int(round(bar_width * frac))
            bar = "#" * filled + "-" * (bar_width - filled)
            elapsed = time.time() - t_progress0
            if done > 0:
                eta = elapsed * (N_samples - done) / done
                eta_s = f"{eta:6.1f}s"
            else:
                eta_s = "   n/a "
            if is_tty:
                msg = f"\r  Sampling: [{bar}] {done:4d}/{N_samples:4d} ({frac*100:5.1f}%) | ETA {eta_s}"
                sys.stdout.write(msg)
                sys.stdout.flush()
                return

            pct = int(frac * 100.0)
            should_print = force or done == N_samples or done == 0 or (pct >= last_reported_pct + 5)
            if should_print:
                print(f"  Sampling: [{bar}] {done:4d}/{N_samples:4d} ({frac*100:5.1f}%) | ETA {eta_s}")
                last_reported_pct = pct

        _print_progress(0, force=True)

        for i in range(N_samples):
            thrust_mult, isp_mult, dens_mult = self._draw_uncertainty_multipliers(rng, model="gaussian_independent")
            result = self._simulate_uncertainty_case(opt_res, thrust_mult, isp_mult, dens_mult)
            tm = result["terminal"]
            alt_err = tm["alt_err_m"] if tm["terminal_valid"] else np.nan
            vel_err = tm["vel_err_m_s"] if tm["terminal_valid"] else np.nan
            fuel_margin = tm["fuel_margin_kg"] if tm["terminal_valid"] else np.nan
            rows.append((
                i + 1, thrust_mult, isp_mult, dens_mult,
                alt_err, vel_err, fuel_margin, int(tm["strict_ok"]), tm["status"]
            ))
            if np.isfinite(alt_err) and np.isfinite(vel_err) and np.isfinite(fuel_margin):
                X.append([thrust_mult, isp_mult, dens_mult])
                y_alt.append(alt_err)
                y_vel.append(vel_err)
                y_fuel.append(fuel_margin)
            done = i + 1
            if (done % update_every == 0) or (done == N_samples):
                _print_progress(done)
        if is_tty:
            print()

        self._save_table(
            "global_sensitivity_samples",
            [
                "run", "thrust_multiplier", "isp_multiplier", "density_multiplier",
                "altitude_error_m", "velocity_error_m_s", "fuel_margin_kg", "strict_success", "status"
            ],
            rows
        )

        strict_count = int(np.sum([r[7] for r in rows]))
        print(f"  Strict-success rate in sampled set: {strict_count}/{N_samples} ({100.0*strict_count/max(N_samples,1):.1f}%)")

        X = np.array(X, dtype=float)
        if X.shape[0] < 40:
            print("  Too few valid terminal states for robust sensitivity ranking. Skipping.")
            return

        var_names = ["thrust_multiplier", "isp_multiplier", "density_multiplier"]

        def prcc(X_in, y_in):
            y_rank = self._rankdata(y_in)
            X_rank = np.column_stack([self._rankdata(X_in[:, j]) for j in range(X_in.shape[1])])
            vals = []
            for j in range(X_rank.shape[1]):
                others = [k for k in range(X_rank.shape[1]) if k != j]
                if len(others) == 0:
                    vals.append(self._corr(X_rank[:, j], y_rank))
                    continue
                Z = np.column_stack([np.ones(len(y_rank)), X_rank[:, others]])
                beta_x = np.linalg.lstsq(Z, X_rank[:, j], rcond=None)[0]
                beta_y = np.linalg.lstsq(Z, y_rank, rcond=None)[0]
                rx = X_rank[:, j] - Z @ beta_x
                ry = y_rank - Z @ beta_y
                vals.append(self._corr(rx, ry))
            return np.array(vals, dtype=float)

        def bootstrap_prcc_ci(X_in, y_in, n_boot=500, alpha=0.05):
            y_arr = np.asarray(y_in, dtype=float)
            n = X_in.shape[0]
            p = X_in.shape[1]
            if n < 10:
                return np.full(p, np.nan), np.full(p, np.nan)
            rng_boot = np.random.default_rng(self.seed_sequence.spawn(1)[0])
            boots = np.full((n_boot, p), np.nan, dtype=float)
            for b in range(n_boot):
                idx = rng_boot.integers(0, n, size=n)
                boots[b, :] = prcc(X_in[idx, :], y_arr[idx])
            lo = np.nanpercentile(boots, 100.0 * (alpha / 2.0), axis=0)
            hi = np.nanpercentile(boots, 100.0 * (1.0 - alpha / 2.0), axis=0)
            return lo.astype(float), hi.astype(float)

        spearman_alt = np.array([self._corr(self._rankdata(X[:, j]), self._rankdata(y_alt)) for j in range(X.shape[1])])
        spearman_vel = np.array([self._corr(self._rankdata(X[:, j]), self._rankdata(y_vel)) for j in range(X.shape[1])])
        spearman_fuel = np.array([self._corr(self._rankdata(X[:, j]), self._rankdata(y_fuel)) for j in range(X.shape[1])])
        prcc_alt = prcc(X, np.array(y_alt, dtype=float))
        prcc_vel = prcc(X, np.array(y_vel, dtype=float))
        prcc_fuel = prcc(X, np.array(y_fuel, dtype=float))
        ci_alt_lo, ci_alt_hi = bootstrap_prcc_ci(X, y_alt)
        ci_vel_lo, ci_vel_hi = bootstrap_prcc_ci(X, y_vel)
        ci_fuel_lo, ci_fuel_hi = bootstrap_prcc_ci(X, y_fuel)

        summary_rows = []
        for i, name in enumerate(var_names):
            summary_rows.append((
                name,
                float(spearman_alt[i]), float(prcc_alt[i]),
                float(ci_alt_lo[i]), float(ci_alt_hi[i]),
                float(spearman_vel[i]), float(prcc_vel[i]),
                float(ci_vel_lo[i]), float(ci_vel_hi[i]),
                float(spearman_fuel[i]), float(prcc_fuel[i]),
                float(ci_fuel_lo[i]), float(ci_fuel_hi[i])
            ))
        self._save_table(
            "global_sensitivity_ranking",
            [
                "variable",
                "spearman_altitude_error", "prcc_altitude_error",
                "prcc_altitude_ci_low_95", "prcc_altitude_ci_high_95",
                "spearman_velocity_error", "prcc_velocity_error",
                "prcc_velocity_ci_low_95", "prcc_velocity_ci_high_95",
                "spearman_fuel_margin", "prcc_fuel_margin",
                "prcc_fuel_ci_low_95", "prcc_fuel_ci_high_95"
            ],
            summary_rows
        )

        print("  PRCC ranking by |value| (Fuel Margin):")
        order = np.argsort(np.abs(prcc_fuel))[::-1]
        for idx in order:
            print(f"    {var_names[idx]:<20}: {prcc_fuel[idx]: .3f}  [95% CI {ci_fuel_lo[idx]: .3f}, {ci_fuel_hi[idx]: .3f}]")

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0), sharey=True)
        panels = [
            ("Altitude Error", prcc_alt, ci_alt_lo, ci_alt_hi, "tab:blue"),
            ("Velocity Error", prcc_vel, ci_vel_lo, ci_vel_hi, "tab:orange"),
            ("Fuel Margin", prcc_fuel, ci_fuel_lo, ci_fuel_hi, "tab:green"),
        ]
        x = np.arange(len(var_names))
        for ax, (title, vals, lo_ci, hi_ci, color) in zip(axes, panels):
            yerr_low = np.maximum(vals - lo_ci, 0.0)
            yerr_high = np.maximum(hi_ci - vals, 0.0)
            yerr = np.vstack([yerr_low, yerr_high])
            ax.bar(x, vals, color=color, alpha=0.75)
            ax.errorbar(x, vals, yerr=yerr, fmt="none", ecolor="k", elinewidth=1.0, capsize=3)
            ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(var_names, rotation=15)
            ax.set_title(title)
            ax.grid(True, axis="y", alpha=0.3)
        axes[0].set_ylabel("PRCC (with 95% bootstrap CI)")
        fig.suptitle("Global Sensitivity Ranking (PRCC + Bootstrap Confidence Intervals)")
        self._finalize_figure(fig, "global_sensitivity_ranking")

    # 21. EVENT-TIME CONVERGENCE
    def analyze_event_time_convergence(self):
        debug._print_sub_header("21. Event-Time Convergence vs Integrator Tolerance")
        print("Checking convergence of key event times (Max-Q, Max-G, staging, depletion).")

        opt_res = self._get_baseline_opt_res()
        tols = [
            (1e-6, 1e-9), (1e-7, 1e-10), (1e-8, 1e-11), (1e-9, 1e-12), (1e-10, 1e-13), (1e-12, 1e-14)
        ]
        rows = []
        diagnostics = []
        for rtol, atol in tols:
            with suppress_stdout():
                sim_res = run_simulation(opt_res, self.veh, self.base_config, rtol=rtol, atol=atol)
            diag = self._trajectory_diagnostics(sim_res, self.veh, self.base_config)
            diagnostics.append(diag)
            rows.append((
                rtol, atol,
                diag["max_q_time_s"], diag["max_g_time_s"], diag["staging_time_s"], diag["depletion_time_s"],
                diag["max_q_pa"], diag["max_g"]
            ))

        ref = diagnostics[-1]
        drift_rows = []
        for i, (rtol, atol) in enumerate(tols):
            d = diagnostics[i]
            def drift(a, b):
                if np.isfinite(a) and np.isfinite(b):
                    return abs(a - b)
                return np.nan
            drift_rows.append((
                rtol, atol,
                drift(d["max_q_time_s"], ref["max_q_time_s"]),
                drift(d["max_g_time_s"], ref["max_g_time_s"]),
                drift(d["staging_time_s"], ref["staging_time_s"]),
                drift(d["depletion_time_s"], ref["depletion_time_s"]),
                drift(d["max_q_pa"], ref["max_q_pa"]),
                drift(d["max_g"], ref["max_g"])
            ))

        self._save_table(
            "event_time_convergence",
            [
                "rtol", "atol",
                "t_max_q_s", "t_max_g_s", "t_staging_s", "t_depletion_s",
                "max_q_pa", "max_g",
                "drift_max_q_s", "drift_max_g_s", "drift_staging_s", "drift_depletion_s",
                "drift_max_q_pa", "drift_max_g"
            ],
            [
                (
                    rows[i][0], rows[i][1], rows[i][2], rows[i][3], rows[i][4], rows[i][5],
                    rows[i][6], rows[i][7],
                    drift_rows[i][2], drift_rows[i][3], drift_rows[i][4], drift_rows[i][5],
                    drift_rows[i][6], drift_rows[i][7]
                )
                for i in range(len(rows))
            ]
        )

        rtols = np.array([r[0] for r in drift_rows[:-1]], dtype=float)
        d_q = np.array([r[2] for r in drift_rows[:-1]], dtype=float)
        d_g = np.array([r[3] for r in drift_rows[:-1]], dtype=float)
        mask_q = np.isfinite(rtols) & np.isfinite(d_q) & (rtols > 0.0) & (d_q > 0.0)
        mask_g = np.isfinite(rtols) & np.isfinite(d_g) & (rtols > 0.0) & (d_g > 0.0)

        fig = plt.figure(figsize=(10, 6))
        if np.any(mask_q):
            plt.loglog(rtols[mask_q], d_q[mask_q], "o-", label="|Δt Max-Q|")
        if np.any(mask_g):
            plt.loglog(rtols[mask_g], d_g[mask_g], "s-", label="|Δt Max-G|")
        if not (np.any(mask_q) or np.any(mask_g)):
            print("  No finite positive drifts available for log-log event-time plot.")
            plt.close(fig)
            return
        finite_q = d_q[np.isfinite(d_q)]
        finite_g = d_g[np.isfinite(d_g)]
        if len(finite_q) > 0:
            print(f"  Max |Δt Max-Q| over sweep: {np.max(finite_q):.3f} s")
        if len(finite_g) > 0:
            print(f"  Max |Δt Max-G| over sweep: {np.max(finite_g):.3f} s")
        dq_val = np.array([r[6] for r in drift_rows[:-1]], dtype=float)
        dg_val = np.array([r[7] for r in drift_rows[:-1]], dtype=float)
        if np.any(np.isfinite(dg_val)) and np.nanmax(dg_val) < 0.05 and np.any(np.isfinite(finite_g)) and np.nanmax(finite_g) > 1.0:
            print("  Note: Max-G time drift is large while peak value drift is tiny (flat-peak timing ambiguity).")
        dep_valid_count = int(np.sum(np.isfinite(np.array([r[5] for r in rows], dtype=float))))
        if dep_valid_count == 0:
            print("  Note: depletion event not triggered in this mission profile (all depletion times are NaN).")
        plt.gca().invert_xaxis()
        plt.xlabel("Integrator Relative Tolerance (rtol)")
        plt.ylabel("Event Time Drift vs Tightest Tolerance (s)")
        plt.title("Event-Time Convergence")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        self._finalize_figure(fig, "event_time_convergence")

    # 22. BIFURCATION 2D MAP
    def analyze_bifurcation_2d_map(self, n_thrust=11, n_density=11):
        debug._print_sub_header("22. Bifurcation 2D Map (Thrust x Density)")
        print("Mapping open-loop strict success and orbit-error structure over a 2D parameter grid.")

        opt_res = self._get_baseline_opt_res()
        if not opt_res.get("success", False):
            print("  Baseline optimization failed. Skipping.")
            return

        n_thrust = max(3, int(n_thrust))
        n_density = max(3, int(n_density))
        thrust_vals = np.linspace(0.94, 1.06, n_thrust)
        density_vals = np.linspace(0.80, 1.20, n_density)
        success_map = np.zeros((len(density_vals), len(thrust_vals)), dtype=float)
        margin_map = np.full((len(density_vals), len(thrust_vals)), np.nan, dtype=float)
        orbit_err_map = np.full((len(density_vals), len(thrust_vals)), np.nan, dtype=float)
        rows = []

        for iy, dens in enumerate(density_vals):
            for ix, thrust in enumerate(thrust_vals):
                result = self._simulate_uncertainty_case(opt_res, thrust, 1.0, dens)
                tm = result["terminal"]
                success = int(tm["strict_ok"])
                margin = tm["fuel_margin_kg"] if tm["terminal_valid"] else np.nan
                orbit_err = normalized_orbit_error(tm)
                success_map[iy, ix] = success
                margin_map[iy, ix] = margin
                orbit_err_map[iy, ix] = orbit_err
                rows.append((
                    float(thrust), float(dens), int(success), float(margin),
                    float(orbit_err), float(tm["alt_err_m"]), float(tm["vel_err_m_s"]), tm["status"]
                ))

        self._save_table(
            "bifurcation_2d_map",
            [
                "thrust_multiplier", "density_multiplier", "strict_success",
                "fuel_margin_kg", "normalized_orbit_error",
                "altitude_error_m", "velocity_error_m_s", "status"
            ],
            rows
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        im1 = ax1.imshow(
            success_map,
            origin="lower",
            aspect="auto",
            extent=[thrust_vals[0], thrust_vals[-1], density_vals[0], density_vals[-1]],
            vmin=0,
            vmax=1,
            cmap="RdYlGn"
        )
        ax1.set_xlabel("Thrust Multiplier")
        ax1.set_ylabel("Density Multiplier")
        ax1.set_title("Strict Success Map")
        fig.colorbar(im1, ax=ax1, label="Success (0/1)")

        # Focus the second panel on orbit-feasibility error, not fuel margin.
        finite_err = orbit_err_map[np.isfinite(orbit_err_map)]
        if finite_err.size > 0:
            vmax = max(1.5, float(np.nanpercentile(finite_err, 95.0)))
        else:
            vmax = 2.0
        im2 = ax2.imshow(
            orbit_err_map,
            origin="lower",
            aspect="auto",
            extent=[thrust_vals[0], thrust_vals[-1], density_vals[0], density_vals[-1]],
            cmap="viridis",
            vmin=0.0,
            vmax=vmax
        )
        ax2.set_xlabel("Thrust Multiplier")
        ax2.set_ylabel("Density Multiplier")
        ax2.set_title("Normalized Orbit Error Map")
        fig.colorbar(im2, ax=ax2, label="Orbit Error (-)")
        finite_mask = np.isfinite(orbit_err_map)
        if np.any(finite_mask):
            finite_vals = orbit_err_map[finite_mask]
            if float(np.min(finite_vals)) <= 1.0 <= float(np.max(finite_vals)):
                ax2.contour(
                    thrust_vals, density_vals, orbit_err_map,
                    levels=[1.0], colors="w", linewidths=1.3
                )
        else:
            print("  Note: no finite orbit-error values available for threshold contour.")

        strict_count = int(np.sum(success_map))
        total_count = int(success_map.size)
        print(f"  Strict-success points: {strict_count}/{total_count}")
        self._finalize_figure(fig, "bifurcation_2d_map")
        print(f">>> {debug.Style.GREEN}PASS: 2D bifurcation map generated.{debug.Style.RESET}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reliability and robustness analysis suite.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for exported figures and CSV tables.")
    parser.add_argument("--seed", type=int, default=1337, help="Global random seed for reproducible Monte Carlo runs.")
    parser.add_argument("--mc-samples", type=int, default=200, help="Monte Carlo sample count used in Monte Carlo-based analyses.")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive plot windows (save figures only).")
    parser.add_argument("--no-save", action="store_true", help="Do not save figures/CSV outputs.")
    args = parser.parse_args()

    suite = ReliabilitySuite(
        output_dir=args.output_dir,
        save_figures=not args.no_save,
        show_plots=not args.no_show,
        random_seed=args.seed
    )
    suite.run_all(monte_carlo_samples=args.mc_samples)
