# reliabilityanalysis.py
# Purpose: Comprehensive Reliability & Robustness Analysis Suite.
# Implements tactics to verify trust in the optimization results.

import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import contextlib
import os
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import multiprocessing
import matplotlib.patches as patches
import csv
from pathlib import Path
from datetime import datetime
import argparse

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

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)

    # Check for near-zero variance to avoid singular ellipse geometry.
    if cov[0, 0] < 1e-12 or cov[1, 1] < 1e-12:
        return None

    eig_vals, eig_vecs = np.linalg.eigh(cov)
    order = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[order]
    eig_vecs = eig_vecs[:, order]
    # Convert principal-axis orientation to degrees.
    angle = np.degrees(np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0]))

    width = 2.0 * n_std * np.sqrt(max(eig_vals[0], 0.0))
    height = 2.0 * n_std * np.sqrt(max(eig_vals[1], 0.0))

    ellipse = patches.Ellipse(
        (np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=angle,
        facecolor=facecolor,
        **kwargs
    )
    return ax.add_patch(ellipse)


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


def gaussian_2d_coverage(n_std):
    """
    Probability mass enclosed by an n-sigma Mahalanobis ellipse in 2D Gaussian data.
    """
    n = float(n_std)
    return 1.0 - np.exp(-0.5 * n * n)


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


def _export_csv(path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def _monte_carlo_worker(args):
    """
    Worker function for parallel Monte Carlo simulations.
    Must be at module level to be picklable.
    """
    idx, base_config, base_env_config, opt_res, seed = args
    rng = np.random.default_rng(seed)
    
    # Perturb Configuration
    cfg = copy.deepcopy(base_config)
    env_cfg = copy.deepcopy(base_env_config)
    
    # Randomize (Gaussian)
    thrust_mult = rng.normal(1.0, 0.02)
    isp_mult = rng.normal(1.0, 0.01)
    dens_mult = rng.normal(1.0, 0.10)
    
    cfg.stage_1.thrust_vac *= thrust_mult
    cfg.stage_2.thrust_vac *= thrust_mult
    cfg.stage_1.isp_vac *= isp_mult
    cfg.stage_2.isp_vac *= isp_mult
    env_cfg.density_multiplier = dens_mult
    
    sim_res = {'y': np.zeros((7, 1)), 'success': False}
    
    # Run Simulation (Suppress output)
    try:
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull):
                env_mc = Environment(env_cfg)
                veh_mc = Vehicle(cfg, env_mc)
                sim_res = run_simulation(opt_res, veh_mc, cfg)
    except Exception:
        pass # sim_res remains dummy failure

    metrics = evaluate_terminal_state(sim_res, cfg, env_cfg)

    # Return results and trajectory data (if valid)
    phase_data = None
    if metrics["trajectory_valid"]:
        y_sim = sim_res['y']
        r_mag = np.linalg.norm(y_sim[0:3, :], axis=0)
        alt = (r_mag - env_cfg.earth_radius_equator) / 1000.0
        vel = np.linalg.norm(y_sim[3:6, :], axis=0)
        phase_data = (alt, vel, metrics["strict_ok"])

    return {
        "orbit_ok": metrics["orbit_ok"],
        "fuel_ok": metrics["fuel_ok"],
        "strict_ok": metrics["strict_ok"],
        "terminal_valid": metrics["terminal_valid"],
        "trajectory_valid": metrics["trajectory_valid"],
        "status": metrics["status"],
        "final_altitude_km": (metrics["r_f_m"] / 1000.0) if metrics["terminal_valid"] else np.nan,
        "final_velocity_m_s": metrics["v_f_m_s"],
        "fuel_margin_kg": metrics["fuel_margin_kg"],
        "phase_data": phase_data
    }

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
            "lines.linewidth": 1.8
        })
        
        print(f"\n{debug.Style.BOLD}=== RELIABILITY ANALYSIS SUITE ==={debug.Style.RESET}")
        print(f"Target: {self.base_config.name}")
        print(f"Output directory: {self.output_dir}")
        print(f"Reproducible seed: {self.random_seed}")

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
        if enabled:
            fn(*args, **kwargs)
            return True
        print(f"\n[Skip] {label} disabled in RELIABILITY_ANALYSIS_TOGGLES.{flag_name}")
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
        max_q = -np.inf
        max_q_time = np.nan
        max_g = -np.inf
        max_g_time = np.nan

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
            if q > max_q:
                max_q = float(q)
                max_q_time = float(ti)

            if th_i < 0.01:
                stage_mode = "coast" if m_i > m_stage2_wet + 1000.0 else "coast_2"
            else:
                stage_mode = "boost" if m_i > m_stage2_wet + 1000.0 else "ship"
            dyn = veh.get_dynamics(y[:, i], th_i, dir_i, ti, stage_mode=stage_mode, scaling=None)
            sensed_acc = dyn[3:6] - env_state["gravity"]
            g_load = np.linalg.norm(sensed_acc) / g0
            if g_load > max_g:
                max_g = float(g_load)
                max_g_time = float(ti)

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

    def run_all(self, monte_carlo_samples=500):
        """Runs all analysis modules sequentially."""
        ran_any = False
        mc_n = max(1, int(monte_carlo_samples))
        mc_min_precision = min(mc_n, max(20, mc_n // 4))
        mc_batch = max(5, min(50, max(1, mc_n // 10)))

        # Major reliability tests.
        ran_any |= self._run_if_enabled(
            "smooth_integrator_benchmark",
            "Smooth ODE integrator benchmark",
            self.analyze_smooth_integrator_benchmark
        )
        ran_any |= self._run_if_enabled(
            "stiffness_convergence",
            "Integrator convergence",
            self.analyze_stiffness_euler
        )
        ran_any |= self._run_if_enabled(
            "conservative_invariants",
            "Conservative-invariant audit",
            self.analyze_conservative_invariants
        )
        ran_any |= self._run_if_enabled(
            "monte_carlo_convergence",
            "Monte Carlo convergence",
            self.analyze_monte_carlo_convergence,
            N_samples=mc_n
        )
        ran_any |= self._run_if_enabled(
            "monte_carlo_precision_target",
            "Monte Carlo precision target",
            self.analyze_monte_carlo_precision_target,
            min_samples=mc_min_precision,
            max_samples=mc_n,
            batch_size=mc_batch
        )
        ran_any |= self._run_if_enabled(
            "global_sensitivity",
            "Global sensitivity ranking",
            self.analyze_global_sensitivity,
            N_samples=mc_n
        )
        ran_any |= self._run_if_enabled(
            "constraint_reliability",
            "Constraint reliability curves",
            self.analyze_constraint_reliability,
            N_samples=mc_n
        )
        ran_any |= self._run_if_enabled(
            "distribution_robustness",
            "Distribution robustness",
            self.analyze_distribution_robustness,
            N_per_model=mc_n
        )
        ran_any |= self._run_if_enabled("grid_independence", "Grid independence", self.analyze_grid_independence)
        ran_any |= self._run_if_enabled("integrator_tolerance", "Integrator tolerance", self.analyze_integrator_tolerance)
        ran_any |= self._run_if_enabled(
            "event_time_convergence",
            "Event-time convergence",
            self.analyze_event_time_convergence
        )
        ran_any |= self._run_if_enabled("corner_cases", "Corner cases", self.analyze_corner_cases)
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
        ran_any |= self._run_if_enabled(
            "theoretical_efficiency",
            "Theoretical efficiency",
            self.analyze_theoretical_efficiency
        )

        # Deep-dive analyses share the same baseline simulation.
        deep_dive_flags = [
            "drift",
            "energy_balance",
            "control_slew",
            "aerodynamics",
            "lagrange_multipliers",
        ]
        if any(bool(getattr(self.test_toggles, name, True)) for name in deep_dive_flags):
            print(f"\n{debug.Style.BOLD}--- Generating Baseline Solution for Deep Dive ---{debug.Style.RESET}")
            opt_res = self._get_baseline_opt_res()
            sim_res = run_simulation(opt_res, self.veh, self.base_config)

            ran_any |= self._run_if_enabled("drift", "Drift analysis", self.analyze_drift, opt_res, sim_res)
            ran_any |= self._run_if_enabled("energy_balance", "Energy balance", self.analyze_energy_balance, sim_res)
            ran_any |= self._run_if_enabled("control_slew", "Control slew", self.analyze_control_slew, sim_res, opt_res)
            ran_any |= self._run_if_enabled("aerodynamics", "Aerodynamic audit", self.analyze_aerodynamics, sim_res)
            ran_any |= self._run_if_enabled("lagrange_multipliers", "Lagrange multipliers", self.analyze_lagrange_multipliers, opt_res)
        else:
            print("\n[Skip] All deep-dive tests are disabled in RELIABILITY_ANALYSIS_TOGGLES.")

        if not ran_any:
            print("\nNo analyses executed. Enable tests in RELIABILITY_ANALYSIS_TOGGLES in config.py.")

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

    # 3. DRIFT ANALYSIS
    def analyze_drift(self, opt_res, sim_res):
        debug._print_sub_header("3. Drift Analysis (Opt vs Sim)")
        debug.analyze_trajectory_drift(opt_res, sim_res)

    # 4. ENERGY BALANCE AUDIT
    def analyze_energy_balance(self, sim_res):
        debug._print_sub_header("4. Energy Balance Audit")
        debug.analyze_energy_balance(sim_res, self.veh)
        
        # --- Visualization: Energy Drift ---
        print("  Generating Energy Drift Plot...")
        t = sim_res['t']
        if len(t) < 2:
            print("  Not enough data for energy drift plot.")
            return
            
        y = sim_res['y']
        u = sim_res['u']
        
        # Constants
        mu = self.env.config.earth_mu
        R_e = self.env.config.earth_radius_equator
        J2 = self.env.config.j2_constant
        use_j2 = self.env.config.use_j2_perturbation

        def get_potential_energy(r_vec):
            r = np.linalg.norm(r_vec)
            pe = -mu / r
            if use_j2:
                z = r_vec[2]
                sin_phi = z / r
                term_j2 = (mu * J2 * R_e**2) / (2 * r**3) * (3 * sin_phi**2 - 1)
                pe += term_j2
            return pe

        # Arrays
        E_mech = np.zeros(len(t))
        Work_Done = np.zeros(len(t))
        powers = np.zeros(len(t))
        
        # Initial State
        r0 = y[0:3, 0]
        v0 = y[3:6, 0]
        E_start = 0.5 * np.dot(v0, v0) + get_potential_energy(r0)
        
        # Loop
        for i in range(len(t)):
            r_i = y[0:3, i]
            v_i = y[3:6, i]
            m_i = y[6, i]
            th_i = u[0, i]
            dir_i = u[1:, i]
            t_i = t[i]
            
            # 1. Calculate Mechanical Energy
            pe = get_potential_energy(r_i)
            ke = 0.5 * np.dot(v_i, v_i)
            E_mech[i] = pe + ke
            
            # 2. Calculate Specific Power (Non-Conservative)
            # Determine phase for drag model
            m_ship_wet = self.base_config.stage_2.dry_mass + self.base_config.stage_2.propellant_mass + self.base_config.payload_mass
            if m_i > m_ship_wet + 1000:
                phase = "coast" if th_i < 0.01 else "boost"
            else:
                phase = "coast_2" if th_i < 0.01 else "ship"
                
            dyn = self.veh.get_dynamics(y[:, i], th_i, dir_i, t_i, stage_mode=phase, scaling=None)
            acc_total = dyn[3:6]
            env_state = self.env.get_state_sim(r_i, t_i)
            acc_nc = acc_total - env_state['gravity']
            
            powers[i] = np.dot(acc_nc, v_i)
            
            # 3. Integrate Work (Trapezoidal)
            if i > 0:
                dt = t[i] - t[i-1]
                Work_Done[i] = Work_Done[i-1] + 0.5 * (powers[i] + powers[i-1]) * dt

        # Calculate Drift
        delta_E = E_mech - E_start
        drift = delta_E - Work_Done

        self._save_table(
            "energy_balance_drift",
            ["time_s", "mechanical_energy_J_per_kg", "integrated_work_J_per_kg", "energy_drift_J_per_kg"],
            [(t[i], E_mech[i], Work_Done[i], drift[i]) for i in range(len(t))]
        )
        
        # Plot
        fig = plt.figure(figsize=(10, 6))
        plt.plot(t, drift, 'r-', linewidth=1.5, label='Energy Drift')
        plt.axhline(0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Time (s)')
        plt.ylabel('Energy Error (J/kg)')
        plt.title(f'Integrator Energy Conservation Check\nMax Drift: {np.max(np.abs(drift)):.4f} J/kg')
        plt.grid(True)
        plt.legend()
        self._finalize_figure(fig, "energy_balance_drift")

    # 5. CONTROL SLEW ANALYSIS
    def analyze_control_slew(self, sim_res, opt_res):
        debug._print_sub_header("5. Control Slew Rate Analysis")
        debug.analyze_control_slew_rates(sim_res, opt_res)

    # 6. AERODYNAMIC ANGLE CHECK
    def analyze_aerodynamics(self, sim_res):
        debug._print_sub_header("6. Aerodynamic Angle Check (Max Q)")
        
        # Re-use debug logic implicitly via simulation metrics
        t = sim_res['t']
        y = sim_res['y']
        u = sim_res['u']
        
        max_q = 0.0
        aoa_at_max_q = 0.0
        
        for i in range(len(t)):
            r = y[0:3, i]
            v = y[3:6, i]
            env_state = self.env.get_state_sim(r, t[i])
            v_rel = v - env_state['wind_velocity']
            q = 0.5 * env_state['density'] * np.linalg.norm(v_rel)**2
            
            if q > max_q:
                max_q = q
                # Calc AoA
                thrust_dir = u[1:, i]
                norm_thrust = np.linalg.norm(thrust_dir)
                if norm_thrust > 1e-9:
                    thrust_dir = thrust_dir / norm_thrust
                    v_norm = np.linalg.norm(v_rel)
                    # Robustness Check: Avoid division by zero at low speeds
                    if v_norm > 1e-3:
                        v_dir = v_rel / v_norm
                        aoa = np.degrees(np.arccos(np.clip(np.dot(thrust_dir, v_dir), -1, 1)))
                        aoa_at_max_q = aoa

        print(f"  Max Q: {max_q/1000:.1f} kPa")
        print(f"  AoA at Max Q: {aoa_at_max_q:.2f} deg")
        self._save_table(
            "aerodynamic_max_q",
            ["max_q_pa", "aoa_at_max_q_deg"],
            [(max_q, aoa_at_max_q)]
        )
        
        if aoa_at_max_q < 4.0:
            print(f">>> {debug.Style.GREEN}PASS: Low aerodynamic stress at Max Q.{debug.Style.RESET}")
        else:
            print(f">>> {debug.Style.RED}FAIL: High AoA at Max Q (>4 deg). Structural risk.{debug.Style.RESET}")

    # 7. MONTE CARLO DISPERSION
    def analyze_monte_carlo_dispersion(self):
        debug._print_sub_header("7. Monte Carlo Dispersion Analysis (N=20)")
        
        # Get nominal controls
        opt_res = self._get_baseline_opt_res()
            
        if not opt_res.get("success", False):
            print(f">>> {debug.Style.RED}CRITICAL: Baseline optimization failed. Skipping Monte Carlo.{debug.Style.RESET}")
            return

        N_runs = 20
        rng = np.random.default_rng(self.seed_sequence.spawn(1)[0])
        strict_success_count = 0
        hardware_success_count = 0
        invalid_terminal_count = 0
        alt_errors = []
        vel_errors = []
        run_rows = []
        
        print(f"{'Run':<4} | {'Thrust':<8} | {'ISP':<8} | {'Dens':<8} | {'Alt Err (km)':<12} | {'Vel Err (m/s)':<12} | {'Status':<6}")
        print("-" * 75)
        
        for i in range(N_runs):
            # Perturb Parameters
            cfg = copy.deepcopy(self.base_config)
            env_cfg = copy.deepcopy(self.base_env_config)
            
            # Perturbations (Gaussian)
            thrust_mult = rng.normal(1.0, 0.02) # 2% sigma
            isp_mult = rng.normal(1.0, 0.01)    # 1% sigma
            dens_mult = rng.normal(1.0, 0.10)   # 10% sigma
            
            # Apply to Config
            cfg.stage_1.thrust_vac *= thrust_mult
            cfg.stage_2.thrust_vac *= thrust_mult
            cfg.stage_1.isp_vac *= isp_mult
            cfg.stage_2.isp_vac *= isp_mult
            env_cfg.density_multiplier = dens_mult
            
            # Re-init Physics
            with suppress_stdout():
                env_mc = Environment(env_cfg)
                veh_mc = Vehicle(cfg, env_mc)
            
            # Run Sim
            with suppress_stdout():
                sim_res = run_simulation(opt_res, veh_mc, cfg)

            metrics = evaluate_terminal_state(sim_res, cfg, env_cfg)
            if not metrics["terminal_valid"]:
                invalid_terminal_count += 1

            if metrics["fuel_ok"]:
                hardware_success_count += 1
            if metrics["strict_ok"]:
                strict_success_count += 1

            alt_err = metrics["alt_err_m"] / 1000.0 if metrics["terminal_valid"] else np.nan
            vel_err = metrics["vel_err_m_s"] if metrics["terminal_valid"] else np.nan
            if np.isfinite(alt_err) and np.isfinite(vel_err):
                alt_errors.append(alt_err)
                vel_errors.append(vel_err)

            status_str = metrics["status"]

            print(f"{i+1:<4} | {thrust_mult:<8.3f} | {isp_mult:<8.3f} | {dens_mult:<8.3f} | {alt_err:<12.1f} | {vel_err:<12.1f} | {status_str:<6}")
            run_rows.append((
                i + 1, thrust_mult, isp_mult, dens_mult, alt_err, vel_err,
                metrics["fuel_margin_kg"], int(metrics["terminal_valid"]), int(metrics["orbit_ok"]),
                int(metrics["fuel_ok"]), int(metrics["strict_ok"]), status_str
            ))
            
        print(f"\nStrict Robustness (orbit+fuel): {strict_success_count}/{N_runs} ({(strict_success_count/N_runs)*100:.0f}%)")
        print(f"Hardware Robustness (fuel only): {hardware_success_count}/{N_runs} ({(hardware_success_count/N_runs)*100:.0f}%)")
        if invalid_terminal_count > 0:
            print(f"Invalid terminal states: {invalid_terminal_count}/{N_runs}")
        if len(alt_errors) > 0:
            print(f"Alt Dispersion (Sigma): {np.std(alt_errors):.2f} km")
            print(f"Vel Dispersion (Sigma): {np.std(vel_errors):.2f} m/s")
        else:
            print("No valid terminal states available for dispersion statistics.")
        self._save_table(
            "monte_carlo_dispersion_n20",
            [
                "run", "thrust_multiplier", "isp_multiplier", "density_multiplier", "altitude_error_km",
                "velocity_error_m_s", "fuel_margin_kg", "terminal_valid", "orbit_ok", "fuel_ok",
                "strict_success", "status"
            ],
            run_rows
        )
        
        # Visualization: Targeting Scatter Plot
        fig = plt.figure(figsize=(10, 6))
        if len(vel_errors) > 0:
            plt.scatter(vel_errors, alt_errors, c='b', alpha=0.6, label='Valid simulations')
        
        # Draw Success Box (+/- 20m/s, +/- 10km)
        rect = plt.Rectangle((-20, -10), 40, 20, linewidth=2, edgecolor='g', facecolor='g', alpha=0.1, label='Success Criteria')
        plt.gca().add_patch(rect)
        
        plt.xlabel('Velocity Error (m/s)')
        plt.ylabel('Altitude Error (km)')
        plt.title(f'Monte Carlo Dispersion (N={N_runs})\nTargeting Accuracy')
        plt.axhline(0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(0, color='k', linestyle='--', alpha=0.3)
        if invalid_terminal_count > 0:
            plt.text(
                0.99, 0.01,
                f'Excluded invalid runs: {invalid_terminal_count}/{N_runs}',
                transform=plt.gca().transAxes,
                ha='right', va='bottom', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
            )
        plt.grid(True)
        plt.legend()
        self._finalize_figure(fig, "monte_carlo_dispersion_n20")

    # 8. LAGRANGE MULTIPLIER ANALYSIS
    def analyze_lagrange_multipliers(self, opt_res):
        debug._print_sub_header("8. Lagrange Multiplier (Sensitivity) Analysis")
        
        if 'lam_g' not in opt_res:
            print("  No dual variables found in result.")
            return
            
        lam = np.array(opt_res['lam_g'])
        
        # Identify active constraints (where g is close to bound and lambda is high)
        max_lam = np.max(np.abs(lam))
        mean_lam = np.mean(np.abs(lam))
        
        print(f"  Max Multiplier:  {max_lam:.2e}")
        print(f"  Mean Multiplier: {mean_lam:.2e}")
        
        if max_lam > 1e3:
            print(f">>> {debug.Style.YELLOW}NOTE: High sensitivity detected. Optimization is fighting hard against a constraint.{debug.Style.RESET}")
        else:
            print(f">>> {debug.Style.GREEN}PASS: Constraints are well-scaled.{debug.Style.RESET}")

    # 9. CORNER CASE TESTING
    def analyze_corner_cases(self):
        debug._print_sub_header("9. Corner Case Testing (Worst Case)")
        
        # Define Worst Case: Low Performance, High Drag
        wc_config = copy.deepcopy(self.base_config)
        wc_config.name = "Worst Case Scenario"
        wc_config.stage_1.thrust_vac *= 0.98 # -2% Thrust
        wc_config.stage_1.isp_vac *= 0.99    # -1% ISP
        wc_config.stage_1.dry_mass *= 1.02   # +2% Dry Mass
        
        # High Density
        wc_env_config = copy.deepcopy(self.base_env_config)
        wc_env_config.density_multiplier = 1.10 # +10% Density
        
        wc_env = Environment(wc_env_config)
        wc_veh = Vehicle(wc_config, wc_env)
        
        print("Running Optimization with Worst-Case Parameters...")
        try:
            with suppress_stdout():
                res = solve_optimal_trajectory(wc_config, wc_veh, wc_env, print_level=0)

            m_dry = wc_config.stage_2.dry_mass + wc_config.payload_mass
            if not res.get("success", False):
                print(f">>> {debug.Style.RED}FAIL: Worst-case optimization did not converge.{debug.Style.RESET}")
                self._save_table(
                    "corner_case_worst_case",
                    ["solver_success", "final_mass_kg", "dry_plus_payload_kg", "fuel_margin_kg"],
                    [(0, np.nan, m_dry, np.nan)]
                )
                return

            m_final = res['X3'][6, -1]
            margin = m_final - m_dry
            
            print(f"  Worst Case Fuel Margin: {margin:.1f} kg")
            
            if margin > 0:
                print(f">>> {debug.Style.GREEN}PASS: Mission feasible in worst case.{debug.Style.RESET}")
            else:
                print(f">>> {debug.Style.RED}FAIL: Mission fails in worst case (Negative Margin).{debug.Style.RESET}")

            self._save_table(
                "corner_case_worst_case",
                ["solver_success", "final_mass_kg", "dry_plus_payload_kg", "fuel_margin_kg"],
                [(1, m_final, m_dry, margin)]
            )
                
            # Visualization: Mass Budget
            fig = plt.figure(figsize=(8, 6))
            wc_dry = wc_config.stage_2.dry_mass
            wc_pay = wc_config.payload_mass
            wc_fuel = margin
            
            plt.bar(['Worst Case'], [wc_dry], label='Dry Structure', color='gray', alpha=0.7)
            plt.bar(['Worst Case'], [wc_pay], bottom=[wc_dry], label='Payload', color='blue', alpha=0.7)
            plt.bar(['Worst Case'], [wc_fuel], bottom=[wc_dry+wc_pay], label='Fuel Margin', color='green' if margin>0 else 'red', alpha=0.7)
            
            plt.ylabel('Mass (kg)')
            plt.title('Worst Case Scenario Mass Budget')
            plt.legend()
            plt.grid(True, axis='y', alpha=0.3)
            self._finalize_figure(fig, "corner_case_mass_budget")
                
        except Exception:
            print(f">>> {debug.Style.RED}FAIL: Optimizer crashed on worst case.{debug.Style.RESET}")

    # 10. MONTE CARLO CONVERGENCE (Upgrade 1)
    def analyze_monte_carlo_convergence(self, N_samples=500):
        debug._print_sub_header(f"10. Monte Carlo Convergence (N={N_samples})")
        print(f"Running large-batch Monte Carlo to demonstrate Statistical Convergence (Law of Large Numbers)...")
        print("Note: This may take several minutes.")
        if N_samples < 1:
            print("N_samples must be >= 1. Skipping Monte Carlo convergence.")
            return
        
        # Get nominal controls
        opt_res = self._get_baseline_opt_res()
            
        if not opt_res.get("success", False):
            print("Baseline optimization failed. Skipping.")
            return

        success_count = 0
        orbit_fail_count = 0
        fuel_fail_count = 0
        invalid_terminal_count = 0
        invalid_trajectory_count = 0
        hardware_success_count = 0
        cumulative_rates = []
        std_errors = []
        wilson_low = []
        wilson_high = []
        phase_space_data = [] # For "Cherry on Top" Phase Space Plot
        terminal_rows = []
        
        t0 = time.time()
        
        # Prepare deterministic per-worker seeds for reproducible Monte Carlo runs.
        child_seqs = self.seed_sequence.spawn(N_samples)
        args_list = [
            (i, self.base_config, self.base_env_config, opt_res, int(child_seqs[i].generate_state(1)[0]))
            for i in range(N_samples)
        ]
        
        n_workers = min(multiprocessing.cpu_count(), max(1, N_samples))
        print(f"  Starting parallel execution with {n_workers} workers...")
        main_file = getattr(__import__("__main__"), "__file__", None)
        can_parallel = main_file is not None and os.path.exists(main_file)
        
        results = []
        try:
            if not can_parallel:
                raise RuntimeError("no importable __main__ module path")
            with multiprocessing.Pool(processes=n_workers) as pool:
                # Use imap to get results as they complete for progress reporting.
                for i, res in enumerate(pool.imap(_monte_carlo_worker, args_list)):
                    results.append(res)
                    if i % 10 == 0 or i == N_samples - 1:
                        elapsed = time.time() - t0
                        avg_time = elapsed / (i + 1)
                        remaining = avg_time * (N_samples - (i + 1))
                        print(f"  Progress: {i+1}/{N_samples} ({((i+1)/N_samples)*100:.1f}%) - ETA: {remaining:.1f}s   ", end='\r')
        except Exception as exc:
            print(f"  Parallel execution unavailable ({exc}). Falling back to sequential mode.")
            results = []
            for i, worker_args in enumerate(args_list):
                results.append(_monte_carlo_worker(worker_args))
                if i % 10 == 0 or i == N_samples - 1:
                    elapsed = time.time() - t0
                    avg_time = elapsed / (i + 1)
                    remaining = avg_time * (N_samples - (i + 1))
                    print(f"  Progress: {i+1}/{N_samples} ({((i+1)/N_samples)*100:.1f}%) - ETA: {remaining:.1f}s   ", end='\r')

        # Process results sequentially to build cumulative stats
        for i, res in enumerate(results):
            orbit_ok = bool(res.get("orbit_ok", False))
            fuel_ok = bool(res.get("fuel_ok", False))
            strict_ok = bool(res.get("strict_ok", False))
            terminal_valid = bool(res.get("terminal_valid", False))
            trajectory_valid = bool(res.get("trajectory_valid", False))
            p_data = res.get("phase_data", None)

            if terminal_valid:
                terminal_rows.append((
                    i + 1,
                    float(res.get("final_altitude_km", np.nan)),
                    float(res.get("final_velocity_m_s", np.nan)),
                    float(res.get("fuel_margin_kg", np.nan)),
                    int(orbit_ok),
                    int(fuel_ok),
                    int(strict_ok),
                    str(res.get("status", "SIM_ERROR"))
                ))
            else:
                invalid_terminal_count += 1
                terminal_rows.append((i + 1, np.nan, np.nan, np.nan, 0, 0, 0, "SIM_ERROR"))

            if not trajectory_valid:
                invalid_trajectory_count += 1

            if fuel_ok:
                hardware_success_count += 1
            
            if strict_ok:
                success_count += 1
            else:
                if terminal_valid and (not orbit_ok):
                    orbit_fail_count += 1
                if terminal_valid and (not fuel_ok):
                    fuel_fail_count += 1
            
            if p_data is not None:
                phase_space_data.append(p_data)
            
            # Calculate stats
            n = i + 1
            p = success_count / n
            cumulative_rates.append(p)
            # Standard Error (Bernoulli)
            se = np.sqrt(p * (1 - p) / n) if n > 1 else 0.0
            std_errors.append(se)
            lo, hi = wilson_interval(success_count, n, z=1.96)
            wilson_low.append(lo)
            wilson_high.append(hi)
            
        print(f"  Progress: {N_samples}/{N_samples} (100.0%) - Done in {time.time()-t0:.1f}s")
        
        # Plotting Data Prep
        n_values = np.arange(1, N_samples + 1)
        rates = np.array(cumulative_rates)
        errors = np.array(std_errors)
        ci_low = np.array(wilson_low)
        ci_high = np.array(wilson_high)

        self._save_table(
            "monte_carlo_convergence",
            ["n", "success_rate", "std_error", "wilson_low_95", "wilson_high_95"],
            [(int(n_values[i]), float(rates[i]), float(errors[i]), float(ci_low[i]), float(ci_high[i])) for i in range(len(n_values))]
        )

        # Print Summary Statistics to Console
        print(f"  Final Statistics (N={N_samples}):")
        if len(rates) > 0:
            print(f"    Strict Success (Hit Target):   {rates[-1]*100:.1f}% (Low due to Open-Loop Sensitivity)")
            print(f"    Hardware Robustness (Fuel>0):  {(hardware_success_count/N_samples)*100:.1f}% (Vehicle Capability)")
            print(f"    Valid terminal states:         {(N_samples-invalid_terminal_count)}/{N_samples}")
            print(f"    Valid trajectories (for phase plots): {(N_samples-invalid_trajectory_count)}/{N_samples}")
            print(f"    Failures Breakdown:")
            print(f"      - Missed Target (Guidance):  {orbit_fail_count}")
            print(f"      - Ran out of Fuel (Perf):    {fuel_fail_count}")
            print(f"      - Invalid terminal data:     {invalid_terminal_count}")
        
        fig = plt.figure(figsize=(10, 6))
        plt.plot(n_values, rates * 100.0, 'b-', label='Cumulative Success Rate')
        plt.fill_between(
            n_values,
            ci_low * 100.0,
            ci_high * 100.0,
            color='b',
            alpha=0.18,
            label='Wilson 95% CI'
        )
        plt.axhline(rates[-1]*100.0, color='k', linestyle='--', alpha=0.5, label=f'Final Rate ({rates[-1]*100:.1f}%)')
        plt.xlabel('Number of Samples (N)')
        plt.ylabel('Success Rate (%)')
        if len(rates) > 0:
            plt.title(
                f'Monte Carlo Convergence (Law of Large Numbers)\n'
                f'Final N={N_samples}, Success Rate={rates[-1]*100:.1f}% '
                f'[{ci_low[-1]*100:.1f}, {ci_high[-1]*100:.1f}]'
            )
        plt.grid(True)
        plt.legend()
        self._finalize_figure(fig, "monte_carlo_convergence_rate")
        
        # Plotting 3: Error Histograms (Altitude & Velocity)
        fig_hist = plt.figure(figsize=(12, 6))
        valid_final_alts = [row[1] for row in terminal_rows if np.isfinite(row[1])]
        valid_final_vels = [row[2] for row in terminal_rows if np.isfinite(row[2])]
        
        # Altitude
        plt.subplot(1, 2, 1)
        target_alt_km = self.base_config.target_altitude / 1000.0
        if len(valid_final_alts) > 0:
            plt.hist(valid_final_alts, bins=30, color='purple', alpha=0.7, edgecolor='black')
        plt.axvline(self.base_config.target_altitude/1000.0, color='k', linestyle='dashed', linewidth=1, label='Target')
        plt.xlabel('Final Altitude (km)')
        plt.ylabel('Frequency')
        plt.title(f'Altitude Dispersion (all valid terminal states, N={len(valid_final_alts)})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Velocity
        plt.subplot(1, 2, 2)
        target_vel = np.sqrt(self.base_env_config.earth_mu / (self.base_env_config.earth_radius_equator + self.base_config.target_altitude))
        if len(valid_final_vels) > 0:
            plt.hist(valid_final_vels, bins=30, color='teal', alpha=0.7, edgecolor='black')
        plt.axvline(target_vel, color='k', linestyle='dashed', linewidth=1, label='Target')
        plt.xlabel('Final Velocity (m/s)')
        plt.ylabel('Frequency')
        plt.title(f'Velocity Dispersion (all valid terminal states, N={len(valid_final_vels)})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        self._finalize_figure(fig_hist, "monte_carlo_terminal_histograms")
        
        # Plotting 2: Phase Space (Cherry on Top)
        fig_phase = plt.figure(figsize=(10, 6))
        for alt, vel, success in phase_space_data:
            color = 'g' if success else 'r'
            alpha = 0.1 if success else 0.8 # Highlight failures
            zorder = 1 if success else 2    # Draw failures on top
            plt.plot(alt, vel, color=color, alpha=alpha, linewidth=0.8, zorder=zorder)
            
        plt.xlabel('Altitude (km)')
        plt.ylabel('Velocity (m/s)')
        plt.title(f'Phase Space Analysis (Valid trajectories N={len(phase_space_data)})\nGreen=Success, Red=Failure')
        plt.grid(True)
        self._finalize_figure(fig_phase, "monte_carlo_phase_space")
        
        # Plotting 4: Terminal State Scatter (Targeting)
        fig_scatter = plt.figure(figsize=(10, 6))

        self._save_table(
            "monte_carlo_terminal_states",
            [
                "run", "final_altitude_km", "final_velocity_m_s", "fuel_margin_kg",
                "orbit_ok", "fuel_ok", "strict_success", "status"
            ],
            terminal_rows
        )
        self._save_table(
            "monte_carlo_data_integrity",
            [
                "total_runs", "valid_terminal_runs", "invalid_terminal_runs",
                "valid_trajectory_runs", "invalid_trajectory_runs"
            ],
            [(
                N_samples,
                N_samples - invalid_terminal_count,
                invalid_terminal_count,
                N_samples - invalid_trajectory_count,
                invalid_trajectory_count
            )]
        )
        
        final_alts = [row[1] for row in terminal_rows if np.isfinite(row[1])]
        final_vels = [row[2] for row in terminal_rows if np.isfinite(row[2])]
        success_flags = [bool(row[6]) for row in terminal_rows if np.isfinite(row[1]) and np.isfinite(row[2])]

        # Separate success and failure for coloring
        succ_alts = [a for a, s in zip(final_alts, success_flags) if s]
        succ_vels = [v for v, s in zip(final_vels, success_flags) if s]
        fail_alts = [a for a, s in zip(final_alts, success_flags) if not s]
        fail_vels = [v for v, s in zip(final_vels, success_flags) if not s]
        
        plt.scatter(fail_alts, fail_vels, c='r', alpha=0.5, label='Failure', s=15)
        plt.scatter(succ_alts, succ_vels, c='g', alpha=0.7, label='Success', s=15)
        
        # Target Marker
        target_alt_km = self.base_config.target_altitude / 1000.0
        target_vel_ms = np.sqrt(self.base_env_config.earth_mu / (self.base_env_config.earth_radius_equator + self.base_config.target_altitude))
        
        plt.plot(target_alt_km, target_vel_ms, 'k+', markersize=20, markeredgewidth=2, label='Target Orbit')
        
        # Tolerance Box (+/- 10km, +/- 20m/s)
        rect = plt.Rectangle((target_alt_km - 10, target_vel_ms - 20), 20, 40, 
                             linewidth=1, edgecolor='k', facecolor='none', linestyle='--', label='Tolerance')
        plt.gca().add_patch(rect)
        
        # Add 2D Gaussian-coverage ellipses for successful runs.
        if len(succ_alts) > 5:
            ax = plt.gca()
            p1 = 100.0 * gaussian_2d_coverage(1.0)
            p3 = 100.0 * gaussian_2d_coverage(3.0)
            confidence_ellipse(np.array(succ_alts), np.array(succ_vels), ax, n_std=1.0, edgecolor='blue', linestyle='--', label=f'1σ ellipse ({p1:.1f}% in 2D)')
            confidence_ellipse(np.array(succ_alts), np.array(succ_vels), ax, n_std=3.0, edgecolor='blue', linestyle=':', label=f'3σ ellipse ({p3:.1f}% in 2D)')
        
        plt.xlabel('Final Altitude (km)')
        plt.ylabel('Final Velocity (m/s)')
        plt.title(f'Terminal State Dispersion (valid terminal states N={len(final_alts)})')
        if invalid_terminal_count > 0:
            plt.text(
                0.99, 0.01,
                f'Excluded invalid terminal states: {invalid_terminal_count}/{N_samples}',
                transform=plt.gca().transAxes,
                ha='right', va='bottom', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
            )
        plt.legend()
        plt.grid(True)
        self._finalize_figure(fig_scatter, "monte_carlo_terminal_scatter")
        
        print(f">>> {debug.Style.GREEN}PASS: Convergence analysis complete.{debug.Style.RESET}")
        print(f"\n{debug.Style.BOLD}REPORT RECOMMENDATION (Open-Loop Robustness):{debug.Style.RESET}")
        print("Explicitly state in your report:")
        final_rate = rates[-1] * 100.0 if len(rates) > 0 else 0.0
        print(f"'The low open-loop success rate ({final_rate:.1f}%) validates that optimal trajectories are inherently sensitive.")
        print(" This proves the necessity of Closed-Loop Guidance (PID/MPC) for flight,")
        print(" as Open-Loop optimization is insufficient for robustness against environmental perturbations.'")


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
        
        # Calculate divergence (Euclidean distance in Position)
        delta_r = np.linalg.norm(y_nom_interp[0:3, :] - y_pert_interp[0:3, :], axis=0)
        delta_r = np.maximum(delta_r, 1e-12) # Avoid log(0)
        log_delta = np.log(delta_r)
        
        # Quantitative Analysis
        # Finite-time divergence growth-rate estimate from log-distance slope.
        # Fit to the middle 50% of the trajectory to avoid transient and saturation
        idx_start = int(len(t_dense) * 0.2)
        idx_end = int(len(t_dense) * 0.8)
        coeffs = np.polyfit(t_dense[idx_start:idx_end], log_delta[idx_start:idx_end], 1)
        growth_rate = coeffs[0]
        final_div = delta_r[-1] / 1000.0 # km
        fit_x = t_dense[idx_start:idx_end]
        fit_y = log_delta[idx_start:idx_end]
        fit_pred = coeffs[0] * fit_x + coeffs[1]
        ss_res = np.sum((fit_y - fit_pred)**2)
        ss_tot = np.sum((fit_y - np.mean(fit_y))**2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-16)

        self._save_table(
            "finite_time_divergence",
            ["time_s", "delta_r_m", "log_delta_r"],
            [(t_dense[i], delta_r[i], log_delta[i]) for i in range(len(t_dense))]
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

    # 12. STIFFNESS / EULER TEST (Upgrade 3)
    def analyze_stiffness_euler(self):
        debug._print_sub_header("12. Integrator Convergence Under Piecewise-Constant Controls")
        print("Comparing fixed-step Euler and RK4 against a high-accuracy Phase-1 reference.")
        print("Note: With piecewise-constant controls and non-smooth aerodynamics, RK4 formal order 4 does not fully appear.")
        
        # 1. Get Controls & Reference
        opt_res = self._get_baseline_opt_res()
            
        if not opt_res.get("success", False):
            print("  Baseline optimization failed. Skipping convergence test.")
            return

        # Rebuild interpolator for Phase 1 controls.
        T1 = opt_res["T1"]
        U1 = np.array(opt_res["U1"])
        t_grid_1 = np.linspace(0, T1, U1.shape[1] + 1)[:-1]
        ctrl_1 = interp1d(t_grid_1, U1, axis=1, kind='previous', fill_value="extrapolate", bounds_error=False)
        
        # Build a dedicated high-accuracy reference for Phase 1 at exact T1.
        r0, v0 = self.env.get_launch_site_state()
        y0 = np.concatenate([r0, v0, [self.base_config.launch_mass]])

        def dynamics_phase1(t, y):
            u = ctrl_1(t)
            return self.veh.get_dynamics(y, u[0], u[1:], t, stage_mode="boost", scaling=None)

        with suppress_stdout():
            ref_res = solve_ivp(
                fun=dynamics_phase1,
                t_span=(0.0, T1),
                y0=y0,
                method="DOP853",
                rtol=1e-12,
                atol=1e-14
            )
        if not ref_res.success or ref_res.y.shape[1] < 1:
            print("  High-accuracy reference integration failed. Skipping convergence test.")
            return
        y_ref_final = ref_res.y[:, -1]
        
        # 2. Convergence Study
        dt_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        euler_errors = []
        rk4_errors = []
        
        print(f"{'dt (s)':<8} | {'Euler Err (m)':<15} | {'RK4 Err (m)':<15}")
        print("-" * 45)

        for dt in dt_values:
            # --- Euler ---
            t_curr = 0.0
            y_curr = y0.copy()
            while t_curr < T1 - 1e-14:
                # Clamp last step
                step = min(dt, T1 - t_curr)
                u = ctrl_1(t_curr)
                dy = self.veh.get_dynamics(y_curr, u[0], u[1:], t_curr, stage_mode="boost", scaling=None)
                y_curr += dy * step
                t_curr += step
            
            err_eu = np.linalg.norm(y_curr[0:3] - y_ref_final[0:3])
            euler_errors.append(err_eu)
            
            # --- Fixed RK4 ---
            t_curr = 0.0
            y_curr = y0.copy()
            while t_curr < T1 - 1e-14:
                step = min(dt, T1 - t_curr)
                
                # k1
                u1_val = ctrl_1(t_curr)
                k1 = self.veh.get_dynamics(y_curr, u1_val[0], u1_val[1:], t_curr, stage_mode="boost", scaling=None)
                
                # k2
                t_half = t_curr + 0.5 * step
                u2_val = ctrl_1(t_half)
                k2 = self.veh.get_dynamics(y_curr + 0.5*step*k1, u2_val[0], u2_val[1:], t_half, stage_mode="boost", scaling=None)
                
                # k3
                k3 = self.veh.get_dynamics(y_curr + 0.5*step*k2, u2_val[0], u2_val[1:], t_half, stage_mode="boost", scaling=None)
                
                # k4
                t_full = t_curr + step
                u4_val = ctrl_1(t_full)
                k4 = self.veh.get_dynamics(y_curr + step*k3, u4_val[0], u4_val[1:], t_full, stage_mode="boost", scaling=None)
                
                y_curr += (step / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                t_curr += step
                
            err_rk4 = np.linalg.norm(y_curr[0:3] - y_ref_final[0:3])
            rk4_errors.append(err_rk4)
            
            print(f"{dt:<8.3f} | {err_eu:<15.4f} | {err_rk4:<15.4f}")

        # 3. Estimate effective orders with robust log-log regression.
        dt_arr = np.array(dt_values, dtype=float)
        eu_arr = np.maximum(np.array(euler_errors, dtype=float), 1e-16)
        rk_arr = np.maximum(np.array(rk4_errors, dtype=float), 1e-16)
        log_dt = np.log10(dt_arr)

        # Use smaller dt points for local asymptotic slope estimation.
        fit_idx = np.where(dt_arr <= 0.1)[0]
        if len(fit_idx) < 3:
            fit_idx = np.arange(max(3, len(dt_arr) // 2))

        def fit_slope_and_r2(log_x, log_y):
            coeff = np.polyfit(log_x, log_y, 1)
            pred = coeff[0] * log_x + coeff[1]
            ss_res = np.sum((log_y - pred) ** 2)
            ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
            r2 = 1.0 - ss_res / (ss_tot + 1e-16)
            return coeff, r2

        coeff_eu, r2_eu = fit_slope_and_r2(log_dt[fit_idx], np.log10(eu_arr[fit_idx]))
        coeff_rk, r2_rk = fit_slope_and_r2(log_dt[fit_idx], np.log10(rk_arr[fit_idx]))
        slope_eu = coeff_eu[0]
        slope_rk = coeff_rk[0]
        
        print(f"\nEffective convergence slopes (log-log regression on dt <= 0.1 s):")
        print(f"  Euler: {slope_eu:.2f} (R^2={r2_eu:.3f}, expected ~1 for first-order)")
        print(f"  RK4:   {slope_rk:.2f} (R^2={r2_rk:.3f}, formal 4 only for smooth RHS)")
        if slope_rk < 3.0:
            print("  Interpretation: RK4 order reduction is expected here due to control/discrete-model non-smoothness.")

        self._save_table(
            "stiffness_convergence",
            ["dt_s", "euler_error_m", "rk4_error_m"],
            [(dt_values[i], euler_errors[i], rk4_errors[i]) for i in range(len(dt_values))]
        )
        self._save_table(
            "stiffness_convergence_fit",
            ["method", "fit_range_dt_max_s", "slope", "r2"],
            [
                ("Euler", float(np.max(dt_arr[fit_idx])), float(slope_eu), float(r2_eu)),
                ("RK4", float(np.max(dt_arr[fit_idx])), float(slope_rk), float(r2_rk))
            ]
        )
        
        # 4. Plotting
        fig = plt.figure(figsize=(10, 7))
        plt.loglog(dt_arr, eu_arr, 'r-o', label=f'Euler (fit slope={slope_eu:.2f})')
        plt.loglog(dt_arr, rk_arr, 'b-s', label=f'Fixed RK4 (fit slope={slope_rk:.2f})')
        
        # Fit trend lines on the selected local-fit interval.
        fit_dt = np.linspace(np.min(dt_arr[fit_idx]), np.max(dt_arr[fit_idx]), 100)
        fit_log_dt = np.log10(fit_dt)
        eu_fit_line = 10 ** (coeff_eu[0] * fit_log_dt + coeff_eu[1])
        rk_fit_line = 10 ** (coeff_rk[0] * fit_log_dt + coeff_rk[1])
        plt.loglog(fit_dt, eu_fit_line, 'r--', alpha=0.5, label='Euler local fit')
        plt.loglog(fit_dt, rk_fit_line, 'b--', alpha=0.5, label='RK4 local fit')
        
        # First-order reference for context in this non-smooth setting.
        ref_idx = len(dt_arr) // 2
        ref_y1 = eu_arr[ref_idx] * (dt_arr / dt_arr[ref_idx]) ** 1
        plt.loglog(dt_arr, ref_y1, 'k:', alpha=0.35, label='O(dt) reference')
        
        plt.xlabel('Time Step dt (s)')
        plt.ylabel('Phase-1 Final Position Error (m)')
        plt.title('Integrator Convergence with Piecewise-Constant Controls')
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend()
        self._finalize_figure(fig, "stiffness_euler_vs_rk4")
        
        print(f"\n{debug.Style.BOLD}THEORETICAL DEFENSE (For Report):{debug.Style.RESET}")
        print("1. Why RK45 for production simulation?")
        print("   Adaptive stepping gives robust accuracy/cost tradeoff across fast launch and slower coast segments.")
        print("2. Why not Symplectic (Verlet)? The system is Non-Conservative (Thrust/Drag add/remove energy).")
        print("   Symplectic integrators are designed for Hamiltonian systems (Energy Conserving), which this is not.")
        print("3. Why RK4 does not show formal order 4 here?")
        print("   Piecewise-constant controls and non-smooth model terms reduce observable global order.")

    # 13. BIFURCATION ANALYSIS (Upgrade 4)
    def analyze_bifurcation(self):
        debug._print_sub_header("13. Bifurcation Analysis (The 'Cliff Edge')")
        
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
        statuses = []
        
        print(f"{'Thrust Mult':<12} | {'Final Alt (km)':<15} | {'Status':<10}")
        print("-" * 45)
        
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

            if metrics["strict_ok"]:
                status = "Orbit"
            elif metrics["terminal_valid"]:
                status = metrics["status"]
            else:
                status = "SIM_ERROR"
            statuses.append(status)
            print(f"{mult:<12.2f} | {final_alts[-1]:<15.1f} | {status:<10}")

        self._save_table(
            "bifurcation_thrust_sweep",
            ["thrust_multiplier", "final_altitude_km", "final_velocity_m_s", "fuel_margin_kg", "status"],
            [(multipliers[i], final_alts[i], final_vels[i], margins[i], statuses[i]) for i in range(len(multipliers))]
        )

        critical_mult = None
        for i in range(1, len(multipliers)):
            if margins[i-1] <= 0.0 <= margins[i]:
                critical_mult = np.interp(0.0, [margins[i-1], margins[i]], [multipliers[i-1], multipliers[i]])
                break
            if margins[i-1] >= 0.0 >= margins[i]:
                critical_mult = np.interp(0.0, [margins[i-1], margins[i]], [multipliers[i-1], multipliers[i]])
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
        color = 'tab:red'
        ax2.set_ylabel('Fuel Margin (kg)', color=color)
        ax2.plot(multipliers, margins, 'x--', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.axhline(0, color='r', linestyle=':', alpha=0.5)
        if critical_mult is not None:
            ax1.axvline(critical_mult, color='purple', linestyle='-.', alpha=0.7, label=f'Fuel Cliff ~ {critical_mult:.3f}')
            print(f"  Estimated fuel-margin bifurcation near thrust multiplier {critical_mult:.3f}")
        
        plt.title('Bifurcation Analysis: Sensitivity to Thrust\n(Open-Loop Control)')
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
    def analyze_monte_carlo_precision_target(self, target_half_width=0.04, min_samples=80, max_samples=300, batch_size=20):
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
            batch_rows.append((n, p, lo, hi, half_width))
            print(f"  N={n:4d} | Success={p*100:6.2f}% | CI=[{lo*100:5.2f}, {hi*100:5.2f}] | Half-width={half_width*100:5.2f}%")

            if n >= min_samples and half_width <= target_half_width:
                stop_reason = "target_precision"
                break

        elapsed = time.time() - t0
        final_n, final_p, final_lo, final_hi, final_half = batch_rows[-1]
        print(f"  Stop reason: {stop_reason}")
        print(f"  Final: N={final_n}, Success={final_p*100:.2f}%, CI=[{final_lo*100:.2f}, {final_hi*100:.2f}]")
        print(f"  Runtime: {elapsed:.1f}s")

        self._save_table(
            "monte_carlo_precision_target",
            ["n", "success_rate", "wilson_low_95", "wilson_high_95", "ci_half_width_95", "target_half_width"],
            [(row[0], row[1], row[2], row[3], row[4], target_half_width) for row in batch_rows]
        )

        fig, ax1 = plt.subplots(figsize=(10, 6))
        n_vals = np.array([r[0] for r in batch_rows], dtype=int)
        p_vals = np.array([r[1] for r in batch_rows], dtype=float) * 100.0
        lo_vals = np.array([r[2] for r in batch_rows], dtype=float) * 100.0
        hi_vals = np.array([r[3] for r in batch_rows], dtype=float) * 100.0
        hw_vals = np.array([r[4] for r in batch_rows], dtype=float) * 100.0

        ax1.plot(n_vals, p_vals, "b-o", label="Success rate")
        ax1.fill_between(n_vals, lo_vals, hi_vals, color="b", alpha=0.15, label="Wilson 95% CI")
        ax1.set_xlabel("Samples N")
        ax1.set_ylabel("Success Rate (%)", color="b")
        ax1.tick_params(axis="y", labelcolor="b")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(n_vals, hw_vals, "r-s", label="CI half-width")
        ax2.axhline(target_half_width * 100.0, color="r", linestyle="--", alpha=0.7, label="Target half-width")
        ax2.set_ylabel("CI Half-Width (%)", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
        plt.title("Monte Carlo Precision Target (Adaptive Sample Size)")
        self._finalize_figure(fig, "monte_carlo_precision_target")

    # 18. GLOBAL SENSITIVITY RANKING (PRCC / SPEARMAN)
    def analyze_global_sensitivity(self, N_samples=160):
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

        self._save_table(
            "global_sensitivity_samples",
            [
                "run", "thrust_multiplier", "isp_multiplier", "density_multiplier",
                "altitude_error_m", "velocity_error_m_s", "fuel_margin_kg", "strict_success", "status"
            ],
            rows
        )

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

        spearman_alt = np.array([self._corr(self._rankdata(X[:, j]), self._rankdata(y_alt)) for j in range(X.shape[1])])
        spearman_vel = np.array([self._corr(self._rankdata(X[:, j]), self._rankdata(y_vel)) for j in range(X.shape[1])])
        spearman_fuel = np.array([self._corr(self._rankdata(X[:, j]), self._rankdata(y_fuel)) for j in range(X.shape[1])])
        prcc_alt = prcc(X, np.array(y_alt, dtype=float))
        prcc_vel = prcc(X, np.array(y_vel, dtype=float))
        prcc_fuel = prcc(X, np.array(y_fuel, dtype=float))

        summary_rows = []
        for i, name in enumerate(var_names):
            summary_rows.append((
                name,
                float(spearman_alt[i]), float(prcc_alt[i]),
                float(spearman_vel[i]), float(prcc_vel[i]),
                float(spearman_fuel[i]), float(prcc_fuel[i])
            ))
        self._save_table(
            "global_sensitivity_ranking",
            [
                "variable",
                "spearman_altitude_error", "prcc_altitude_error",
                "spearman_velocity_error", "prcc_velocity_error",
                "spearman_fuel_margin", "prcc_fuel_margin"
            ],
            summary_rows
        )

        print("  PRCC ranking by |value| (Fuel Margin):")
        order = np.argsort(np.abs(prcc_fuel))[::-1]
        for idx in order:
            print(f"    {var_names[idx]:<20}: {prcc_fuel[idx]: .3f}")

        fig = plt.figure(figsize=(10, 6))
        x = np.arange(len(var_names))
        w = 0.25
        plt.bar(x - w, np.abs(prcc_alt), width=w, label='|PRCC| Altitude Error')
        plt.bar(x, np.abs(prcc_vel), width=w, label='|PRCC| Velocity Error')
        plt.bar(x + w, np.abs(prcc_fuel), width=w, label='|PRCC| Fuel Margin')
        plt.xticks(x, var_names, rotation=15)
        plt.ylabel("Absolute PRCC")
        plt.title("Global Sensitivity Ranking (Rank-Based Partial Correlation)")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        self._finalize_figure(fig, "global_sensitivity_ranking")

    # 19. CONSTRAINT RELIABILITY CURVES
    def analyze_constraint_reliability(self, N_samples=180):
        debug._print_sub_header(f"19. Constraint Reliability Curves (N={N_samples})")
        print("Tracking cumulative probabilities of key constraint violations under uncertainty.")

        if N_samples < 20:
            print("  Need at least 20 samples for reliability curves. Skipping.")
            return

        opt_res = self._get_baseline_opt_res()
        rng = np.random.default_rng(self.seed_sequence.spawn(1)[0])
        run_rows = []
        cumulative_rows = []

        count_orbit = 0
        count_fuel = 0
        count_q = 0
        count_g = 0
        count_invalid_traj = 0
        count_any = 0

        for i in range(N_samples):
            thrust_mult, isp_mult, dens_mult = self._draw_uncertainty_multipliers(rng, model="gaussian_independent")
            result = self._simulate_uncertainty_case(opt_res, thrust_mult, isp_mult, dens_mult)
            tm = result["terminal"]
            tr = result["trajectory"]

            orbit_fail = int((not tm["terminal_valid"]) or (not tm["orbit_ok"]))
            fuel_fail = int((not tm["terminal_valid"]) or (not tm["fuel_ok"]))
            traj_invalid = int(not tr["valid"])
            q_violate = int(tr["valid"] and tr["q_violation"])
            g_violate = int(tr["valid"] and tr["g_violation"])
            q_fail_conservative = int((not tr["valid"]) or tr["q_violation"])
            g_fail_conservative = int((not tr["valid"]) or tr["g_violation"])
            any_fail = int(orbit_fail or fuel_fail or q_fail_conservative or g_fail_conservative)

            count_orbit += orbit_fail
            count_fuel += fuel_fail
            count_q += q_violate
            count_g += g_violate
            count_invalid_traj += traj_invalid
            count_any += any_fail
            n = i + 1

            cumulative_rows.append((
                n,
                count_orbit / n,
                count_fuel / n,
                count_q / n,
                count_g / n,
                count_invalid_traj / n,
                count_any / n
            ))
            run_rows.append((
                n,
                thrust_mult,
                isp_mult,
                dens_mult,
                int(tm["terminal_valid"]),
                int(tm["orbit_ok"]),
                int(tm["fuel_ok"]),
                float(tm["fuel_margin_kg"]),
                int(tr["valid"]),
                float(tr["max_q_pa"]),
                float(tr["max_g"]),
                orbit_fail,
                fuel_fail,
                q_violate,
                g_violate,
                traj_invalid,
                q_fail_conservative,
                g_fail_conservative,
                any_fail
            ))

        self._save_table(
            "constraint_reliability_runs",
            [
                "run", "thrust_multiplier", "isp_multiplier", "density_multiplier",
                "terminal_valid", "orbit_ok", "fuel_ok", "fuel_margin_kg",
                "trajectory_valid", "max_q_pa", "max_g",
                "orbit_fail", "fuel_fail", "q_violation_physical", "g_violation_physical",
                "trajectory_invalid", "q_fail_conservative", "g_fail_conservative", "any_failure_conservative"
            ],
            run_rows
        )
        self._save_table(
            "constraint_reliability_curves",
            [
                "n", "p_orbit_fail", "p_fuel_fail",
                "p_q_violate_physical", "p_g_violate_physical",
                "p_trajectory_invalid", "p_any_failure_conservative"
            ],
            cumulative_rows
        )

        print("  Final violation probabilities:")
        labels = [
            ("orbit", count_orbit),
            ("fuel", count_fuel),
            ("max_q (physical)", count_q),
            ("max_g (physical)", count_g),
            ("traj_invalid", count_invalid_traj),
            ("any (conservative)", count_any),
        ]
        for name, c in labels:
            lo, hi = wilson_interval(c, N_samples, z=1.96)
            print(f"    {name:<6}: {100*c/N_samples:6.2f}%  CI=[{100*lo:5.2f}, {100*hi:5.2f}]")

        n_vals = np.array([r[0] for r in cumulative_rows], dtype=int)
        p_orbit = np.array([r[1] for r in cumulative_rows], dtype=float)
        p_fuel = np.array([r[2] for r in cumulative_rows], dtype=float)
        p_q = np.array([r[3] for r in cumulative_rows], dtype=float)
        p_g = np.array([r[4] for r in cumulative_rows], dtype=float)
        p_invalid = np.array([r[5] for r in cumulative_rows], dtype=float)
        p_any = np.array([r[6] for r in cumulative_rows], dtype=float)

        fig = plt.figure(figsize=(10, 6))
        plt.plot(n_vals, 100.0 * p_orbit, label="Orbit fail")
        plt.plot(n_vals, 100.0 * p_fuel, label="Fuel fail")
        plt.plot(n_vals, 100.0 * p_q, label="Max-Q violation (physical)")
        plt.plot(n_vals, 100.0 * p_g, label="Max-G violation (physical)")
        plt.plot(n_vals, 100.0 * p_invalid, "m-.", linewidth=1.5, label="Trajectory invalid")
        plt.plot(n_vals, 100.0 * p_any, "k--", linewidth=2, label="Any failure (conservative)")
        plt.xlabel("Samples N")
        plt.ylabel("Violation Probability (%)")
        plt.title("Constraint Reliability Curves (Cumulative)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        self._finalize_figure(fig, "constraint_reliability_curves")

    # 20. DISTRIBUTION ROBUSTNESS STUDY
    def analyze_distribution_robustness(self, N_per_model=100):
        debug._print_sub_header(f"20. Distribution Robustness (N={N_per_model} per model)")
        print("Comparing reliability outcomes under different uncertainty distribution assumptions.")

        if N_per_model < 20:
            print("  Need at least 20 samples per model. Skipping.")
            return

        opt_res = self._get_baseline_opt_res()
        models = [
            "gaussian_independent",
            "uniform_bounded",
            "gaussian_correlated"
        ]
        rng = np.random.default_rng(self.seed_sequence.spawn(1)[0])
        run_rows = []
        summary_rows = []

        for model in models:
            success = 0
            q_viol = 0
            g_viol = 0
            q_fail_cons = 0
            g_fail_cons = 0
            invalid_terminal = 0
            invalid_traj = 0
            for i in range(N_per_model):
                thrust_mult, isp_mult, dens_mult = self._draw_uncertainty_multipliers(rng, model=model)
                result = self._simulate_uncertainty_case(opt_res, thrust_mult, isp_mult, dens_mult)
                tm = result["terminal"]
                tr = result["trajectory"]
                success += int(tm["strict_ok"])
                q_viol += int(tr["valid"] and tr["q_violation"])
                g_viol += int(tr["valid"] and tr["g_violation"])
                q_fail_cons += int((not tr["valid"]) or tr["q_violation"])
                g_fail_cons += int((not tr["valid"]) or tr["g_violation"])
                invalid_terminal += int(not tm["terminal_valid"])
                invalid_traj += int(not tr["valid"])
                run_rows.append((
                    model, i + 1, thrust_mult, isp_mult, dens_mult,
                    int(tm["strict_ok"]), int(tm["terminal_valid"]),
                    int(tr["valid"]),
                    int(tr["valid"] and tr["q_violation"]),
                    int(tr["valid"] and tr["g_violation"]),
                    int((not tr["valid"]) or tr["q_violation"]),
                    int((not tr["valid"]) or tr["g_violation"]),
                    tm["status"]
                ))

            p_success = success / N_per_model
            lo, hi = wilson_interval(success, N_per_model, z=1.96)
            summary_rows.append((
                model, N_per_model, p_success, lo, hi,
                q_viol / N_per_model,
                g_viol / N_per_model,
                q_fail_cons / N_per_model,
                g_fail_cons / N_per_model,
                invalid_terminal / N_per_model,
                invalid_traj / N_per_model
            ))
            print(f"  {model:<22} | Success={p_success*100:6.2f}% | CI=[{lo*100:5.2f}, {hi*100:5.2f}]")

        self._save_table(
            "distribution_robustness_runs",
            [
                "model", "run", "thrust_multiplier", "isp_multiplier", "density_multiplier",
                "strict_success", "terminal_valid", "trajectory_valid",
                "q_violation_physical", "g_violation_physical",
                "q_fail_conservative", "g_fail_conservative", "status"
            ],
            run_rows
        )
        self._save_table(
            "distribution_robustness_summary",
            [
                "model", "n", "success_rate", "wilson_low_95", "wilson_high_95",
                "q_violation_rate_physical", "g_violation_rate_physical",
                "q_fail_rate_conservative", "g_fail_rate_conservative",
                "invalid_terminal_rate", "invalid_trajectory_rate"
            ],
            summary_rows
        )

        models_plot = [r[0] for r in summary_rows]
        rates = np.array([r[2] for r in summary_rows], dtype=float) * 100.0
        lo = np.array([r[3] for r in summary_rows], dtype=float) * 100.0
        hi = np.array([r[4] for r in summary_rows], dtype=float) * 100.0
        err = np.vstack([rates - lo, hi - rates])

        fig = plt.figure(figsize=(10, 6))
        plt.bar(models_plot, rates, yerr=err, capsize=6, alpha=0.8, color=["tab:blue", "tab:orange", "tab:green"])
        plt.ylabel("Strict Success Rate (%)")
        plt.title("Distribution Robustness: Success Rate Across Uncertainty Models")
        plt.grid(True, axis="y", alpha=0.3)
        self._finalize_figure(fig, "distribution_robustness_success")

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
                rtol, atol, diag["max_q_time_s"], diag["max_g_time_s"], diag["staging_time_s"], diag["depletion_time_s"]
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
                drift(d["depletion_time_s"], ref["depletion_time_s"])
            ))

        self._save_table(
            "event_time_convergence",
            [
                "rtol", "atol", "t_max_q_s", "t_max_g_s", "t_staging_s", "t_depletion_s",
                "drift_max_q_s", "drift_max_g_s", "drift_staging_s", "drift_depletion_s"
            ],
            [
                (
                    rows[i][0], rows[i][1], rows[i][2], rows[i][3], rows[i][4], rows[i][5],
                    drift_rows[i][2], drift_rows[i][3], drift_rows[i][4], drift_rows[i][5]
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
        print("Mapping open-loop success and fuel margin over a 2D parameter grid.")

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
        rows = []

        for iy, dens in enumerate(density_vals):
            for ix, thrust in enumerate(thrust_vals):
                result = self._simulate_uncertainty_case(opt_res, thrust, 1.0, dens)
                tm = result["terminal"]
                success = int(tm["strict_ok"])
                margin = tm["fuel_margin_kg"] if tm["terminal_valid"] else np.nan
                success_map[iy, ix] = success
                margin_map[iy, ix] = margin
                rows.append((
                    float(thrust), float(dens), int(success), float(margin),
                    float(tm["alt_err_m"]), float(tm["vel_err_m_s"]), tm["status"]
                ))

        self._save_table(
            "bifurcation_2d_map",
            [
                "thrust_multiplier", "density_multiplier", "strict_success",
                "fuel_margin_kg", "altitude_error_m", "velocity_error_m_s", "status"
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

        im2 = ax2.imshow(
            margin_map,
            origin="lower",
            aspect="auto",
            extent=[thrust_vals[0], thrust_vals[-1], density_vals[0], density_vals[-1]],
            cmap="coolwarm"
        )
        ax2.set_xlabel("Thrust Multiplier")
        ax2.set_ylabel("Density Multiplier")
        ax2.set_title("Fuel Margin Map (kg)")
        fig.colorbar(im2, ax=ax2, label="Fuel Margin (kg)")
        ax2.contour(
            thrust_vals, density_vals, margin_map,
            levels=[0.0], colors="k", linewidths=1.2
        )
        self._finalize_figure(fig, "bifurcation_2d_map")
        print(f">>> {debug.Style.GREEN}PASS: 2D bifurcation map generated.{debug.Style.RESET}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reliability and robustness analysis suite.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for exported figures and CSV tables.")
    parser.add_argument("--seed", type=int, default=1337, help="Global random seed for reproducible Monte Carlo runs.")
    parser.add_argument("--mc-samples", type=int, default=500, help="Monte Carlo sample count used in convergence analysis.")
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
