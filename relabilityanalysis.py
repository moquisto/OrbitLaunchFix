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
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import csv
from pathlib import Path
from datetime import datetime
import argparse
import platform
from matplotlib.ticker import StrMethodFormatter

# Project Imports
from config import StarshipBlock2, EARTH_CONFIG, RELIABILITY_ANALYSIS_TOGGLES
from vehicle import Vehicle
from environment import Environment
from main import (
    OPTIMIZATION_G_GUARD_BAND,
    OPTIMIZATION_Q_GUARD_BAND_PA,
    solve_optimal_trajectory,
)
from simulation import run_simulation
import debug
import guidance

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
        toggles.collocation_defect_audit = True
        toggles.theoretical_efficiency = True
        toggles.integrator_tolerance = True
        toggles.monte_carlo_precision_target = False
        toggles.q2_uncertainty_budget = False
        toggles.smooth_integrator_benchmark = True
        toggles.bifurcation_2d_map = False
        toggles.drift = True
        toggles.model_limitations = False
        toggles.q7_conclusion_support = False
    return toggles




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
    alt_tol_m=TERMINAL_ALT_TOL_M,
    vel_tol_m_s=TERMINAL_VEL_TOL_M_S,
    radial_vel_tol_m_s=TERMINAL_RADIAL_VEL_TOL_M_S,
    inc_tol_deg=TERMINAL_INC_TOL_DEG,
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


def _ms_refine_peak_time(t_series, y_series, idx_peak):
    """
    Worker-safe peak refinement used by parallel multi-start diagnostics.
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


def _run_multistart_trial_worker(payload):
    """
    One randomized multi-start trial executed in its own process.
    """
    idx = int(payload["trial_index"])
    label = str(payload["label"])
    params = payload["params"]
    cfg = payload["cfg"]
    env_cfg = payload["env_cfg"]
    guess_i = payload["guess"]
    q_slack_pa = float(payload["q_slack_pa"])
    g_slack = float(payload["g_slack"])

    t0 = time.time()
    solver_ok = False
    mission_ok = False
    terminal_strict_ok = False
    path_ok = False
    q_ok = False
    g_ok = False
    m_final = np.nan
    status = "FAILED"

    try:
        env_i = Environment(env_cfg)
        veh_i = Vehicle(cfg, env_i)
    except Exception:
        return {
            "trial_index": idx,
            "label": label,
            "params": params,
            "solver_success": 0,
            "mission_success": 0,
            "terminal_strict_success": 0,
            "path_ok": 0,
            "q_ok": 0,
            "g_ok": 0,
            "status": "SETUP_ERR",
            "final_mass_kg": np.nan,
            "runtime_s": float(time.time() - t0),
        }

    try:
        with suppress_stdout():
            res = solve_optimal_trajectory(
                cfg,
                veh_i,
                env_i,
                print_level=0,
                initial_guess_override=guess_i
            )
        solver_ok = bool(res.get("success", False))
        X3 = res.get("X3", None)
        if solver_ok and isinstance(X3, np.ndarray) and X3.ndim == 2 and X3.shape[0] >= 7 and X3.shape[1] >= 1:
            m_try = float(X3[6, -1])
            m_final = m_try if np.isfinite(m_try) and m_try > 0.0 else np.nan

        if solver_ok and np.isfinite(m_final):
            try:
                with suppress_stdout():
                    sim_res = run_simulation(res, veh_i, cfg, rtol=1e-9, atol=1e-12)
                terminal = evaluate_terminal_state(sim_res, cfg, env_cfg)
                traj = _ms_trajectory_diagnostics(sim_res, veh_i, cfg)
                path = _ms_evaluate_path_compliance(traj, cfg, q_slack_pa, g_slack)
                terminal_strict_ok = bool(terminal.get("strict_ok", False))
                path_ok = bool(path.get("path_ok", False))
                q_ok = bool(path.get("q_ok", False))
                g_ok = bool(path.get("g_ok", False))
                mission_ok = bool(terminal_strict_ok and path_ok)
                if mission_ok:
                    status = "MISSION_OK"
                else:
                    if terminal_strict_ok and not path_ok:
                        status = "PATH"
                    elif not terminal_strict_ok:
                        status = terminal.get("status", "MISS")
                    else:
                        status = "MISSION_FAIL"
            except Exception:
                status = "SIM_ERROR"
        elif solver_ok:
            solver_ok = False
            status = "BAD_SOL"
        else:
            status = "SOLVE_FAIL"
    except Exception:
        status = "SOLVE_ERR"

    return {
        "trial_index": idx,
        "label": label,
        "params": params,
        "solver_success": int(solver_ok),
        "mission_success": int(mission_ok),
        "terminal_strict_success": int(terminal_strict_ok),
        "path_ok": int(path_ok),
        "q_ok": int(q_ok),
        "g_ok": int(g_ok),
        "status": status,
        "final_mass_kg": float(m_final) if np.isfinite(m_final) else np.nan,
        "runtime_s": float(time.time() - t0),
    }


def _run_monte_carlo_batch_worker(payload):
    """
    Evaluate a batch of Monte Carlo uncertainty samples in one worker process.
    Returns aggregate counts needed by Wilson-CI stopping.
    """
    opt_res = payload["opt_res"]
    base_cfg = payload["base_cfg"]
    base_env_cfg = payload["base_env_cfg"]
    q_slack_pa = float(payload["q_slack_pa"])
    g_slack = float(payload["g_slack"])
    samples = payload["samples"]

    successes = 0
    path_failures = 0
    orbit_errors = []
    mission_flags = []

    for thrust_mult, isp_mult, dens_mult in samples:
        cfg = copy.deepcopy(base_cfg)
        env_cfg = copy.deepcopy(base_env_cfg)
        cfg.stage_1.thrust_vac *= float(thrust_mult)
        cfg.stage_2.thrust_vac *= float(thrust_mult)
        cfg.stage_1.isp_vac *= float(isp_mult)
        cfg.stage_2.isp_vac *= float(isp_mult)
        env_cfg.density_multiplier = float(dens_mult)

        try:
            with suppress_stdout():
                env_mc = Environment(env_cfg)
                veh_mc = Vehicle(cfg, env_mc)
                sim_res = run_simulation(opt_res, veh_mc, cfg)
            terminal = evaluate_terminal_state(sim_res, cfg, env_cfg)
            traj = _ms_trajectory_diagnostics(sim_res, veh_mc, cfg)
            path = _ms_evaluate_path_compliance(traj, cfg, q_slack_pa, g_slack)
            strict_path_ok = bool(terminal.get("strict_ok", False) and path.get("path_ok", False))
            orbit_err = normalized_orbit_error(terminal)
            if strict_path_ok:
                successes += 1
            if bool(terminal.get("strict_ok", False)) and (not strict_path_ok):
                path_failures += 1
            orbit_errors.append(float(orbit_err) if np.isfinite(orbit_err) else np.nan)
            mission_flags.append(int(strict_path_ok))
        except Exception:
            # Treat simulation failures as non-success samples.
            orbit_errors.append(np.nan)
            mission_flags.append(0)
            continue

    return {
        "successes": int(successes),
        "path_failures": int(path_failures),
        "orbit_errors": orbit_errors,
        "mission_flags": mission_flags,
    }


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


def _run_collocation_phase_defects_worker(payload):
    """
    Compute collocation defects for one phase.
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

    pos_def = np.zeros(N_loc)
    vel_def = np.zeros(N_loc)
    mass_def = np.zeros(N_loc)
    rel_state_max = np.zeros(N_loc)

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

        def dyn(x, t):
            return veh_i.get_dynamics(x, throttle, u_dir, t, stage_mode=stage_mode, scaling=None)

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

    return {
        "phase_name": phase_name,
        "position_defect": pos_def,
        "velocity_defect": vel_def,
        "mass_defect": mass_def,
        "relative_defect": rel_state_max,
    }


def _run_bifurcation_batch_worker(payload):
    """
    Evaluate a chunk of (thrust, density) bifurcation points.
    """
    opt_res = payload["opt_res"]
    base_cfg = payload["base_cfg"]
    base_env_cfg = payload["base_env_cfg"]
    q_slack_pa = float(payload["q_slack_pa"])
    g_slack = float(payload["g_slack"])
    samples = payload["samples"]

    rows = []
    for iy, ix, thrust, dens in samples:
        cfg = copy.deepcopy(base_cfg)
        env_cfg = copy.deepcopy(base_env_cfg)
        cfg.stage_1.thrust_vac *= float(thrust)
        cfg.stage_2.thrust_vac *= float(thrust)
        env_cfg.density_multiplier = float(dens)

        try:
            with suppress_stdout():
                env_i = Environment(env_cfg)
                veh_i = Vehicle(cfg, env_i)
                sim_res = run_simulation(opt_res, veh_i, cfg)
            terminal = evaluate_terminal_state(sim_res, cfg, env_cfg)
            traj = _ms_trajectory_diagnostics(sim_res, veh_i, cfg)
            path = _ms_evaluate_path_compliance(traj, cfg, q_slack_pa, g_slack)
            strict_path_ok = bool(terminal.get("strict_ok", False) and path.get("path_ok", False))
            margin = terminal["fuel_margin_kg"] if terminal["terminal_valid"] else np.nan
            orbit_err = normalized_orbit_error(terminal)
            status = terminal.get("status", "SIM_ERROR")
            if bool(terminal.get("strict_ok", False)) and not strict_path_ok:
                status = "PATH"
            rows.append(
                {
                    "iy": int(iy),
                    "ix": int(ix),
                    "thrust": float(thrust),
                    "density": float(dens),
                    "strict_success": int(strict_path_ok),
                    "fuel_margin_kg": float(margin) if np.isfinite(margin) else np.nan,
                    "normalized_orbit_error": float(orbit_err) if np.isfinite(orbit_err) else np.nan,
                    "altitude_error_m": float(terminal.get("alt_err_m", np.nan)),
                    "velocity_error_m_s": float(terminal.get("vel_err_m_s", np.nan)),
                    "q_ok": int(path.get("q_ok", False)),
                    "g_ok": int(path.get("g_ok", False)),
                    "status": str(status),
                }
            )
        except Exception:
            rows.append(
                {
                    "iy": int(iy),
                    "ix": int(ix),
                    "thrust": float(thrust),
                    "density": float(dens),
                    "strict_success": 0,
                    "fuel_margin_kg": np.nan,
                    "normalized_orbit_error": np.nan,
                    "altitude_error_m": np.nan,
                    "velocity_error_m_s": np.nan,
                    "q_ok": 0,
                    "g_ok": 0,
                    "status": "SIM_ERROR",
                }
            )
    return {"rows": rows}


def _run_q2_grid_point_worker(payload):
    """
    Q2 uncertainty budget: solve one node-count case.
    """
    node_count = int(payload["num_nodes"])
    base_cfg = payload["base_cfg"]
    base_env_cfg = payload["base_env_cfg"]

    cfg = copy.deepcopy(base_cfg)
    cfg.num_nodes = node_count

    t0 = time.time()
    ok = False
    fuel_i = np.nan
    status = "Failed"
    try:
        with suppress_stdout():
            env_i = Environment(copy.deepcopy(base_env_cfg))
            veh_i = Vehicle(cfg, env_i)
            opt_i = solve_optimal_trajectory(cfg, veh_i, env_i, print_level=0)
        ok = bool(opt_i.get("success", False))
        if ok:
            X3 = opt_i.get("X3", None)
            if isinstance(X3, np.ndarray) and X3.ndim == 2 and X3.shape[0] >= 7 and X3.shape[1] >= 1:
                m_final_i = float(X3[6, -1])
            else:
                m_final_i = np.nan
            if np.isfinite(m_final_i) and m_final_i > 0.0:
                fuel_i = float(cfg.launch_mass - m_final_i)
            else:
                fuel_i = np.nan
            status = "Converged" if np.isfinite(fuel_i) else "Failed"
    except Exception:
        ok = False
        fuel_i = np.nan
        status = "Error"

    return {
        "nodes": node_count,
        "solver_success": int(ok),
        "status": status,
        "minimum_fuel_kg": float(fuel_i) if np.isfinite(fuel_i) else np.nan,
        "runtime_s": float(time.time() - t0),
    }


def _run_q2_integrator_point_worker(payload):
    """
    Q2 uncertainty budget: evaluate one integrator-tolerance verification run.
    """
    opt_nom = payload["opt_nom"]
    base_cfg = payload["base_cfg"]
    base_env_cfg = payload["base_env_cfg"]
    rtol = float(payload["rtol"])
    atol = float(payload["atol"])
    q_slack_pa = float(payload["q_slack_pa"])
    g_slack = float(payload["g_slack"])

    cfg = copy.deepcopy(base_cfg)
    env_cfg = copy.deepcopy(base_env_cfg)

    term_i = {"terminal_valid": False, "strict_ok": False, "status": "SIM_ERROR"}
    path_i = {"q_ok": False, "g_ok": False, "path_ok": False}
    strict_path_i = 0
    fuel_sim = np.nan
    status = "SIM_ERROR"
    try:
        with suppress_stdout():
            env_i = Environment(env_cfg)
            veh_i = Vehicle(cfg, env_i)
            sim_i = run_simulation(opt_nom, veh_i, cfg, rtol=rtol, atol=atol)
        term_i = evaluate_terminal_state(sim_i, cfg, env_cfg)
        traj_i = _ms_trajectory_diagnostics(sim_i, veh_i, cfg)
        path_i = _ms_evaluate_path_compliance(traj_i, cfg, q_slack_pa, g_slack)
        strict_path_i = int(bool(term_i.get("strict_ok", False)) and bool(path_i.get("path_ok", False)))
        y = sim_i.get("y", None)
        if isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[0] >= 7 and y.shape[1] >= 1:
            m_final_sim = float(y[6, -1])
            if np.isfinite(m_final_sim) and m_final_sim > 0.0:
                fuel_sim = float(cfg.launch_mass - m_final_sim)
        status = term_i.get("status", "SIM_ERROR")
        if bool(term_i.get("strict_ok", False)) and not bool(strict_path_i):
            status = "PATH"
    except Exception:
        status = "SIM_ERROR"

    return {
        "rtol": rtol,
        "atol": atol,
        "terminal_valid": int(term_i.get("terminal_valid", False)),
        "strict_ok": int(term_i.get("strict_ok", False)),
        "strict_path_ok": int(strict_path_i),
        "q_ok": int(path_i.get("q_ok", False)),
        "g_ok": int(path_i.get("g_ok", False)),
        "status": status,
        "fuel_used_kg": float(fuel_sim) if np.isfinite(fuel_sim) else np.nan,
    }


def _run_q2_parameter_sample_worker(payload):
    """
    Q2 uncertainty budget: one parameter-perturbed re-optimization sample.
    """
    run_index = int(payload["run_index"])
    thrust_mult = float(payload["thrust_multiplier"])
    isp_mult = float(payload["isp_multiplier"])
    dens_mult = float(payload["density_multiplier"])
    base_cfg = payload["base_cfg"]
    base_env_cfg = payload["base_env_cfg"]
    q_slack_pa = float(payload["q_slack_pa"])
    g_slack = float(payload["g_slack"])

    cfg_i = copy.deepcopy(base_cfg)
    env_cfg_i = copy.deepcopy(base_env_cfg)
    cfg_i.stage_1.thrust_vac *= thrust_mult
    cfg_i.stage_2.thrust_vac *= thrust_mult
    cfg_i.stage_1.isp_vac *= isp_mult
    cfg_i.stage_2.isp_vac *= isp_mult
    env_cfg_i.density_multiplier = dens_mult

    t0 = time.time()
    opt_success = False
    strict_ok = False
    strict_path_ok = False
    fuel_i = np.nan
    status = "OPT_FAIL"
    path_i = {"q_ok": False, "g_ok": False, "path_ok": False}

    try:
        with suppress_stdout():
            env_i = Environment(env_cfg_i)
            veh_i = Vehicle(cfg_i, env_i)
            opt_i = solve_optimal_trajectory(cfg_i, veh_i, env_i, print_level=0)
        opt_success = bool(opt_i.get("success", False))
        if opt_success:
            X3 = opt_i.get("X3", None)
            m_final_i = float(X3[6, -1]) if isinstance(X3, np.ndarray) and X3.ndim == 2 and X3.shape[0] >= 7 and X3.shape[1] >= 1 else np.nan
            if np.isfinite(m_final_i) and m_final_i > 0.0:
                fuel_i = float(cfg_i.launch_mass - m_final_i)
            try:
                with suppress_stdout():
                    sim_i = run_simulation(opt_i, veh_i, cfg_i, rtol=1e-9, atol=1e-12)
                term_i = evaluate_terminal_state(sim_i, cfg_i, env_cfg_i)
                traj_i = _ms_trajectory_diagnostics(sim_i, veh_i, cfg_i)
                path_i = _ms_evaluate_path_compliance(traj_i, cfg_i, q_slack_pa, g_slack)
                strict_ok = bool(term_i.get("strict_ok", False))
                strict_path_ok = bool(strict_ok and path_i.get("path_ok", False))
                status = term_i.get("status", "SIM_ERROR")
                if strict_ok and not strict_path_ok:
                    status = "PATH"
            except Exception:
                status = "ERROR"
    except Exception:
        status = "ERROR"
        path_i = {"q_ok": False, "g_ok": False, "path_ok": False}

    return {
        "run": run_index + 1,
        "thrust_multiplier": thrust_mult,
        "isp_multiplier": isp_mult,
        "density_multiplier": dens_mult,
        "optimizer_success": int(opt_success),
        "strict_ok": int(strict_ok),
        "strict_path_ok": int(strict_path_ok),
        "q_ok": int(path_i["q_ok"]),
        "g_ok": int(path_i["g_ok"]),
        "status": status,
        "minimum_fuel_kg": float(fuel_i) if np.isfinite(fuel_i) else np.nan,
        "runtime_s": float(time.time() - t0),
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
            path = self._evaluate_path_compliance(traj, cfg)
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
            path = self._evaluate_path_compliance(traj, cfg)
        strict_path_ok = self._is_strict_path_success(terminal, traj, cfg)

        return {
            "thrust_multiplier": float(thrust_mult),
            "isp_multiplier": float(isp_mult),
            "density_multiplier": float(dens_mult),
            "terminal": terminal,
            "trajectory": traj,
            "path": path,
            "strict_path_ok": int(strict_path_ok)
        }

    @staticmethod
    def _extract_final_mass_from_opt(opt_res):
        if not isinstance(opt_res, dict):
            return np.nan
        X3 = opt_res.get("X3", None)
        if not isinstance(X3, np.ndarray) or X3.ndim != 2 or X3.shape[0] < 7 or X3.shape[1] < 1:
            return np.nan
        m_final = float(X3[6, -1])
        if not np.isfinite(m_final) or m_final <= 0.0:
            return np.nan
        return m_final

    @staticmethod
    def _extract_final_mass_from_sim(sim_res):
        if not isinstance(sim_res, dict):
            return np.nan
        y = sim_res.get("y", None)
        if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[0] < 7 or y.shape[1] < 1:
            return np.nan
        m_final = float(y[6, -1])
        if not np.isfinite(m_final) or m_final <= 0.0:
            return np.nan
        return m_final

    @staticmethod
    def _rss(values):
        arr = np.array([float(v) for v in values if np.isfinite(v)], dtype=float)
        if arr.size == 0:
            return np.nan
        return float(np.sqrt(np.sum(arr * arr)))

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

    def _get_saved_metric(self, table_name, key):
        """
        Best-effort helper for cross-test summaries.
        Reads `metric,value` style CSVs if they exist.
        """
        table_path = self.data_dir / f"{table_name}.csv"
        if not table_path.exists():
            return np.nan
        try:
            with open(table_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if str(row.get("metric", "")).strip() == str(key):
                        val = row.get("value", "nan")
                        try:
                            return float(val)
                        except Exception:
                            return np.nan
        except Exception:
            return np.nan
        return np.nan

    @staticmethod
    def _normalize_direction_history(td):
        arr = np.array(td, dtype=float, copy=True)
        if arr.ndim != 2 or arr.shape[0] != 3:
            return arr
        norms = np.linalg.norm(arr, axis=0)
        good = norms > 1e-12
        if np.any(good):
            arr[:, good] = arr[:, good] / norms[good]
        if np.any(~good):
            arr[:, ~good] = np.array([[1.0], [0.0], [0.0]])
        return arr

    @staticmethod
    def _perturb_state_history(X, rng, pos_sigma, vel_sigma, mass_sigma):
        arr = np.array(X, dtype=float, copy=True)
        if arr.ndim != 2 or arr.shape[0] < 7 or arr.shape[1] < 1:
            return arr

        def _apply_rowwise_relative_jitter(row_slice, sigma):
            # Relative jitter for non-zero entries + additive fallback for near-zero entries.
            # This avoids "zero stays zero" degeneracy while keeping perturbations scale-aware.
            for j in range(row_slice.start, row_slice.stop):
                row = np.array(arr[j, :], dtype=float, copy=True)
                noise = rng.uniform(-sigma, sigma, size=row.shape)
                pert = row * (1.0 + noise)
                near_zero = np.abs(row) <= 1e-12
                if np.any(near_zero):
                    ref_scale = float(np.nanmedian(np.abs(row)))
                    if not np.isfinite(ref_scale) or ref_scale <= 0.0:
                        ref_scale = float(np.nanmax(np.abs(row)))
                    if not np.isfinite(ref_scale) or ref_scale <= 0.0:
                        ref_scale = 1.0
                    pert[near_zero] = row[near_zero] + noise[near_zero] * ref_scale
                arr[j, :] = pert

        _apply_rowwise_relative_jitter(slice(0, 3), pos_sigma)
        _apply_rowwise_relative_jitter(slice(3, 6), vel_sigma)
        _apply_rowwise_relative_jitter(slice(6, 7), mass_sigma)

        # Preserve phase start node to stay near boundary/linkage constraints.
        arr[:, 0] = X[:, 0]
        # Keep phase mass physically plausible and monotone decreasing.
        arr[6, :] = np.minimum.accumulate(arr[6, :])
        arr[6, :] = np.clip(arr[6, :], 1.0e3, None)
        return arr

    def _build_randomized_initial_guess(self, base_guess, cfg, rng):
        if not isinstance(base_guess, dict):
            raise ValueError("Initial guess must be a dict.")

        guess = copy.deepcopy(base_guess)

        # Uniform +/-1% perturbation regime.
        frac = 0.01
        time_scale = float(1.0 + rng.uniform(-frac, frac))
        pos_sigma = float(frac)
        vel_sigma = float(frac)
        mass_sigma = float(frac)
        throttle_sigma = float(frac)
        direction_sigma = float(frac)
        direction_sigma_deg = float(np.degrees(np.arcsin(direction_sigma)))

        # Time variables: perturb around guidance and clamp to physically valid minima.
        t1 = float(guess.get("T1", cfg.sequence.min_stage_1_burn))
        t3 = float(guess.get("T3", cfg.sequence.min_stage_2_burn))
        guess["T1"] = max(cfg.sequence.min_stage_1_burn, t1 * time_scale)
        guess["T3"] = max(cfg.sequence.min_stage_2_burn, t3 * time_scale)
        if guess.get("T2", None) is not None:
            guess["T2"] = max(0.0, float(guess["T2"]) * (1.0 + float(rng.uniform(-frac, frac))))

        # State trajectories.
        for key in ("X1", "X2", "X3"):
            val = guess.get(key, None)
            if val is not None:
                guess[key] = self._perturb_state_history(val, rng, pos_sigma, vel_sigma, mass_sigma)

        # Throttle histories.
        for key in ("TH1", "TH3"):
            val = guess.get(key, None)
            if val is None:
                continue
            th = np.array(val, dtype=float, copy=True)
            th *= 1.0 + rng.uniform(-throttle_sigma, throttle_sigma, size=th.shape)
            th = np.clip(th, cfg.sequence.min_throttle, 1.0)
            guess[key] = th

        # Direction histories.
        for key in ("TD1", "TD3"):
            val = guess.get(key, None)
            if val is None:
                continue
            td = np.array(val, dtype=float, copy=True)
            td += rng.normal(0.0, direction_sigma, size=td.shape)
            guess[key] = self._normalize_direction_history(td)

        params = {
            "time_scale": time_scale,
            "pos_sigma": pos_sigma,
            "vel_sigma": vel_sigma,
            "mass_sigma": mass_sigma,
            "throttle_sigma": throttle_sigma,
            "direction_sigma_deg": direction_sigma_deg,
        }
        return guess, params

    def run_all(self, monte_carlo_samples=200, max_workers=None):
        """Runs enabled analyses from shortest expected runtime to longest."""
        self._analysis_execution = []
        ran_any = False
        mc_n = max(1, int(monte_carlo_samples))
        mc_max_precision = max(400, mc_n, int(np.ceil(1.30 * mc_n)))
        # Keep a conservative minimum before CI-based stopping to avoid over-optimistic early stops.
        mc_min_precision = min(mc_max_precision, max(80, mc_n // 4))
        mc_batch = max(5, min(50, max(1, mc_n // 10)))
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
            "collocation_defect_audit",
            "bifurcation_2d_map",
            "monte_carlo_precision_target",
            "q2_uncertainty_budget",
        )
        try:
            if any(bool(getattr(self.test_toggles, name, True)) for name in baseline_flags):
                print(f"\n{debug.Style.BOLD}--- Precomputing Shared Baseline Solution ---{debug.Style.RESET}")
                self._get_baseline_opt_res()

            # Run grid independence first so the main discretization check is available early.
            ordered_tests = [
                ("grid_independence", "Grid independence", self.analyze_grid_independence, (), {"max_workers": worker_cap}),
                ("model_limitations", "Model limitations", self.analyze_model_limitations, (), {}),
                ("smooth_integrator_benchmark", "Smooth ODE integrator benchmark", self.analyze_smooth_integrator_benchmark, (), {}),
                ("drift", "Drift analysis", _run_drift_with_cached_baseline, (), {}),
                ("theoretical_efficiency", "Theoretical efficiency", self.analyze_theoretical_efficiency, (), {}),
                ("integrator_tolerance", "Integrator tolerance", self.analyze_integrator_tolerance, (), {}),
                ("collocation_defect_audit", "Collocation defect audit", self.analyze_collocation_defect_audit, (), {"max_workers": worker_cap}),
                ("bifurcation_2d_map", "Bifurcation 2D map", self.analyze_bifurcation_2d_map, (), {"max_workers": worker_cap}),
                ("randomized_multistart", "Randomized multi-start robustness", self.analyze_randomized_multistart, (), {"max_workers": worker_cap}),
                (
                    "monte_carlo_precision_target",
                    "Monte Carlo precision target",
                    self.analyze_monte_carlo_precision_target,
                    (),
                    {"min_samples": mc_min_precision, "max_samples": mc_max_precision, "batch_size": mc_batch, "max_workers": worker_cap},
                ),
                (
                    "q2_uncertainty_budget",
                    "Q2 uncertainty budget",
                    self.analyze_q2_uncertainty_budget,
                    (),
                    {"parameter_samples": max(30, min(60, mc_n // 6)), "max_workers": worker_cap},
                ),
            ]
            total_steps = len(ordered_tests) + 1  # + synthesis step
            for i, (flag, label, fn, args, kwargs) in enumerate(ordered_tests, start=1):
                progress = f"{i}/{total_steps}"
                ran_any |= self._run_if_enabled(flag, label, fn, *args, progress=progress, **kwargs)

            # Keep synthesis last because it consumes outputs from earlier analyses.
            ran_any |= self._run_if_enabled(
                "q7_conclusion_support",
                "Q7 conclusion support",
                self.analyze_q7_conclusion_support,
                progress=f"{total_steps}/{total_steps}",
            )

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
            f"{'RTOL':<10} | {'ATOL':<10} | {'Alt err (m)':<12} | {'Vel err (m/s)':<14} | "
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
                terminal.get("r_f_m", np.nan),
                terminal.get("v_f_m_s", np.nan),
                terminal.get("alt_err_m", np.nan),
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
                f"{rtol:<10.1e} | {atol:<10.1e} | {terminal.get('alt_err_m', np.nan):<12.2f} | "
                f"{terminal.get('vel_err_m_s', np.nan):<14.3f} | {path['max_q_pa']/1000.0:<12.3f} | "
                f"{path['max_g']:<8.3f} | {strict_path_ok:<6d} | {status:<10}"
            )

        self._save_table(
            "integrator_tolerance",
            [
                "rtol", "atol",
                "final_altitude_m", "final_velocity_m_s",
                "altitude_error_m", "velocity_error_m_s", "radial_velocity_m_s",
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

    def analyze_randomized_multistart(self, n_trials=8, max_workers=None):
        debug._print_sub_header("Randomized Multi-Start Robustness")
        n_use = max(4, int(n_trials))
        rng = np.random.default_rng(self.seed_sequence.spawn(1)[0])
        if max_workers is None:
            max_workers = min(4, max(1, os.cpu_count() or 1))
        max_workers = max(1, int(max_workers))

        rows = []
        solver_success_flags = []
        mission_success_flags = []
        masses_scored = []
        runtimes = []
        labels = []
        trial_results = {}

        print(
            f"{'Trial':<7} | {'Guess':<8} | {'Time x':<8} | {'Pos %':<7} | {'Vel %':<7} | "
            f"{'Mass %':<8} | {'Throt %':<8} | {'Dir (deg)':<10} | {'Solve':<5} | {'Mission':<7} | {'Status':<12} | "
            f"{'Final Mass (kg)':<15} | {'Runtime (s)':<10}"
        )
        print("-" * 166)

        # Guidance warm start is deterministic for the nominal configuration; compute once.
        guidance_guess = None
        try:
            cfg_g = copy.deepcopy(self.base_config)
            env_cfg_g = copy.deepcopy(self.base_env_config)
            env_g = Environment(env_cfg_g)
            veh_g = Vehicle(cfg_g, env_g)
            with suppress_stdout():
                guidance_guess = guidance.get_initial_guess(cfg_g, veh_g, env_g, num_nodes=cfg_g.num_nodes)
        except Exception:
            guidance_guess = None

        trial_jobs = []
        for i in range(n_use):
            cfg = copy.deepcopy(self.base_config)
            env_cfg = copy.deepcopy(self.base_env_config)

            if i == 0:
                # Always include nominal guidance warm start as deterministic anchor.
                label = "nominal"
                params = {
                    "time_scale": 1.0,
                    "pos_sigma": 0.0,
                    "vel_sigma": 0.0,
                    "mass_sigma": 0.0,
                    "throttle_sigma": 0.0,
                    "direction_sigma_deg": 0.0,
                }
            else:
                label = f"rand_{i:02d}"
                params = {
                    "time_scale": np.nan,
                    "pos_sigma": np.nan,
                    "vel_sigma": np.nan,
                    "mass_sigma": np.nan,
                    "throttle_sigma": np.nan,
                    "direction_sigma_deg": np.nan,
                }

            if guidance_guess is None:
                trial_results[i] = {
                    "trial_index": i,
                    "label": label,
                    "params": params,
                    "solver_success": 0,
                    "mission_success": 0,
                    "terminal_strict_success": 0,
                    "path_ok": 0,
                    "q_ok": 0,
                    "g_ok": 0,
                    "status": "GUESS_FAIL",
                    "final_mass_kg": np.nan,
                    "runtime_s": 0.0,
                }
                continue

            try:
                if i == 0:
                    guess_i = copy.deepcopy(guidance_guess)
                else:
                    guess_i, params = self._build_randomized_initial_guess(guidance_guess, cfg, rng)
            except Exception:
                trial_results[i] = {
                    "trial_index": i,
                    "label": label,
                    "params": params,
                    "solver_success": 0,
                    "mission_success": 0,
                    "terminal_strict_success": 0,
                    "path_ok": 0,
                    "q_ok": 0,
                    "g_ok": 0,
                    "status": "PERTURB",
                    "final_mass_kg": np.nan,
                    "runtime_s": 0.0,
                }
                continue

            trial_jobs.append(
                {
                    "trial_index": i,
                    "label": label,
                    "params": params,
                    "cfg": cfg,
                    "env_cfg": env_cfg,
                    "guess": guess_i,
                    "q_slack_pa": float(self.path_q_slack_pa),
                    "g_slack": float(self.path_g_slack),
                }
            )

        parallel_active = bool(max_workers > 1 and len(trial_jobs) > 1)
        if parallel_active:
            print(f"  Parallel execution enabled (workers={max_workers}).")
            try:
                failed_trial_payloads = []
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    idx_to_payload = {int(payload["trial_index"]): payload for payload in trial_jobs}
                    fut_to_idx = {
                        executor.submit(_run_multistart_trial_worker, payload): int(payload["trial_index"])
                        for payload in trial_jobs
                    }
                    total_trials = len(fut_to_idx)
                    done_trials = 0
                    for fut in as_completed(fut_to_idx):
                        done_trials += 1
                        self._print_parallel_progress("Multistart trials", done_trials, total_trials)
                        idx = fut_to_idx[fut]
                        payload = idx_to_payload[idx]
                        try:
                            trial_results[idx] = fut.result()
                        except Exception:
                            failed_trial_payloads.append(payload)
                if len(failed_trial_payloads) > 0:
                    print(f"  Retrying {len(failed_trial_payloads)} failed worker trial(s) in serial mode.")
                    retry_total = len(failed_trial_payloads)
                    for retry_i, payload in enumerate(failed_trial_payloads, start=1):
                        idx = int(payload["trial_index"])
                        try:
                            trial_results[idx] = _run_multistart_trial_worker(payload)
                        except Exception:
                            trial_results[idx] = {
                                "trial_index": idx,
                                "label": payload["label"],
                                "params": payload["params"],
                                "solver_success": 0,
                                "mission_success": 0,
                                "terminal_strict_success": 0,
                                "path_ok": 0,
                                "q_ok": 0,
                                "g_ok": 0,
                                "status": "WORKER_ERR",
                                "final_mass_kg": np.nan,
                                "runtime_s": np.nan,
                            }
                        self._print_parallel_progress("Multistart retry trials", retry_i, retry_total)
            except Exception:
                print("  Parallel execution unavailable; falling back to serial trial evaluation.")
                parallel_active = False

        if not parallel_active:
            print("  Serial execution mode.")
            for payload in trial_jobs:
                idx = int(payload["trial_index"])
                try:
                    trial_results[idx] = _run_multistart_trial_worker(payload)
                except Exception:
                    trial_results[idx] = {
                        "trial_index": idx,
                        "label": payload["label"],
                        "params": payload["params"],
                        "solver_success": 0,
                        "mission_success": 0,
                        "terminal_strict_success": 0,
                        "path_ok": 0,
                        "q_ok": 0,
                        "g_ok": 0,
                        "status": "WORKER_ERR",
                        "final_mass_kg": np.nan,
                        "runtime_s": np.nan,
                    }

        for i in range(n_use):
            r = trial_results.get(i, None)
            if r is None:
                # Defensive fallback to preserve table shape on unexpected edge cases.
                r = {
                    "trial_index": i,
                    "label": f"rand_{i:02d}" if i > 0 else "nominal",
                    "params": {
                        "time_scale": np.nan,
                        "pos_sigma": np.nan,
                        "vel_sigma": np.nan,
                        "mass_sigma": np.nan,
                        "throttle_sigma": np.nan,
                        "direction_sigma_deg": np.nan,
                    },
                    "solver_success": 0,
                    "mission_success": 0,
                    "terminal_strict_success": 0,
                    "path_ok": 0,
                    "q_ok": 0,
                    "g_ok": 0,
                    "status": "INTERNAL_ERR",
                    "final_mass_kg": np.nan,
                    "runtime_s": np.nan,
                }

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

            labels.append(label)
            masses_scored.append(m_final if mission_ok and np.isfinite(m_final) else np.nan)
            runtimes.append(dt)
            solver_success_flags.append(solver_ok)
            mission_success_flags.append(mission_ok)

            rows.append({
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
            })
            print(
                f"{i+1:<7} | {label:<8} | {params['time_scale']:<8.3f} | "
                f"{100.0 * params['pos_sigma']:<7.2f} | {100.0 * params['vel_sigma']:<7.2f} | "
                f"{100.0 * params['mass_sigma']:<8.2f} | {100.0 * params['throttle_sigma']:<8.2f} | "
                f"{params['direction_sigma_deg']:<10.2f} | {solver_ok:<5d} | {mission_ok:<7d} | "
                f"{status:<12} | {m_final:<15.2f} | {dt:<10.2f}"
            )

        solver_rate = float(np.mean(solver_success_flags)) if len(solver_success_flags) > 0 else 0.0
        success_rate = float(np.mean(mission_success_flags)) if len(mission_success_flags) > 0 else 0.0
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

        self._save_table(
            "randomized_multistart",
            [
                "trial", "label", "time_scale", "position_noise_sigma", "velocity_noise_sigma",
                "mass_noise_sigma", "throttle_noise_sigma", "direction_noise_sigma_deg",
                "solver_success", "mission_success", "terminal_strict_success",
                "path_ok", "q_ok", "g_ok", "status",
                "final_mass_kg", "final_mass_scored_kg", "final_mass_gap_to_best_kg", "runtime_s"
            ],
            [
                (
                    rows[i]["trial"], rows[i]["label"], rows[i]["time_scale"],
                    rows[i]["position_noise_sigma"], rows[i]["velocity_noise_sigma"],
                    rows[i]["mass_noise_sigma"], rows[i]["throttle_noise_sigma"],
                    rows[i]["direction_noise_sigma_deg"], rows[i]["solver_success"],
                    rows[i]["mission_success"], rows[i]["terminal_strict_success"],
                    rows[i]["path_ok"], rows[i]["q_ok"], rows[i]["g_ok"], rows[i]["status"],
                    rows[i]["final_mass_kg"], masses_scored[i], mass_gap_kg[i], rows[i]["runtime_s"]
                )
                for i in range(len(rows))
            ]
        )

        x = np.arange(n_use)
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [0.9, 1.2, 1.0]}
        )

        ax1.bar(x, mission_success_flags, color=["tab:green" if s else "tab:red" for s in mission_success_flags], alpha=0.7)
        # Explicit markers make failed trials visible even when bar height is zero.
        success_idx_top = [i for i, s in enumerate(mission_success_flags) if s]
        fail_idx_top = [i for i, s in enumerate(mission_success_flags) if not s]
        if len(success_idx_top) > 0:
            ax1.scatter(success_idx_top, [1.0] * len(success_idx_top), s=20, c="tab:green", marker="o", label="Mission OK")
        if len(fail_idx_top) > 0:
            ax1.scatter(fail_idx_top, [0.03] * len(fail_idx_top), s=40, c="tab:red", marker="x", label="Failed")
        ax1.set_ylim(-0.05, 1.15)
        ax1.set_ylabel("Mission Success (0/1)")
        ax1.set_title("Randomized Multi-Start Robustness")
        ax1.grid(True, axis="y", alpha=0.3)
        if len(success_idx_top) > 0 or len(fail_idx_top) > 0:
            ax1.legend(loc="upper right")

        success_idx = [i for i, m in enumerate(masses_scored) if np.isfinite(m)]
        if len(success_idx) > 0:
            ax2.plot(
                success_idx,
                [mass_gap_kg[i] for i in success_idx],
                "o-",
                color="tab:blue",
                label="Fuel gap to best"
            )
            ax2.axhline(0.0, color="tab:blue", linestyle="--", alpha=0.6, label="Best run")
        fail_idx = [i for i, ok in enumerate(mission_success_flags) if not ok]
        if len(fail_idx) > 0:
            ax2.scatter(fail_idx, [0.0] * len(fail_idx), marker="x", color="tab:red", label="Failed")
        ax2.set_ylabel("Gap to Best (kg)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best")

        ax3.bar(x, runtimes, color="tab:purple", alpha=0.55)
        ax3.set_ylabel("Runtime (s)")
        ax3.set_xlabel("Multi-start Trial")
        ax3.grid(True, axis="y", alpha=0.3)

        xticks = [i for i in range(n_use) if i == 0 or (i % 2 == 1)]
        ax3.set_xticks(xticks)
        ax3.set_xticklabels([labels[i] for i in xticks], rotation=25)

        summary = (
            f"Trials: {n_use}\n"
            f"Solver: {solver_rate:.1%}\n"
            f"Mission: {success_rate:.1%}\n"
            + (f"Mass spread: {mass_spread:.3f} kg" if np.isfinite(mass_spread) else "Mass spread: n/a")
        )
        ax2.text(
            0.02,
            0.95,
            summary,
            transform=ax2.transAxes,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.7"),
        )
        self._finalize_figure(fig, "randomized_multistart")

    def analyze_collocation_defect_audit(self, max_workers=None):
        debug._print_sub_header("Collocation Defect Audit")
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
                        executor.submit(_run_collocation_phase_defects_worker, payload): str(payload["phase_name"])
                        for payload in phase_jobs
                    }
                    parallel_ok = True
                    total_phases = len(fut_to_name)
                    done_phases = 0
                    for fut in as_completed(fut_to_name):
                        done_phases += 1
                        self._print_parallel_progress("Collocation phases", done_phases, total_phases)
                        key = fut_to_name[fut]
                        try:
                            phase_results[key] = fut.result()
                        except Exception:
                            parallel_ok = False
                            break
                if not parallel_ok:
                    print("  Worker failure during collocation audit; switching to serial mode.")
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
                phase_results[key] = _run_collocation_phase_defects_worker(payload)

        phase_series = []
        for payload in phase_jobs:
            key = str(payload["phase_name"])
            r = phase_results.get(key, None)
            if r is None:
                continue
            pos_s = np.asarray(r.get("position_defect", []), dtype=float)
            vel_s = np.asarray(r.get("velocity_defect", []), dtype=float)
            mass_s = np.asarray(r.get("mass_defect", []), dtype=float)
            rel_s = np.asarray(r.get("relative_defect", []), dtype=float)
            phase_series.append((key, pos_s, vel_s, mass_s, rel_s))

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
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        total_intervals = int(sum(len(s[1]) for s in phase_series))
        offset = 0
        for phase_name, pos_s, vel_s, m_s, rel_s in phase_series:
            idx = np.arange(offset, offset + len(pos_s))
            ax1.semilogy(idx, np.maximum(pos_s, eps), label=f"{phase_name.title()} position defect")
            ax2.semilogy(idx, np.maximum(rel_s, eps), label=f"{phase_name.title()} max relative defect")
            offset += len(pos_s)
            if offset < total_intervals:
                ax1.axvline(offset - 0.5, color="0.5", linestyle="--", linewidth=0.8, alpha=0.6)
                ax2.axvline(offset - 0.5, color="0.5", linestyle="--", linewidth=0.8, alpha=0.6)

        ax1.set_ylabel("Position Defect (m)")
        ax1.set_title("Collocation Defect Audit")
        ax1.legend(loc="best")
        ax1.grid(True, which="both", ls="-", alpha=0.3)

        ax2.axhline(1e-4, color="tab:orange", linestyle="--", alpha=0.8, label="Pass threshold (1e-4)")
        ax2.set_xlabel("Global Interval Index")
        ax2.set_ylabel("Max Relative Defect (-)")
        ax2.legend(loc="best")
        ax2.grid(True, which="both", ls="-", alpha=0.3)
        self._finalize_figure(fig, "collocation_defect_audit")


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

    # 17. MONTE CARLO PRECISION TARGET
    def analyze_monte_carlo_precision_target(
        self,
        target_half_width=0.04,
        target_relative_half_width=0.60,
        min_samples=80,
        max_samples=400,
        batch_size=20,
        max_workers=None,
    ):
        debug._print_sub_header("17. Monte Carlo Precision Target (CI-Based Stop)")
        print("Running Monte Carlo until Wilson 95% CI half-width reaches target precision.")

        if min_samples < 1 or max_samples < min_samples or batch_size < 1:
            print("  Invalid parameters for precision-target Monte Carlo. Skipping.")
            return

        opt_res = self._get_baseline_opt_res()
        rng = np.random.default_rng(self.seed_sequence.spawn(1)[0])
        if max_workers is None:
            max_workers = min(4, max(1, os.cpu_count() or 1))
        max_workers = max(1, int(max_workers))

        successes = 0
        path_failures = 0
        orbit_errors_all = []
        mission_flags_all = []
        n = 0
        batch_rows = []
        stop_reason = "max_samples"
        t0 = time.time()
        parallel_active = bool(max_workers > 1 and batch_size > 1)
        executor = None
        if parallel_active:
            try:
                executor = ProcessPoolExecutor(max_workers=max_workers)
                print(f"  Parallel execution enabled (workers={max_workers}).")
            except Exception:
                parallel_active = False
                executor = None
                print("  Parallel execution unavailable; using serial mode.")
        if not parallel_active:
            print("  Serial execution mode.")

        try:
            while n < max_samples:
                n_batch = min(batch_size, max_samples - n)
                draws = [
                    self._draw_uncertainty_multipliers(rng, model="gaussian_independent")
                    for _ in range(n_batch)
                ]
                batch_successes = 0
                batch_path_failures = 0

                if parallel_active and executor is not None:
                    n_chunks = min(max_workers, n_batch)
                    chunk_size = int(np.ceil(n_batch / max(n_chunks, 1)))
                    fut_to_chunk_size = {}
                    for i0 in range(0, n_batch, chunk_size):
                        chunk = draws[i0:i0 + chunk_size]
                        payload = {
                            "opt_res": opt_res,
                            "base_cfg": self.base_config,
                            "base_env_cfg": self.base_env_config,
                            "q_slack_pa": float(self.path_q_slack_pa),
                            "g_slack": float(self.path_g_slack),
                            "samples": chunk,
                        }
                        fut = executor.submit(_run_monte_carlo_batch_worker, payload)
                        fut_to_chunk_size[fut] = len(chunk)

                    parallel_ok = True
                    parallel_successes = 0
                    parallel_path_failures = 0
                    parallel_orbit_errors = []
                    parallel_mission_flags = []
                    completed_batch_samples = 0
                    for fut in as_completed(fut_to_chunk_size):
                        completed_batch_samples += int(fut_to_chunk_size[fut])
                        self._print_parallel_progress(
                            "Monte Carlo samples (toward max)",
                            n + completed_batch_samples,
                            max_samples,
                        )
                        try:
                            res = fut.result()
                            parallel_successes += int(res.get("successes", 0))
                            parallel_path_failures += int(res.get("path_failures", 0))
                            batch_orbit_errors = res.get("orbit_errors", [])
                            batch_mission_flags = res.get("mission_flags", [])
                            parallel_orbit_errors.extend([float(v) if np.isfinite(v) else np.nan for v in batch_orbit_errors])
                            parallel_mission_flags.extend([int(v) for v in batch_mission_flags])
                        except Exception:
                            parallel_ok = False
                            break

                    if not parallel_ok:
                        print("  Worker failure during Monte Carlo batch; switching to serial mode.")
                        parallel_active = False
                        if executor is not None:
                            executor.shutdown(wait=True)
                            executor = None
                        batch_successes = 0
                        batch_path_failures = 0
                        for thrust_mult, isp_mult, dens_mult in draws:
                            result = self._simulate_uncertainty_case(opt_res, thrust_mult, isp_mult, dens_mult)
                            batch_successes += int(result["strict_path_ok"])
                            batch_path_failures += int(
                                bool(result["terminal"].get("strict_ok", False)) and not bool(result["strict_path_ok"])
                            )
                            orbit_err = normalized_orbit_error(result["terminal"])
                            orbit_errors_all.append(float(orbit_err) if np.isfinite(orbit_err) else np.nan)
                            mission_flags_all.append(int(result["strict_path_ok"]))
                    else:
                        batch_successes = parallel_successes
                        batch_path_failures = parallel_path_failures
                        orbit_errors_all.extend(parallel_orbit_errors)
                        mission_flags_all.extend(parallel_mission_flags)
                else:
                    for thrust_mult, isp_mult, dens_mult in draws:
                        result = self._simulate_uncertainty_case(opt_res, thrust_mult, isp_mult, dens_mult)
                        batch_successes += int(result["strict_path_ok"])
                        batch_path_failures += int(
                            bool(result["terminal"].get("strict_ok", False)) and not bool(result["strict_path_ok"])
                        )
                        orbit_err = normalized_orbit_error(result["terminal"])
                        orbit_errors_all.append(float(orbit_err) if np.isfinite(orbit_err) else np.nan)
                        mission_flags_all.append(int(result["strict_path_ok"]))

                successes += batch_successes
                path_failures += batch_path_failures
                n += n_batch

                lo, hi = wilson_interval(successes, n, z=1.96)
                p = successes / n
                half_width = 0.5 * (hi - lo)
                rel_half_width = half_width / max(p, 1.0 / max(n, 1))
                finite_orbit = np.array([x for x in orbit_errors_all if np.isfinite(x)], dtype=float)
                if finite_orbit.size > 0:
                    orbit_p50 = float(np.percentile(finite_orbit, 50.0))
                    orbit_p90 = float(np.percentile(finite_orbit, 90.0))
                    orbit_p95 = float(np.percentile(finite_orbit, 95.0))
                    orbit_feasible_frac = float(np.mean(finite_orbit <= 1.0))
                else:
                    orbit_p50 = np.nan
                    orbit_p90 = np.nan
                    orbit_p95 = np.nan
                    orbit_feasible_frac = np.nan

                batch_rows.append(
                    (
                        n, p, lo, hi, half_width, rel_half_width,
                        orbit_p50, orbit_p90, orbit_p95, orbit_feasible_frac
                    )
                )
                print(
                    f"  N={n:4d} | Success={p*100:6.2f}% | CI=[{lo*100:5.2f}, {hi*100:5.2f}] | "
                    f"Half-width={half_width*100:5.2f}% | Relative={rel_half_width*100:5.1f}% | "
                    f"OrbitErr p90={orbit_p90:6.2f} | Feasible<=1: {orbit_feasible_frac*100 if np.isfinite(orbit_feasible_frac) else np.nan:5.1f}%"
                )

                abs_ok = half_width <= target_half_width
                rel_ok = True
                if target_relative_half_width is not None:
                    rel_ok = rel_half_width <= float(target_relative_half_width)
                if n >= min_samples and abs_ok and rel_ok:
                    stop_reason = "target_precision"
                    break
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        elapsed = time.time() - t0
        final_n, final_p, final_lo, final_hi, final_half, final_rel_half, final_orbit_p50, final_orbit_p90, final_orbit_p95, final_orbit_feas = batch_rows[-1]
        print(f"  Stop reason: {stop_reason}")
        print(
            f"  Final: N={final_n}, Success={final_p*100:.2f}%, "
            f"CI=[{final_lo*100:.2f}, {final_hi*100:.2f}], "
            f"Half-width={final_half*100:.2f}% ({final_rel_half*100:.1f}% relative)"
        )
        print(
            f"  Orbit-error quantiles: p50={final_orbit_p50:.3f}, "
            f"p90={final_orbit_p90:.3f}, p95={final_orbit_p95:.3f}, "
            f"feasible(error<=1)={final_orbit_feas*100 if np.isfinite(final_orbit_feas) else np.nan:.1f}%"
        )
        p_eff = float(np.clip(final_p, 1e-9, 1.0 - 1e-9))
        n_req_abs = int(np.ceil((1.96 ** 2) * p_eff * (1.0 - p_eff) / (target_half_width ** 2)))
        n_req_rel = np.nan
        if target_relative_half_width is not None:
            n_req_rel = int(np.ceil((1.96 ** 2) * (1.0 - p_eff) / ((target_relative_half_width ** 2) * p_eff)))
        n_req = max(min_samples, n_req_abs, int(n_req_rel) if np.isfinite(n_req_rel) else min_samples)
        print(f"  Path-violation reclassifications: {path_failures}")
        print(f"  Runtime: {elapsed:.1f}s")
        print(
            f"  Approx. required N from current success rate: "
            f"abs~{n_req_abs}, rel~{n_req_rel if np.isfinite(n_req_rel) else 'n/a'}, combined~{n_req}"
        )
        if stop_reason != "target_precision":
            print(
                f">>> {debug.Style.YELLOW}WARN: Precision targets not met before max samples "
                f"(N={max_samples}). Consider increasing max_samples toward ~{n_req}.{debug.Style.RESET}"
            )

        self._save_table(
            "monte_carlo_precision_target",
            [
                "n", "success_rate", "wilson_low_95", "wilson_high_95",
                "ci_half_width_95", "relative_half_width", "target_half_width",
                "target_relative_half_width",
                "orbit_error_p50", "orbit_error_p90", "orbit_error_p95",
                "orbit_feasible_fraction_error_leq_1"
            ],
            [
                (
                    row[0], row[1], row[2], row[3], row[4], row[5],
                    target_half_width,
                    np.nan if target_relative_half_width is None else target_relative_half_width,
                    row[6], row[7], row[8], row[9]
                )
                for row in batch_rows
            ]
        )

        self._save_table(
            "monte_carlo_orbit_error_samples",
            ["sample_index", "mission_success", "normalized_orbit_error"],
            [
                (i + 1, mission_flags_all[i], orbit_errors_all[i])
                for i in range(min(len(orbit_errors_all), len(mission_flags_all)))
            ],
        )

        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
        n_vals = np.array([r[0] for r in batch_rows], dtype=int)
        p_vals = np.array([r[1] for r in batch_rows], dtype=float) * 100.0
        lo_vals = np.array([r[2] for r in batch_rows], dtype=float) * 100.0
        hi_vals = np.array([r[3] for r in batch_rows], dtype=float) * 100.0
        hw_vals = np.array([r[4] for r in batch_rows], dtype=float) * 100.0
        rel_hw_vals = np.array([r[5] for r in batch_rows], dtype=float) * 100.0
        orbit_p50_vals = np.array([r[6] for r in batch_rows], dtype=float)
        orbit_p90_vals = np.array([r[7] for r in batch_rows], dtype=float)
        orbit_p95_vals = np.array([r[8] for r in batch_rows], dtype=float)
        orbit_feas_vals = np.array([r[9] for r in batch_rows], dtype=float) * 100.0

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
        ax1.set_title("Monte Carlo Precision Target (Adaptive Sample Size)")

        finite_mask = np.isfinite(orbit_p50_vals) & np.isfinite(orbit_p90_vals) & np.isfinite(orbit_p95_vals)
        if np.any(finite_mask):
            ax3.plot(n_vals[finite_mask], orbit_p50_vals[finite_mask], "o-", color="tab:green", label="Orbit error p50")
            ax3.plot(n_vals[finite_mask], orbit_p90_vals[finite_mask], "s-", color="tab:orange", label="Orbit error p90")
            ax3.plot(n_vals[finite_mask], orbit_p95_vals[finite_mask], "^-", color="tab:red", label="Orbit error p95")
            ax3.axhline(1.0, color="k", linestyle="--", alpha=0.7, label="Feasible threshold (=1)")
        else:
            ax3.text(0.5, 0.5, "No finite orbit-error values", transform=ax3.transAxes, ha="center", va="center")
        ax3.set_ylabel("Normalized Orbit Error (-)")
        ax3.grid(True, alpha=0.3)

        ax4 = ax3.twinx()
        if np.any(np.isfinite(orbit_feas_vals)):
            ax4.plot(n_vals[np.isfinite(orbit_feas_vals)], orbit_feas_vals[np.isfinite(orbit_feas_vals)], "d-.", color="tab:purple", label="Feasible fraction (error<=1)")
        ax4.set_ylabel("Orbit-Feasible Fraction (%)", color="tab:purple")
        ax4.tick_params(axis="y", labelcolor="tab:purple")
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines4, labels4 = ax4.get_legend_handles_labels()
        if len(lines3) > 0 or len(lines4) > 0:
            ax3.legend(lines3 + lines4, labels3 + labels4, loc="best")

        ax3.set_xlabel("Samples N")
        self._finalize_figure(fig, "monte_carlo_precision_target")

    # 19. Q2 UNCERTAINTY BUDGET (MINIMUM FUEL)
    def analyze_q2_uncertainty_budget(self, parameter_samples=30, max_workers=None):
        debug._print_sub_header("19. Q2 Uncertainty Budget (Minimum Fuel)")
        print("Quantifying minimum-fuel uncertainty split: numerical vs parameter sources.")

        opt_nom = self._get_baseline_opt_res()
        m_final_nom = self._extract_final_mass_from_opt(opt_nom)
        if not np.isfinite(m_final_nom):
            print("  Baseline optimization did not return a valid final mass. Skipping.")
            return
        fuel_nominal = float(self.base_config.launch_mass - m_final_nom)
        if max_workers is None:
            max_workers = min(4, max(1, os.cpu_count() or 1))
        max_workers = max(1, int(max_workers))

        # --- A) Numerical component 1: transcription/grid sensitivity of objective ---
        node_counts = [120, 130, 140]
        grid_rows = []
        grid_fuels = []

        print("\n  [A] Grid sensitivity on minimum fuel:")
        print(f"  {'Nodes':<6} | {'Status':<10} | {'Min Fuel (kg)':<15} | {'Runtime (s)':<10}")
        print("  " + "-" * 52)
        grid_jobs = [
            {"num_nodes": int(N), "base_cfg": self.base_config, "base_env_cfg": self.base_env_config}
            for N in node_counts
        ]
        grid_results = {}
        grid_parallel = bool(max_workers > 1 and len(grid_jobs) > 1)
        if grid_parallel:
            print(f"  [A] Parallel execution enabled (workers={max_workers}).")
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    fut_to_n = {executor.submit(_run_q2_grid_point_worker, payload): int(payload["num_nodes"]) for payload in grid_jobs}
                    parallel_ok = True
                    total_grid_jobs = len(fut_to_n)
                    done_grid_jobs = 0
                    for fut in as_completed(fut_to_n):
                        done_grid_jobs += 1
                        self._print_parallel_progress("[A] Grid jobs", done_grid_jobs, total_grid_jobs)
                        n_key = fut_to_n[fut]
                        try:
                            grid_results[n_key] = fut.result()
                        except Exception:
                            parallel_ok = False
                            break
                if not parallel_ok:
                    print("  [A] Worker failure; switching to serial mode.")
                    grid_parallel = False
                    grid_results = {}
            except Exception:
                print("  [A] Parallel execution unavailable; switching to serial mode.")
                grid_parallel = False
                grid_results = {}
        if not grid_parallel:
            print("  [A] Serial execution mode.")
            for payload in grid_jobs:
                n_key = int(payload["num_nodes"])
                grid_results[n_key] = _run_q2_grid_point_worker(payload)

        for N in node_counts:
            r = grid_results.get(int(N), None)
            if r is None:
                r = {
                    "nodes": int(N),
                    "solver_success": 0,
                    "status": "Error",
                    "minimum_fuel_kg": np.nan,
                    "runtime_s": np.nan,
                }
            ok = int(r["solver_success"])
            status = str(r["status"])
            fuel_i = float(r["minimum_fuel_kg"]) if np.isfinite(r["minimum_fuel_kg"]) else np.nan
            dt = float(r["runtime_s"]) if np.isfinite(r["runtime_s"]) else np.nan
            if np.isfinite(fuel_i):
                grid_fuels.append(fuel_i)
            grid_rows.append((int(N), ok, status, fuel_i, dt))
            print(f"  {int(N):<6d} | {status:<10} | {fuel_i:<15.3f} | {dt:<10.2f}")

        sigma_grid = float(np.std(grid_fuels, ddof=1)) if len(grid_fuels) >= 2 else np.nan

        # --- B) Numerical component 2: integrator tolerance sensitivity in verification ---
        integrator_tols = [
            (1e-9, 1e-12),
            (1e-10, 1e-13),
            (1e-11, 1e-14),
            (1e-12, 1e-14),
        ]
        integ_rows = []
        integ_fuels = []

        print("\n  [B] Integrator sensitivity on verified terminal fuel:")
        print(f"  {'RTOL':<8} | {'ATOL':<8} | {'Status':<10} | {'Path':<6} | {'Fuel Used (kg)':<15}")
        print("  " + "-" * 62)
        integ_jobs = [
            {
                "rtol": float(rtol),
                "atol": float(atol),
                "opt_nom": opt_nom,
                "base_cfg": self.base_config,
                "base_env_cfg": self.base_env_config,
                "q_slack_pa": float(self.path_q_slack_pa),
                "g_slack": float(self.path_g_slack),
            }
            for rtol, atol in integrator_tols
        ]
        integ_results = {}
        integ_parallel = bool(max_workers > 1 and len(integ_jobs) > 1)
        if integ_parallel:
            print(f"  [B] Parallel execution enabled (workers={max_workers}).")
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    fut_to_key = {
                        executor.submit(_run_q2_integrator_point_worker, payload): (float(payload["rtol"]), float(payload["atol"]))
                        for payload in integ_jobs
                    }
                    parallel_ok = True
                    total_integ_jobs = len(fut_to_key)
                    done_integ_jobs = 0
                    for fut in as_completed(fut_to_key):
                        done_integ_jobs += 1
                        self._print_parallel_progress("[B] Integrator jobs", done_integ_jobs, total_integ_jobs)
                        key = fut_to_key[fut]
                        try:
                            integ_results[key] = fut.result()
                        except Exception:
                            parallel_ok = False
                            break
                if not parallel_ok:
                    print("  [B] Worker failure; switching to serial mode.")
                    integ_parallel = False
                    integ_results = {}
            except Exception:
                print("  [B] Parallel execution unavailable; switching to serial mode.")
                integ_parallel = False
                integ_results = {}
        if not integ_parallel:
            print("  [B] Serial execution mode.")
            for payload in integ_jobs:
                key = (float(payload["rtol"]), float(payload["atol"]))
                integ_results[key] = _run_q2_integrator_point_worker(payload)

        for rtol, atol in integrator_tols:
            key = (float(rtol), float(atol))
            r = integ_results.get(key, None)
            if r is None:
                r = {
                    "rtol": float(rtol),
                    "atol": float(atol),
                    "terminal_valid": 0,
                    "strict_ok": 0,
                    "strict_path_ok": 0,
                    "q_ok": 0,
                    "g_ok": 0,
                    "status": "SIM_ERROR",
                    "fuel_used_kg": np.nan,
                }
            status = str(r["status"])
            strict_path_i = int(r["strict_path_ok"])
            fuel_sim = float(r["fuel_used_kg"]) if np.isfinite(r["fuel_used_kg"]) else np.nan
            integ_rows.append((
                float(rtol), float(atol),
                int(r["terminal_valid"]),
                int(r["strict_ok"]),
                strict_path_i,
                int(r["q_ok"]),
                int(r["g_ok"]),
                status,
                fuel_sim,
            ))
            if np.isfinite(fuel_sim):
                integ_fuels.append(fuel_sim)
            print(f"  {rtol:<8.0e} | {atol:<8.0e} | {status:<10} | {strict_path_i:<6d} | {fuel_sim:<15.3f}")

        sigma_integrator = float(np.std(integ_fuels, ddof=1)) if len(integ_fuels) >= 2 else np.nan
        fuel_sim_nom = integ_rows[0][8] if len(integ_rows) > 0 else np.nan
        bias_opt_vs_sim = abs(fuel_nominal - fuel_sim_nom) if np.isfinite(fuel_sim_nom) else np.nan

        sigma_numerical = self._rss([sigma_grid, sigma_integrator, bias_opt_vs_sim])

        # --- C) Parameter uncertainty on the minimum-fuel estimate ---
        n_param = max(30, int(parameter_samples))
        rng = np.random.default_rng(self.seed_sequence.spawn(1)[0])
        param_rows = []
        param_fuels_all = []
        param_fuels_feasible = []
        print(f"\n  [C] Parameter uncertainty via re-optimization (N={n_param}):")
        print(
            f"  {'Run':<4} | {'Thrust':<7} | {'ISP':<7} | {'Density':<7} | "
            f"{'Status':<10} | {'Min Fuel (kg)':<15} | {'Runtime (s)':<10}"
        )
        print("  " + "-" * 88)
        param_draws = [self._draw_uncertainty_multipliers(rng, model="gaussian_independent") for _ in range(n_param)]
        param_jobs = []
        for i, draw in enumerate(param_draws):
            thrust_mult, isp_mult, dens_mult = draw
            param_jobs.append(
                {
                    "run_index": int(i),
                    "thrust_multiplier": float(thrust_mult),
                    "isp_multiplier": float(isp_mult),
                    "density_multiplier": float(dens_mult),
                    "base_cfg": self.base_config,
                    "base_env_cfg": self.base_env_config,
                    "q_slack_pa": float(self.path_q_slack_pa),
                    "g_slack": float(self.path_g_slack),
                }
            )

        param_results = {}
        param_parallel = bool(max_workers > 1 and len(param_jobs) > 1)
        if param_parallel:
            print(f"  [C] Parallel execution enabled (workers={max_workers}).")
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    fut_to_i = {
                        executor.submit(_run_q2_parameter_sample_worker, payload): int(payload["run_index"])
                        for payload in param_jobs
                    }
                    parallel_ok = True
                    total_param_jobs = len(fut_to_i)
                    done_param_jobs = 0
                    for fut in as_completed(fut_to_i):
                        done_param_jobs += 1
                        self._print_parallel_progress("[C] Parameter samples", done_param_jobs, total_param_jobs)
                        idx = fut_to_i[fut]
                        try:
                            param_results[idx] = fut.result()
                        except Exception:
                            parallel_ok = False
                            break
                if not parallel_ok:
                    print("  [C] Worker failure; switching to serial mode.")
                    param_parallel = False
                    param_results = {}
            except Exception:
                print("  [C] Parallel execution unavailable; switching to serial mode.")
                param_parallel = False
                param_results = {}
        if not param_parallel:
            print("  [C] Serial execution mode.")
            for payload in param_jobs:
                idx = int(payload["run_index"])
                param_results[idx] = _run_q2_parameter_sample_worker(payload)

        for i in range(n_param):
            r = param_results.get(i, None)
            if r is None:
                thrust_mult, isp_mult, dens_mult = param_draws[i]
                r = {
                    "run": i + 1,
                    "thrust_multiplier": float(thrust_mult),
                    "isp_multiplier": float(isp_mult),
                    "density_multiplier": float(dens_mult),
                    "optimizer_success": 0,
                    "strict_ok": 0,
                    "strict_path_ok": 0,
                    "q_ok": 0,
                    "g_ok": 0,
                    "status": "ERROR",
                    "minimum_fuel_kg": np.nan,
                    "runtime_s": np.nan,
                }
            run_i = int(r["run"])
            thrust_mult = float(r["thrust_multiplier"])
            isp_mult = float(r["isp_multiplier"])
            dens_mult = float(r["density_multiplier"])
            opt_success = int(r["optimizer_success"])
            strict_ok = int(r["strict_ok"])
            strict_path_ok = int(r["strict_path_ok"])
            q_ok = int(r["q_ok"])
            g_ok = int(r["g_ok"])
            status = str(r["status"])
            fuel_i = float(r["minimum_fuel_kg"]) if np.isfinite(r["minimum_fuel_kg"]) else np.nan
            dt = float(r["runtime_s"]) if np.isfinite(r["runtime_s"]) else np.nan

            if opt_success and np.isfinite(fuel_i):
                param_fuels_all.append(fuel_i)
            if opt_success and strict_path_ok and np.isfinite(fuel_i):
                param_fuels_feasible.append(fuel_i)

            param_rows.append((
                run_i,
                thrust_mult,
                isp_mult,
                dens_mult,
                opt_success,
                strict_ok,
                strict_path_ok,
                q_ok,
                g_ok,
                status,
                fuel_i,
                dt,
            ))
            print(
                f"  {run_i:<4d} | {thrust_mult:<7.3f} | {isp_mult:<7.3f} | {dens_mult:<7.3f} | "
                f"{status:<10} | {fuel_i:<15.3f} | {dt:<10.2f}"
            )

        sigma_parameter = float(np.std(param_fuels_all, ddof=1)) if len(param_fuels_all) >= 2 else np.nan
        param_mean = float(np.mean(param_fuels_all)) if len(param_fuels_all) > 0 else np.nan
        param_p025 = float(np.percentile(param_fuels_all, 2.5)) if len(param_fuels_all) > 0 else np.nan
        param_p975 = float(np.percentile(param_fuels_all, 97.5)) if len(param_fuels_all) > 0 else np.nan
        valid_param_rate = float(len(param_fuels_all) / max(n_param, 1))

        sigma_parameter_feasible = float(np.std(param_fuels_feasible, ddof=1)) if len(param_fuels_feasible) >= 2 else np.nan
        feasible_param_rate = float(len(param_fuels_feasible) / max(n_param, 1))
        sigma_param_ci_low = np.nan
        sigma_param_ci_high = np.nan
        if len(param_fuels_all) >= 8:
            rng_boot = np.random.default_rng(self.seed_sequence.spawn(1)[0])
            arr = np.array(param_fuels_all, dtype=float)
            boots = np.zeros(1200, dtype=float)
            for b in range(len(boots)):
                idx = rng_boot.integers(0, len(arr), size=len(arr))
                boots[b] = np.std(arr[idx], ddof=1)
            sigma_param_ci_low = float(np.percentile(boots, 2.5))
            sigma_param_ci_high = float(np.percentile(boots, 97.5))

        sigma_param_feasible_ci_low = np.nan
        sigma_param_feasible_ci_high = np.nan
        if len(param_fuels_feasible) >= 8:
            rng_boot = np.random.default_rng(self.seed_sequence.spawn(1)[0])
            arr_f = np.array(param_fuels_feasible, dtype=float)
            boots_f = np.zeros(1200, dtype=float)
            for b in range(len(boots_f)):
                idx = rng_boot.integers(0, len(arr_f), size=len(arr_f))
                boots_f[b] = np.std(arr_f[idx], ddof=1)
            sigma_param_feasible_ci_low = float(np.percentile(boots_f, 2.5))
            sigma_param_feasible_ci_high = float(np.percentile(boots_f, 97.5))

        # --- D) Total uncertainty and contribution split ---
        sigma_total = self._rss([sigma_numerical, sigma_parameter])
        ci95_half_width = 1.96 * sigma_total if np.isfinite(sigma_total) else np.nan
        sigma_total_ci_low = self._rss([sigma_numerical, sigma_param_ci_low]) if np.isfinite(sigma_param_ci_low) else np.nan
        sigma_total_ci_high = self._rss([sigma_numerical, sigma_param_ci_high]) if np.isfinite(sigma_param_ci_high) else np.nan

        if np.isfinite(sigma_total) and sigma_total > 0.0:
            var_total = sigma_total * sigma_total
            num_var = sigma_numerical * sigma_numerical if np.isfinite(sigma_numerical) else 0.0
            param_var = sigma_parameter * sigma_parameter if np.isfinite(sigma_parameter) else 0.0
            numerical_share_pct = 100.0 * num_var / var_total
            parameter_share_pct = 100.0 * param_var / var_total
        else:
            numerical_share_pct = np.nan
            parameter_share_pct = np.nan

        if np.isfinite(ci95_half_width) and ci95_half_width > 0.0:
            rounding_step_kg = float(10.0 ** np.floor(np.log10(ci95_half_width)))
            reported_uncertainty_kg = float(np.round(ci95_half_width / rounding_step_kg) * rounding_step_kg)
            reported_fuel_kg = float(np.round(fuel_nominal / rounding_step_kg) * rounding_step_kg)
        else:
            rounding_step_kg = np.nan
            reported_uncertainty_kg = np.nan
            reported_fuel_kg = np.nan

        print("\n  Q2 uncertainty budget summary (minimum fuel):")
        print(f"    Baseline minimum fuel (nominal): {fuel_nominal:.3f} kg")
        print(f"    Numerical sigma (kg):            {sigma_numerical:.3f}")
        print(f"      - Grid sigma (kg):             {sigma_grid:.3f}")
        print(f"      - Integrator sigma (kg):       {sigma_integrator:.3f}")
        print(f"      - Opt-vs-sim bias (kg):        {bias_opt_vs_sim:.3f}")
        print(f"    Parameter sigma (kg):            {sigma_parameter:.3f}")
        if np.isfinite(sigma_param_ci_low) and np.isfinite(sigma_param_ci_high):
            print(f"      - Parameter sigma 95% CI (kg): [{sigma_param_ci_low:.3f}, {sigma_param_ci_high:.3f}]")
        print(f"    Feasible-only parameter sigma:   {sigma_parameter_feasible:.3f} kg")
        if np.isfinite(sigma_param_feasible_ci_low) and np.isfinite(sigma_param_feasible_ci_high):
            print(
                f"      - Feasible-only sigma 95% CI:   "
                f"[{sigma_param_feasible_ci_low:.3f}, {sigma_param_feasible_ci_high:.3f}] kg"
            )
        print(f"    Total sigma (kg):                {sigma_total:.3f}")
        if np.isfinite(sigma_total_ci_low) and np.isfinite(sigma_total_ci_high):
            print(f"      - Total sigma range from sigma_param CI (kg): [{sigma_total_ci_low:.3f}, {sigma_total_ci_high:.3f}]")
        print(f"    95% half-width (kg):             {ci95_half_width:.3f}")
        print(f"    Variance share numerical:        {numerical_share_pct:.1f}%")
        print(f"    Variance share parameter:        {parameter_share_pct:.1f}%")
        print(f"    Converged parameter solves:      {len(param_fuels_all)}/{n_param} ({100.0*valid_param_rate:.1f}%)")
        print(f"    Feasible parameter solves:       {len(param_fuels_feasible)}/{n_param} ({100.0*feasible_param_rate:.1f}%)")
        if np.isfinite(reported_fuel_kg) and np.isfinite(reported_uncertainty_kg):
            print(
                f"    Suggested report form:            "
                f"{reported_fuel_kg:.0f} +/- {reported_uncertainty_kg:.0f} kg (95%)"
            )

        if len(param_fuels_all) < 20:
            print(
                f">>> {debug.Style.YELLOW}WARN: Too few converged parameter re-optimizations for a stable "
                f"unconditional parameter-uncertainty estimate.{debug.Style.RESET}"
            )
        if len(param_fuels_feasible) < 20:
            print(
                f">>> {debug.Style.YELLOW}WARN: Feasible-only parameter subset is small; "
                f"feasible-conditioned uncertainty is high-variance.{debug.Style.RESET}"
            )

        self._save_table(
            "q2_uncertainty_grid",
            ["nodes", "solver_success", "status", "minimum_fuel_kg", "runtime_s"],
            grid_rows
        )
        self._save_table(
            "q2_uncertainty_integrator",
            [
                "rtol", "atol", "terminal_valid", "strict_ok",
                "strict_path_ok", "q_ok", "g_ok", "status", "fuel_used_kg"
            ],
            integ_rows
        )
        self._save_table(
            "q2_uncertainty_parameter_samples",
            [
                "run", "thrust_multiplier", "isp_multiplier", "density_multiplier",
                "optimizer_success", "strict_ok", "strict_path_ok", "q_ok", "g_ok",
                "status", "minimum_fuel_kg", "runtime_s"
            ],
            param_rows
        )
        self._save_table(
            "q2_uncertainty_budget",
            ["metric", "value"],
            [
                ("baseline_minimum_fuel_kg", fuel_nominal),
                ("sigma_grid_kg", sigma_grid),
                ("sigma_integrator_kg", sigma_integrator),
                ("bias_opt_vs_sim_kg", bias_opt_vs_sim),
                ("sigma_numerical_kg", sigma_numerical),
                ("sigma_parameter_kg", sigma_parameter),
                ("sigma_parameter_ci_low_95_kg", sigma_param_ci_low),
                ("sigma_parameter_ci_high_95_kg", sigma_param_ci_high),
                ("sigma_parameter_feasible_only_kg", sigma_parameter_feasible),
                ("sigma_parameter_feasible_only_ci_low_95_kg", sigma_param_feasible_ci_low),
                ("sigma_parameter_feasible_only_ci_high_95_kg", sigma_param_feasible_ci_high),
                ("sigma_total_kg", sigma_total),
                ("sigma_total_from_sigma_param_ci_low_kg", sigma_total_ci_low),
                ("sigma_total_from_sigma_param_ci_high_kg", sigma_total_ci_high),
                ("ci95_half_width_kg", ci95_half_width),
                ("numerical_variance_share_percent", numerical_share_pct),
                ("parameter_variance_share_percent", parameter_share_pct),
                ("parameter_converged_rate", valid_param_rate),
                ("parameter_feasible_rate", feasible_param_rate),
                ("parameter_mean_minimum_fuel_kg", param_mean),
                ("parameter_p2p5_minimum_fuel_kg", param_p025),
                ("parameter_p97p5_minimum_fuel_kg", param_p975),
                ("report_rounding_step_kg", rounding_step_kg),
                ("report_value_kg", reported_fuel_kg),
                ("report_uncertainty_kg", reported_uncertainty_kg),
            ]
        )

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        ax1, ax2, ax3, ax4 = axes.flatten()

        # Grid objective sensitivity.
        grid_nodes_plot = np.array([r[0] for r in grid_rows if np.isfinite(r[3])], dtype=float)
        grid_fuels_plot = np.array([r[3] for r in grid_rows if np.isfinite(r[3])], dtype=float)
        if len(grid_nodes_plot) > 0:
            grid_delta = grid_fuels_plot - fuel_nominal
            ax1.plot(grid_nodes_plot, grid_delta, "o-", color="tab:blue")
            ax1.axhline(0.0, color="tab:blue", linestyle="--", alpha=0.5, label="Nominal N=140")
            ax1.set_xlabel("Collocation Nodes")
            ax1.set_ylabel("Delta Minimum Fuel (kg)")
            ax1.legend(loc="best")
        else:
            ax1.text(0.5, 0.5, "No converged grid runs", transform=ax1.transAxes, ha="center", va="center")
        ax1.set_title("Numerical: Grid Sensitivity")
        ax1.grid(True, alpha=0.3)

        # Integrator sensitivity.
        tol_plot = np.array([r[0] for r in integ_rows if np.isfinite(r[8])], dtype=float)
        fuel_plot = np.array([r[8] for r in integ_rows if np.isfinite(r[8])], dtype=float)
        if len(tol_plot) > 0:
            ref_fuel = fuel_plot[-1]
            drift = np.abs(fuel_plot - ref_fuel)
            ax2.loglog(tol_plot, np.maximum(drift, 1e-12), "s-", color="tab:orange")
            ax2.invert_xaxis()
            ax2.set_xlabel("Integrator rtol")
            ax2.set_ylabel("|Fuel Drift| vs tightest (kg)")
        else:
            ax2.text(0.5, 0.5, "No valid integrator runs", transform=ax2.transAxes, ha="center", va="center")
        ax2.set_title("Numerical: Integrator Sensitivity")
        ax2.grid(True, which="both", alpha=0.3)

        # Parameter uncertainty distribution.
        if len(param_fuels_all) > 0:
            ax3.hist(
                param_fuels_all,
                bins=min(12, max(5, len(param_fuels_all))),
                color="tab:green",
                alpha=0.55,
                edgecolor="k",
                label="All converged"
            )
            if len(param_fuels_feasible) > 0:
                ax3.hist(
                    param_fuels_feasible,
                    bins=min(12, max(5, len(param_fuels_feasible))),
                    color="tab:blue",
                    alpha=0.45,
                    edgecolor="k",
                    label="Feasible subset"
                )
            ax3.axvline(fuel_nominal, color="tab:blue", linestyle="--", linewidth=1.5, label="Nominal")
            ax3.set_xlabel("Minimum Fuel (kg)")
            ax3.set_ylabel("Count")
            ax3.legend(loc="best")
        else:
            ax3.text(0.5, 0.5, "No valid parameter re-optimizations", transform=ax3.transAxes, ha="center", va="center")
        ax3.set_title("Parameter Uncertainty (Re-optimization)")
        ax3.grid(True, axis="y", alpha=0.3)

        # Contribution split.
        comp_labels = ["Grid", "Integrator", "Opt-Sim bias", "Numerical", "Parameter", "Total"]
        comp_vals = [sigma_grid, sigma_integrator, bias_opt_vs_sim, sigma_numerical, sigma_parameter, sigma_total]
        comp_plot = [1e-6 if (not np.isfinite(v) or v <= 0.0) else float(v) for v in comp_vals]
        colors = ["tab:blue", "tab:orange", "tab:gray", "tab:purple", "tab:green", "tab:red"]
        ax4.bar(comp_labels, comp_plot, color=colors, alpha=0.8)
        ax4.set_ylabel("Uncertainty sigma (kg, log scale)")
        ax4.set_yscale("log")
        ax4.set_title("Q2 Uncertainty Budget")
        ax4.tick_params(axis="x", rotation=20)
        ax4.grid(True, axis="y", alpha=0.3)
        if np.isfinite(numerical_share_pct) and np.isfinite(parameter_share_pct):
            ax4.text(
                0.02,
                0.97,
                (
                    f"Variance shares:\n"
                    f"Numerical {numerical_share_pct:.3f}%\n"
                    f"Parameter {parameter_share_pct:.3f}%\n"
                    f"95% half-width: {ci95_half_width:.0f} kg"
                ),
                transform=ax4.transAxes,
                va="top",
                ha="left",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.7")
            )

        fig.suptitle("Q2: Minimum-Fuel Accuracy and Uncertainty Split")
        self._finalize_figure(fig, "q2_uncertainty_budget")
        self._summary["q2_budget"] = {
            "baseline_minimum_fuel_kg": fuel_nominal,
            "sigma_numerical_kg": sigma_numerical,
            "sigma_parameter_kg": sigma_parameter,
            "sigma_parameter_ci_low_95_kg": sigma_param_ci_low,
            "sigma_parameter_ci_high_95_kg": sigma_param_ci_high,
            "sigma_parameter_feasible_only_kg": sigma_parameter_feasible,
            "sigma_parameter_feasible_only_ci_low_95_kg": sigma_param_feasible_ci_low,
            "sigma_parameter_feasible_only_ci_high_95_kg": sigma_param_feasible_ci_high,
            "sigma_total_kg": sigma_total,
            "ci95_half_width_kg": ci95_half_width,
            "numerical_variance_share_percent": numerical_share_pct,
            "parameter_variance_share_percent": parameter_share_pct,
            "parameter_converged_rate": valid_param_rate,
            "parameter_feasible_rate": feasible_param_rate,
        }

    # 22. BIFURCATION 2D MAP
    def analyze_bifurcation_2d_map(self, n_thrust=11, n_density=11, max_workers=None):
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

        if max_workers is None:
            max_workers = min(4, max(1, os.cpu_count() or 1))
        max_workers = max(1, int(max_workers))

        samples = []
        for iy, dens in enumerate(density_vals):
            for ix, thrust in enumerate(thrust_vals):
                samples.append((int(iy), int(ix), float(thrust), float(dens)))

        sample_rows = []
        parallel_active = bool(max_workers > 1 and len(samples) > 1)
        if parallel_active:
            print(f"  Parallel execution enabled (workers={max_workers}).")
            try:
                n_chunks = min(max_workers, len(samples))
                chunk_size = int(np.ceil(len(samples) / max(n_chunks, 1)))
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    fut_to_chunk_size = {}
                    for i0 in range(0, len(samples), chunk_size):
                        chunk = samples[i0:i0 + chunk_size]
                        payload = {
                            "opt_res": opt_res,
                            "base_cfg": self.base_config,
                            "base_env_cfg": self.base_env_config,
                            "q_slack_pa": float(self.path_q_slack_pa),
                            "g_slack": float(self.path_g_slack),
                            "samples": chunk,
                        }
                        fut = executor.submit(_run_bifurcation_batch_worker, payload)
                        fut_to_chunk_size[fut] = len(chunk)

                    parallel_ok = True
                    completed_samples = 0
                    for fut in as_completed(fut_to_chunk_size):
                        completed_samples += int(fut_to_chunk_size[fut])
                        self._print_parallel_progress("Bifurcation samples", completed_samples, len(samples))
                        try:
                            r = fut.result()
                            sample_rows.extend(r.get("rows", []))
                        except Exception:
                            parallel_ok = False
                            break
                if not parallel_ok:
                    print("  Worker failure during bifurcation map; switching to serial mode.")
                    parallel_active = False
                    sample_rows = []
            except Exception:
                print("  Parallel execution unavailable; switching to serial mode.")
                parallel_active = False
                sample_rows = []

        if not parallel_active:
            print("  Serial execution mode.")
            r = _run_bifurcation_batch_worker(
                {
                    "opt_res": opt_res,
                    "base_cfg": self.base_config,
                    "base_env_cfg": self.base_env_config,
                    "q_slack_pa": float(self.path_q_slack_pa),
                    "g_slack": float(self.path_g_slack),
                    "samples": samples,
                }
            )
            sample_rows = list(r.get("rows", []))

        sample_rows.sort(key=lambda d: (int(d.get("iy", 0)), int(d.get("ix", 0))))
        for item in sample_rows:
            iy = int(item.get("iy", 0))
            ix = int(item.get("ix", 0))
            thrust = float(item.get("thrust", np.nan))
            dens = float(item.get("density", np.nan))
            success = int(item.get("strict_success", 0))
            margin = float(item.get("fuel_margin_kg", np.nan))
            orbit_err = float(item.get("normalized_orbit_error", np.nan))
            alt_err = float(item.get("altitude_error_m", np.nan))
            vel_err = float(item.get("velocity_error_m_s", np.nan))
            q_ok = int(item.get("q_ok", 0))
            g_ok = int(item.get("g_ok", 0))
            status = str(item.get("status", "SIM_ERROR"))

            if 0 <= iy < success_map.shape[0] and 0 <= ix < success_map.shape[1]:
                success_map[iy, ix] = success
                margin_map[iy, ix] = margin
                orbit_err_map[iy, ix] = orbit_err

            rows.append((
                thrust,
                dens,
                success,
                margin,
                orbit_err,
                alt_err,
                vel_err,
                q_ok,
                g_ok,
                status,
            ))

        self._save_table(
            "bifurcation_2d_map",
            [
                "thrust_multiplier", "density_multiplier", "strict_success",
                "fuel_margin_kg", "normalized_orbit_error",
                "altitude_error_m", "velocity_error_m_s", "q_ok", "g_ok", "status"
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
        strict_fraction = strict_count / max(total_count, 1)
        orbit_feasible = np.isfinite(orbit_err_map) & (orbit_err_map <= 1.0)
        orbit_feasible_count = int(np.sum(orbit_feasible))
        orbit_feasible_fraction = orbit_feasible_count / max(total_count, 1)

        dens_idx_nom = int(np.argmin(np.abs(density_vals - 1.0)))
        row_nom = orbit_err_map[dens_idx_nom, :]
        row_lower = np.nan
        row_upper = np.nan

        # Interpolate threshold crossings on nominal-density row.
        row_crossings = []
        for i in range(1, len(thrust_vals)):
            e0 = row_nom[i - 1]
            e1 = row_nom[i]
            if not (np.isfinite(e0) and np.isfinite(e1)):
                continue
            f0 = e0 - 1.0
            f1 = e1 - 1.0
            if abs(f0) < 1e-12:
                row_crossings.append(float(thrust_vals[i - 1]))
            if f0 * f1 < 0.0:
                row_crossings.append(float(np.interp(0.0, [f0, f1], [thrust_vals[i - 1], thrust_vals[i]])))
            if i == len(thrust_vals) - 1 and abs(f1) < 1e-12:
                row_crossings.append(float(thrust_vals[i]))
        row_crossings = sorted(row_crossings)
        if len(row_crossings) > 0:
            row_lower = float(min(row_crossings))
            row_upper = float(max(row_crossings))
        else:
            # Fallback: use discrete feasible band if no crossing can be interpolated.
            row_mask = np.isfinite(row_nom) & (row_nom <= 1.0)
            if np.any(row_mask):
                row_lower = float(np.min(thrust_vals[row_mask]))
                row_upper = float(np.max(thrust_vals[row_mask]))
        margin_lower_pct = (1.0 - row_lower) * 100.0 if np.isfinite(row_lower) else np.nan
        margin_upper_pct = (row_upper - 1.0) * 100.0 if np.isfinite(row_upper) else np.nan
        margin_candidates = [m for m in [margin_lower_pct, margin_upper_pct] if np.isfinite(m)]
        min_margin_pct = float(np.min(margin_candidates)) if len(margin_candidates) > 0 else np.nan

        self._save_table(
            "bifurcation_2d_summary",
            ["metric", "value"],
            [
                ("strict_success_fraction", strict_fraction),
                ("orbit_feasible_fraction", orbit_feasible_fraction),
                ("nominal_density_row_lower_thrust", row_lower),
                ("nominal_density_row_upper_thrust", row_upper),
                ("nominal_density_row_lower_margin_percent", margin_lower_pct),
                ("nominal_density_row_upper_margin_percent", margin_upper_pct),
                ("nominal_density_row_min_margin_percent", min_margin_pct),
            ]
        )
        self._summary["q5_bifurcation_2d"] = {
            "strict_success_fraction": strict_fraction,
            "orbit_feasible_fraction": orbit_feasible_fraction,
            "nominal_row_min_margin_percent": min_margin_pct,
        }

        print(f"  Strict-success points: {strict_count}/{total_count}")
        print(f"  Orbit-feasible points (error<=1): {orbit_feasible_count}/{total_count}")
        if np.isfinite(min_margin_pct):
            print(f"  Nominal-density row minimum thrust margin: {min_margin_pct:.2f}%")
        self._finalize_figure(fig, "bifurcation_2d_map")
        print(f">>> {debug.Style.GREEN}PASS: 2D bifurcation map generated.{debug.Style.RESET}")

    # 23. MODEL LIMITATIONS REGISTER (Q6)
    def analyze_model_limitations(self):
        debug._print_sub_header("23. Model Limitations and Validity Bounds")
        print("Documenting modeling assumptions, likely bias directions, and mitigation evidence.")

        rows = [
            (
                "open_loop_control",
                "No feedback controller; controls are replayed open-loop.",
                "Tends to overstate fragility under perturbations.",
                "bifurcation_2d_map",
                "High",
            ),
            (
                "reduced_dof",
                "Translational dynamics only (no full attitude/actuator dynamics).",
                "Can misestimate controllability and steering losses.",
                "drift, smooth_integrator_benchmark",
                "High",
            ),
            (
                "aero_prop_models",
                "Aerodynamics and propulsion use reduced lookup/parametric models.",
                "Bias in drag/thrust loss estimates and fuel optimum.",
                "theoretical_efficiency, q2_uncertainty_budget",
                "Medium",
            ),
            (
                "atmosphere_uncertainty",
                "Atmospheric uncertainty modeled with scalar density multiplier.",
                "May underrepresent profile/wind-structure uncertainty.",
                "monte_carlo_precision_target, q2_uncertainty_budget",
                "Medium",
            ),
            (
                "numerical_transcription",
                "Finite collocation grid and finite solver tolerances.",
                "Can introduce discretization/integration bias in objective.",
                "grid_independence, integrator_tolerance, q2_uncertainty_budget",
                "Medium",
            ),
            (
                "ideal_staging",
                "Stage transitions are idealized without hardware faults.",
                "Underestimates real mission risk and contingency fuel.",
                "integrator_tolerance, drift",
                "Medium",
            ),
        ]

        self._save_table(
            "model_limitations",
            ["id", "assumption", "likely_impact", "mitigation_evidence", "severity"],
            rows,
        )

        sev_map = {"High": 3, "Medium": 2, "Low": 1}
        severity_score = int(np.sum([sev_map.get(r[4], 1) for r in rows]))
        high_count = int(np.sum([1 for r in rows if r[4] == "High"]))
        print(f"  Documented limitations: {len(rows)}")
        print(f"  High-severity limitations: {high_count}")
        self._save_table(
            "model_limitations_summary",
            ["metric", "value"],
            [
                ("n_limitations", len(rows)),
                ("n_high_severity", high_count),
                ("severity_score", severity_score),
            ],
        )
        self._summary["q6_limitations"] = {
            "n_limitations": len(rows),
            "n_high_severity": high_count,
            "severity_score": severity_score,
        }

    # 24. CONCLUSION SUPPORT SYNTHESIS (Q7)
    def analyze_q7_conclusion_support(self):
        debug._print_sub_header("24. Q7 Conclusion Support Synthesis")
        print("Synthesizing evidence for the final engineering conclusion.")

        q2 = self._summary.get("q2_budget", {})
        q5 = self._summary.get("q5_bifurcation_2d", {})

        # Fallback to saved summary tables if upstream tests were skipped.
        ci95_half_width_kg = q2.get("ci95_half_width_kg", self._get_saved_metric("q2_uncertainty_budget", "ci95_half_width_kg"))
        param_share_pct = q2.get(
            "parameter_variance_share_percent",
            self._get_saved_metric("q2_uncertainty_budget", "parameter_variance_share_percent"),
        )
        num_share_pct = q2.get(
            "numerical_variance_share_percent",
            self._get_saved_metric("q2_uncertainty_budget", "numerical_variance_share_percent"),
        )
        min_margin_pct = q5.get(
            "nominal_row_min_margin_percent",
            self._get_saved_metric("bifurcation_2d_summary", "nominal_density_row_min_margin_percent")
        )
        strict_success_fraction = q5.get(
            "strict_success_fraction",
            self._get_saved_metric("bifurcation_2d_summary", "strict_success_fraction"),
        )
        orbit_feasible_fraction = q5.get(
            "orbit_feasible_fraction",
            self._get_saved_metric("bifurcation_2d_summary", "orbit_feasible_fraction"),
        )

        evidence_rows = [
            ("q2_ci95_half_width_kg", ci95_half_width_kg),
            ("q2_parameter_variance_share_percent", param_share_pct),
            ("q2_numerical_variance_share_percent", num_share_pct),
            ("q5_nominal_row_min_thrust_margin_percent", min_margin_pct),
            ("q5_strict_success_fraction", strict_success_fraction),
            ("q5_orbit_feasible_fraction", orbit_feasible_fraction),
        ]

        # Pragmatic decision rules from retained tests only.
        cliff_supported = bool(
            np.isfinite(min_margin_pct) and min_margin_pct > 0.0 and min_margin_pct <= 3.0
        )
        uncertainty_supported = bool(
            np.isfinite(param_share_pct) and np.isfinite(num_share_pct) and param_share_pct > num_share_pct
        )
        map_coverage_supported = bool(
            np.isfinite(strict_success_fraction)
            and np.isfinite(orbit_feasible_fraction)
            and strict_success_fraction > 0.0
            and orbit_feasible_fraction > 0.0
        )
        support_score = int(cliff_supported) + int(uncertainty_supported) + int(map_coverage_supported)
        conclusion_supported = int(support_score >= 2)

        self._save_table(
            "q7_conclusion_support",
            ["metric", "value"],
            evidence_rows
            + [
                ("cliff_supported", int(cliff_supported)),
                ("uncertainty_supported", int(uncertainty_supported)),
                ("map_coverage_supported", int(map_coverage_supported)),
                ("support_score", support_score),
                ("conclusion_supported", conclusion_supported),
            ],
        )

        print(f"  Cliff-edge evidence (2D margin): {int(cliff_supported)}")
        print(f"  Uncertainty evidence (parameter > numerical): {int(uncertainty_supported)}")
        print(f"  2D map coverage evidence: {int(map_coverage_supported)}")
        if conclusion_supported:
            print(f">>> {debug.Style.GREEN}PASS: Evidence supports the Q7 engineering conclusion.{debug.Style.RESET}")
        else:
            print(f">>> {debug.Style.YELLOW}WARN: Evidence is incomplete for a strong Q7 claim.{debug.Style.RESET}")

        self._summary["q7_conclusion"] = {
            "support_score": support_score,
            "conclusion_supported": conclusion_supported,
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reliability and robustness analysis suite.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for exported figures and CSV tables.")
    parser.add_argument("--seed", type=int, default=1337, help="Global random seed for reproducible Monte Carlo runs.")
    parser.add_argument("--mc-samples", type=int, default=200, help="Monte Carlo sample count used in Monte Carlo-based analyses.")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum parallel workers (set 1 for fully serial mode).")
    parser.add_argument("--course-core", action="store_true", help="Run only the recommended paper-safe course-core evidence set.")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive plot windows (save figures only).")
    parser.add_argument("--no-save", action="store_true", help="Do not save figures/CSV outputs.")
    args = parser.parse_args()

    suite = ReliabilitySuite(
        output_dir=args.output_dir,
        save_figures=not args.no_save,
        show_plots=not args.no_show,
        random_seed=args.seed,
        analysis_profile="course_core" if args.course_core else "default",
    )
    suite.run_all(monte_carlo_samples=args.mc_samples, max_workers=args.max_workers)
