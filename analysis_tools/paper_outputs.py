import argparse
import contextlib
import copy
import csv
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from orbit_launch.config import EARTH_CONFIG, StarshipBlock2
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
)
from orbit_launch.vehicle import Vehicle

from analysis_tools.relabilityanalysis import (
    ReliabilitySuite,
    WarmStartMultiseedAnalyzer,
    evaluate_terminal_state,
)


OUTPUT_GUIDANCE_SOURCES = [
    "https://matplotlib.org/stable/users/explain/axes/constrainedlayout_guide.html",
    "https://matplotlib.org/stable/users/explain/colors/colormaps.html",
    "https://matplotlib.org/stable/users/explain/axes/legend_guide.html",
    "https://aiaa.org/publications/journals/Journal-Author/Guidelines-for-Journal-Figures-and-Tables/",
    "https://research-figure-guide.nature.com/figures/",
    "https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/",
]

MAIN_FIGURES = [
    "fig_main_01_nominal_profile",
    "fig_main_02_constraints",
    "fig_main_03_grid_independence",
    "fig_main_04_integrator_convergence",
    "fig_main_06_warm_start_multiseed",
]

APPENDIX_FIGURES = [
    "fig_app_01_interval_replay_audit",
    "fig_app_02_smooth_integrator_order",
    "fig_app_03_theoretical_efficiency",
    "fig_app_04_3d_trajectory",
    "fig_app_05_global_launch_cost",
]

REQUIRED_RELIABILITY_FILES = (
    "data/grid_independence.csv",
    "data/integrator_tolerance.csv",
    "data/drift_phase_metrics.csv",
    "data/drift_summary.csv",
    "data/interval_replay_audit.csv",
    "data/randomized_multistart.csv",
    "data/randomized_multistart_deltas.csv",
    "data/randomized_multistart_trajectory_summary.csv",
    "data/run_metadata.csv",
    "data/smooth_integrator_benchmark.csv",
    "data/smooth_integrator_benchmark_fit.csv",
    "data/theoretical_efficiency.csv",
    "figures/randomized_multistart.png",
    "figures/randomized_multistart.pdf",
)

STALE_PAPER_PACK_ARTIFACTS = (
    "figures/fig_app_01_collocation_defect.png",
    "figures/fig_app_01_collocation_defect.pdf",
    "figures/fig_main_05_replay_drift.png",
    "figures/fig_main_05_replay_drift.pdf",
    "reliability_raw/data/collocation_defect_audit.csv",
    "reliability_raw/figures/collocation_defect_audit.png",
    "reliability_raw/figures/collocation_defect_audit.pdf",
)

HEATMAP_ARTIFACTS = (
    "figures/fig_app_05_global_launch_cost.png",
    "figures/fig_app_05_global_launch_cost.pdf",
    "data/global_launch_cost_latitude_sweep.csv",
)

MULTISTART_ARTIFACTS = (
    "figures/fig_main_06_warm_start_multiseed.png",
    "figures/fig_main_06_warm_start_multiseed.pdf",
    "data/randomized_multistart.csv",
    "data/randomized_multistart_deltas.csv",
    "data/randomized_multistart_trajectory_summary.csv",
)


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w", encoding="utf-8") as fnull:
        with contextlib.redirect_stdout(fnull):
            yield


def _read_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_key_value_csv(path):
    out = {}
    for row in _read_csv_rows(path):
        out[row["key"]] = row["value"]
    return out


def _read_metric_value_csv(path):
    out = {}
    for row in _read_csv_rows(path):
        out[row["metric"]] = row["value"]
    return out


def _save_csv(path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def _save_markdown_table(path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(cell) for cell in row) + " |\n")


def _save_figure(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _add_panel_labels(axes, x=0.01, y=0.99):
    axes_flat = list(np.ravel(axes))
    for idx, ax in enumerate(axes_flat):
        label = f"{chr(ord('a') + idx)})"
        if getattr(ax, "name", "") == "3d" and hasattr(ax, "text2D"):
            ax.text2D(
                x,
                y,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                fontweight="bold",
                color="0.2",
            )
        else:
            ax.text(
                x,
                y,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                fontweight="bold",
                color="0.2",
            )


def _fmt_float(value, digits=3):
    if value is None or not np.isfinite(value):
        return "nan"
    if abs(value) >= 1e4 or (abs(value) > 0 and abs(value) < 1e-2):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def _refine_peak_time_value(t_series, y_series):
    t = np.asarray(t_series, dtype=float)
    y = np.asarray(y_series, dtype=float)
    if t.ndim != 1 or y.ndim != 1 or len(t) != len(y) or len(t) == 0:
        return np.nan, np.nan, None
    if not np.any(np.isfinite(y)):
        return np.nan, np.nan, None

    idx = int(np.nanargmax(y))
    if len(t) < 3 or idx <= 0 or idx >= len(t) - 1:
        return float(t[idx]), float(y[idx]), idx

    t_w = t[idx - 1:idx + 2]
    y_w = y[idx - 1:idx + 2]
    if not (np.all(np.isfinite(t_w)) and np.all(np.isfinite(y_w))):
        return float(t[idx]), float(y[idx]), idx

    x = t_w - t[idx]
    A = np.column_stack([x * x, x, np.ones_like(x)])
    try:
        a, b, c = np.linalg.lstsq(A, y_w, rcond=None)[0]
    except np.linalg.LinAlgError:
        return float(t[idx]), float(y[idx]), idx
    if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(c):
        return float(t[idx]), float(y[idx]), idx
    if a >= 0.0 or abs(a) < 1e-18:
        return float(t[idx]), float(y[idx]), idx

    x_peak = -b / (2.0 * a)
    if x_peak < x[0] or x_peak > x[-1]:
        return float(t[idx]), float(y[idx]), idx

    y_peak = a * x_peak * x_peak + b * x_peak + c
    if not np.isfinite(y_peak):
        return float(t[idx]), float(y[idx]), idx
    sampled_peak = float(np.nanmax(y_w))
    tol = max(1.0e-12, 1.0e-9 * abs(sampled_peak))
    if y_peak > sampled_peak + tol:
        return float(t[idx]), float(y[idx]), idx
    return float(t[idx] + x_peak), float(y_peak), idx


def _deduplicate_last_samples(times, *series, tol=1e-12):
    t = np.asarray(times, dtype=float)
    keep_idx = []
    i = 0
    while i < len(t):
        j = i + 1
        while j < len(t) and abs(float(t[j]) - float(t[i])) <= tol:
            j += 1
        keep_idx.append(j - 1)
        i = j

    deduped = []
    for values in series:
        arr = np.asarray(values)
        if arr.ndim == 1:
            deduped.append(arr[keep_idx])
        else:
            deduped.append(arr[:, keep_idx])
    return t[keep_idx], deduped


def _build_optimizer_phase_histories(opt_res, env_cfg):
    phase_histories = []
    current_t = 0.0
    for duration_key, state_key in (("T1", "X1"), ("T2", "X2"), ("T3", "X3")):
        duration = float(opt_res.get(duration_key, 0.0))
        if duration <= 1e-8 or state_key not in opt_res:
            continue
        state = np.asarray(opt_res[state_key], dtype=float)
        phase_t = np.linspace(current_t, current_t + duration, state.shape[1])
        phase_histories.append(
            {
                "start": float(current_t),
                "end": float(current_t + duration),
                "t": phase_t,
                "spherical_height_km": np.array(
                    [spherical_altitude_m(state[0:3, idx], env_cfg) / 1000.0 for idx in range(state.shape[1])],
                    dtype=float,
                ),
                "ellipsoidal_altitude_km": np.array(
                    [ellipsoidal_altitude_m(state[0:3, idx], env_cfg) / 1000.0 for idx in range(state.shape[1])],
                    dtype=float,
                ),
                "mass_kg": np.asarray(state[6, :], dtype=float),
            }
        )
        current_t += duration
    return phase_histories


def _phasewise_interp(sim_t, phase_histories, key):
    out = np.full_like(np.asarray(sim_t, dtype=float), np.nan, dtype=float)
    sim_t = np.asarray(sim_t, dtype=float)
    tol = 1e-12
    for idx, phase in enumerate(phase_histories):
        start = float(phase["start"])
        end = float(phase["end"])
        if idx < len(phase_histories) - 1:
            mask = (sim_t >= start - tol) & (sim_t < end - tol)
        else:
            mask = (sim_t >= start - tol) & (sim_t <= end + tol)
        if not np.any(mask):
            continue
        out[mask] = np.interp(sim_t[mask], phase["t"], phase[key])
    return out


def _setup_plot_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": False,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "lines.linewidth": 1.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def _apply_paper_grid(ax, axis="y", which="major", alpha=0.10):
    ax.grid(True, axis=axis, which=which, alpha=alpha, linewidth=0.6)


def _effective_target_inclination_deg(cfg, env_cfg):
    if cfg.target_inclination is None:
        return abs(float(env_cfg.launch_latitude))
    return float(cfg.target_inclination)


def _format_ratio_from_value(value):
    if value == 0:
        return "0"
    return f"1/{(1.0 / float(value)):.3f}"

def _flatten_optimal_trajectory(opt_res):
    t_parts = []
    x_parts = []

    n1 = opt_res["X1"].shape[1] - 1
    t1 = np.linspace(0.0, float(opt_res["T1"]), n1 + 1)
    t_parts.append(t1)
    x_parts.append(np.asarray(opt_res["X1"], dtype=float))

    current_t = float(opt_res["T1"])
    if float(opt_res.get("T2", 0.0)) > 1e-8 and "X2" in opt_res:
        n2 = opt_res["X2"].shape[1] - 1
        t2 = np.linspace(current_t, current_t + float(opt_res["T2"]), n2 + 1)
        t_parts.append(t2)
        x_parts.append(np.asarray(opt_res["X2"], dtype=float))
        current_t += float(opt_res["T2"])

    n3 = opt_res["X3"].shape[1] - 1
    t3 = np.linspace(current_t, current_t + float(opt_res["T3"]), n3 + 1)
    t_parts.append(t3)
    x_parts.append(np.asarray(opt_res["X3"], dtype=float))

    return np.concatenate(t_parts), np.hstack(x_parts)


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
        dot_val = np.dot(r_ecef_0, r_ecef) / max(denom, 1e-12)
        angle = np.arccos(np.clip(dot_val, -1.0, 1.0))
        out.append(angle * env_cfg.earth_radius_equator / 1000.0)
    return np.array(out, dtype=float)


def _compute_nominal_timeseries(sim_res, env, veh, cfg):
    t = np.asarray(sim_res["t"], dtype=float)
    y = np.asarray(sim_res["y"], dtype=float)
    u = np.asarray(sim_res["u"], dtype=float)
    r_hist = y[0:3, :]
    v_hist = y[3:6, :]
    mass_hist = y[6, :]

    spherical_height_km = np.zeros_like(t)
    ellipsoidal_altitude_km = np.zeros_like(t)
    vel_m_s = np.zeros_like(t)
    q_kpa = np.zeros_like(t)
    g_load = np.zeros_like(t)
    mach = np.zeros_like(t)

    m_stage2_wet = cfg.stage_2.dry_mass + cfg.stage_2.propellant_mass + cfg.payload_mass
    for i, t_i in enumerate(t):
        r_i = r_hist[:, i]
        v_i = v_hist[:, i]
        env_state = env.get_state_sim(r_i, t_i)
        v_mag = np.linalg.norm(v_i)
        v_rel = v_i - env_state["wind_velocity"]
        v_rel_mag = np.linalg.norm(v_rel)
        q_pa = 0.5 * env_state["density"] * v_rel_mag**2
        spherical_height_km[i] = spherical_altitude_m(r_i, env.config) / 1000.0
        ellipsoidal_altitude_km[i] = ellipsoidal_altitude_m(r_i, env.config) / 1000.0
        vel_m_s[i] = v_mag
        q_kpa[i] = q_pa / 1000.0
        mach[i] = v_rel_mag / max(env_state["speed_of_sound"], 1.0)

        th_i = float(u[0, i])
        dir_i = np.asarray(u[1:, i], dtype=float)
        if th_i < 0.01:
            stage_mode = "coast" if mass_hist[i] > m_stage2_wet + 1000.0 else "coast_2"
        else:
            stage_mode = "boost" if mass_hist[i] > m_stage2_wet + 1000.0 else "ship"
        dyn = veh.get_dynamics(y[:, i], th_i, dir_i, t_i, stage_mode=stage_mode, scaling=None)
        sensed_acc = dyn[3:6] - env_state["gravity"]
        g_load[i] = np.linalg.norm(sensed_acc) / env.config.g0

    q_peak_time_s, q_peak_kpa, q_peak_idx = _refine_peak_time_value(t, q_kpa)
    if q_peak_idx is not None and np.isfinite(q_peak_kpa):
        q_kpa[q_peak_idx] = max(float(q_kpa[q_peak_idx]), q_peak_kpa)

    g_peak_time_s, g_peak_value, g_peak_idx = _refine_peak_time_value(t, g_load)
    if g_peak_idx is not None and np.isfinite(g_peak_value):
        g_load[g_peak_idx] = max(float(g_load[g_peak_idx]), g_peak_value)

    downrange_km = _compute_downrange_km(r_hist, env.config, t)
    return {
        "t_s": t,
        "spherical_height_km": spherical_height_km,
        "ellipsoidal_altitude_km": ellipsoidal_altitude_km,
        "velocity_m_s": vel_m_s,
        "mass_kg": mass_hist,
        "dynamic_pressure_kpa": q_kpa,
        "g_load": g_load,
        "mach": mach,
        "downrange_km": downrange_km,
        "throttle": u[0, :],
        "state": y,
        "max_q_time_s": q_peak_time_s,
        "max_q_kpa": q_peak_kpa,
        "max_g_time_s": g_peak_time_s,
        "max_g": g_peak_value,
    }


def _compute_orbit_projection(state_hist, env_cfg):
    r_final = state_hist[0:3, -1]
    v_final = state_hist[3:6, -1]
    mu = env_cfg.earth_mu
    r_mag = np.linalg.norm(r_final)
    v_mag = np.linalg.norm(v_final)
    specific_energy = (v_mag**2) / 2.0 - mu / r_mag
    if specific_energy >= 0.0:
        return None

    a = -mu / (2.0 * specific_energy)
    period = 2.0 * np.pi * np.sqrt(a**3 / mu)
    j2 = env_cfg.j2_constant if bool(env_cfg.use_j2_perturbation) else 0.0
    r_eq = env_cfg.earth_radius_equator

    def dyn(_, y):
        r = y[0:3]
        v = y[3:6]
        r_norm = np.linalg.norm(r)
        g_central = -mu / (r_norm**3) * r
        factor = -1.5 * j2 * mu * (r_eq**2) / (r_norm**5)
        z_sq = (r[2] / r_norm) ** 2
        g_j2 = np.array(
            [
                factor * r[0] * (1.0 - 5.0 * z_sq),
                factor * r[1] * (1.0 - 5.0 * z_sq),
                factor * r[2] * (3.0 - 5.0 * z_sq),
            ],
            dtype=float,
        )
        return np.concatenate([v, g_central + g_j2])

    sol = solve_ivp(
        dyn,
        (0.0, period),
        np.concatenate([r_final, v_final]),
        t_eval=np.linspace(0.0, period, 300),
        rtol=1e-6,
        atol=1e-9,
    )
    if not bool(sol.success):
        return None
    return sol.y


def _orbit_plane_basis(state_hist):
    r_final = np.asarray(state_hist[0:3, -1], dtype=float)
    v_final = np.asarray(state_hist[3:6, -1], dtype=float)
    r_norm = np.linalg.norm(r_final)
    h_vec = np.cross(r_final, v_final)
    h_norm = np.linalg.norm(h_vec)
    if r_norm <= 1e-12 or h_norm <= 1e-12:
        return None
    e_u = r_final / r_norm
    e_w = h_vec / h_norm
    e_v = np.cross(e_w, e_u)
    e_v_norm = np.linalg.norm(e_v)
    if e_v_norm <= 1e-12:
        return None
    e_v = e_v / e_v_norm
    return e_u, e_v


def _project_to_orbit_plane(r_hist, basis):
    e_u, e_v = basis
    x = np.asarray(e_u @ r_hist, dtype=float) / 1000.0
    y = np.asarray(e_v @ r_hist, dtype=float) / 1000.0
    return x, y


def _earth_cross_section_in_plane(env_cfg, basis, n_samples=361):
    e_u, e_v = basis
    a = float(env_cfg.earth_radius_equator)
    c = float(env_cfg.earth_radius_equator * (1.0 - env_cfg.earth_flattening))
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples)
    dirs = np.outer(e_u, np.cos(theta)) + np.outer(e_v, np.sin(theta))
    denom = (dirs[0, :] ** 2 + dirs[1, :] ** 2) / (a * a) + (dirs[2, :] ** 2) / (c * c)
    rho = 1.0 / np.sqrt(np.maximum(denom, 1e-18))
    return rho * np.cos(theta) / 1000.0, rho * np.sin(theta) / 1000.0


def _summarize_grid(rows):
    ordered = sorted(rows, key=lambda row: int(row["nodes"]))
    nodes = np.array([int(row["nodes"]) for row in ordered], dtype=int)
    masses = np.array([float(row["final_mass_kg"]) for row in ordered], dtype=float)
    raw_ok = np.array([bool(int(row["raw_path_ok"])) for row in ordered], dtype=bool)
    slack_ok = np.array([bool(int(row["strict_path_ok"])) for row in ordered], dtype=bool)
    ref_mass = masses[-1]
    mass_err = np.abs(masses - ref_mass)
    criterion_kg = 100.0

    def _select(mask):
        for idx, node in enumerate(nodes):
            suffix = slice(idx, len(nodes))
            if np.all(mask[suffix]) and np.all(np.isfinite(mass_err[suffix])) and np.all(mass_err[suffix] <= criterion_kg):
                return int(node)
        return None

    return {
        "nodes": nodes,
        "masses": masses,
        "raw_ok": raw_ok,
        "slack_ok": slack_ok,
        "mass_err": mass_err,
        "reference_node": int(nodes[-1]),
        "reference_mass_kg": float(ref_mass),
        "criterion_kg": criterion_kg,
        "selected_raw_node": _select(raw_ok),
        "selected_slack_node": _select(slack_ok),
    }


def _summarize_integrator(rows):
    ordered = sorted(rows, key=lambda row: float(row["rtol"]), reverse=True)
    rtol = np.array([float(row["rtol"]) for row in ordered], dtype=float)
    atol = np.array([float(row["atol"]) for row in ordered], dtype=float)
    alt = np.array([float(row["final_spherical_height_m"]) for row in ordered], dtype=float)
    vel = np.array([float(row["final_velocity_m_s"]) for row in ordered], dtype=float)
    radial = np.array([float(row["radial_velocity_m_s"]) for row in ordered], dtype=float)
    raw_ok = np.array([bool(int(row["raw_path_ok"])) for row in ordered], dtype=bool)
    q_max = np.array([float(row["max_q_pa"]) for row in ordered], dtype=float)
    g_max = np.array([float(row["max_g"]) for row in ordered], dtype=float)

    ref_alt = alt[-1]
    ref_vel = vel[-1]
    ref_rad = radial[-1]
    ref_q = q_max[-1]
    ref_g = g_max[-1]

    d_alt = np.abs(alt - ref_alt)
    d_vel = np.abs(vel - ref_vel)
    d_rad = np.abs(radial - ref_rad)
    d_q = np.abs(q_max - ref_q)
    d_g = np.abs(g_max - ref_g)

    thresholds = {
        "altitude_m": 50.0,
        "velocity_m_s": 0.5,
        "radial_velocity_m_s": 0.5,
        "dynamic_pressure_pa": 50.0,
        "g_load": 0.02,
    }

    selected_idx = None
    for idx in range(len(rtol)):
        if (
            raw_ok[idx]
            and d_alt[idx] < thresholds["altitude_m"]
            and d_vel[idx] < thresholds["velocity_m_s"]
            and d_rad[idx] < thresholds["radial_velocity_m_s"]
            and d_q[idx] < thresholds["dynamic_pressure_pa"]
            and d_g[idx] < thresholds["g_load"]
        ):
            selected_idx = idx
            break

    return {
        "rtol": rtol,
        "atol": atol,
        "altitude_drift_m": d_alt,
        "velocity_drift_m_s": d_vel,
        "radial_velocity_drift_m_s": d_rad,
        "dynamic_pressure_drift_pa": d_q,
        "g_load_drift": d_g,
        "selected_idx": selected_idx,
        "thresholds": thresholds,
    }


class PaperPackBuilder:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.figure_dir = self.output_dir / "figures"
        self.table_dir = self.output_dir / "tables"
        self.data_dir = self.output_dir / "data"
        self.notes_dir = self.output_dir / "notes"
        self.raw_reliability_dir = self.output_dir / "reliability_raw"
        self.has_heatmap_outputs = False
        self._reliability_suite = None

    def build(self):
        _setup_plot_style()
        self._ensure_dirs()
        self._cleanup_stale_outputs()

        cfg = copy.deepcopy(StarshipBlock2)
        env_cfg = copy.deepcopy(EARTH_CONFIG)
        env = Environment(env_cfg)
        veh = Vehicle(cfg, env)

        print("[PaperPack] Solving baseline nominal trajectory...")
        opt_res = solve_optimal_trajectory(cfg, veh, env, print_level=self.args.print_level)
        if not bool(opt_res.get("success", False)):
            raise RuntimeError("Baseline optimization failed; paper pack cannot be generated.")

        print("[PaperPack] Running baseline forward replay...")
        sim_res = run_simulation(
            opt_res,
            veh,
            cfg,
            rtol=self.args.nominal_rtol,
            atol=self.args.nominal_atol,
        )

        print("[PaperPack] Preparing reliability evidence...")
        reliability_dir = self._prepare_reliability(opt_res)
        timeseries = _compute_nominal_timeseries(sim_res, env, veh, cfg)
        self._write_time_history_csv(opt_res, env_cfg, timeseries)
        self._write_paper_tables(cfg, env_cfg, env, veh, opt_res, sim_res, reliability_dir)
        self._render_main_figures(cfg, env_cfg, opt_res, sim_res, timeseries, reliability_dir)
        self._render_appendix_figures(cfg, env_cfg, opt_res, sim_res, timeseries, reliability_dir)

        if not self.args.skip_heatmap:
            self.has_heatmap_outputs = self._build_launch_cost_heatmap()
        else:
            self.has_heatmap_outputs = self._heatmap_artifacts_exist()
            if self.has_heatmap_outputs:
                print("[PaperPack] Skipping global launch-cost heatmap recompute; keeping existing heatmap artifacts.")
            else:
                print("[PaperPack] Skipping global launch-cost heatmap; no existing heatmap artifacts found.")
        if not self._copy_multistart_artifacts(reliability_dir):
            raise FileNotFoundError("Warm-start multiseed artifacts are missing from the reliability outputs.")
        self._write_notes()

        print(f"[PaperPack] Completed: {self.output_dir}")

    def _ensure_dirs(self):
        for path in (self.figure_dir, self.table_dir, self.data_dir, self.notes_dir):
            path.mkdir(parents=True, exist_ok=True)

    def _cleanup_stale_outputs(self):
        for rel_path in STALE_PAPER_PACK_ARTIFACTS:
            path = self.output_dir / rel_path
            if path.exists():
                path.unlink()

    def _heatmap_artifacts_exist(self):
        return all((self.output_dir / rel_path).exists() for rel_path in HEATMAP_ARTIFACTS)

    def _multistart_artifacts_exist(self):
        return all((self.output_dir / rel_path).exists() for rel_path in MULTISTART_ARTIFACTS)

    def _prepare_reliability(self, opt_res):
        if self.args.reliability_dir:
            reliability_dir = Path(self.args.reliability_dir)
            required = tuple(reliability_dir / rel_path for rel_path in REQUIRED_RELIABILITY_FILES)
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError("Reliability directory is missing required files:\n" + "\n".join(missing))
            return reliability_dir

        suite = ReliabilitySuite(
            output_dir=self.raw_reliability_dir,
            save_figures=True,
            show_plots=False,
            analysis_profile="course_core",
        )
        suite._baseline_opt_res = copy.deepcopy(opt_res)
        suite.test_toggles.randomized_multistart = True
        suite._save_table(
            "enabled_test_toggles",
            ["toggle", "enabled"],
            [(k, int(v)) for k, v in suite.test_toggles.__dict__.items()],
        )
        suite.run_all(max_workers=self.args.max_workers)
        self._reliability_suite = suite
        return self.raw_reliability_dir

    def _write_time_history_csv(self, opt_res, env_cfg, timeseries):
        phase_histories = _build_optimizer_phase_histories(opt_res, env_cfg)
        sim_t, deduped = _deduplicate_last_samples(
            timeseries["t_s"],
            timeseries["spherical_height_km"],
            timeseries["ellipsoidal_altitude_km"],
            timeseries["velocity_m_s"],
            timeseries["mass_kg"],
            timeseries["dynamic_pressure_kpa"],
            timeseries["g_load"],
            timeseries["downrange_km"],
            timeseries["throttle"],
        )
        (
            sim_spherical_height_km,
            sim_ellipsoidal_altitude_km,
            sim_velocity_m_s,
            sim_mass_kg,
            sim_dynamic_pressure_kpa,
            sim_g_load,
            sim_downrange_km,
            sim_throttle,
        ) = deduped

        opt_spherical_height_interp = _phasewise_interp(sim_t, phase_histories, "spherical_height_km")
        opt_ellipsoidal_altitude_interp = _phasewise_interp(sim_t, phase_histories, "ellipsoidal_altitude_km")
        opt_mass_interp = _phasewise_interp(sim_t, phase_histories, "mass_kg")

        rows = []
        for idx, t_val in enumerate(sim_t):
            rows.append(
                (
                    _fmt_float(t_val, 6),
                    _fmt_float(opt_spherical_height_interp[idx], 6),
                    _fmt_float(sim_spherical_height_km[idx], 6),
                    _fmt_float(opt_ellipsoidal_altitude_interp[idx], 6),
                    _fmt_float(sim_ellipsoidal_altitude_km[idx], 6),
                    _fmt_float(sim_velocity_m_s[idx], 6),
                    _fmt_float(opt_mass_interp[idx], 6),
                    _fmt_float(sim_mass_kg[idx], 6),
                    _fmt_float(sim_dynamic_pressure_kpa[idx], 6),
                    _fmt_float(sim_g_load[idx], 6),
                    _fmt_float(sim_downrange_km[idx], 6),
                    _fmt_float(sim_throttle[idx], 6),
                )
            )

        _save_csv(
            self.data_dir / "nominal_time_history.csv",
            [
                "time_s",
                "optimizer_spherical_height_km",
                "simulation_spherical_height_km",
                "optimizer_ellipsoidal_altitude_km",
                "simulation_ellipsoidal_altitude_km",
                "simulation_velocity_m_s",
                "optimizer_mass_kg",
                "simulation_mass_kg",
                "dynamic_pressure_kpa",
                "g_load",
                "downrange_km",
                "throttle",
            ],
            rows,
        )

    def _write_paper_tables(self, cfg, env_cfg, env, veh, opt_res, sim_res, reliability_dir):
        run_metadata = _read_key_value_csv(reliability_dir / "data" / "run_metadata.csv")
        grid_rows = _read_csv_rows(reliability_dir / "data" / "grid_independence.csv")
        replay_rows = _read_csv_rows(reliability_dir / "data" / "interval_replay_audit.csv")
        integ_rows = _read_csv_rows(reliability_dir / "data" / "integrator_tolerance.csv")
        drift_phase_rows = _read_csv_rows(reliability_dir / "data" / "drift_phase_metrics.csv")
        multistart_rows = _read_csv_rows(reliability_dir / "data" / "randomized_multistart.csv")
        multistart_delta_rows = _read_csv_rows(reliability_dir / "data" / "randomized_multistart_deltas.csv")

        suite = ReliabilitySuite(output_dir=self.output_dir / "_tmp_suite", save_figures=False, show_plots=False, analysis_profile="course_core")
        traj_diag = suite._trajectory_diagnostics(sim_res, veh, cfg)
        path_diag = suite._evaluate_path_compliance(traj_diag, cfg, q_slack_pa=0.0, g_slack=0.0)
        terminal = evaluate_terminal_state(sim_res, cfg, env_cfg)
        drift = suite._compute_drift_metrics(opt_res, sim_res)
        grid_summary = _summarize_grid(grid_rows)
        integ_summary = _summarize_integrator(integ_rows)

        max_rel_defect = max(float(row["max_relative_error"]) for row in replay_rows)
        max_pos_defect = max(float(row["end_position_error_m"]) for row in replay_rows)

        nominal_rows = [
            ("Target spherical height", _fmt_float(cfg.target_altitude / 1000.0, 3), "km", "Optimizer enforces norm(r) = R_eq + target"),
            ("Effective target inclination", _fmt_float(_effective_target_inclination_deg(cfg, env_cfg), 3), "deg", "Defaults to absolute launch latitude"),
            ("Nodes in each burn", str(cfg.num_nodes), "-", "Node-based RK4 direct transcription grid"),
            ("Booster burn time", _fmt_float(float(opt_res["T1"]), 3), "s", "Optimized"),
            ("Ship burn time", _fmt_float(float(opt_res["T3"]), 3), "s", "Optimized"),
            ("Final mass", _fmt_float(float(opt_res["X3"][6, -1]), 3), "kg", "Optimizer result"),
            (
                "Propellant margin",
                _fmt_float(float(opt_res["X3"][6, -1] - (cfg.stage_2.dry_mass + cfg.payload_mass)), 3),
                "kg",
                "Final mass above dry+payload",
            ),
            ("Final spherical-height error", _fmt_float(float(terminal["spherical_height_err_m"]), 3), "m", "Forward replay against the optimizer target"),
            ("Final ellipsoidal altitude", _fmt_float(float(terminal["ellipsoidal_altitude_m"]) / 1000.0, 3), "km", "Ground-referenced diagnostic only"),
            ("Final velocity error", _fmt_float(float(terminal["vel_err_m_s"]), 3), "m/s", "Forward replay"),
            ("Final radial velocity", _fmt_float(float(terminal["radial_velocity_m_s"]), 3), "m/s", "Forward replay"),
            ("Inclination error", _fmt_float(float(terminal["inclination_err_deg"]), 4), "deg", "Forward replay"),
            ("Max dynamic pressure", _fmt_float(float(traj_diag["max_q_pa"]) / 1000.0, 3), "kPa", "Forward replay"),
            ("Max g-load", _fmt_float(float(traj_diag["max_g"]), 3), "g", "Forward replay"),
            ("Solver return status", str(opt_res.get("solver_return_status", "")), "-", "CasADi/IPOPT"),
            ("Solver iterations", str(opt_res.get("solver_iter_count", "")), "-", "CasADi/IPOPT"),
        ]

        verification_rows = [
            ("Raw max-Q compliant", str(int(path_diag["q_ok"])), "-", "Forward replay without slack"),
            ("Raw max-g compliant", str(int(path_diag["g_ok"])), "-", "Forward replay without slack"),
            ("Coarsest raw-valid grid", str(grid_summary["selected_raw_node"]), "nodes", "All finer grids stay raw-valid and within 100 kg of N=140"),
            ("Coarsest slack-valid grid", str(grid_summary["selected_slack_node"]), "nodes", "All finer grids stay slack-valid and within 100 kg of N=140"),
            (
                "Coarsest converged replay tolerance",
                "nan" if integ_summary["selected_idx"] is None else f"{integ_summary['rtol'][integ_summary['selected_idx']]:.0e}",
                "rtol",
                "Against tightest replay baseline",
            ),
            (
                "Nominal replay tolerance",
                f"{self.args.nominal_rtol:.0e} / {self.args.nominal_atol:.0e}",
                "rtol/atol",
                "Used for baseline forward replay",
            ),
            ("Max position drift", _fmt_float(float(drift["max_pos_m"]), 4), "m", "Optimizer vs replay"),
            ("Max velocity drift", _fmt_float(float(drift["max_vel_m_s"]), 6), "m/s", "Optimizer vs replay"),
            ("Max mass drift", _fmt_float(float(drift["max_mass_kg"]), 6), "kg", "Optimizer vs replay"),
            ("Max interval replay relative error", _fmt_float(max_rel_defect, 3), "-", "All intervals"),
            ("Max interval replay position error", _fmt_float(max_pos_defect, 3), "m", "Independent DOP853 interval replay"),
            ("Reliability profile", run_metadata.get("analysis_profile", "unknown"), "-", "Raw evidence bundle"),
        ]

        assumptions_rows = [
            ("Vehicle case", "Two-stage Starship proxy", "-", cfg.name),
            ("Payload", _fmt_float(float(cfg.payload_mass), 1), "kg", "Nominal case solved in this report"),
            (
                "Launch site",
                f"{float(env_cfg.launch_latitude):.3f}, {float(env_cfg.launch_longitude):.3f}",
                "deg",
                "Geodetic latitude/longitude; launch altitude 5 m",
            ),
            (
                "Earth / gravity model",
                "Rotating WGS84 ellipsoid + J2",
                "-",
                f"R_eq={float(env_cfg.earth_radius_equator):.0f} m, f={_format_ratio_from_value(env_cfg.earth_flattening)}",
            ),
            (
                "Atmosphere model",
                "USSA1976 lookup",
                "-",
                f"10 m tabulation to {env_cfg.atmosphere_max_alt/1000.0:.0f} km",
            ),
            ("Wind assumption", "Corotating atmosphere", "-", "No explicit local wind model"),
            (
                "Aerodynamic model",
                "Cd(M) + crossflow term",
                "-",
                f"A_ref={cfg.stage_1.aero.reference_area:.2f} m^2, Cd_crossflow={cfg.stage_1.aero.cd_crossflow_factor:.1f}",
            ),
            (
                "Orbit target",
                _fmt_float(cfg.target_altitude / 1000.0, 3),
                "km",
                "Target spherical height above R_eq; inclination defaults to absolute launch latitude when unspecified",
            ),
            (
                "Objective",
                "Maximize final mass",
                "-",
                "Equivalent to minimizing propellant consumed for the nominal ascent",
            ),
            (
                "Transcription",
                "Node-based RK4 direct transcription",
                "-",
                f"{cfg.num_nodes} nodes in each burn with piecewise-constant controls",
            ),
            (
                "Path constraints",
                "Ellipsoidal ground clearance, max-Q, max-g, mass, throttle",
                "-",
                f"Published limits: {cfg.max_q_limit/1000.0:.1f} kPa and {cfg.max_g_load:.1f} g",
            ),
            (
                "Internal optimization buffer",
                f"{OPTIMIZATION_Q_GUARD_BAND_PA:.0f} Pa / {OPTIMIZATION_G_GUARD_BAND:.2f} g",
                "-",
                "Small guard bands used so replay respects the published hard limits",
            ),
            (
                "Warm start",
                "Deterministic guidance guess",
                "-",
                "Generated by guidance.py before IPOPT solve",
            ),
            (
                "Optimizer",
                "IPOPT via CasADi Opti",
                "-",
                f"expand=True, max_iter={cfg.max_iter}, tol=1e-6",
            ),
            (
                "Replay integrator",
                "SciPy RK45",
                "-",
                f"Nominal tolerances rtol={self.args.nominal_rtol:.0e}, atol={self.args.nominal_atol:.0e}",
            ),
        ]

        self._write_table_files("table_01_nominal_case", nominal_rows)
        self._write_table_files("table_02_verification_summary", verification_rows)
        self._write_table_files("table_03_assumptions_method", assumptions_rows)
        replay_drift_rows = []
        for row in drift_phase_rows:
            replay_drift_rows.extend(
                [
                    (
                        row["phase"],
                        "Max position drift",
                        _fmt_float(float(row["max_position_drift_m"]), 3),
                        "m",
                        "Optimizer vs replay",
                    ),
                    (
                        row["phase"],
                        "RMS position drift",
                        _fmt_float(float(row["rms_position_drift_m"]), 3),
                        "m",
                        "Optimizer vs replay",
                    ),
                    (
                        row["phase"],
                        "Max velocity drift",
                        _fmt_float(float(row["max_velocity_drift_m_s"]), 6),
                        "m/s",
                        "Optimizer vs replay",
                    ),
                    (
                        row["phase"],
                        "RMS velocity drift",
                        _fmt_float(float(row["rms_velocity_drift_m_s"]), 6),
                        "m/s",
                        "Optimizer vs replay",
                    ),
                    (
                        row["phase"],
                        "Max mass drift",
                        _fmt_float(float(row["max_mass_drift_kg"]), 6),
                        "kg",
                        "Optimizer vs replay",
                    ),
                    (
                        row["phase"],
                        "RMS mass drift",
                        _fmt_float(float(row["rms_mass_drift_kg"]), 6),
                        "kg",
                        "Optimizer vs replay",
                    ),
                ]
            )
        self._write_table_files(
            "table_04_replay_drift_by_phase",
            replay_drift_rows,
            headers=["Phase", "Metric", "Value", "Unit", "Notes"],
        )
        multistart_delta_summary = self._summarize_multistart_deltas(multistart_delta_rows)
        self._write_table_files(
            "table_05_multistart_consistency",
            self._build_multistart_consistency_rows(multistart_rows, multistart_delta_summary),
            headers=[
                "Seed",
                "Status",
                "Final-mass gap to best (mg)",
                "Max position separation (m)",
                "Max velocity delta (m/s)",
            ],
        )

        plt.close("all")

    def _write_table_files(self, stem, rows, headers=None):
        if headers is None:
            headers = ["Quantity", "Value", "Unit", "Notes"]
        csv_rows = [tuple(row) for row in rows]
        _save_csv(self.table_dir / f"{stem}.csv", headers, csv_rows)
        _save_markdown_table(self.table_dir / f"{stem}.md", headers, csv_rows)

    def _render_main_figures(self, cfg, env_cfg, opt_res, sim_res, timeseries, reliability_dir):
        t_sim = timeseries["t_s"]
        stage_time = float(opt_res["T1"])
        max_q_idx = int(np.argmax(timeseries["dynamic_pressure_kpa"]))
        phase_histories = _build_optimizer_phase_histories(opt_res, env_cfg)
        opt_altitude_km = _phasewise_interp(t_sim, phase_histories, "ellipsoidal_altitude_km")
        opt_mass_kg = _phasewise_interp(t_sim, phase_histories, "mass_kg")
        has_optimizer_altitude = np.count_nonzero(np.isfinite(opt_altitude_km)) > 1
        has_optimizer_mass = np.count_nonzero(np.isfinite(opt_mass_kg)) > 1
        replay_color = "tab:blue"
        optimizer_color = "tab:orange"
        reference_color = "0.35"
        legend_handles = []
        legend_labels = []

        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6), layout="constrained")
        ax = axes[0, 0]
        if has_optimizer_altitude:
            optimizer_line, = ax.plot(t_sim, opt_altitude_km, color=optimizer_color, linestyle="--", label="Optimizer profile")
            if "Optimizer profile" not in legend_labels:
                legend_handles.append(optimizer_line)
                legend_labels.append("Optimizer profile")
        replay_line, = ax.plot(t_sim, timeseries["ellipsoidal_altitude_km"], color=replay_color, label="Forward replay")
        if "Forward replay" not in legend_labels:
            legend_handles.append(replay_line)
            legend_labels.append("Forward replay")
        ax.axvline(stage_time, color="0.55", linestyle=":", linewidth=1.0)
        ax.annotate("Staging", (stage_time, 0.98), xycoords=("data", "axes fraction"), xytext=(4, -4), textcoords="offset points", fontsize=8, color="0.2", va="top")
        ax.set_ylabel("Ellipsoidal altitude (km)")

        ax = axes[0, 1]
        ax.plot(t_sim, timeseries["velocity_m_s"], color=replay_color, label="Forward replay")
        target_line = ax.axhline(
            circular_target_speed_m_s(cfg.target_altitude, env_cfg),
            color=reference_color,
            linestyle=":",
            label="Circular-orbit speed",
        )
        if "Circular-orbit speed" not in legend_labels:
            legend_handles.append(target_line)
            legend_labels.append("Circular-orbit speed")
        ax.axvline(stage_time, color="0.55", linestyle=":", linewidth=1.0)
        ax.set_ylabel("Velocity (m/s)")

        ax = axes[1, 0]
        if has_optimizer_mass:
            ax.plot(t_sim, opt_mass_kg / 1000.0, color=optimizer_color, linestyle="--", label="Optimizer profile")
        ax.plot(t_sim, timeseries["mass_kg"] / 1000.0, color=replay_color, label="Forward replay")
        ax.axvline(stage_time, color="0.55", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mass (t)")

        ax = axes[1, 1]
        ax.plot(timeseries["downrange_km"], timeseries["ellipsoidal_altitude_km"], color=replay_color, linewidth=2.0)
        stage_idx = int(np.abs(t_sim - stage_time).argmin())
        event_specs = [
            ("Launch", 0, "tab:green", (6, -10)),
            ("Max Q", max_q_idx, "tab:orange", (6, 8)),
            ("Staging", stage_idx, "tab:red", (6, -12)),
            ("Injection", len(t_sim) - 1, "black", (-44, -2)),
        ]
        for label, idx, color, offset in event_specs:
            x_event = float(timeseries["downrange_km"][idx])
            y_event = float(timeseries["ellipsoidal_altitude_km"][idx])
            ax.scatter([x_event], [y_event], color=color, s=20, zorder=3)
            ax.annotate(
                label,
                (x_event, y_event),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                color="0.2",
            )
        ax.set_xlabel("Downrange (km)")
        ax.set_ylabel("Ellipsoidal altitude (km)")
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(legend_handles),
            frameon=True,
            columnspacing=1.4,
            handlelength=2.0,
        )
        _add_panel_labels(axes)
        _save_figure(fig, self.figure_dir / "fig_main_01_nominal_profile")

        fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.8), sharex=True, layout="constrained")
        axes[0].plot(t_sim, timeseries["dynamic_pressure_kpa"], color="tab:blue")
        axes[0].axhline(cfg.max_q_limit / 1000.0, color="tab:red", linestyle="--", label="Max-Q limit")
        axes[0].axvline(stage_time, color="0.55", linestyle=":", linewidth=1.0)
        axes[0].annotate("Staging", (stage_time, 0.98), xycoords=("data", "axes fraction"), xytext=(4, -4), textcoords="offset points", fontsize=8, color="0.2", va="top")
        axes[0].set_ylabel("Dynamic pressure (kPa)")
        axes[0].legend(loc="upper right")

        axes[1].plot(t_sim, timeseries["g_load"], color="tab:orange")
        axes[1].axhline(cfg.max_g_load, color="tab:red", linestyle="--", label="Max g-load limit")
        axes[1].axvline(stage_time, color="0.55", linestyle=":", linewidth=1.0)
        axes[1].set_ylabel("g-load (g)")
        axes[1].legend(loc="upper right")

        axes[2].step(t_sim, timeseries["throttle"], where="post", color="tab:green")
        axes[2].axhline(cfg.sequence.min_throttle, color="0.35", linestyle=":", label="Min throttle")
        axes[2].axvline(stage_time, color="0.55", linestyle=":", linewidth=1.0)
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Throttle (-)")
        axes[2].set_ylim(-0.02, 1.05)
        axes[2].legend(loc="upper right")
        _add_panel_labels(axes)
        _save_figure(fig, self.figure_dir / "fig_main_02_constraints")

        self._render_grid_independence(reliability_dir / "data" / "grid_independence.csv", self.figure_dir / "fig_main_03_grid_independence")
        self._render_integrator_convergence(reliability_dir / "data" / "integrator_tolerance.csv", self.figure_dir / "fig_main_04_integrator_convergence")
        self._render_drift(reliability_dir / "data" / "drift_phase_metrics.csv", self.figure_dir / "fig_main_05_replay_drift")

    def _render_appendix_figures(self, cfg, env_cfg, opt_res, sim_res, timeseries, reliability_dir):
        self._render_interval_replay_audit(reliability_dir / "data" / "interval_replay_audit.csv", self.figure_dir / "fig_app_01_interval_replay_audit")
        self._render_smooth_order(
            reliability_dir / "data" / "smooth_integrator_benchmark.csv",
            reliability_dir / "data" / "smooth_integrator_benchmark_fit.csv",
            self.figure_dir / "fig_app_02_smooth_integrator_order",
        )
        self._render_theoretical_efficiency(reliability_dir / "data" / "theoretical_efficiency.csv", self.figure_dir / "fig_app_03_theoretical_efficiency")
        self._render_3d_trajectory(env_cfg, timeseries, self.figure_dir / "fig_app_04_3d_trajectory")

    def _render_grid_independence(self, csv_path, stem):
        rows = _read_csv_rows(csv_path)
        summary = _summarize_grid(rows)
        nodes = summary["nodes"]
        masses = summary["masses"]
        mass_err = summary["mass_err"]

        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(7.2, 5.8),
            sharex=True,
            layout="constrained",
            gridspec_kw={"height_ratios": [1.15, 1.0]},
        )

        ax1.plot(nodes, masses, marker="o", color="tab:blue", label="Final mass")
        ax1.axhline(summary["reference_mass_kg"], color="tab:blue", linestyle="--", alpha=0.55, label=f"Reference (N={summary['reference_node']})")
        ax1.set_ylabel("Final mass (kg)")
        ax1.ticklabel_format(style="plain", axis="y", useOffset=False)
        ax1.legend(loc="upper right")

        mask = np.arange(len(nodes)) != len(nodes) - 1
        ax2.semilogy(nodes[mask], np.maximum(mass_err[mask], 1e-12), marker="o", color="tab:purple", label="Mass error")
        ax2.axhline(summary["criterion_kg"], color="tab:orange", linestyle="--", label="Criterion")
        ax2.set_xlabel("Nodes in each burn")
        ax2.set_ylabel("|m(N)-m_ref| (kg)")
        ax2.legend(loc="upper right")

        for ax in (ax1, ax2):
            _apply_paper_grid(ax, axis="y")
        _add_panel_labels((ax1, ax2))
        _save_figure(fig, stem)

    def _render_integrator_convergence(self, csv_path, stem):
        rows = _read_csv_rows(csv_path)
        summary = _summarize_integrator(rows)
        if len(summary["rtol"]) > 1:
            plot_slice = slice(0, -1)
        else:
            plot_slice = slice(None)
        rtol = summary["rtol"][plot_slice]
        altitude_drift = summary["altitude_drift_m"][plot_slice]
        velocity_drift = summary["velocity_drift_m_s"][plot_slice]
        radial_velocity_drift = summary["radial_velocity_drift_m_s"][plot_slice]

        fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.4), sharex=True, layout="constrained")
        axes[0].loglog(rtol, np.maximum(altitude_drift, 1e-12), marker="o", color="tab:blue", label="Altitude drift")
        axes[0].axhline(summary["thresholds"]["altitude_m"], color="tab:orange", linestyle="--", label="Criterion")
        axes[0].set_ylabel("Altitude drift (m)")
        axes[0].legend(loc="upper right")

        axes[1].loglog(rtol, np.maximum(velocity_drift, 1e-12), marker="s", color="tab:orange", label="Velocity drift")
        axes[1].axhline(summary["thresholds"]["velocity_m_s"], color="tab:red", linestyle="--", label="Criterion")
        axes[1].set_ylabel("Velocity drift (m/s)")
        axes[1].legend(loc="upper right")

        axes[2].loglog(rtol, np.maximum(radial_velocity_drift, 1e-12), marker="^", color="tab:green", label="Radial velocity drift")
        axes[2].axhline(summary["thresholds"]["radial_velocity_m_s"], color="tab:red", linestyle="--", label="Criterion")
        axes[2].set_xlabel("Integrator relative tolerance (rtol)")
        axes[2].set_ylabel("Radial velocity drift (m/s)")
        axes[2].legend(loc="upper right")

        for ax in axes:
            ax.invert_xaxis()
            _apply_paper_grid(ax, axis="y")
        _add_panel_labels(axes)
        _save_figure(fig, stem)

    def _render_drift(self, csv_path, stem):
        rows = _read_csv_rows(csv_path)
        phases = [row["phase"] for row in rows]
        pos = np.array([float(row["max_position_drift_m"]) for row in rows], dtype=float)
        vel = np.array([float(row["max_velocity_drift_m_s"]) for row in rows], dtype=float)
        mass = np.array([float(row["max_mass_drift_kg"]) for row in rows], dtype=float)

        x = np.arange(len(phases))
        fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.1), sharex=True, layout="constrained")

        panels = [
            (pos / 2000.0, "tab:blue", "Position drift / threshold"),
            (vel / 30.0, "tab:green", "Velocity drift / threshold"),
            (mass / 500.0, "tab:purple", "Mass drift / threshold"),
        ]
        for ax, (vals, color, ylabel) in zip(axes, panels):
            ax.bar(x, np.maximum(vals, 1e-12), color=color, alpha=0.82)
            ax.axhline(1.0, color="tab:orange", linestyle="--", label="Acceptance threshold")
            ax.set_yscale("log")
            ax.set_ylabel(ylabel)
            ax.legend(loc="best")
            _apply_paper_grid(ax, axis="y")
        axes[-1].set_xticks(x, phases)
        axes[-1].set_xlabel("Mission phase")
        _add_panel_labels(axes)
        _save_figure(fig, stem)

    def _render_interval_replay_audit(self, csv_path, stem):
        rows = _read_csv_rows(csv_path)
        pos = np.array([float(row["end_position_error_m"]) for row in rows], dtype=float)
        rel = np.array([float(row["max_relative_error"]) for row in rows], dtype=float)
        phase = np.array([row["phase"] for row in rows])

        global_idx = np.arange(len(rows))
        phase_order = []
        for name in phase:
            if name not in phase_order:
                phase_order.append(name)
        transition_points = np.where(phase[1:] != phase[:-1])[0] + 1 if len(phase) > 1 else np.array([], dtype=int)
        colors = plt.get_cmap("tab10")

        fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True, layout="constrained")
        for idx, name in enumerate(phase_order):
            mask = phase == name
            color = colors(idx % colors.N)
            label = name.replace("_", " ").title()
            axes[0].semilogy(global_idx[mask], np.maximum(pos[mask], 1e-18), color=color, label=label)
            axes[1].semilogy(global_idx[mask], np.maximum(rel[mask], 1e-18), color=color, label=label)
        axes[0].set_ylabel("End-position error (m)")
        axes[0].legend(loc="best")

        axes[1].axhline(1e-6, color="tab:orange", linestyle="--", label="Reference threshold")
        axes[1].set_xlabel("Node interval index")
        axes[1].set_ylabel("Max relative error (-)")
        axes[1].legend(loc="best")
        for point in transition_points:
            for ax in axes:
                ax.axvline(point - 0.5, color="0.4", linestyle=":", linewidth=1.0)
        for ax in axes:
            _apply_paper_grid(ax, axis="y")
        _add_panel_labels(axes)
        _save_figure(fig, stem)

    def _render_smooth_order(self, bench_csv, fit_csv, stem):
        rows = _read_csv_rows(bench_csv)
        fit_rows = _read_csv_rows(fit_csv)
        dt = np.array([float(row["dt_s"]) for row in rows], dtype=float)
        euler = np.array([float(row["euler_state_error"]) for row in rows], dtype=float)
        rk4 = np.array([float(row["rk4_state_error"]) for row in rows], dtype=float)
        slopes = {row["method"]: float(row["slope"]) for row in fit_rows}

        fig, ax = plt.subplots(figsize=(6.6, 4.8), layout="constrained")
        ax.loglog(dt, euler, marker="o", color="tab:red", label=f"Euler (slope={slopes['Euler']:.2f})")
        ax.loglog(dt, rk4, marker="s", color="tab:blue", label=f"RK4 (slope={slopes['RK4']:.2f})")
        ax.set_xlabel("Time step (s)")
        ax.set_ylabel("Absolute state error at T = 4 s (-)")
        ax.legend(loc="best")
        _apply_paper_grid(ax, axis="both", which="major")
        _save_figure(fig, stem)

    def _render_theoretical_efficiency(self, csv_path, stem):
        row = _read_csv_rows(csv_path)[0]
        dv_theory = float(row["dv_theoretical_m_s"])
        dv_actual = float(row["dv_actual_m_s"])
        efficiency = float(row["efficiency_percent"])
        losses = float(row["losses_m_s"])

        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), layout="constrained")
        axes[0].bar(["Idealized\nreference", "Actual\ntrajectory"], [dv_theory, dv_actual], color=["tab:blue", "tab:green"])
        axes[0].set_ylabel("Delta-v (m/s)")

        axes[1].barh(["Mission\nefficiency"], [efficiency], color="tab:orange")
        axes[1].set_xlim(0.0, max(100.0, efficiency + 8.0))
        axes[1].set_xlabel("Efficiency (%)")
        axes[1].text(
            0.03,
            0.95,
            f"Losses: {losses:.0f} m/s\nDiagnostic only",
            transform=axes[1].transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.7"},
        )
        _add_panel_labels(axes)
        _save_figure(fig, stem)

    def _render_3d_trajectory(self, env_cfg, timeseries, stem):
        state_hist = timeseries["state"]
        r_hist = state_hist[0:3, :]
        r_hist_km = r_hist / 1000.0
        orbit_hist = _compute_orbit_projection(state_hist, env_cfg)
        orbit_hist_km = None if orbit_hist is None else orbit_hist[0:3, :] / 1000.0
        r_eq = env_cfg.earth_radius_equator
        r_pol = r_eq * (1.0 - env_cfg.earth_flattening)

        fig = plt.figure(figsize=(10.6, 5.6), layout="constrained")
        gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])
        ax = fig.add_subplot(gs[0, 0], projection="3d")

        u = np.linspace(0.0, 2.0 * np.pi, 30)
        v = np.linspace(0.0, np.pi, 30)
        x_earth = (r_eq * np.outer(np.cos(u), np.sin(v))) / 1000.0
        y_earth = (r_eq * np.outer(np.sin(u), np.sin(v))) / 1000.0
        z_earth = (r_pol * np.outer(np.ones_like(u), np.cos(v))) / 1000.0
        ax.plot_wireframe(x_earth, y_earth, z_earth, color="0.75", alpha=0.28, linewidth=0.5)

        ax.plot(r_hist_km[0, :], r_hist_km[1, :], r_hist_km[2, :], color="tab:red", label="Flight path")
        if orbit_hist_km is not None:
            ax.plot(orbit_hist_km[0, :], orbit_hist_km[1, :], orbit_hist_km[2, :], linestyle="--", color="0.25", label="Projected orbit")
        ax.scatter(r_hist_km[0, 0], r_hist_km[1, 0], r_hist_km[2, 0], color="tab:green", s=26, label="Launch")
        ax.scatter(r_hist_km[0, -1], r_hist_km[1, -1], r_hist_km[2, -1], color="black", marker="x", s=34, label="Injection")

        max_val = max(np.max(np.abs(r_hist_km)), r_eq / 1000.0) * 1.05
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.set_zlim(-max_val, max_val)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("ECI X (km)")
        ax.set_ylabel("ECI Y (km)")
        ax.set_zlabel("ECI Z (km)")
        ax.legend(loc="upper left")

        ax2 = fig.add_subplot(gs[0, 1])
        basis = _orbit_plane_basis(state_hist)
        if basis is not None:
            earth_x_km, earth_y_km = _earth_cross_section_in_plane(env_cfg, basis)
            traj_x_km, traj_y_km = _project_to_orbit_plane(r_hist, basis)
            ax2.fill(earth_x_km, earth_y_km, color="0.88", zorder=0)
            ax2.plot(earth_x_km, earth_y_km, color="0.55", linewidth=1.0, zorder=1)
            if orbit_hist is not None:
                orbit_x_km, orbit_y_km = _project_to_orbit_plane(orbit_hist[0:3, :], basis)
                ax2.plot(orbit_x_km, orbit_y_km, linestyle="--", color="0.25", linewidth=1.5, label="Projected orbit")
            ax2.plot(traj_x_km, traj_y_km, color="tab:red", linewidth=2.0, label="Ascent path")
            ax2.scatter(traj_x_km[0], traj_y_km[0], color="tab:green", s=24, zorder=3)
            ax2.scatter(traj_x_km[-1], traj_y_km[-1], color="black", marker="x", s=34, zorder=3)
            ax2.annotate("Launch", (traj_x_km[0], traj_y_km[0]), xytext=(6, -12), textcoords="offset points", fontsize=8, color="0.2")
            ax2.annotate("Injection", (traj_x_km[-1], traj_y_km[-1]), xytext=(-48, -2), textcoords="offset points", fontsize=8, color="0.2")
            ax2.set_xlabel("Orbit-plane x (km)")
            ax2.set_ylabel("Orbit-plane y (km)")
            ax2.set_aspect("equal", adjustable="box")
            ax2.legend(loc="upper left")
        else:
            downrange_km = np.asarray(timeseries["downrange_km"], dtype=float)
            altitude_km = np.asarray(timeseries["ellipsoidal_altitude_km"], dtype=float)
            ax2.plot(downrange_km, altitude_km, color="tab:red", linewidth=2.0)
            ax2.scatter(downrange_km[0], altitude_km[0], color="tab:green", s=24)
            ax2.scatter(downrange_km[-1], altitude_km[-1], color="black", marker="x", s=34)
            ax2.annotate("Launch", (downrange_km[0], altitude_km[0]), xytext=(6, -12), textcoords="offset points", fontsize=8, color="0.2")
            ax2.annotate("Injection", (downrange_km[-1], altitude_km[-1]), xytext=(-48, -2), textcoords="offset points", fontsize=8, color="0.2")
            ax2.set_xlabel("Downrange (km)")
            ax2.set_ylabel("Ellipsoidal altitude (km)")
        _add_panel_labels((ax, ax2))
        _save_figure(fig, stem)

    def _build_launch_cost_heatmap(self):
        print("[PaperPack] Building global launch-cost heatmap...")
        source_root = None
        if self._reliability_suite is not None:
            source_root = Path(self._reliability_suite.output_dir)
        elif self.args.reliability_dir:
            source_root = Path(self.args.reliability_dir)

        source_csv = None if source_root is None else source_root / "data" / "global_launch_cost_latitude_sweep.csv"
        source_png = None if source_root is None else source_root / "figures" / "global_launch_cost_heatmap.png"
        source_pdf = None if source_root is None else source_root / "figures" / "global_launch_cost_heatmap.pdf"

        if source_csv is not None and source_csv.exists() and not self.args.force_heatmap:
            rows = self._load_launch_cost_rows(source_csv)
            if self._render_cached_heatmap_figure(rows, source_root):
                if source_png is not None and source_pdf is not None:
                    return self._copy_heatmap_artifacts(source_csv, source_png, source_pdf)
        output_csv = self.data_dir / "global_launch_cost_latitude_sweep.csv"
        if output_csv.exists() and not self.args.force_heatmap:
            rows = self._load_launch_cost_rows(output_csv)
            if self._render_cached_heatmap_figure(rows, self.output_dir, stem="fig_app_05_global_launch_cost"):
                return self._heatmap_artifacts_exist()

        suite = self._reliability_suite
        if suite is None:
            suite = ReliabilitySuite(
                output_dir=self.output_dir / "_tmp_heatmap_suite",
                save_figures=True,
                show_plots=False,
                analysis_profile="course_core",
            )
        suite.analyze_global_launch_cost(
            lat_step=self.args.heatmap_lat_step,
            solver_max_cpu_time=self.args.heatmap_solver_cpu_time,
        )
        source_root = Path(suite.output_dir)
        return self._copy_heatmap_artifacts(
            source_root / "data" / "global_launch_cost_latitude_sweep.csv",
            source_root / "figures" / "global_launch_cost_heatmap.png",
            source_root / "figures" / "global_launch_cost_heatmap.pdf",
        )

    def _copy_heatmap_artifacts(self, csv_path, png_path, pdf_path):
        required = (csv_path, png_path, pdf_path)
        if not all(path.exists() for path in required):
            return False
        shutil.copy2(csv_path, self.data_dir / "global_launch_cost_latitude_sweep.csv")
        shutil.copy2(png_path, self.figure_dir / "fig_app_05_global_launch_cost.png")
        shutil.copy2(pdf_path, self.figure_dir / "fig_app_05_global_launch_cost.pdf")
        return True

    def _copy_multistart_artifacts(self, reliability_dir):
        source_root = Path(reliability_dir)
        src_csv = source_root / "data" / "randomized_multistart.csv"
        src_deltas = source_root / "data" / "randomized_multistart_deltas.csv"
        src_summary = source_root / "data" / "randomized_multistart_trajectory_summary.csv"
        required = (src_csv, src_deltas, src_summary)
        if not all(path.exists() for path in required):
            return False
        shutil.copy2(src_csv, self.data_dir / "randomized_multistart.csv")
        shutil.copy2(src_deltas, self.data_dir / "randomized_multistart_deltas.csv")
        shutil.copy2(src_summary, self.data_dir / "randomized_multistart_trajectory_summary.csv")
        rows = self._load_multistart_rows(src_csv)
        delta_summary = self._load_multistart_delta_summary(src_deltas)
        summary = _read_metric_value_csv(src_summary)
        self._render_multistart_overview(rows, summary, delta_summary, self.figure_dir / "fig_main_06_warm_start_multiseed")
        return self._multistart_artifacts_exist()

    def _load_multistart_rows(self, csv_path):
        return _read_csv_rows(csv_path)

    def _load_multistart_delta_summary(self, csv_path):
        return self._summarize_multistart_deltas(_read_csv_rows(csv_path))

    @staticmethod
    def _float_from_row(row, key):
        try:
            value = float(row.get(key, np.nan))
        except (TypeError, ValueError):
            return np.nan
        return value

    @staticmethod
    def _int_from_row(row, key):
        try:
            return int(float(row.get(key, 0)))
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _summarize_multistart_deltas(cls, rows):
        summary = {}
        for row in rows:
            label = str(row.get("label", "")).strip()
            if not label:
                continue
            entry = summary.setdefault(
                label,
                {
                    "max_position_separation_m": np.nan,
                    "max_velocity_delta_m_s": np.nan,
                },
            )
            position_sep_km = cls._float_from_row(row, "position_separation_km")
            velocity_delta_m_s = cls._float_from_row(row, "delta_velocity_m_s")
            if np.isfinite(position_sep_km):
                position_sep_m = abs(position_sep_km) * 1000.0
                current = entry["max_position_separation_m"]
                entry["max_position_separation_m"] = position_sep_m if not np.isfinite(current) else max(current, position_sep_m)
            if np.isfinite(velocity_delta_m_s):
                velocity_delta = abs(velocity_delta_m_s)
                current = entry["max_velocity_delta_m_s"]
                entry["max_velocity_delta_m_s"] = velocity_delta if not np.isfinite(current) else max(current, velocity_delta)
        return summary

    @staticmethod
    def _multistart_status_label(status):
        status_key = str(status).strip()
        if status_key == "PASS_RAW":
            return "Raw pass"
        if status_key == "PASS_SLACK":
            return "Slack pass"
        if not status_key:
            return "Unknown"
        return status_key.replace("_", " ").title()

    def _build_multistart_consistency_rows(self, rows, delta_summary):
        if not rows:
            return []
        raw_seed_labels = [str(row.get("label", "seed")) for row in rows]
        seed_labels = WarmStartMultiseedAnalyzer._seed_display_labels(raw_seed_labels)
        table_rows = []
        for display_label, row in zip(seed_labels, rows):
            label = str(row.get("label", ""))
            delta_row = delta_summary.get(label, {})
            table_rows.append(
                (
                    display_label,
                    self._multistart_status_label(row.get("status", "")),
                    _fmt_float(self._float_from_row(row, "final_mass_gap_to_best_kg") * 1.0e6, 3),
                    _fmt_float(float(delta_row.get("max_position_separation_m", np.nan)), 3),
                    _fmt_float(float(delta_row.get("max_velocity_delta_m_s", np.nan)), 4),
                )
            )
        return table_rows

    def _render_multistart_overview(self, rows, summary, delta_summary, stem):
        if not rows:
            return False

        seed_positions = np.arange(len(rows), dtype=float)
        raw_seed_labels = [str(row.get("label", "seed")) for row in rows]
        seed_labels = WarmStartMultiseedAnalyzer._seed_display_labels(raw_seed_labels)
        group_breaks = WarmStartMultiseedAnalyzer._seed_group_break_indices(raw_seed_labels)
        mass_gap_mg = np.array(
            [self._float_from_row(row, "final_mass_gap_to_best_kg") * 1.0e6 for row in rows],
            dtype=float,
        )
        position_sep_m = np.array(
            [float(delta_summary.get(str(row.get("label", "")), {}).get("max_position_separation_m", np.nan)) for row in rows],
            dtype=float,
        )
        velocity_delta_m_s = np.array(
            [float(delta_summary.get(str(row.get("label", "")), {}).get("max_velocity_delta_m_s", np.nan)) for row in rows],
            dtype=float,
        )
        status_colors = [WarmStartMultiseedAnalyzer._outcome_color(row.get("status", "")) for row in rows]
        raw_success_rate = float(summary.get("mission_success_rate_raw", np.nan)) if summary else np.nan

        fig, axes = plt.subplots(1, 3, figsize=(9.4, 4.8), sharey=True, layout="constrained")
        metric_specs = (
            ("Final-mass gap to best (mg)", mass_gap_mg),
            ("Max position separation (m)", position_sep_m),
            ("Max velocity delta (m/s)", velocity_delta_m_s),
        )
        for ax, (xlabel, values) in zip(axes, metric_specs):
            finite_mask = np.isfinite(values)
            if np.any(finite_mask):
                ax.scatter(
                    values[finite_mask],
                    seed_positions[finite_mask],
                    c=[status_colors[idx] for idx in np.flatnonzero(finite_mask)],
                    s=44,
                    edgecolors="0.20",
                    linewidths=0.6,
                    zorder=3,
                )
            for break_idx in group_breaks:
                ax.axhline(break_idx - 0.5, color="0.65", linestyle=":", linewidth=1.0, zorder=0)
            ax.set_xlabel(xlabel)
            ax.set_ylim(-0.6, len(rows) - 0.4)
            _apply_paper_grid(ax, axis="x")
            ax.tick_params(axis="y", length=0)

        positive_mass_gap = mass_gap_mg[mass_gap_mg > 0.0]
        mass_axis = axes[0]
        if positive_mass_gap.size:
            linthresh = max(1.0e-3, float(np.nanmin(positive_mass_gap)) * 0.75)
            mass_axis.set_xscale("symlog", linthresh=linthresh)
            mass_axis.set_xlim(left=0.0, right=max(20.0, float(np.nanmax(positive_mass_gap)) * 1.15))
        else:
            mass_axis.set_xlim(0.0, 1.0)

        for ax, values in zip(axes[1:], (position_sep_m, velocity_delta_m_s)):
            finite_vals = values[np.isfinite(values)]
            max_val = float(np.nanmax(finite_vals)) if finite_vals.size else 1.0
            if not np.isfinite(max_val) or max_val <= 0.0:
                max_val = 1.0
            ax.set_xlim(0.0, 1.12 * max_val)

        axes[0].set_yticks(seed_positions)
        axes[0].set_yticklabels(seed_labels)
        axes[0].invert_yaxis()
        for ax in axes[1:]:
            ax.tick_params(axis="y", labelleft=False)

        if np.isfinite(raw_success_rate):
            mass_axis.text(
                0.99,
                0.02,
                f"Raw-feasible: {int(round(raw_success_rate * len(rows)))}/{len(rows)}",
                transform=mass_axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="0.25",
            )

        _add_panel_labels(axes)
        _save_figure(fig, stem)
        return True

    def _load_launch_cost_rows(self, csv_path):
        rows = []
        for row in _read_csv_rows(csv_path):
            rows.append(
                {
                    "latitude_deg": float(row["latitude_deg"]),
                    "fuel_consumed_kg": float(row["fuel_consumed_kg"]),
                    "final_mass_kg": float(row["final_mass_kg"]),
                    "earth_rotation_speed_m_s": float(row["earth_rotation_speed_m_s"]),
                    "solver_success": int(row["solver_success"]),
                }
            )
        return rows

    def _render_cached_heatmap_figure(self, rows, output_root, stem="global_launch_cost_heatmap"):
        if output_root is None:
            return False
        fake_suite = SimpleNamespace(
            save_outputs=True,
            show_plots=False,
            fig_dir=Path(output_root) / "figures",
        )

        def finalize(fig, stem):
            return ReliabilitySuite._finalize_figure(fake_suite, fig, stem)

        fake_suite._finalize_figure = finalize
        return ReliabilitySuite._render_global_launch_cost_heatmap(fake_suite, rows, stem)

    def _write_notes(self):
        appendix_figures = [
            stem
            for stem in APPENDIX_FIGURES
            if stem != "fig_app_05_global_launch_cost" or self.has_heatmap_outputs
        ]
        manifest_lines = [
            "# Final Paper Pack",
            "",
            "## Main-text figure shortlist",
            "",
        ]
        for stem in MAIN_FIGURES:
            manifest_lines.append(f"- `{stem}`")
        manifest_lines.extend(
            [
                "",
                "## Appendix / supplementary figures",
                "",
            ]
        )
        for stem in appendix_figures:
            manifest_lines.append(f"- `{stem}`")
        manifest_lines.extend(
            [
                "",
                "## Tables",
                "",
                "- `table_01_nominal_case`",
                "- `table_02_verification_summary`",
                "- `table_03_assumptions_method`",
                "- `table_04_replay_drift_by_phase`",
                "- `table_05_multistart_consistency`",
                "",
                "## Figure and table guidance used",
                "",
            ]
        )
        for source in OUTPUT_GUIDANCE_SOURCES:
            manifest_lines.append(f"- {source}")
        manifest_lines.extend(
            [
                "",
                "## Output constraints applied",
                "",
                "- Figure titles are omitted; use report captions instead.",
                "- Multi-panel figures use in-panel a), b), c) labels instead of repeated subplot titles.",
                "- Axis labels include physical units.",
                "- Legends are only used where they add information.",
                "- Perceptually uniform colormaps are preferred over rainbow maps.",
                "- Use 3D views only when spatial geometry is part of the message; otherwise prefer simpler 2D encodings.",
                "- Main-text figures are trimmed to one argument per figure.",
                "- Vector exports keep text editable for downstream publication workflows.",
                "- Tables are exported as editable CSV and Markdown, not as images.",
                "- Tables use explicit quantity/value/unit/notes columns to avoid undefined abbreviations.",
            ]
        )

        with open(self.notes_dir / "paper_pack_manifest.md", "w", encoding="utf-8") as f:
            f.write("\n".join(manifest_lines) + "\n")

        checklist_lines = [
            "# Output Style Checklist",
            "",
            "## Figures",
            "",
            "- Cite every figure in numerical order near the paragraph that uses it.",
            "- Put the explanation in the report caption, not in the figure title.",
            "- Label multi-panel figures in-panel with a), b), c) markers.",
            "- Keep one main argument per figure; move supporting diagnostics to the appendix.",
            "- Keep all axis labels in physical units and define any abbreviations in the caption.",
            "- Reserve 3D views for outputs where spatial context is the point of the figure.",
            "- Prefer editable vector output (`.pdf`) in the report workflow; use `.png` for quick review.",
            "- Keep text editable in vector outputs for journal submission workflows.",
            "",
            "## Tables",
            "",
            "- Keep tables editable; use the generated `.csv` or `.md`, not screenshots.",
            "- Do not repeat a table if the same number can be stated once in the text.",
            "- Keep units in a dedicated column or in the table caption, not implied.",
            "- Define acronyms or non-obvious symbols in the caption or notes column.",
            "",
            "## Sources",
            "",
        ]
        for source in OUTPUT_GUIDANCE_SOURCES:
            checklist_lines.append(f"- {source}")

        with open(self.notes_dir / "output_style_checklist.md", "w", encoding="utf-8") as f:
            f.write("\n".join(checklist_lines) + "\n")


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate a curated output pack for the final paper.")
    parser.add_argument(
        "--output-dir",
        default="paper_outputs/final_paper_pack",
        help="Directory where the curated paper pack will be written.",
    )
    parser.add_argument(
        "--reliability-dir",
        default="",
        help="Reuse an existing reliability output directory instead of running a fresh course-core sweep.",
    )
    parser.add_argument(
        "--skip-heatmap",
        action="store_true",
        help="Skip the expensive global launch-cost sweep.",
    )
    parser.add_argument(
        "--force-heatmap",
        action="store_true",
        help="Recompute the heatmap sweep even if cached latitude-sweep data already exists.",
    )
    parser.add_argument(
        "--heatmap-lat-step",
        type=float,
        default=5.0,
        help="Latitude step in degrees for the global launch-cost sweep.",
    )
    parser.add_argument(
        "--heatmap-solver-cpu-time",
        type=float,
        default=120.0,
        help="Per-latitude CPU time cap for the heatmap optimization solves.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum worker count for the reliability suite.",
    )
    parser.add_argument(
        "--nominal-rtol",
        type=float,
        default=1e-9,
        help="Relative tolerance for the baseline forward replay.",
    )
    parser.add_argument(
        "--nominal-atol",
        type=float,
        default=1e-12,
        help="Absolute tolerance for the baseline forward replay.",
    )
    parser.add_argument(
        "--print-level",
        type=int,
        default=0,
        help="IPOPT print level for the baseline solve.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    builder = PaperPackBuilder(args)
    builder.build()
