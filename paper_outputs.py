import argparse
import contextlib
import copy
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.integrate import solve_ivp

from config import EARTH_CONFIG, StarshipBlock2
from environment import Environment
from main import (
    OPTIMIZATION_G_GUARD_BAND,
    OPTIMIZATION_Q_GUARD_BAND_PA,
    solve_optimal_trajectory,
)
from relabilityanalysis import ReliabilitySuite, evaluate_terminal_state
from simulation import run_simulation
from vehicle import Vehicle


OUTPUT_GUIDANCE_SOURCES = [
    "https://matplotlib.org/stable/users/explain/axes/constrainedlayout_guide.html",
    "https://matplotlib.org/stable/users/explain/colors/colormaps.html",
    "https://matplotlib.org/stable/users/explain/axes/legend_guide.html",
    "https://aiaa.org/publications/journals/Journal-Author/Guidelines-for-Journal-Figures-and-Tables/",
    "https://www.nature.com/palcomms/author-instructions/submission-instructions",
    "https://www.nature.com/dpn/authors-and-referees/artwork-figures-tables",
]

MAIN_FIGURES = [
    "fig_main_01_nominal_profile",
    "fig_main_02_constraints",
    "fig_main_03_grid_independence",
    "fig_main_04_integrator_convergence",
    "fig_main_05_replay_drift",
]

APPENDIX_FIGURES = [
    "fig_app_01_collocation_defect",
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
    "data/collocation_defect_audit.csv",
    "data/run_metadata.csv",
    "data/smooth_integrator_benchmark.csv",
    "data/smooth_integrator_benchmark_fit.csv",
    "data/theoretical_efficiency.csv",
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


def _fmt_float(value, digits=3):
    if value is None or not np.isfinite(value):
        return "nan"
    if abs(value) >= 1e4 or (abs(value) > 0 and abs(value) < 1e-2):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def _setup_plot_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "lines.linewidth": 1.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def _effective_target_inclination_deg(cfg, env_cfg):
    if cfg.target_inclination is None:
        return abs(float(env_cfg.launch_latitude))
    return float(cfg.target_inclination)


def _format_ratio_from_value(value):
    if value == 0:
        return "0"
    return f"1/{(1.0 / float(value)):.3f}"


def _ellipsoidal_altitude_m(position_vector, env_cfg):
    r = np.asarray(position_vector, dtype=float)
    r_sq = float(np.dot(r, r))
    r_mag = np.sqrt(max(r_sq, 1e-16))
    r_eq = float(env_cfg.earth_radius_equator)
    r_pol = r_eq * (1.0 - float(env_cfg.earth_flattening))
    rho_sq = float(r[0] ** 2 + r[1] ** 2)
    z_sq = float(r[2] ** 2)
    denom = np.sqrt((r_pol**2) * rho_sq + (r_eq**2) * z_sq + 1e-16)
    r_local = (r_eq * r_pol * r_mag) / denom
    return r_mag - r_local


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

    alt_km = np.zeros_like(t)
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
        alt_km[i] = _ellipsoidal_altitude_m(r_i, env.config) / 1000.0
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

    downrange_km = _compute_downrange_km(r_hist, env.config, t)
    return {
        "t_s": t,
        "altitude_km": alt_km,
        "velocity_m_s": vel_m_s,
        "mass_kg": mass_hist,
        "dynamic_pressure_kpa": q_kpa,
        "g_load": g_load,
        "mach": mach,
        "downrange_km": downrange_km,
        "throttle": u[0, :],
        "state": y,
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
    alt = np.array([float(row["final_altitude_m"]) for row in ordered], dtype=float)
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
        self.generated_heatmap = False

    def build(self):
        _setup_plot_style()
        self._ensure_dirs()

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
            self.generated_heatmap = self._build_launch_cost_heatmap()
        else:
            print("[PaperPack] Skipping global launch-cost heatmap.")
        self._write_notes()

        print(f"[PaperPack] Completed: {self.output_dir}")

    def _ensure_dirs(self):
        for path in (self.figure_dir, self.table_dir, self.data_dir, self.notes_dir):
            path.mkdir(parents=True, exist_ok=True)

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
        suite.run_all(max_workers=self.args.max_workers)
        return self.raw_reliability_dir

    def _write_time_history_csv(self, opt_res, env_cfg, timeseries):
        t_opt, x_opt = _flatten_optimal_trajectory(opt_res)
        opt_alt_km = np.array([_ellipsoidal_altitude_m(x_opt[0:3, idx], env_cfg) / 1000.0 for idx in range(x_opt.shape[1])], dtype=float)
        opt_mass_kg = x_opt[6, :]
        sim_t = timeseries["t_s"]
        opt_alt_interp = np.interp(sim_t, t_opt, opt_alt_km)
        opt_mass_interp = np.interp(sim_t, t_opt, opt_mass_kg)

        rows = []
        for idx, t_val in enumerate(sim_t):
            rows.append(
                (
                    _fmt_float(t_val, 6),
                    _fmt_float(opt_alt_interp[idx], 6),
                    _fmt_float(timeseries["altitude_km"][idx], 6),
                    _fmt_float(timeseries["velocity_m_s"][idx], 6),
                    _fmt_float(opt_mass_interp[idx], 6),
                    _fmt_float(timeseries["mass_kg"][idx], 6),
                    _fmt_float(timeseries["dynamic_pressure_kpa"][idx], 6),
                    _fmt_float(timeseries["g_load"][idx], 6),
                    _fmt_float(timeseries["downrange_km"][idx], 6),
                    _fmt_float(timeseries["throttle"][idx], 6),
                )
            )

        _save_csv(
            self.data_dir / "nominal_time_history.csv",
            [
                "time_s",
                "optimizer_altitude_km",
                "simulation_altitude_km",
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
        defect_rows = _read_csv_rows(reliability_dir / "data" / "collocation_defect_audit.csv")
        integ_rows = _read_csv_rows(reliability_dir / "data" / "integrator_tolerance.csv")

        suite = ReliabilitySuite(output_dir=self.output_dir / "_tmp_suite", save_figures=False, show_plots=False, analysis_profile="course_core")
        traj_diag = suite._trajectory_diagnostics(sim_res, veh, cfg)
        path_diag = suite._evaluate_path_compliance(traj_diag, cfg, q_slack_pa=0.0, g_slack=0.0)
        terminal = evaluate_terminal_state(sim_res, cfg, env_cfg)
        drift = suite._compute_drift_metrics(opt_res, sim_res)
        grid_summary = _summarize_grid(grid_rows)
        integ_summary = _summarize_integrator(integ_rows)

        max_rel_defect = max(float(row["max_relative_defect"]) for row in defect_rows)
        max_pos_defect = max(float(row["position_defect_m"]) for row in defect_rows)

        nominal_rows = [
            ("Target altitude", _fmt_float(cfg.target_altitude / 1000.0, 3), "km", "Configured circular target"),
            ("Effective target inclination", _fmt_float(_effective_target_inclination_deg(cfg, env_cfg), 3), "deg", "Defaults to |launch latitude|"),
            ("Nodes per powered phase", str(cfg.num_nodes), "-", "Direct-collocation grid"),
            ("Booster burn time", _fmt_float(float(opt_res["T1"]), 3), "s", "Optimized"),
            ("Ship burn time", _fmt_float(float(opt_res["T3"]), 3), "s", "Optimized"),
            ("Final mass", _fmt_float(float(opt_res["X3"][6, -1]), 3), "kg", "Optimizer result"),
            (
                "Propellant margin",
                _fmt_float(float(opt_res["X3"][6, -1] - (cfg.stage_2.dry_mass + cfg.payload_mass)), 3),
                "kg",
                "Final mass above dry+payload",
            ),
            ("Final altitude error", _fmt_float(float(terminal["alt_err_m"]), 3), "m", "Forward replay"),
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
            ("Selected raw-valid grid", str(grid_summary["selected_raw_node"]), "nodes", "Stable converged tail"),
            ("Selected slack-valid grid", str(grid_summary["selected_slack_node"]), "nodes", "Stable converged tail"),
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
            ("Max collocation relative defect", _fmt_float(max_rel_defect, 3), "-", "All intervals"),
            ("Max collocation position defect", _fmt_float(max_pos_defect, 3), "m", "All intervals"),
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
                "Circular target; inclination defaults to |launch latitude| when unspecified",
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
                f"{cfg.num_nodes} nodes per powered phase with piecewise-constant controls",
            ),
            (
                "Path constraints",
                "Altitude, max-Q, max-g, mass, throttle",
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

        plt.close("all")

    def _write_table_files(self, stem, rows):
        headers = ["Quantity", "Value", "Unit", "Notes"]
        csv_rows = [tuple(row) for row in rows]
        _save_csv(self.table_dir / f"{stem}.csv", headers, csv_rows)
        _save_markdown_table(self.table_dir / f"{stem}.md", headers, csv_rows)

    def _render_main_figures(self, cfg, env_cfg, opt_res, sim_res, timeseries, reliability_dir):
        t_opt, x_opt = _flatten_optimal_trajectory(opt_res)
        opt_alt_km = np.array([_ellipsoidal_altitude_m(x_opt[0:3, idx], env_cfg) / 1000.0 for idx in range(x_opt.shape[1])], dtype=float)
        opt_mass_t = x_opt[6, :]

        t_sim = timeseries["t_s"]
        stage_time = float(opt_res["T1"])
        max_q_idx = int(np.argmax(timeseries["dynamic_pressure_kpa"]))

        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6), layout="constrained")
        ax = axes[0, 0]
        ax.plot(t_opt, opt_alt_km, linestyle="--", color="0.35", label="Optimization nodes")
        ax.plot(t_sim, timeseries["altitude_km"], color="tab:blue", label="Forward replay")
        ax.axhline(cfg.target_altitude / 1000.0, color="tab:orange", linestyle=":", label="Target altitude")
        ax.axvline(stage_time, color="0.55", linestyle=":", linewidth=1.0)
        ax.annotate("Staging", (stage_time, 0.98), xycoords=("data", "axes fraction"), xytext=(4, -4), textcoords="offset points", fontsize=8, color="0.35", va="top")
        ax.set_ylabel("Altitude (km)")
        ax.legend(loc="best")

        ax = axes[0, 1]
        ax.plot(t_sim, timeseries["velocity_m_s"], color="tab:green", label="Forward replay")
        ax.axhline(
            np.sqrt(env_cfg.earth_mu / (env_cfg.earth_radius_equator + cfg.target_altitude)),
            color="tab:orange",
            linestyle=":",
            label="Circular-orbit speed",
        )
        ax.axvline(stage_time, color="0.55", linestyle=":", linewidth=1.0)
        ax.set_ylabel("Velocity (m/s)")
        ax.legend(loc="best")

        ax = axes[1, 0]
        ax.plot(t_opt, opt_mass_t / 1000.0, linestyle="--", color="0.35", label="Optimization nodes")
        ax.plot(t_sim, timeseries["mass_kg"] / 1000.0, color="tab:purple", label="Forward replay")
        ax.axvline(stage_time, color="0.55", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mass (t)")
        ax.legend(loc="best")

        ax = axes[1, 1]
        ax.plot(timeseries["downrange_km"], timeseries["altitude_km"], color="tab:blue", linewidth=2.0)
        stage_idx = int(np.abs(t_sim - stage_time).argmin())
        event_specs = [
            ("Launch", 0, "tab:green", (6, -10)),
            ("Max Q", max_q_idx, "tab:orange", (6, 8)),
            ("Staging", stage_idx, "tab:red", (6, -12)),
            ("Injection", len(t_sim) - 1, "black", (-44, -2)),
        ]
        for label, idx, color, offset in event_specs:
            x_event = float(timeseries["downrange_km"][idx])
            y_event = float(timeseries["altitude_km"][idx])
            ax.scatter([x_event], [y_event], color=color, s=20, zorder=3)
            ax.annotate(
                label,
                (x_event, y_event),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                color="0.2" if color == "black" else color,
            )
        ax.set_xlabel("Downrange (km)")
        ax.set_ylabel("Altitude (km)")
        _save_figure(fig, self.figure_dir / "fig_main_01_nominal_profile")

        fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.8), sharex=True, layout="constrained")
        axes[0].plot(t_sim, timeseries["dynamic_pressure_kpa"], color="tab:blue")
        axes[0].axhline(cfg.max_q_limit / 1000.0, color="tab:red", linestyle="--", label="Max-Q limit")
        axes[0].axvline(stage_time, color="0.55", linestyle=":", linewidth=1.0)
        axes[0].annotate("Staging", (stage_time, 0.98), xycoords=("data", "axes fraction"), xytext=(4, -4), textcoords="offset points", fontsize=8, color="0.35", va="top")
        axes[0].set_ylabel("Dynamic pressure (kPa)")
        axes[0].legend(loc="best")

        axes[1].plot(t_sim, timeseries["g_load"], color="tab:orange")
        axes[1].axhline(cfg.max_g_load, color="tab:red", linestyle="--", label="Max-g limit")
        axes[1].axvline(stage_time, color="0.55", linestyle=":", linewidth=1.0)
        axes[1].set_ylabel("G-load (g)")
        axes[1].legend(loc="best")

        axes[2].step(t_sim, timeseries["throttle"], where="post", color="tab:green")
        axes[2].axhline(cfg.sequence.min_throttle, color="0.35", linestyle=":", label="Min throttle")
        axes[2].axvline(stage_time, color="0.55", linestyle=":", linewidth=1.0)
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Throttle (-)")
        axes[2].set_ylim(-0.02, 1.05)
        axes[2].legend(loc="best")
        _save_figure(fig, self.figure_dir / "fig_main_02_constraints")

        self._render_grid_independence(reliability_dir / "data" / "grid_independence.csv", self.figure_dir / "fig_main_03_grid_independence")
        self._render_integrator_convergence(reliability_dir / "data" / "integrator_tolerance.csv", self.figure_dir / "fig_main_04_integrator_convergence")
        self._render_drift(reliability_dir / "data" / "drift_phase_metrics.csv", self.figure_dir / "fig_main_05_replay_drift")

    def _render_appendix_figures(self, cfg, env_cfg, opt_res, sim_res, timeseries, reliability_dir):
        self._render_collocation_defect(reliability_dir / "data" / "collocation_defect_audit.csv", self.figure_dir / "fig_app_01_collocation_defect")
        self._render_smooth_order(
            reliability_dir / "data" / "smooth_integrator_benchmark.csv",
            reliability_dir / "data" / "smooth_integrator_benchmark_fit.csv",
            self.figure_dir / "fig_app_02_smooth_integrator_order",
        )
        self._render_theoretical_efficiency(reliability_dir / "data" / "theoretical_efficiency.csv", self.figure_dir / "fig_app_03_theoretical_efficiency")
        self._render_3d_trajectory(env_cfg, timeseries["state"], self.figure_dir / "fig_app_04_3d_trajectory")

    def _render_grid_independence(self, csv_path, stem):
        rows = _read_csv_rows(csv_path)
        summary = _summarize_grid(rows)
        nodes = summary["nodes"]
        masses = summary["masses"]
        mass_err = summary["mass_err"]

        fig, (ax1, ax2, ax3) = plt.subplots(
            3,
            1,
            figsize=(7.2, 7.4),
            sharex=True,
            layout="constrained",
            gridspec_kw={"height_ratios": [1.15, 1.0, 0.9]},
        )

        ax1.plot(nodes, masses, marker="o", color="tab:blue", label="Final mass")
        ax1.axhline(summary["reference_mass_kg"], color="tab:blue", linestyle="--", alpha=0.55, label=f"Reference (N={summary['reference_node']})")
        ax1.set_ylabel("Final mass (kg)")
        ax1.ticklabel_format(style="plain", axis="y", useOffset=False)
        ax1.legend(loc="best")

        mask = np.arange(len(nodes)) != len(nodes) - 1
        ax2.semilogy(nodes[mask], np.maximum(mass_err[mask], 1e-12), marker="o", color="tab:purple", label="Mass error")
        ax2.axhline(summary["criterion_kg"], color="tab:orange", linestyle="--", label="Criterion")
        ax2.set_ylabel("|m(N)-m_ref| (kg)")
        ax2.legend(loc="best")

        ax3.scatter(nodes, summary["slack_ok"].astype(float) + 0.02, color="tab:green", marker="o", label="Replay-valid (slack)")
        ax3.scatter(nodes, summary["raw_ok"].astype(float) - 0.02, color="tab:purple", marker="x", label="Replay-valid (raw)")
        ax3.set_xlabel("Nodes per powered phase")
        ax3.set_ylabel("Replay validity")
        ax3.set_ylim(-0.1, 1.1)
        ax3.set_yticks([0.0, 1.0], labels=["Invalid", "Valid"])
        ax3.legend(loc="best")
        _save_figure(fig, stem)

    def _render_integrator_convergence(self, csv_path, stem):
        rows = _read_csv_rows(csv_path)
        summary = _summarize_integrator(rows)
        rtol = summary["rtol"]

        fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.4), sharex=True, layout="constrained")
        axes[0].loglog(rtol, np.maximum(summary["altitude_drift_m"], 1e-12), marker="o", color="tab:blue", label="Altitude drift")
        axes[0].axhline(summary["thresholds"]["altitude_m"], color="tab:orange", linestyle="--", label="Criterion")
        axes[0].set_ylabel("Altitude drift (m)")
        axes[0].legend(loc="best")

        axes[1].loglog(rtol, np.maximum(summary["velocity_drift_m_s"], 1e-12), marker="s", color="tab:orange", label="Velocity drift")
        axes[1].axhline(summary["thresholds"]["velocity_m_s"], color="tab:red", linestyle="--", label="Criterion")
        axes[1].set_ylabel("Velocity drift (m/s)")
        axes[1].legend(loc="best")

        axes[2].loglog(rtol, np.maximum(summary["radial_velocity_drift_m_s"], 1e-12), marker="^", color="tab:green", label="Radial velocity drift")
        axes[2].axhline(summary["thresholds"]["radial_velocity_m_s"], color="tab:red", linestyle="--", label="Criterion")
        axes[2].set_xlabel("Integrator relative tolerance (rtol)")
        axes[2].set_ylabel("Radial velocity drift (m/s)")
        axes[2].legend(loc="best")

        for ax in axes:
            ax.invert_xaxis()
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
            (pos, 2000.0, "tab:blue", "Position drift (m)"),
            (vel, 30.0, "tab:green", "Velocity drift (m/s)"),
            (mass, 500.0, "tab:purple", "Mass drift (kg)"),
        ]
        for ax, (vals, threshold, color, ylabel) in zip(axes, panels):
            ax.bar(x, np.maximum(vals, 1e-12), color=color, alpha=0.82)
            ax.axhline(threshold, color="tab:orange", linestyle="--", label="Threshold")
            ax.set_yscale("log")
            ax.set_ylabel(ylabel)
            ax.legend(loc="best")
        axes[-1].set_xticks(x, phases)
        axes[-1].set_xlabel("Mission phase")
        _save_figure(fig, stem)

    def _render_collocation_defect(self, csv_path, stem):
        rows = _read_csv_rows(csv_path)
        pos = np.array([float(row["position_defect_m"]) for row in rows], dtype=float)
        rel = np.array([float(row["max_relative_defect"]) for row in rows], dtype=float)
        phase = np.array([row["phase"] for row in rows])

        global_idx = np.arange(len(rows))
        boost_mask = phase == "boost"
        ship_mask = phase == "ship"

        fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True, layout="constrained")
        axes[0].semilogy(global_idx[boost_mask], np.maximum(pos[boost_mask], 1e-18), color="tab:blue", label="Boost")
        axes[0].semilogy(global_idx[ship_mask], np.maximum(pos[ship_mask], 1e-18), color="tab:orange", label="Ship")
        axes[0].set_ylabel("Position defect (m)")
        axes[0].legend(loc="best")

        axes[1].semilogy(global_idx[boost_mask], np.maximum(rel[boost_mask], 1e-18), color="tab:blue", label="Boost")
        axes[1].semilogy(global_idx[ship_mask], np.maximum(rel[ship_mask], 1e-18), color="tab:orange", label="Ship")
        axes[1].axhline(1e-4, color="tab:orange", linestyle="--", label="Pass threshold")
        axes[1].set_xlabel("Global interval index")
        axes[1].set_ylabel("Max relative defect (-)")
        axes[1].legend(loc="best")
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
        ax.set_xlabel("Time step dt (s)")
        ax.set_ylabel("State error at T")
        ax.legend(loc="best")
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
        _save_figure(fig, stem)

    def _render_3d_trajectory(self, env_cfg, state_hist, stem):
        r_hist = state_hist[0:3, :]
        orbit_hist = _compute_orbit_projection(state_hist, env_cfg)
        r_eq = env_cfg.earth_radius_equator
        r_pol = r_eq * (1.0 - env_cfg.earth_flattening)

        fig = plt.figure(figsize=(7.0, 7.0), layout="constrained")
        ax = fig.add_subplot(111, projection="3d")

        u = np.linspace(0.0, 2.0 * np.pi, 30)
        v = np.linspace(0.0, np.pi, 30)
        x_earth = r_eq * np.outer(np.cos(u), np.sin(v))
        y_earth = r_eq * np.outer(np.sin(u), np.sin(v))
        z_earth = r_pol * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(x_earth, y_earth, z_earth, color="0.75", alpha=0.28, linewidth=0.5)

        ax.plot(r_hist[0, :], r_hist[1, :], r_hist[2, :], color="tab:red", label="Flight path")
        if orbit_hist is not None:
            ax.plot(orbit_hist[0, :], orbit_hist[1, :], orbit_hist[2, :], linestyle="--", color="0.25", label="Projected orbit")
        ax.scatter(r_hist[0, 0], r_hist[1, 0], r_hist[2, 0], color="tab:green", s=26, label="Launch")
        ax.scatter(r_hist[0, -1], r_hist[1, -1], r_hist[2, -1], color="black", marker="x", s=34, label="Injection")

        max_val = max(np.max(np.abs(r_hist)), r_eq) * 1.05
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.set_zlim(-max_val, max_val)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("X (ECI) [m]")
        ax.set_ylabel("Y (ECI) [m]")
        ax.set_zlabel("Z (ECI) [m]")
        ax.legend(loc="upper left")
        _save_figure(fig, stem)

    def _build_launch_cost_heatmap(self):
        print("[PaperPack] Building global launch-cost heatmap...")
        cache_path = self.data_dir / "global_launch_cost_latitude_sweep.csv"
        if cache_path.exists() and not self.args.force_heatmap:
            rows = _read_csv_rows(cache_path)
        else:
            rows = self._run_launch_cost_sweep()
            _save_csv(
                cache_path,
                ["latitude_deg", "fuel_consumed_kg", "final_mass_kg", "earth_rotation_speed_m_s", "solver_success"],
                [
                    (
                        _fmt_float(float(row["latitude_deg"]), 6),
                        _fmt_float(float(row["fuel_consumed_kg"]), 6),
                        _fmt_float(float(row["final_mass_kg"]), 6),
                        _fmt_float(float(row["earth_rotation_speed_m_s"]), 6),
                        int(bool(row["solver_success"])),
                    )
                    for row in rows
                ],
            )
        return self._render_launch_cost_heatmap(rows, self.figure_dir / "fig_app_05_global_launch_cost")

    def _run_launch_cost_sweep(self):
        lat_step = float(self.args.heatmap_lat_step)
        if lat_step <= 0.0:
            raise ValueError("--heatmap-lat-step must be positive.")
        lat_values = np.arange(-90.0, 90.0 + 1e-9, lat_step, dtype=float)
        rows = []

        for lat in lat_values:
            cfg_i = copy.deepcopy(StarshipBlock2)
            env_cfg_i = copy.deepcopy(EARTH_CONFIG)
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
                        solver_max_cpu_time=self.args.heatmap_solver_cpu_time,
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
                f"fuel={_fmt_float(fuel_used, 2)} kg"
            )

        return rows

    def _render_launch_cost_heatmap(self, rows, stem):
        ordered = sorted(rows, key=lambda row: float(row["latitude_deg"]))
        lat_values = np.array([float(row["latitude_deg"]) for row in ordered], dtype=float)
        fuel_values = np.array([float(row["fuel_consumed_kg"]) for row in ordered], dtype=float)
        lon_values = np.linspace(-180.0, 180.0, 73)

        lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
        fuel_grid = np.repeat(fuel_values[:, None], len(lon_values), axis=1)

        rad_lat = np.radians(lat_grid)
        rad_lon = np.radians(lon_grid)
        x = np.cos(rad_lat) * np.cos(rad_lon)
        y = np.cos(rad_lat) * np.sin(rad_lon)
        z = np.sin(rad_lat)

        valid = np.isfinite(fuel_grid)
        if not np.any(valid):
            return False

        norm = Normalize(vmin=np.nanmin(fuel_grid), vmax=np.nanmax(fuel_grid))
        colors = matplotlib.colormaps["cividis"](norm(fuel_grid))

        fig = plt.figure(figsize=(7.0, 6.4), layout="constrained")
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1, linewidth=0.0, antialiased=True, shade=False)
        mappable = cm.ScalarMappable(norm=norm, cmap="cividis")
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.04, pad=0.04)
        cbar.set_label("Fuel consumed (kg)")
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        _save_figure(fig, stem)
        return True

    def _write_notes(self):
        appendix_figures = [
            stem
            for stem in APPENDIX_FIGURES
            if stem != "fig_app_05_global_launch_cost" or self.generated_heatmap
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
                "- Axis labels include physical units.",
                "- Legends are only used where they add information.",
                "- Perceptually uniform colormaps are preferred over rainbow maps.",
                "- Main-text figures are trimmed to one argument per figure.",
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
            "- Keep one main argument per figure; move supporting diagnostics to the appendix.",
            "- Keep all axis labels in physical units and define any abbreviations in the caption.",
            "- Prefer editable vector output (`.pdf`) in the report workflow; use `.png` for quick review.",
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
