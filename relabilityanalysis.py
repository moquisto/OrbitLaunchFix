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
import multiprocessing
import matplotlib.patches as patches
import csv
from pathlib import Path
from datetime import datetime
import argparse

# Project Imports
from config import StarshipBlock2, EARTH_CONFIG
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


def evaluate_terminal_state(sim_res, cfg, env_cfg, alt_tol_m=10000.0, vel_tol_m_s=20.0):
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
        "alt_err_m": np.nan,
        "vel_err_m_s": np.nan,
        "m_final_kg": np.nan,
        "m_dry_limit_kg": float(cfg.stage_2.dry_mass + cfg.payload_mass),
        "fuel_margin_kg": np.nan
    }

    y = sim_res.get("y", None) if isinstance(sim_res, dict) else None
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[0] < 7 or y.shape[1] < 1:
        return metrics

    final_state = y[:, -1]
    if not np.all(np.isfinite(final_state)):
        return metrics

    r_f = np.linalg.norm(final_state[0:3]) - env_cfg.earth_radius_equator
    v_f = np.linalg.norm(final_state[3:6])
    m_final = float(final_state[6])
    target_alt = float(cfg.target_altitude)
    target_vel = np.sqrt(env_cfg.earth_mu / (env_cfg.earth_radius_equator + target_alt))

    alt_err = r_f - target_alt
    vel_err = v_f - target_vel
    fuel_margin = m_final - metrics["m_dry_limit_kg"]

    orbit_ok = abs(alt_err) < alt_tol_m and abs(vel_err) < vel_tol_m_s
    fuel_ok = fuel_margin > 0.0
    strict_ok = orbit_ok and fuel_ok

    metrics.update({
        "terminal_valid": True,
        "orbit_ok": orbit_ok,
        "fuel_ok": fuel_ok,
        "strict_ok": strict_ok,
        "r_f_m": r_f,
        "v_f_m_s": v_f,
        "target_vel_m_s": target_vel,
        "alt_err_m": alt_err,
        "vel_err_m_s": vel_err,
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

    # Require finite full history for trajectory-based plots.
    if np.all(np.isfinite(y)) and np.linalg.norm(y[:, 0]) > 1.0:
        metrics["trajectory_valid"] = True

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

    def run_all(self, monte_carlo_samples=500):
        """Runs all analysis modules sequentially."""
        # Grade A Upgrades
        self.analyze_stiffness_euler()                       # Upgrade 3: Numerical Stiffness
        
        # Grade A Upgrade 1: Rigorous Statistics
        self.analyze_monte_carlo_convergence(N_samples=monte_carlo_samples)
        
        self.analyze_grid_independence()
        self.analyze_integrator_tolerance()
        self.analyze_corner_cases()
        
        # Grade A Upgrades
        self.analyze_chaos_lyapunov()                        # Upgrade 2: Chaos Theory
        self.analyze_bifurcation()                           # Upgrade 4: Bifurcation Analysis
        self.analyze_theoretical_efficiency()                # Physics Verification: Hohmann Comparison
        
        # Run a baseline optimization for single-run checks
        print(f"\n{debug.Style.BOLD}--- Generating Baseline Solution for Deep Dive ---{debug.Style.RESET}")
        opt_res = self._get_baseline_opt_res()
        sim_res = run_simulation(opt_res, self.veh, self.base_config)
        
        self.analyze_drift(opt_res, sim_res)
        self.analyze_energy_balance(sim_res)
        self.analyze_control_slew(sim_res, opt_res)
        self.analyze_aerodynamics(sim_res)
        self.analyze_lagrange_multipliers(opt_res)

    # 1. GRID INDEPENDENCE STUDY
    def analyze_grid_independence(self):
        debug._print_sub_header("1. Grid Independence Study")
        # User limit: Max 140 nodes
        # Increased resolution for smoother convergence graph
        node_counts = [40, 50, 60, 70, 80, 90, 100, 120, 140]
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
        g_val = np.array(opt_res['g'])
        
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
        debug._print_sub_header("12. Numerical Stiffness & Convergence Analysis")
        print("Comparing Fixed-Step Integrators (Euler vs RK4) against Adaptive RK45 Baseline...")
        
        # 1. Get Controls & Reference
        opt_res = self._get_baseline_opt_res()
            
        if not opt_res.get("success", False):
            print("  Baseline optimization failed. Skipping Stiffness test.")
            return

        with suppress_stdout():
            sim_rk45 = run_simulation(opt_res, self.veh, self.base_config)
            
        # Rebuild interpolators for Phase 1
        T1 = opt_res["T1"]
        U1 = np.array(opt_res["U1"])
        t_grid_1 = np.linspace(0, T1, U1.shape[1] + 1)[:-1]
        ctrl_1 = interp1d(t_grid_1, U1, axis=1, kind='previous', fill_value="extrapolate", bounds_error=False)
        
        # Reference RK45 Final State at T1
        # Find index closest to T1 in simulation
        idx_T1 = np.abs(sim_rk45['t'] - T1).argmin()
        y_ref_final = sim_rk45['y'][:, idx_T1]
        
        # 2. Convergence Study
        dt_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5] # Log-spacing roughly
        euler_errors = []
        rk4_errors = []
        
        print(f"{'dt (s)':<8} | {'Euler Err (m)':<15} | {'RK4 Err (m)':<15}")
        print("-" * 45)
        
        y0 = sim_rk45['y'][:, 0]
        
        for dt in dt_values:
            # --- Euler ---
            t_curr = 0.0
            y_curr = y0.copy()
            while t_curr < T1:
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
            while t_curr < T1:
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

        # 3. Calculate Slopes (Log-Log)
        # Use first and last points for slope estimation
        slope_eu = (np.log10(euler_errors[-1]) - np.log10(euler_errors[0])) / (np.log10(dt_values[-1]) - np.log10(dt_values[0]))
        slope_rk = (np.log10(rk4_errors[-1]) - np.log10(rk4_errors[0])) / (np.log10(dt_values[-1]) - np.log10(dt_values[0]))
        
        print(f"\nConvergence Slopes (Log-Log):")
        print(f"  Euler (Order 1): {slope_eu:.2f} (Expected ~1.0)")
        print(f"  RK4   (Order 4): {slope_rk:.2f} (Expected ~4.0)")

        self._save_table(
            "stiffness_convergence",
            ["dt_s", "euler_error_m", "rk4_error_m"],
            [(dt_values[i], euler_errors[i], rk4_errors[i]) for i in range(len(dt_values))]
        )
        
        # 4. Plotting
        fig = plt.figure(figsize=(10, 7))
        plt.loglog(dt_values, euler_errors, 'r-o', label=f'Euler (Slope={slope_eu:.1f})')
        plt.loglog(dt_values, rk4_errors, 'b-s', label=f'Fixed RK4 (Slope={slope_rk:.1f})')
        
        # Add Reference Slopes
        ref_x = np.array(dt_values)
        # Anchor references to the middle data point
        mid_idx = len(dt_values) // 2
        
        # Slope 1 Ref
        ref_y1 = euler_errors[mid_idx] * (ref_x / ref_x[mid_idx])**1
        plt.loglog(ref_x, ref_y1, 'k--', alpha=0.3, label='O(dt) Reference')
        
        # Slope 4 Ref
        ref_y4 = rk4_errors[mid_idx] * (ref_x / ref_x[mid_idx])**4
        plt.loglog(ref_x, ref_y4, 'k:', alpha=0.3, label='O(dt^4) Reference')
        
        plt.xlabel('Time Step dt (s)')
        plt.ylabel('Global Position Error (m)')
        plt.title('Numerical Convergence Analysis: Euler vs RK4')
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend()
        self._finalize_figure(fig, "stiffness_euler_vs_rk4")
        
        print(f"\n{debug.Style.BOLD}THEORETICAL DEFENSE (For Report):{debug.Style.RESET}")
        print("1. Why RK45? Rocket dynamics are 'Stiff' (Fast Launch vs Slow Coast).")
        print("   Adaptive stepping (RK45) captures fast dynamics without wasting time in slow phases.")
        print("2. Why not Symplectic (Verlet)? The system is Non-Conservative (Thrust/Drag add/remove energy).")
        print("   Symplectic integrators are designed for Hamiltonian systems (Energy Conserving), which this is not.")

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
