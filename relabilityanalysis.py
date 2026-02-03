# relabilityanalysis.py
# Purpose: Comprehensive Reliability & Robustness Analysis Suite.
# Implements tactics to verify trust in the optimization results.

import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import contextlib
import os

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

class ReliabilitySuite:
    def __init__(self):
        self.base_config = copy.deepcopy(StarshipBlock2)
        self.base_env_config = copy.deepcopy(EARTH_CONFIG)
        
        # Initialize Baseline Physics
        self.env = Environment(self.base_env_config)
        self.veh = Vehicle(self.base_config, self.env)
        
        print(f"\n{debug.Style.BOLD}=== RELIABILITY ANALYSIS SUITE ==={debug.Style.RESET}")
        print(f"Target: {self.base_config.name}")

    def run_all(self):
        """Runs all analysis modules sequentially."""
        self.analyze_grid_independence()
        self.analyze_integrator_tolerance()
        self.analyze_corner_cases()
        self.analyze_monte_carlo_dispersion()
        
        # Run a baseline optimization for single-run checks
        print(f"\n{debug.Style.BOLD}--- Generating Baseline Solution for Deep Dive ---{debug.Style.RESET}")
        opt_res = solve_optimal_trajectory(self.base_config, self.veh, self.env, print_level=0)
        sim_res = run_simulation(opt_res, self.veh, self.base_config)
        
        self.analyze_drift(opt_res, sim_res)
        self.analyze_energy_balance(sim_res)
        self.analyze_control_slew(sim_res)
        self.analyze_aerodynamics(sim_res)
        self.analyze_lagrange_multipliers(opt_res)

    # 1. GRID INDEPENDENCE STUDY
    def analyze_grid_independence(self):
        debug._print_sub_header("1. Grid Independence Study")
        # User limit: Max 140 nodes
        node_counts = [40, 70, 100, 140]
        masses = []
        runtimes = []
        
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
                else:
                    raise Exception("Solver failed")
            except:
                m_final = 0.0
                status = "Failed"
            
            dt = time.time() - t0
            masses.append(m_final)
            runtimes.append(dt)
            print(f"{N:<6} | {m_final:<15.1f} | {dt:<10.2f} | {status:<10}")
            
        # Convergence Check
        if len(masses) >= 2:
            diff = abs(masses[-1] - masses[-2])
            print(f"\nMass Delta (140 vs 100 nodes): {diff:.1f} kg")
            if diff < 100.0:
                print(f">>> {debug.Style.GREEN}PASS: Grid Independent (<100kg change).{debug.Style.RESET}")
            else:
                print(f">>> {debug.Style.YELLOW}WARN: Grid Dependent (Solution still changing).{debug.Style.RESET}")

    # 2. INTEGRATOR TOLERANCE SWEEP
    def analyze_integrator_tolerance(self):
        debug._print_sub_header("2. Integrator Tolerance Sweep")
        
        # Get a solution first
        with suppress_stdout():
            opt_res = solve_optimal_trajectory(self.base_config, self.veh, self.env, print_level=0)
            
        tols = [(1e-6, 1e-9), (1e-9, 1e-12), (1e-12, 1e-14)]
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
            
        # Check consistency
        alt_diff = abs(results[-1][0] - results[0][0])
        if alt_diff < 50.0:
            print(f"\n>>> {debug.Style.GREEN}PASS: Numerical Stability Confirmed (Drift < 50m).{debug.Style.RESET}")
        else:
            print(f"\n>>> {debug.Style.RED}FAIL: Numerical Instability (Drift {alt_diff:.1f}m).{debug.Style.RESET}")

    # 3. DRIFT ANALYSIS
    def analyze_drift(self, opt_res, sim_res):
        debug._print_sub_header("3. Drift Analysis (Opt vs Sim)")
        debug.analyze_trajectory_drift(opt_res, sim_res)

    # 4. ENERGY BALANCE AUDIT
    def analyze_energy_balance(self, sim_res):
        debug._print_sub_header("4. Energy Balance Audit")
        debug.analyze_energy_balance(sim_res, self.veh)

    # 5. CONTROL SLEW ANALYSIS
    def analyze_control_slew(self, sim_res):
        debug._print_sub_header("5. Control Slew Rate Analysis")
        debug.analyze_control_slew_rates(sim_res)

    # 6. AERODYNAMIC ANGLE CHECK
    def analyze_aerodynamics(self, sim_res):
        debug._print_sub_header("6. Aerodynamic Angle Check (Max Q)")
        # Extract Max Q region
        debug.analyze_instantaneous_orbit(sim_res, self.env) # This prints orbit table
        
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
        
        if aoa_at_max_q < 4.0:
            print(f">>> {debug.Style.GREEN}PASS: Low aerodynamic stress at Max Q.{debug.Style.RESET}")
        else:
            print(f">>> {debug.Style.RED}FAIL: High AoA at Max Q (>4 deg). Structural risk.{debug.Style.RESET}")

    # 7. MONTE CARLO DISPERSION
    def analyze_monte_carlo_dispersion(self):
        debug._print_sub_header("7. Monte Carlo Dispersion Analysis (N=20)")
        
        # Get nominal controls
        with suppress_stdout():
            opt_res = solve_optimal_trajectory(self.base_config, self.veh, self.env, print_level=0)
            
        if not opt_res.get("success", False):
            print(f">>> {debug.Style.RED}CRITICAL: Baseline optimization failed. Skipping Monte Carlo.{debug.Style.RESET}")
            return

        N_runs = 20
        success_count = 0
        alt_errors = []
        vel_errors = []
        
        print(f"{'Run':<4} | {'Thrust':<8} | {'ISP':<8} | {'Dens':<8} | {'Alt Err (km)':<12} | {'Vel Err (m/s)':<12} | {'Status':<6}")
        print("-" * 75)
        
        for i in range(N_runs):
            # Perturb Parameters
            cfg = copy.deepcopy(self.base_config)
            env_cfg = copy.deepcopy(self.base_env_config)
            
            # Perturbations (Gaussian)
            thrust_mult = np.random.normal(1.0, 0.02) # 2% sigma
            isp_mult = np.random.normal(1.0, 0.01)    # 1% sigma
            dens_mult = np.random.normal(1.0, 0.10)   # 10% sigma
            
            # Apply to Config
            cfg.stage_1.thrust_vac *= thrust_mult
            cfg.stage_2.thrust_vac *= thrust_mult
            cfg.stage_1.isp_vac *= isp_mult
            cfg.stage_2.isp_vac *= isp_mult
            env_cfg.density_multiplier = dens_mult
            
            # Re-init Physics
            env_mc = Environment(env_cfg)
            veh_mc = Vehicle(cfg, env_mc)
            
            # Run Sim
            with suppress_stdout():
                sim_res = run_simulation(opt_res, veh_mc, cfg)
                
            # Check Orbit
            r_f = np.linalg.norm(sim_res['y'][0:3, -1]) - env_cfg.earth_radius_equator
            v_f = np.linalg.norm(sim_res['y'][3:6, -1])
            m_final = sim_res['y'][6, -1]
            m_dry_limit = cfg.stage_2.dry_mass + cfg.payload_mass
            
            target_alt = cfg.target_altitude
            target_vel = np.sqrt(env_cfg.earth_mu / (env_cfg.earth_radius_equator + target_alt))
            
            alt_err = (r_f - target_alt) / 1000.0
            vel_err = v_f - target_vel
            
            alt_errors.append(alt_err)
            vel_errors.append(vel_err)
            
            # Success Criteria: +/- 10km, +/- 20 m/s AND Positive Fuel Margin
            fuel_margin = m_final - m_dry_limit
            
            status_str = "FAIL"
            if fuel_margin < 0:
                status_str = "FUEL" # Ran out of gas
            elif abs(alt_err) < 10.0 and abs(vel_err) < 20.0:
                success_count += 1
                status_str = "PASS"
                
            print(f"{i+1:<4} | {thrust_mult:<8.3f} | {isp_mult:<8.3f} | {dens_mult:<8.3f} | {alt_err:<12.1f} | {vel_err:<12.1f} | {status_str:<6}")
            
        print(f"\nRobustness: {success_count}/{N_runs} ({(success_count/N_runs)*100:.0f}%)")
        print(f"Alt Dispersion (Sigma): {np.std(alt_errors):.2f} km")
        print(f"Vel Dispersion (Sigma): {np.std(vel_errors):.2f} m/s")

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
            
            m_final = res['X3'][6, -1]
            m_dry = wc_config.stage_2.dry_mass + wc_config.payload_mass
            margin = m_final - m_dry
            
            print(f"  Worst Case Fuel Margin: {margin:.1f} kg")
            
            if margin > 0:
                print(f">>> {debug.Style.GREEN}PASS: Mission feasible in worst case.{debug.Style.RESET}")
            else:
                print(f">>> {debug.Style.RED}FAIL: Mission fails in worst case (Negative Margin).{debug.Style.RESET}")
                
        except Exception as e:
            print(f">>> {debug.Style.RED}FAIL: Optimizer crashed on worst case.{debug.Style.RESET}")

if __name__ == "__main__":
    suite = ReliabilitySuite()
    suite.run_all()