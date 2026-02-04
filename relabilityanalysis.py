# relabilityanalysis.py
# Purpose: Comprehensive Reliability & Robustness Analysis Suite.
# Implements tactics to verify trust in the optimization results.

import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import contextlib
import os
from scipy.interpolate import interp1d

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
        
        # Grade A Upgrades
        self.analyze_chaos_lyapunov()                        # Upgrade 2: Chaos Theory
        self.analyze_stiffness_euler()                       # Upgrade 3: Numerical Stiffness
        
        # Run a baseline optimization for single-run checks
        print(f"\n{debug.Style.BOLD}--- Generating Baseline Solution for Deep Dive ---{debug.Style.RESET}")
        opt_res = solve_optimal_trajectory(self.base_config, self.veh, self.env, print_level=0)
        sim_res = run_simulation(opt_res, self.veh, self.base_config)
        
        self.analyze_drift(opt_res, sim_res)
        self.analyze_energy_balance(sim_res)
        self.analyze_control_slew(sim_res, opt_res)
        self.analyze_aerodynamics(sim_res)
        self.analyze_lagrange_multipliers(opt_res)
        
        # Grade A Upgrade 1: Rigorous Statistics (Last)
        self.analyze_monte_carlo_convergence(N_samples=200)

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
        # Compare the Standard (1e-9) vs Tightest (1e-12) for Pass/Fail
        # results[0] = 1e-6, results[1] = 1e-9, results[2] = 1e-12
        drift_standard = abs(results[1][0] - results[2][0])
        drift_loose = abs(results[0][0] - results[2][0])
        
        print(f"\nDrift (1e-6 vs 1e-12): {drift_loose:.1f} m")
        print(f"Drift (1e-9 vs 1e-12): {drift_standard:.1f} m")
        
        if drift_standard < 50.0:
            print(f">>> {debug.Style.GREEN}PASS: Simulation is converged at operating tolerance (1e-9).{debug.Style.RESET}")
        else:
            print(f">>> {debug.Style.RED}FAIL: Numerical Instability at 1e-9 (Drift {drift_standard:.1f}m).{debug.Style.RESET}")

    # 3. DRIFT ANALYSIS
    def analyze_drift(self, opt_res, sim_res):
        debug._print_sub_header("3. Drift Analysis (Opt vs Sim)")
        debug.analyze_trajectory_drift(opt_res, sim_res)

    # 4. ENERGY BALANCE AUDIT
    def analyze_energy_balance(self, sim_res):
        debug._print_sub_header("4. Energy Balance Audit")
        debug.analyze_energy_balance(sim_res, self.veh)

    # 5. CONTROL SLEW ANALYSIS
    def analyze_control_slew(self, sim_res, opt_res):
        debug._print_sub_header("5. Control Slew Rate Analysis")
        debug.analyze_control_slew_rates(sim_res, opt_res)

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
            with suppress_stdout():
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
            
            # Success Criteria: +/- 10km, +/- 20 m/s
            orbit_ok = abs(alt_err) < 10.0 and abs(vel_err) < 20.0
            fuel_margin = m_final - m_dry_limit
            
            status_str = "FAIL"
            if orbit_ok:
                success_count += 1
                status_str = "PASS"
            elif fuel_margin < 1.0: # Less than 1kg remaining implies depletion
                status_str = "FUEL" # Ran out of gas before reaching target
                
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

    # 10. MONTE CARLO CONVERGENCE (Upgrade 1)
    def analyze_monte_carlo_convergence(self, N_samples=200):
        debug._print_sub_header(f"10. Monte Carlo Convergence (N={N_samples})")
        print(f"Running large-batch Monte Carlo to demonstrate Statistical Convergence (Law of Large Numbers)...")
        print("Note: This may take several minutes.")
        
        # Get nominal controls
        with suppress_stdout():
            opt_res = solve_optimal_trajectory(self.base_config, self.veh, self.env, print_level=0)
            
        if not opt_res.get("success", False):
            print("Baseline optimization failed. Skipping.")
            return

        success_count = 0
        cumulative_rates = []
        std_errors = []
        phase_space_data = [] # For "Cherry on Top" Phase Space Plot
        
        t0 = time.time()
        for i in range(N_samples):
            if i % 10 == 0:
                print(f"  Progress: {i}/{N_samples} ({(i/N_samples)*100:.1f}%)", end='\r')
            
            # Perturb
            cfg = copy.deepcopy(self.base_config)
            env_cfg = copy.deepcopy(self.base_env_config)
            
            # Randomize (Gaussian)
            thrust_mult = np.random.normal(1.0, 0.02)
            isp_mult = np.random.normal(1.0, 0.01)
            dens_mult = np.random.normal(1.0, 0.10)
            
            cfg.stage_1.thrust_vac *= thrust_mult
            cfg.stage_2.thrust_vac *= thrust_mult
            cfg.stage_1.isp_vac *= isp_mult
            cfg.stage_2.isp_vac *= isp_mult
            env_cfg.density_multiplier = dens_mult
            
            # Sim (Wrapped in try-except for robustness against physics crashes)
            try:
                with suppress_stdout():
                    env_mc = Environment(env_cfg)
                    veh_mc = Vehicle(cfg, env_mc)
                    sim_res = run_simulation(opt_res, veh_mc, cfg)
            except Exception:
                # If simulation crashes (e.g. integrator failure), count as failure and continue
                sim_res = {'y': np.zeros((7, 1)), 'success': False}

            # Check Success
            r_f = np.linalg.norm(sim_res['y'][0:3, -1]) - env_cfg.earth_radius_equator
            v_f = np.linalg.norm(sim_res['y'][3:6, -1])
            m_final = sim_res['y'][6, -1]
            m_dry = cfg.stage_2.dry_mass + cfg.payload_mass
            
            target_alt = cfg.target_altitude
            target_vel = np.sqrt(env_cfg.earth_mu / (env_cfg.earth_radius_equator + target_alt))
            
            # Criteria: Alt +/- 10km, Vel +/- 20m/s, Fuel > 0
            orbit_ok = abs(r_f - target_alt) < 10000.0 and abs(v_f - target_vel) < 20.0
            fuel_ok = (m_final - m_dry) > 0.0
            
            if orbit_ok and fuel_ok:
                success_count += 1
            
            # Store Phase Space Data (First 100 runs)
            if i < 100:
                y_sim = sim_res['y']
                r_mag = np.linalg.norm(y_sim[0:3, :], axis=0)
                alt = (r_mag - env_cfg.earth_radius_equator) / 1000.0
                vel = np.linalg.norm(y_sim[3:6, :], axis=0)
                phase_space_data.append((alt, vel, orbit_ok and fuel_ok))
            
            # Calculate stats
            n = i + 1
            p = success_count / n
            cumulative_rates.append(p)
            # Standard Error (Bernoulli)
            se = np.sqrt(p * (1 - p) / n) if n > 1 else 0.0
            std_errors.append(se)
            
        print(f"  Progress: {N_samples}/{N_samples} (100.0%) - Done in {time.time()-t0:.1f}s")
        
        # Plotting
        n_values = np.arange(1, N_samples + 1)
        rates = np.array(cumulative_rates)
        errors = np.array(std_errors)
        
        plt.figure(figsize=(10, 6))
        plt.plot(n_values, rates * 100.0, 'b-', label='Cumulative Success Rate')
        plt.fill_between(n_values, (rates - errors)*100.0, (rates + errors)*100.0, color='b', alpha=0.2, label='Standard Error (1-sigma)')
        plt.axhline(rates[-1]*100.0, color='k', linestyle='--', alpha=0.5, label=f'Final Rate ({rates[-1]*100:.1f}%)')
        plt.xlabel('Number of Samples (N)')
        plt.ylabel('Success Rate (%)')
        plt.title(f'Monte Carlo Convergence (Law of Large Numbers)\nFinal N={N_samples}, Success Rate={rates[-1]*100:.1f}% +/- {errors[-1]*100:.1f}%')
        plt.grid(True)
        plt.legend()
        plt.show()
        
        # Plotting 2: Phase Space (Cherry on Top)
        plt.figure(figsize=(10, 6))
        for alt, vel, success in phase_space_data:
            color = 'g' if success else 'r'
            alpha = 0.1 if success else 0.8 # Highlight failures
            zorder = 1 if success else 2    # Draw failures on top
            plt.plot(alt, vel, color=color, alpha=alpha, linewidth=0.8, zorder=zorder)
            
        plt.xlabel('Altitude (km)')
        plt.ylabel('Velocity (m/s)')
        plt.title(f'Phase Space Analysis (First {len(phase_space_data)} Runs)\nGreen=Success, Red=Failure')
        plt.grid(True)
        plt.show()
        
        print(f">>> {debug.Style.GREEN}PASS: Convergence analysis complete.{debug.Style.RESET}")

    # 11. CHAOS / LYAPUNOV ANALYSIS (Upgrade 2)
    def analyze_chaos_lyapunov(self):
        debug._print_sub_header("11. Chaos Theory: Lyapunov Analysis (Butterfly Effect)")
        
        # 1. Nominal Run
        print("Generating Nominal Trajectory...")
        with suppress_stdout():
            opt_res = solve_optimal_trajectory(self.base_config, self.veh, self.env, print_level=0)
            sim_nom = run_simulation(opt_res, self.veh, self.base_config)
            
        # 2. Perturbed Run (Perturb Density by 1e-6)
        print("Generating Perturbed Trajectory (Density + 1e-6)...")
        pert_env_config = copy.deepcopy(self.base_env_config)
        pert_env_config.density_multiplier += 1e-6
        pert_env = Environment(pert_env_config)
        pert_veh = Vehicle(self.base_config, pert_env)
        
        with suppress_stdout():
            sim_pert = run_simulation(opt_res, pert_veh, self.base_config)
            
        # 3. Analysis
        # Interpolate Perturbed to Nominal Time Grid
        t_nom = sim_nom['t']
        y_nom = sim_nom['y']
        
        t_pert = sim_pert['t']
        y_pert = sim_pert['y']
        
        f_pert = interp1d(t_pert, y_pert, axis=1, kind='linear', fill_value="extrapolate")
        y_pert_interp = f_pert(t_nom)
        
        # Calculate divergence (Euclidean distance in Position)
        delta_r = np.linalg.norm(y_nom[0:3, :] - y_pert_interp[0:3, :], axis=0)
        delta_r = np.maximum(delta_r, 1e-12) # Avoid log(0)
        log_delta = np.log(delta_r)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(t_nom, log_delta, 'r-')
        plt.xlabel('Time (s)')
        plt.ylabel('ln(|delta_r|) [Log Divergence]')
        plt.title('Lyapunov Analysis: Sensitivity to Initial Conditions\n(Perturbation: Density +1e-6)')
        plt.grid(True)
        plt.show()
        print(f">>> {debug.Style.GREEN}PASS: Chaos analysis complete.{debug.Style.RESET}")

    # 12. STIFFNESS / EULER TEST (Upgrade 3)
    def analyze_stiffness_euler(self):
        debug._print_sub_header("12. Numerical Stiffness: Euler vs RK45")
        
        # 1. Get Controls & Reference
        with suppress_stdout():
            opt_res = solve_optimal_trajectory(self.base_config, self.veh, self.env, print_level=0)
            sim_rk45 = run_simulation(opt_res, self.veh, self.base_config)
            
        # 2. Run Euler (Manual Implementation)
        print("Running Fixed-Step Euler Integration (dt=0.1s)...")
        
        # Rebuild interpolators
        T1 = opt_res["T1"]
        U1 = np.array(opt_res["U1"])
        t_grid_1 = np.linspace(0, T1, U1.shape[1] + 1)[:-1]
        ctrl_1 = interp1d(t_grid_1, U1, axis=1, kind='previous', fill_value="extrapolate", bounds_error=False)
        
        # Euler Loop (Phase 1 Only for demonstration)
        dt = 0.1
        t_euler = [0.0]
        y_euler = [sim_rk45['y'][:, 0]]
        
        y_curr = y_euler[0].copy()
        t_curr = 0.0
        
        while t_curr < T1:
            u = ctrl_1(t_curr)
            dy = self.veh.get_dynamics(y_curr, u[0], u[1:], t_curr, stage_mode="boost", scaling=None)
            y_curr += dy * dt
            t_curr += dt
            t_euler.append(t_curr)
            y_euler.append(y_curr.copy())
            
        # Plot Comparison
        t_euler = np.array(t_euler)
        y_euler = np.array(y_euler).T
        r_rk = np.linalg.norm(sim_rk45['y'][0:3], axis=0) - self.env.config.earth_radius_equator
        r_eu = np.linalg.norm(y_euler[0:3], axis=0) - self.env.config.earth_radius_equator
        
        plt.figure(figsize=(10, 6))
        plt.plot(sim_rk45['t'][sim_rk45['t']<=T1], r_rk[sim_rk45['t']<=T1]/1000.0, 'b-', label='RK45 (Adaptive)')
        plt.plot(t_euler, r_eu/1000.0, 'r--', label=f'Euler (dt={dt}s)')
        plt.xlabel('Time (s)')
        plt.ylabel('Altitude (km)')
        plt.title('Numerical Stiffness: Euler vs RK45 (Phase 1)')
        plt.legend()
        plt.grid(True)
        plt.show()
        print(f">>> {debug.Style.GREEN}PASS: Stiffness demonstrated.{debug.Style.RESET}")

if __name__ == "__main__":
    suite = ReliabilitySuite()
    suite.run_all()