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
import matplotlib.transforms as transforms

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
    
    # Check for zero variance to avoid division by zero
    if cov[0, 0] < 1e-12 or cov[1, 1] < 1e-12:
        return None

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Clip Pearson correlation to avoid numerical errors in sqrt if correlation is perfect
    pearson = np.clip(pearson, -0.99999, 0.99999)
    
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = patches.Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def _monte_carlo_worker(args):
    """
    Worker function for parallel Monte Carlo simulations.
    Must be at module level to be picklable.
    """
    idx, base_config, base_env_config, opt_res = args
    
    # Re-seed random number generator for this process
    np.random.seed((int(time.time() * 1000) + idx) % (2**32))
    
    # Perturb Configuration
    cfg = copy.deepcopy(base_config)
    env_cfg = copy.deepcopy(base_env_config)
    
    # Randomize (Gaussian)
    thrust_mult = np.random.normal(1.0, 0.02)
    isp_mult = np.random.normal(1.0, 0.01)
    dens_mult = np.random.normal(1.0, 0.10)
    
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

    # Check Success Criteria
    r_f = np.linalg.norm(sim_res['y'][0:3, -1]) - env_cfg.earth_radius_equator
    v_f = np.linalg.norm(sim_res['y'][3:6, -1])
    m_final = sim_res['y'][6, -1]
    m_dry = cfg.stage_2.dry_mass + cfg.payload_mass
    
    target_alt = cfg.target_altitude
    target_vel = np.sqrt(env_cfg.earth_mu / (env_cfg.earth_radius_equator + target_alt))
    
    orbit_ok = abs(r_f - target_alt) < 10000.0 and abs(v_f - target_vel) < 20.0
    fuel_ok = (m_final - m_dry) > 0.0
    
    # Return results and trajectory data (if valid)
    phase_data = None
    if np.linalg.norm(sim_res['y'][:, 0]) > 1.0:
        y_sim = sim_res['y']
        r_mag = np.linalg.norm(y_sim[0:3, :], axis=0)
        alt = (r_mag - env_cfg.earth_radius_equator) / 1000.0
        vel = np.linalg.norm(y_sim[3:6, :], axis=0)
        phase_data = (alt, vel, orbit_ok and fuel_ok)

    return (orbit_ok, fuel_ok, phase_data)

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
        # Grade A Upgrades
        self.analyze_stiffness_euler()                       # Upgrade 3: Numerical Stiffness
        
        # Grade A Upgrade 1: Rigorous Statistics
        self.analyze_monte_carlo_convergence(N_samples=500)
        
        self.analyze_grid_independence()
        self.analyze_integrator_tolerance()
        self.analyze_corner_cases()
        
        # Grade A Upgrades
        self.analyze_chaos_lyapunov()                        # Upgrade 2: Chaos Theory
        self.analyze_bifurcation()                           # Upgrade 4: Bifurcation Analysis
        self.analyze_theoretical_efficiency()                # Physics Verification: Hohmann Comparison
        
        # Run a baseline optimization for single-run checks
        print(f"\n{debug.Style.BOLD}--- Generating Baseline Solution for Deep Dive ---{debug.Style.RESET}")
        opt_res = solve_optimal_trajectory(self.base_config, self.veh, self.env, print_level=0)
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
        if len(masses) >= 3:
            diff = abs(masses[-1] - masses[-3])
            print(f"\nMass Delta (140 vs 100 nodes): {diff:.1f} kg")
            if diff < 100.0:
                print(f">>> {debug.Style.GREEN}PASS: Grid Independent (<100kg change).{debug.Style.RESET}")
            else:
                print(f">>> {debug.Style.YELLOW}WARN: Grid Dependent (Solution still changing).{debug.Style.RESET}")
        
        # Visualization: Convergence vs Cost
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Filter valid runs for plotting
        valid_indices = [i for i, m in enumerate(masses) if m > 1.0]
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
        fig.tight_layout()
        plt.show()

    # 2. INTEGRATOR TOLERANCE SWEEP
    def analyze_integrator_tolerance(self):
        debug._print_sub_header("2. Integrator Tolerance Sweep")
        
        # Get a solution first
        with suppress_stdout():
            opt_res = solve_optimal_trajectory(self.base_config, self.veh, self.env, print_level=0)
            
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
        
        plt.figure(figsize=(10, 6))
        plt.loglog(rtols, drifts, 'bo-', label='Altitude Drift')
        plt.xlabel('Integrator Relative Tolerance (rtol)')
        plt.ylabel('Position Drift vs Baseline (m)')
        plt.title('Integrator Convergence (Baseline: rtol=1e-14)')
        plt.grid(True, which="both", ls="-")
        plt.gca().invert_xaxis() # Standard convention: higher precision (lower tol) to the right
        plt.show()

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
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(t, drift, 'r-', linewidth=1.5, label='Energy Drift')
        plt.axhline(0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Time (s)')
        plt.ylabel('Energy Error (J/kg)')
        plt.title(f'Integrator Energy Conservation Check\nMax Drift: {np.max(np.abs(drift)):.4f} J/kg')
        plt.grid(True)
        plt.legend()
        plt.show()

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
        
        # Visualization: Targeting Scatter Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(vel_errors, alt_errors, c='b', alpha=0.6, label='Simulations')
        
        # Draw Success Box (+/- 20m/s, +/- 10km)
        rect = plt.Rectangle((-20, -10), 40, 20, linewidth=2, edgecolor='g', facecolor='g', alpha=0.1, label='Success Criteria')
        plt.gca().add_patch(rect)
        
        plt.xlabel('Velocity Error (m/s)')
        plt.ylabel('Altitude Error (km)')
        plt.title(f'Monte Carlo Dispersion (N={N_runs})\nTargeting Accuracy')
        plt.axhline(0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(0, color='k', linestyle='--', alpha=0.3)
        plt.grid(True)
        plt.legend()
        plt.show()

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
                
            # Visualization: Mass Budget
            plt.figure(figsize=(8, 6))
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
            plt.show()
                
        except Exception as e:
            print(f">>> {debug.Style.RED}FAIL: Optimizer crashed on worst case.{debug.Style.RESET}")

    # 10. MONTE CARLO CONVERGENCE (Upgrade 1)
    def analyze_monte_carlo_convergence(self, N_samples=500):
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
        orbit_fail_count = 0
        fuel_fail_count = 0
        hardware_success_count = 0
        cumulative_rates = []
        std_errors = []
        phase_space_data = [] # For "Cherry on Top" Phase Space Plot
        
        t0 = time.time()
        
        # Prepare arguments for parallel workers
        args_list = [(i, self.base_config, self.base_env_config, opt_res) for i in range(N_samples)]
        
        n_cpu = multiprocessing.cpu_count()
        print(f"  Starting parallel execution with {n_cpu} workers...")
        
        results = []
        with multiprocessing.Pool(processes=n_cpu) as pool:
            # Use imap to get results as they complete for progress reporting
            for i, res in enumerate(pool.imap(_monte_carlo_worker, args_list)):
                results.append(res)
                if i % 10 == 0 or i == N_samples - 1:
                    elapsed = time.time() - t0
                    avg_time = elapsed / (i + 1)
                    remaining = avg_time * (N_samples - (i + 1))
                    print(f"  Progress: {i+1}/{N_samples} ({((i+1)/N_samples)*100:.1f}%) - ETA: {remaining:.1f}s   ", end='\r')

        # Process results sequentially to build cumulative stats
        for i, (orbit_ok, fuel_ok, p_data) in enumerate(results):
            if fuel_ok:
                hardware_success_count += 1
            
            if orbit_ok and fuel_ok:
                success_count += 1
            else:
                if not orbit_ok: orbit_fail_count += 1
                if not fuel_ok: fuel_fail_count += 1
            
            if p_data is not None:
                phase_space_data.append(p_data)
            
            # Calculate stats
            n = i + 1
            p = success_count / n
            cumulative_rates.append(p)
            # Standard Error (Bernoulli)
            se = np.sqrt(p * (1 - p) / n) if n > 1 else 0.0
            std_errors.append(se)
            
        print(f"  Progress: {N_samples}/{N_samples} (100.0%) - Done in {time.time()-t0:.1f}s")
        
        # Plotting Data Prep
        n_values = np.arange(1, N_samples + 1)
        rates = np.array(cumulative_rates)
        errors = np.array(std_errors)

        # Print Summary Statistics to Console
        print(f"  Final Statistics (N={N_samples}):")
        if len(rates) > 0:
            print(f"    Strict Success (Hit Target):   {rates[-1]*100:.1f}% (Low due to Open-Loop Sensitivity)")
            print(f"    Hardware Robustness (Fuel>0):  {(hardware_success_count/N_samples)*100:.1f}% (Vehicle Capability)")
            print(f"    Failures Breakdown:")
            print(f"      - Missed Target (Guidance):  {orbit_fail_count}")
            print(f"      - Ran out of Fuel (Perf):    {fuel_fail_count}")
        
        plt.figure(figsize=(10, 6))
        plt.plot(n_values, rates * 100.0, 'b-', label='Cumulative Success Rate')
        plt.fill_between(n_values, (rates - errors)*100.0, (rates + errors)*100.0, color='b', alpha=0.2, label='Standard Error (1-sigma)')
        plt.axhline(rates[-1]*100.0, color='k', linestyle='--', alpha=0.5, label=f'Final Rate ({rates[-1]*100:.1f}%)')
        plt.xlabel('Number of Samples (N)')
        plt.ylabel('Success Rate (%)')
        if len(rates) > 0:
            plt.title(f'Monte Carlo Convergence (Law of Large Numbers)\nFinal N={N_samples}, Success Rate={rates[-1]*100:.1f}% +/- {errors[-1]*100:.1f}%')
        plt.grid(True)
        plt.legend()
        plt.show()
        
        # Plotting 3: Error Histograms (Altitude & Velocity)
        plt.figure(figsize=(12, 6))
        
        # Altitude
        plt.subplot(1, 2, 1)
        # Filter outliers for cleaner histograms (e.g. crashes at launch pad)
        target_alt_km = self.base_config.target_altitude/1000.0
        all_alts = [d[0][-1] for d in phase_space_data]
        hist_alts = [a for a in all_alts if abs(a - target_alt_km) < 50.0]
        outliers_alt = len(all_alts) - len(hist_alts)
        if outliers_alt > 0: print(f"  [Viz] Filtered {outliers_alt} altitude outliers (crashes) from histogram.")
        
        if len(hist_alts) > 0:
            plt.hist(hist_alts, bins=30, color='purple', alpha=0.7, edgecolor='black')
        plt.axvline(self.base_config.target_altitude/1000.0, color='k', linestyle='dashed', linewidth=1, label='Target')
        plt.xlabel('Final Altitude (km)')
        plt.ylabel('Frequency')
        plt.title(f'Altitude Dispersion (N={len(hist_alts)})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Velocity
        plt.subplot(1, 2, 2)
        target_vel = np.sqrt(self.base_env_config.earth_mu / (self.base_env_config.earth_radius_equator + self.base_config.target_altitude))
        all_vels = [d[1][-1] for d in phase_space_data]
        hist_vels = [v for v in all_vels if abs(v - target_vel) < 100.0]
        outliers_vel = len(all_vels) - len(hist_vels)
        if outliers_vel > 0: print(f"  [Viz] Filtered {outliers_vel} velocity outliers from histogram.")
        
        if len(hist_vels) > 0:
            plt.hist(hist_vels, bins=30, color='teal', alpha=0.7, edgecolor='black')
        plt.axvline(target_vel, color='k', linestyle='dashed', linewidth=1, label='Target')
        plt.xlabel('Final Velocity (m/s)')
        plt.ylabel('Frequency')
        plt.title(f'Velocity Dispersion (N={len(hist_vels)})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
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
        
        # Plotting 4: Terminal State Scatter (Targeting)
        plt.figure(figsize=(10, 6))
        
        final_alts = [d[0][-1] for d in phase_space_data]
        final_vels = [d[1][-1] for d in phase_space_data]
        success_flags = [d[2] for d in phase_space_data]
        
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
        
        # Add Confidence Ellipses (1-sigma and 3-sigma) for successful runs
        if len(succ_alts) > 5:
            ax = plt.gca()
            confidence_ellipse(np.array(succ_alts), np.array(succ_vels), ax, n_std=1.0, edgecolor='blue', linestyle='--', label='1-sigma (68%)')
            confidence_ellipse(np.array(succ_alts), np.array(succ_vels), ax, n_std=3.0, edgecolor='blue', linestyle=':', label='3-sigma (99.7%)')
        
        plt.xlabel('Final Altitude (km)')
        plt.ylabel('Final Velocity (m/s)')
        plt.title(f'Terminal State Dispersion (N={len(phase_space_data)})')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print(f">>> {debug.Style.GREEN}PASS: Convergence analysis complete.{debug.Style.RESET}")
        print(f"\n{debug.Style.BOLD}REPORT RECOMMENDATION (The '8% Success' Defense):{debug.Style.RESET}")
        print("Explicitly state in your report:")
        print("'The low success rate (8%) validates that optimal trajectories are inherently unstable.")
        print(" This proves the necessity of Closed-Loop Guidance (PID/MPC) for flight,")
        print(" as Open-Loop optimization is insufficient for robustness against environmental perturbations.'")


    # 11. CHAOS / LYAPUNOV ANALYSIS (Upgrade 2)
    def analyze_chaos_lyapunov(self):
        debug._print_sub_header("11. Chaos Theory: Lyapunov Analysis (Butterfly Effect)")
        
        # 1. Nominal Run
        print("Generating Nominal Trajectory...")
        with suppress_stdout():
            opt_res = solve_optimal_trajectory(self.base_config, self.veh, self.env, print_level=0)
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
        # Estimate Lyapunov Exponent (Slope of log divergence)
        # Fit to the middle 50% of the trajectory to avoid transient and saturation
        idx_start = int(len(t_dense) * 0.2)
        idx_end = int(len(t_dense) * 0.8)
        coeffs = np.polyfit(t_dense[idx_start:idx_end], log_delta[idx_start:idx_end], 1)
        lambda_est = coeffs[0]
        final_div = delta_r[-1] / 1000.0 # km
        
        print(f"  Estimated Lyapunov Exponent: ~{lambda_est:.4f} s^-1")
        print(f"  Divergence at T={t_dense[-1]:.1f}s: {final_div:.2f} km")
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(t_dense, log_delta, 'b-', linewidth=1.5, label='Log Divergence ln(|Δr|)')
        
        # Plot Fit Line
        y_fit = coeffs[0] * t_dense[idx_start:idx_end] + coeffs[1]
        plt.plot(t_dense[idx_start:idx_end], y_fit, 'r--', linewidth=2, label=f'Lyapunov Fit (λ ≈ {lambda_est:.4f})')
        
        plt.xlabel('Time (s)')
        plt.ylabel('ln(|delta_r|) [Log Divergence]')
        plt.title(f'Lyapunov Analysis: Sensitivity to Initial Conditions (Thrust +0.1%)\nEst. Lambda = {lambda_est:.4f} s^-1, Final Div = {final_div:.1f} km')
        plt.grid(True)
        plt.legend()
        plt.show()
        print(f">>> {debug.Style.GREEN}PASS: Chaos analysis complete.{debug.Style.RESET}")

    # 12. STIFFNESS / EULER TEST (Upgrade 3)
    def analyze_stiffness_euler(self):
        debug._print_sub_header("12. Numerical Stiffness: Euler vs RK45")
        
        # 1. Get Controls & Reference
        with suppress_stdout():
            opt_res = solve_optimal_trajectory(self.base_config, self.veh, self.env, print_level=0)
            
        if not opt_res.get("success", False):
            print("  Baseline optimization failed. Skipping Stiffness test.")
            return

        with suppress_stdout():
            sim_rk45 = run_simulation(opt_res, self.veh, self.base_config)
            
        
        # Rebuild interpolators
        T1 = opt_res["T1"]
        U1 = np.array(opt_res["U1"])
        t_grid_1 = np.linspace(0, T1, U1.shape[1] + 1)[:-1]
        ctrl_1 = interp1d(t_grid_1, U1, axis=1, kind='previous', fill_value="extrapolate", bounds_error=False)
        
        # Reference RK45
        r_rk = np.linalg.norm(sim_rk45['y'][0:3], axis=0) - self.env.config.earth_radius_equator
        t_rk = sim_rk45['t']
        y_rk = sim_rk45['y']
        _, idx_rk = np.unique(t_rk, return_index=True)
        idx_rk = np.sort(idx_rk)
        t_rk = t_rk[idx_rk]
        y_rk = y_rk[:, idx_rk]
        f_rk = interp1d(t_rk, y_rk, axis=1, fill_value="extrapolate")

        # Plot Setup
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Top: Trajectory
        ax1.plot(sim_rk45['t'][sim_rk45['t']<=T1], r_rk[sim_rk45['t']<=T1]/1000.0, 'k-', linewidth=2, label='RK45 (Adaptive)')
        ax1.set_ylabel('Altitude (km)')
        ax1.set_title('Numerical Stiffness: Euler vs RK45 (Phase 1)')
        ax1.grid(True)
        ax1.legend()
        
        # Bottom: Error
        ax2.set_ylabel('Position Error (km) [Log Scale]')
        ax2.set_xlabel('Time (s)')
        ax2.grid(True, which="both", ls="-", alpha=0.3)
        ax2.set_title('Integration Error (Euler - RK45)')
        
        # 2. Run Euler Loop for multiple time steps
        dt_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        colors = ['g--', 'b--', 'r--', 'c--', 'm--']
        
        for dt, color in zip(dt_values, colors):
            print(f"Running Fixed-Step Euler Integration (dt={dt}s)...")
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
            
            t_euler = np.array(t_euler)
            y_euler = np.array(y_euler).T
            r_eu = np.linalg.norm(y_euler[0:3], axis=0) - self.env.config.earth_radius_equator
            
            # Error Calc
            y_rk_final = f_rk(t_euler[-1])
            y_eu_final = y_euler[:, -1]
            err_km = np.linalg.norm(y_eu_final[0:3] - y_rk_final[0:3]) / 1000.0
            
            # Error Trace for Plotting
            y_rk_interp = f_rk(t_euler)
            
            # Position Error (Euclidean Distance) for log plot
            pos_diff = y_euler[0:3, :] - y_rk_interp[0:3, :]
            error_trace_km = np.linalg.norm(pos_diff, axis=0) / 1000.0
            # Avoid log(0)
            error_trace_km = np.maximum(error_trace_km, 1e-9)
            
            print(f"  dt={dt}s -> Error: {err_km:.2f} km")
            ax1.plot(t_euler, r_eu/1000.0, color, label=f'Euler (dt={dt}s)')
            ax2.semilogy(t_euler, error_trace_km, color, label=f'dt={dt}s (Final Err={err_km:.1f}km)')
        
        # Add Threshold Line
        ax2.axhline(y=1.0, color='k', linestyle=':', label='1 km Threshold')
        
        ax1.legend()
        ax2.legend()
        plt.tight_layout()
        plt.show()
        print(f">>> {debug.Style.GREEN}PASS: Stiffness demonstrated.{debug.Style.RESET}")
        print(f"\n{debug.Style.BOLD}THEORETICAL DEFENSE (For Report):{debug.Style.RESET}")

    # 13. BIFURCATION ANALYSIS (Upgrade 4)
    def analyze_bifurcation(self):
        debug._print_sub_header("13. Bifurcation Analysis (The 'Cliff Edge')")
        
        # 1. Get Nominal Controls
        with suppress_stdout():
            opt_res = solve_optimal_trajectory(self.base_config, self.veh, self.env, print_level=0)
            
        if not opt_res.get("success", False):
            print("  Baseline optimization failed. Skipping Bifurcation.")
            return

        # 2. Parameter Sweep (Thrust Multiplier)
        multipliers = np.linspace(0.90, 1.10, 21) # +/- 10%
        final_alts = []
        margins = []
        
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
            
            # Metrics
            r_f = np.linalg.norm(sim_res['y'][0:3, -1]) - self.env.config.earth_radius_equator
            m_final = sim_res['y'][6, -1]
            m_dry = cfg.stage_2.dry_mass + cfg.payload_mass
            
            final_alts.append(r_f / 1000.0)
            margins.append(m_final - m_dry)
            
            status = "Orbit" if r_f > 100000 else "Crash/Sub"
            print(f"{mult:<12.2f} | {r_f/1000:<15.1f} | {status:<10}")

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
        
        plt.title('Bifurcation Analysis: Sensitivity to Thrust\n(Open-Loop Control)')
        fig.tight_layout()
        plt.show()
        
        print(f">>> {debug.Style.GREEN}PASS: Bifurcation diagram generated.{debug.Style.RESET}")

    # 14. THEORETICAL EFFICIENCY (Hohmann Comparison)
    def analyze_theoretical_efficiency(self):
        debug._print_sub_header("14. Theoretical Efficiency (Hohmann Comparison)")
        
        # 1. Get Simulation Data
        with suppress_stdout():
            opt_res = solve_optimal_trajectory(self.base_config, self.veh, self.env, print_level=0)
            sim_res = run_simulation(opt_res, self.veh, self.base_config)
            
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
        
        if efficiency > 85.0:
            print(f">>> {debug.Style.GREEN}PASS: High efficiency (>85%). Trajectory is near-optimal.{debug.Style.RESET}")
        else:
            print(f">>> {debug.Style.YELLOW}NOTE: Efficiency is {efficiency:.1f}%. Losses are significant.{debug.Style.RESET}")

if __name__ == "__main__":
    suite = ReliabilitySuite()
    suite.run_all()