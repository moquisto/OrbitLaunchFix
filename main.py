# main.py
# Purpose: The Optimization Orchestrator.

import casadi as ca
import numpy as np
from config import ScalingConfig
import warnings
import time
import guidance

def check_path_constraints(sol, X, U, scaling, config, phase_name):
    """
    Checks for violations of path constraints in the optimizer solution.
    """
    print(f"  [{phase_name}] Checking Path Constraints...")
    
    # Unpack
    r = sol.value(X)[0:3, :] * scaling.length
    u_val = sol.value(U)
    
    # 1. Altitude Constraint
    r_mags = np.linalg.norm(r, axis=0)
    min_r = np.min(r_mags)
    if min_r < scaling.length - 100.0: # Approx check against Earth Radius
        print(f"    ! ALTITUDE VIOLATION: Min Radius = {min_r:.1f} m")
        
    # 2. Control Constraints
    throttles = u_val[0, :]
    if np.any(throttles < config.sequence.min_throttle - 1e-3) or np.any(throttles > 1.0 + 1e-3):
        print(f"    ! THROTTLE VIOLATION: Range [{np.min(throttles):.2f}, {np.max(throttles):.2f}]")
        
    # 3. Direction Constraint
    dirs = u_val[1:, :]
    norms = np.linalg.norm(dirs, axis=0)
    err_norm = np.max(np.abs(norms - 1.0))
    if err_norm > 1e-3:
        print(f"    ! CONTROL VECTOR VIOLATION: Max Norm Error = {err_norm:.4f}")

def print_debug_info(opti, sol, scaling, config, environment, X1, U1, T1_scaled, X3, U3, T3_scaled):
    """
    Analyzes the failed optimization result to identify the cause.
    """
    print("\n" + "="*40)
    print("       OPTIMIZATION FAILURE DIAGNOSIS")
    print("="*40)
    
    # 1. Terminal State Analysis
    x_final = sol.value(X3)[:, -1]
    r_f = x_final[0:3] * scaling.length
    v_f = x_final[3:6] * scaling.speed
    m_f = x_final[6] * scaling.mass
    
    R_earth = environment.config.earth_radius_equator
    r_mag = np.linalg.norm(r_f)
    alt_f = r_mag - R_earth
    v_mag = np.linalg.norm(v_f)
    
    target_alt = config.target_altitude
    target_v = np.sqrt(environment.config.earth_mu / (R_earth + target_alt))
    
    print(f"TERMINAL STATE (Optimizer Output):")
    print(f"  Altitude:   {alt_f/1000:.2f} km  (Target: {target_alt/1000:.2f} km) | Error: {(alt_f - target_alt)/1000:.2f} km")
    print(f"  Velocity:   {v_mag:.2f} m/s    (Target: {target_v:.2f} m/s)   | Error: {v_mag - target_v:.2f} m/s")
    
    # 2. Mass / Fuel Analysis
    m_dry_s2 = config.stage_2.dry_mass + config.payload_mass
    print(f"\nMASS / PROPELLANT CHECK:")
    print(f"  Final Mass: {m_f:.2f} kg")
    print(f"  Dry Limit:  {m_dry_s2:.2f} kg")
    
    if m_f < m_dry_s2:
        print(f"  >>> CRITICAL FAILURE: Final mass is BELOW dry mass limit by {m_dry_s2 - m_f:.2f} kg.")
        print(f"  >>> The vehicle lacks the Delta-V to reach the target orbit with current constraints.")
    else:
        print(f"  Mass constraint satisfied. Remaining fuel: {m_f - m_dry_s2:.2f} kg")

    # 3. Staging Analysis
    t1 = sol.value(T1_scaled) * scaling.time
    x1_f = sol.value(X1)[:, -1]
    v1_f = x1_f[3:6] * scaling.speed
    v1_mag = np.linalg.norm(v1_f)
    
    print(f"\nSTAGING ANALYSIS:")
    print(f"  MECO Time:      {t1:.2f} s")
    print(f"  MECO Velocity:  {v1_mag:.2f} m/s")
    
    # Check Stage 2 T/W
    m_ship_wet = config.stage_2.dry_mass + config.stage_2.propellant_mass + config.payload_mass
    weight_ship = m_ship_wet * 9.81 # Approx gravity
    thrust_ship = config.stage_2.thrust_vac
    tw_ship = thrust_ship / weight_ship
    print(f"  Ship Init T/W:  {tw_ship:.2f}")
    
    # Flight Path Angle (Gamma) Analysis
    # Gamma = asin( r dot v / (|r||v|) )
    r_s = x1_f[0:3] * scaling.length
    v_s = x1_f[3:6] * scaling.speed
    r_mag_s = np.linalg.norm(r_s)
    v_mag_s = np.linalg.norm(v_s)
    sin_gamma = np.dot(r_s, v_s) / (r_mag_s * v_mag_s)
    gamma_deg = np.degrees(np.arcsin(np.clip(sin_gamma, -1.0, 1.0)))
    
    print(f"  Staging FPA (Gamma): {gamma_deg:.2f} deg")
    
    if tw_ship < 1.0 and gamma_deg < 5.0:
        print(f"  >>> CRITICAL PHYSICS ISSUE: Staging T/W is {tw_ship:.2f} (< 1.0) and Flight Path Angle is low ({gamma_deg:.1f} deg).")
        print(f"      The upper stage cannot fight gravity and will sink into the atmosphere.")
        print(f"      SUGGESTION: Increase Booster 'meco_cutoff_mach' or force a steeper ascent.")
    
    # 4. Solver Metrics
    # If we are here, the solver likely failed to satisfy constraints.
    print(f"\nSOLVER STATUS:")
    print(f"  The optimizer could not find a feasible solution.")
    
    # 5. Path Constraint Check
    check_path_constraints(sol, X1, U1, scaling, config, "Phase 1")
    check_path_constraints(sol, X3, U3, scaling, config, "Phase 3")
    print("="*40 + "\n")

def solve_optimal_trajectory(config, vehicle, environment):
    """
    Formulates and solves the trajectory optimization problem using CasADi.
    """
    print(f"\n[Optimizer] Setting up CasADi problem for {config.name}...")
    t_start = time.time()
    
    # --- 1. SETUP ---
    # Initialize CasADi Opti stack
    opti = ca.Opti()
    scaling = ScalingConfig(mass=1.0e6) # Fixed reference mass (1000t) for numerical stability
    
    mu = environment.config.earth_mu
    R_earth = environment.config.earth_radius_equator
    
    # Target Orbit
    Target_Alt = config.target_altitude
    Target_Inc_Deg = config.target_inclination
    
    # Ensure Target Inclination is physically possible (>= Latitude)
    lat_deg = environment.config.launch_latitude
    if Target_Inc_Deg < lat_deg - 1e-6:
        warnings.warn(f"Target inclination {Target_Inc_Deg:.2f} deg is less than launch latitude {lat_deg:.2f} deg. Clamping to latitude.")
        Target_Inc = np.radians(lat_deg)
    else:
        Target_Inc = np.radians(Target_Inc_Deg)

    # Mass Constants
    # Ship Wet Mass (Start of Phase 3)
    m_ship_wet = config.stage_2.dry_mass + config.stage_2.propellant_mass + config.payload_mass
    # Minimum Mass at end of Phase 1 (Must have enough fuel to reach staging)
    m_stage1_min = config.stage_1.dry_mass + m_ship_wet
    
    # --- 2. DECISION VARIABLES ---
    N = config.num_nodes # Nodes per phase
    
    # Phase 1: Booster Ascent
    T1_scaled = opti.variable()
    X1 = opti.variable(7, N + 1) # [rx, ry, rz, vx, vy, vz, m]
    U1 = opti.variable(4, N)     # [throttle, ux, uy, uz]
    
    # Phase 2: Coast (Optional)
    use_coast = config.sequence.separation_delay > 1e-4
    if use_coast:
        T2_scaled = opti.variable()
        X2 = opti.variable(7, N + 1)
    else:
        T2_scaled = 0.0
    
    # Phase 3: Ship Ascent
    T3_scaled = opti.variable()
    X3 = opti.variable(7, N + 1)
    U3 = opti.variable(4, N)
    
    # --- 3. CONSTRAINTS ---
    
    # A. Boundary Conditions (Initial State)
    r0, v0 = environment.get_launch_site_state()
    m0 = config.launch_mass
    x0_scaled = np.concatenate([
        r0 / scaling.length,
        v0 / scaling.speed,
        [m0 / scaling.mass]
    ])
    opti.subject_to(X1[:, 0] == x0_scaled)
    
    # Time Constraints
    opti.subject_to(T1_scaled >= 60.0 / scaling.time)
    opti.subject_to(T3_scaled >= 10.0 / scaling.time)
    if use_coast:
        opti.subject_to(T2_scaled == config.sequence.separation_delay / scaling.time)
    
    # B. Dynamics & Path Constraints (RK4 Integration)
    def add_phase_dynamics(X, U, T_scaled, phase_mode, t_start_scaled):
        dt_scaled = T_scaled / N
        
        for k in range(N):
            # Current time (scaled & physical)
            t_k_scaled = t_start_scaled + k * dt_scaled
            
            x_k = X[:, k]
            
            # Control Inputs
            if "coast" in phase_mode:
                u_throttle = 0.0
                u_dir = ca.vertcat(1, 0, 0) # Dummy direction
            else:
                u_k = U[:, k]
                u_throttle = u_k[0]
                u_dir = u_k[1:]
                
                # Control Constraints
                opti.subject_to(u_throttle >= config.sequence.min_throttle)
                opti.subject_to(u_throttle <= 1.0)
                opti.subject_to(ca.dot(u_dir, u_dir) == 1.0) # Unit vector
            
            # RK4 Integration Wrapper
            def dyn(x, t):
                return vehicle.get_dynamics(x, u_throttle, u_dir, t, stage_mode=phase_mode, scaling=scaling)

            k1 = dyn(x_k, t_k_scaled)
            k2 = dyn(x_k + 0.5 * dt_scaled * k1, t_k_scaled + 0.5 * dt_scaled)
            k3 = dyn(x_k + 0.5 * dt_scaled * k2, t_k_scaled + 0.5 * dt_scaled)
            k4 = dyn(x_k + dt_scaled * k3, t_k_scaled + dt_scaled)
            
            x_next = x_k + (dt_scaled / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Continuity Constraint
            opti.subject_to(X[:, k+1] == x_next)
            
            # Path Constraints (Safety)
            # Altitude > -100m (allow slight dip at launch due to numerics)
            r_k_scaled = x_k[0:3]
            R_earth_scaled = R_earth / scaling.length
            opti.subject_to(ca.norm_2(r_k_scaled) >= R_earth_scaled - (100.0 / scaling.length))
            
            # Mass Constraints (Don't burn more than available)
            m_k_scaled = x_k[6]
            if phase_mode == "boost":
                opti.subject_to(m_k_scaled >= m_stage1_min / scaling.mass)
            elif phase_mode == "ship":
                opti.subject_to(m_k_scaled >= (config.stage_2.dry_mass + config.payload_mass) / scaling.mass)

    # Apply Dynamics
    # Phase 1: Boost
    add_phase_dynamics(X1, U1, T1_scaled, "boost", 0.0)
    
    # STAGING FIX: Force Booster to provide minimum velocity (e.g. 1500 m/s)
    # This prevents the solver from cutting the booster phase too short.
    v_meco_scaled = ca.norm_2(X1[3:6, -1])
    opti.subject_to(v_meco_scaled >= 1500.0 / scaling.speed)
    
    # Phase 2: Coast / Linkage
    t_end_p1 = T1_scaled
    if use_coast:
        add_phase_dynamics(X2, None, T2_scaled, "coast", t_end_p1)
        opti.subject_to(X2[:, 0] == X1[:, -1]) # Link P1 -> P2
        t_end_p2 = t_end_p1 + T2_scaled
        x_final_prev = X2[:, -1]
    else:
        t_end_p2 = t_end_p1
        x_final_prev = X1[:, -1]
    
    # Phase 3: Ship Linkage (Staging)
    # Position/Velocity match, Mass resets to Ship Wet Mass
    opti.subject_to(X3[0:6, 0] == x_final_prev[0:6])
    opti.subject_to(X3[6, 0] == m_ship_wet / scaling.mass)
    
    # Phase 3 Dynamics
    add_phase_dynamics(X3, U3, T3_scaled, "ship", t_end_p2)
    
    # C. Terminal Constraints (Target Orbit)
    x_final = X3[:, -1]
    r_f_scaled = x_final[0:3]
    v_f_scaled = x_final[3:6]
    
    r_mag_scaled = ca.norm_2(r_f_scaled)
    v_mag_scaled = ca.norm_2(v_f_scaled)
    
    # 1. Altitude
    target_r_scaled = (R_earth + Target_Alt) / scaling.length
    opti.subject_to(r_mag_scaled == target_r_scaled)
    # 2. Velocity (Circular)
    v_target_phys = np.sqrt(mu / (R_earth + Target_Alt))
    v_target_scaled = v_target_phys / scaling.speed
    opti.subject_to(v_mag_scaled == v_target_scaled)
    # 3. Eccentricity (Flight Path Angle = 0)
    opti.subject_to(ca.dot(r_f_scaled, v_f_scaled) == 0.0)
    # 4. Inclination
    h_vec_scaled = ca.cross(r_f_scaled, v_f_scaled)
    h_mag_scaled = ca.norm_2(h_vec_scaled)
    opti.subject_to(h_vec_scaled[2] == h_mag_scaled * np.cos(Target_Inc))

    # --- 4. OBJECTIVE ---
    # Maximize Final Mass
    opti.minimize(-X3[6, -1])
    
    # --- 5. INITIALIZATION ---
    print(f"[Optimizer] Calling Guidance module for Warm Start...")
    t_guess = time.time()
    # FIX: Pass N, not N+1. Guidance generates N+1 points (nodes) from N intervals.
    guess = guidance.get_initial_guess(config, vehicle, environment, num_nodes=N)
    print(f"[Optimizer] Guidance generated in {time.time() - t_guess:.2f}s")
    
    def set_guess(var, val):
        # FIX: Remove try-except to expose shape mismatches immediately
        opti.set_initial(var, val)

    s_vec = np.array([scaling.length]*3 + [scaling.speed]*3 + [scaling.mass])
    
    # Phase 1
    set_guess(T1_scaled, guess["T1"] / scaling.time)
    set_guess(X1, guess["X1"] / s_vec[:, None])
    u1_guess = np.vstack([np.atleast_2d(guess["TH1"]), guess["TD1"]])
    set_guess(U1, u1_guess[:, :N])
    
    # Phase 2
    if use_coast and "X2" in guess and guess["X2"] is not None:
        set_guess(T2_scaled, guess["T2"] / scaling.time)
        set_guess(X2, guess["X2"] / s_vec[:, None])
        
    # Phase 3
    set_guess(T3_scaled, guess["T3"] / scaling.time)
    set_guess(X3, guess["X3"] / s_vec[:, None])
    u3_guess = np.vstack([np.atleast_2d(guess["TH3"]), guess["TD3"]])
    set_guess(U3, u3_guess[:, :N])
    
    # --- 6. SOLVE ---
    print(f"[Optimizer] Starting IPOPT solver (Max Iter={1000})...")
    t_solve = time.time()
    opti.solver("ipopt", {"expand": True}, {"max_iter": 1000, "tol": 1e-4, "print_level": 5})
    
    try:
        sol = opti.solve()
        print(f"[Optimizer] SUCCESS: Optimal solution found in {time.time() - t_solve:.2f}s.")
    except:
        print(f"[Optimizer] FAILURE: Solver did not converge after {time.time() - t_solve:.2f}s. Returning debug values.")
        sol = opti.debug
        print_debug_info(opti, sol, scaling, config, environment, X1, U1, T1_scaled, X3, U3, T3_scaled)
    
    # --- 7. OUTPUT ---
    res = {}
    res["T1"] = sol.value(T1_scaled) * scaling.time
    res["X1"] = sol.value(X1) * s_vec[:, None]
    res["U1"] = sol.value(U1)
    
    if use_coast:
        res["T2"] = sol.value(T2_scaled) * scaling.time
        res["X2"] = sol.value(X2) * s_vec[:, None]
    else:
        res["T2"] = 0.0
        
    res["T3"] = sol.value(T3_scaled) * scaling.time
    res["X3"] = sol.value(X3) * s_vec[:, None]
    res["U3"] = sol.value(U3)
    
    print(f"[Optimizer] Total optimization time: {time.time() - t_start:.2f}s")
    
    return res

def verify_physics_consistency(vehicle, config):
    """
    Debugs the interface between Optimizer (CasADi) and Simulation (NumPy).
    Ensures that both 'modes' of the physics engine return identical results
    given the same inputs.
    """
    print("\n" + "="*40)
    print("DEBUG: PHYSICS ENGINE CONSISTENCY CHECK")
    print("="*40)
    
    R_e = vehicle.env.config.earth_radius_equator
    
    # Define Test Scenarios covering edge cases
    scenarios = [
        {
            "name": "Launch Pad (Static)",
            "r": [R_e, 0, 0],
            "v": [0, 0, 0],          # Singularity check (v=0)
            "m": config.launch_mass,
            "th": 1.0,
            "dir": [1, 0, 0],
            "t": 0.0,
            "mode": "boost"
        },
        {
            "name": "Max Q (Transonic)",
            "r": [R_e + 12000.0, 0, 0],
            "v": [0, 300.0, 300.0],  # High Drag
            "m": config.launch_mass * 0.8,
            "th": 1.0,
            "dir": [0, 0.707, 0.707],
            "t": 60.0,
            "mode": "boost"
        },
        {
            "name": "Coast (Staging)",
            "r": [R_e + 80000.0, 0, 0],
            "v": [0, 2000.0, 1000.0],
            "m": config.launch_mass * 0.4,
            "th": 0.0,               # Zero Throttle check
            "dir": [0, 1, 0],
            "t": 150.0,
            "mode": "coast"
        },
        {
            "name": "Ship (Vacuum)",
            "r": [R_e + 400000.0, 0, 0],
            "v": [0, 7500.0, 0.0],   # Orbital Velocity
            "m": config.stage_2.dry_mass + config.stage_2.propellant_mass,
            "th": 1.0,
            "dir": [0, 1, 0],
            "t": 500.0,
            "mode": "ship"           # Stage 2 Parameters
        }
    ]

    # Symbolic placeholders
    x_sym = ca.MX.sym('x', 7)
    u_th_sym = ca.MX.sym('th', 1)
    u_dir_sym = ca.MX.sym('dir', 3)
    t_sym = ca.MX.sym('t', 1)
    
    overall_success = True

    print(f"{'Scenario':<25} | {'Max Diff':<12} | {'Status':<10}")
    print("-" * 55)

    for case in scenarios:
        # 1. Inputs
        r_test = np.array(case['r'])
        v_test = np.array(case['v'])
        m_test = case['m']
        state_num = np.concatenate([r_test, v_test, [m_test]])
        
        th_test = case['th']
        dir_test = np.array(case['dir'])
        t_test = case['t']
        mode = case['mode']
        
        # 2. Run Numeric (Simulation Mode)
        dyn_sim = vehicle.get_dynamics(
            state_num, th_test, dir_test, t_test, stage_mode=mode, scaling=None
        )
        
        # 3. Run Symbolic (Optimizer Mode)
        # Re-build graph for specific mode (drag area changes etc)
        dyn_sym_expr = vehicle.get_dynamics(x_sym, u_th_sym, u_dir_sym, t_sym, stage_mode=mode, scaling=None)
        f_dyn = ca.Function('f_dyn', [x_sym, u_th_sym, u_dir_sym, t_sym], [dyn_sym_expr])
        res_sym = np.array(f_dyn(state_num, th_test, dir_test, t_test)).flatten()
        
        # 4. Compare
        max_diff = np.max(np.abs(dyn_sim - res_sym))
        
        status = "PASS"
        if max_diff > 1e-9:
            status = "FAIL"
            overall_success = False
            
        print(f"{case['name']:<25} | {max_diff:.2e}     | {status:<10}")
        
        if status == "FAIL":
            labels = ['vx', 'vy', 'vz', 'ax', 'ay', 'az', 'mdot']
            for i, label in enumerate(labels):
                d = abs(dyn_sim[i] - res_sym[i])
                if d > 1e-9:
                    print(f"    ! {label}: Sim={dyn_sim[i]:.4e} Opt={res_sym[i]:.4e}")

    if overall_success:
        print("\n>>> SUCCESS: Physics engines are consistent across all regimes.")
    else:
        print("\n>>> CRITICAL WARNING: Physics engines disagree!")
    print("="*40 + "\n")

def verify_environment_consistency(vehicle):
    """
    Debugs the Environment model to ensure Symbolic (Optimizer) and 
    Numeric (Simulation) implementations return identical values.
    """
    print("\n" + "="*40)
    print("DEBUG: ENVIRONMENT MODEL CONSISTENCY CHECK")
    print("="*40)
    
    # Test points: Surface, Max Q (~12km), Space (~200km), High Lat
    R_e = vehicle.env.config.earth_radius_equator
    test_points = [
        ("Surface Eq", 0.0, 0.0),
        ("Max Q Eq", 12000.0, 0.0),
        ("Space Eq", 200000.0, 0.0),
        ("Mid Lat", 0.0, 45.0)
    ]
    
    # Construct CasADi Function for Symbolic Evaluation
    r_sym = ca.MX.sym('r', 3)
    t_sym = ca.MX.sym('t', 1)
    env_sym = vehicle.env.get_state_opti(r_sym, t_sym)
    
    # Output vector: [rho, p, sos, gravity(3), wind(3)]
    f_env_sym = ca.Function('f_env_debug', [r_sym, t_sym], 
                            [env_sym['density'], 
                             env_sym['pressure'], 
                             env_sym['speed_of_sound'], 
                             env_sym['gravity'],
                             env_sym['wind_velocity']])
    
    print(f"{'Location':<15} | {'Var':<8} | {'Simulation':<12} | {'Optimizer':<12} | {'Diff':<10}")
    print("-" * 70)
    
    overall_success = True
    
    for name, alt, lat in test_points:
        # Position vector
        lat_rad = np.radians(lat)
        r_test = np.array([
            (R_e + alt) * np.cos(lat_rad), 
            0, 
            (R_e + alt) * np.sin(lat_rad)
        ])
        t_test = 0.0
        
        # 1. Run Numeric (Simulation Mode)
        env_sim = vehicle.env.get_state_sim(r_test, t_test)
        
        # 2. Run Symbolic (Optimizer Mode)
        res_opt = f_env_sym(r_test, t_test)
        rho_opt = float(res_opt[0])
        p_opt   = float(res_opt[1])
        sos_opt = float(res_opt[2])
        g_opt   = np.array(res_opt[3]).flatten()
        w_opt   = np.array(res_opt[4]).flatten()
        
        # 3. Compare
        # Density
        diff_rho = abs(env_sim['density'] - rho_opt)
        if diff_rho > 1e-9: overall_success = False
        print(f"{name:<15} | {'rho':<8} | {env_sim['density']:<12.4e} | {rho_opt:<12.4e} | {diff_rho:.2e}")
        
        # Pressure
        diff_p = abs(env_sim['pressure'] - p_opt)
        if diff_p > 1e-9: overall_success = False
        print(f"{'':<15} | {'p':<8} | {env_sim['pressure']:<12.4e} | {p_opt:<12.4e} | {diff_p:.2e}")
        
        # Speed of Sound
        diff_sos = abs(env_sim['speed_of_sound'] - sos_opt)
        if diff_sos > 1e-9: overall_success = False
        print(f"{'':<15} | {'sos':<8} | {env_sim['speed_of_sound']:<12.4f} | {sos_opt:<12.4f} | {diff_sos:.2e}")
        
        # Gravity Magnitude
        g_sim_mag = np.linalg.norm(env_sim['gravity'])
        g_opt_mag = np.linalg.norm(g_opt)
        diff_g = abs(g_sim_mag - g_opt_mag)
        if diff_g > 1e-9: overall_success = False
        print(f"{'':<15} | {'g_mag':<8} | {g_sim_mag:<12.4f} | {g_opt_mag:<12.4f} | {diff_g:.2e}")
        
        # Wind Velocity Magnitude
        w_sim_mag = np.linalg.norm(env_sim['wind_velocity'])
        w_opt_mag = np.linalg.norm(w_opt)
        diff_w = abs(w_sim_mag - w_opt_mag)
        if diff_w > 1e-9: overall_success = False
        print(f"{'':<15} | {'w_mag':<8} | {w_sim_mag:<12.4f} | {w_opt_mag:<12.4f} | {diff_w:.2e}")
        print("-" * 70)

    if overall_success:
        print("\n>>> SUCCESS: Environment models are consistent.")
    else:
        print("\n>>> CRITICAL WARNING: Environment models disagree!")
    print("="*40 + "\n")

if __name__ == "__main__":
    from config import StarshipBlock2, EARTH_CONFIG
    from vehicle import Vehicle
    from environment import Environment
    from simulation import run_simulation
    import analysis
    
    print("--- Setting up Mission ---")
    StarshipBlock2.print_summary()
    env = Environment(EARTH_CONFIG)
    veh = Vehicle(StarshipBlock2, env)
    
    print("--- Verifying Physics Model ---")
    verify_physics_consistency(veh, StarshipBlock2)
    verify_environment_consistency(veh)
    
    print("--- Running Optimization ---")
    opt_res = solve_optimal_trajectory(StarshipBlock2, veh, env)
    
    print("--- Running Verification Simulation ---")
    sim_res = run_simulation(opt_res, veh, StarshipBlock2)
    
    print("--- Validating Trajectory ---")
    analysis.validate_trajectory(sim_res, StarshipBlock2, env)
    
    print("--- Analyzing Efficiency ---")
    analysis.analyze_delta_v_budget(sim_res, veh, StarshipBlock2)
    
    print("--- Plotting Results ---")
    analysis.plot_mission(opt_res, sim_res, env)
    print("Done.")