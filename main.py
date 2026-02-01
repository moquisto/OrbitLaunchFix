# main.py
# Purpose: The Optimization Orchestrator.

import casadi as ca
import numpy as np
from config import ScalingConfig
import warnings
import time
import guidance

def check_path_constraints(sol, X, U, scaling, config, environment, phase_name):
    """
    Checks for violations of path constraints in the optimizer solution.
    """
    print(f"  [{phase_name}] Checking Path Constraints...")
    
    # Unpack
    r = sol.value(X)[0:3, :] * scaling.length
    u_val = sol.value(U)
    
    # 1. Altitude Constraint (Ellipsoidal WGS84)
    R_eq = environment.config.earth_radius_equator
    f = environment.config.earth_flattening
    R_pol = R_eq * (1.0 - f)
    
    # Ellipsoid equation: (x/a)^2 + (y/a)^2 + (z/b)^2 >= 1
    val = (r[0,:]/R_eq)**2 + (r[1,:]/R_eq)**2 + (r[2,:]/R_pol)**2
    min_val = np.min(val)
    
    if min_val < 1.0 - 1e-6: # Strict check
        min_idx = np.argmin(val)
        # Approx violation in meters (geometric mean radius)
        violation = (1.0 - np.sqrt(min_val)) * R_eq 
        print(f"    ! ALTITUDE VIOLATION: Ellipsoid Metric = {min_val:.6f} (~{violation:.1f} m under) at Node {min_idx}")
        
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
    check_path_constraints(sol, X1, U1, scaling, config, environment, "Phase 1")
    check_path_constraints(sol, X3, U3, scaling, config, environment, "Phase 3")
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
    scaling = ScalingConfig() # Use default from config.py (1000t)
    
    mu = environment.config.earth_mu
    R_earth = environment.config.earth_radius_equator
    
    # Target Orbit
    Target_Alt = config.target_altitude
    
    # Resolve Target Inclination
    if config.target_inclination is None:
        print(f"[Optimizer] Target Inclination not set. Defaulting to Launch Latitude ({environment.config.launch_latitude:.2f} deg) for min-energy orbit.")
        config.target_inclination = environment.config.launch_latitude
        
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
        
        # Apply Mass Constraints to ALL nodes (0 to N)
        # This ensures the final node (result of last integration step) also respects the limit.
        m_all_scaled = X[6, :]
        if phase_mode == "boost":
            opti.subject_to(m_all_scaled >= m_stage1_min / scaling.mass)
        elif phase_mode == "ship":
            opti.subject_to(m_all_scaled >= (config.stage_2.dry_mass + config.payload_mass) / scaling.mass)

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
            # Altitude > 0m (Strict constraint, no buffer, WGS84 Ellipsoid)
            r_k_scaled = x_k[0:3]
            Re_s = R_earth / scaling.length
            f = environment.config.earth_flattening
            Rp_s = (R_earth * (1.0 - f)) / scaling.length
            
            # (x/Re)^2 + (y/Re)^2 + (z/Rp)^2 >= 1
            ellipsoid_metric = (r_k_scaled[0]/Re_s)**2 + (r_k_scaled[1]/Re_s)**2 + (r_k_scaled[2]/Rp_s)**2
            opti.subject_to(ellipsoid_metric >= 1.0)

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
    
    # DEBUG: Analyze Guidance Guess
    x1_end = guess["X1"][:, -1]
    x3_end = guess["X3"][:, -1]
    r_meco = np.linalg.norm(x1_end[0:3]) - R_earth
    v_meco = np.linalg.norm(x1_end[3:6])
    r_seco = np.linalg.norm(x3_end[0:3]) - R_earth
    v_seco = np.linalg.norm(x3_end[3:6])
    
    print(f"[Optimizer] Guidance Guess Summary:")
    print(f"  MECO: T={guess['T1']:.1f}s, Alt={r_meco/1000:.1f}km, Vel={v_meco:.1f}m/s")
    print(f"  SECO: T={guess['T3']:.1f}s, Alt={r_seco/1000:.1f}km, Vel={v_seco:.1f}m/s")

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
    print(f"[Optimizer] Starting IPOPT solver (Max Iter={2000})...")
    t_solve = time.time()
    opti.solver("ipopt", {"expand": True}, {"max_iter": 2000, "tol": 1e-6, "print_level": 5})
    
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

def verify_scaling_consistency(config):
    """
    Verifies that scaling factors produce numerically friendly values (O(1))
    and that round-trip conversion preserves data integrity.
    """
    print("\n" + "="*40)
    print("DEBUG: SCALING CONSISTENCY CHECK")
    print("="*40)
    
    # Replicate the scaling config used in the optimizer
    scaling = ScalingConfig() 
    
    print(f"Scaling Factors:")
    print(f"  Length: {scaling.length:.4e} m")
    print(f"  Speed:  {scaling.speed:.4e} m/s")
    print(f"  Time:   {scaling.time:.4e} s")
    print(f"  Mass:   {scaling.mass:.4e} kg")
    print(f"  Force:  {scaling.force:.4e} N")
    print("-" * 75)

    # 0. Internal Consistency Check (Derived Units)
    # Ensure F = ma and v = d/t relationships hold in the config itself
    calc_speed = scaling.length / scaling.time
    calc_force = scaling.mass * (scaling.length / scaling.time**2)
    
    if abs(scaling.speed - calc_speed) > 1e-9:
        print(f"WARNING: Scaling speed mismatch! Config: {scaling.speed}, Calc: {calc_speed}")
    
    if abs(scaling.force - calc_force) > 1e-9:
        print(f"WARNING: Scaling force mismatch! Config: {scaling.force}, Calc: {calc_force}")

    # Define test cases: (Name, Physical Value, Scale Factor)
    tests = [
        ("Radius (Surface)", scaling.length, scaling.length),
        ("Radius (Orbit)",   scaling.length + config.target_altitude, scaling.length),
        ("Velocity (Orbit)", scaling.speed, scaling.speed),
        ("Mass (Launch)",    config.launch_mass, scaling.mass),
        ("Mass (Dry Stg2)",  config.stage_2.dry_mass, scaling.mass),
        ("Thrust (Booster)", config.stage_1.thrust_vac, scaling.force),
        ("Thrust (Ship)",    config.stage_2.thrust_vac, scaling.force),
        ("Time (Mission)",   600.0, scaling.time),
        ("Gravity (Surface)", 9.81, scaling.length / scaling.time**2) # Acceleration scaling
    ]
    
    print(f"{'Variable':<20} | {'Physical':<12} | {'Scaled':<10} | {'Unscaled':<12} | {'Rel Err':<10} | {'Status':<6}")
    print("-" * 85)
    
    overall_success = True
    
    for name, val_phys, scale_factor in tests:
        # 1. Scale
        val_scaled = val_phys / scale_factor
        
        # 2. Unscale
        val_recovered = val_scaled * scale_factor
        
        # 3. Check Magnitude (Should be roughly O(1), e.g. 0.01 to 100)
        is_reasonable = 1e-2 <= abs(val_scaled) <= 1e2
        
        # 4. Check Round Trip
        err = abs(val_phys - val_recovered)
        
        # Robust check: Relative for non-zero, Absolute for zero
        if abs(val_phys) > 1e-15:
            rel_err = err / abs(val_phys)
            is_fail = rel_err > 1e-12
        else:
            # For zero values, relative error is undefined/infinite if error exists.
            # We check absolute error instead.
            is_fail = err > 1e-12
            rel_err = float('inf') if is_fail else 0.0
        
        status = "OK"
        if not is_reasonable:
            status = "WARN MAG"
        if is_fail:
            status = "FAIL"
            overall_success = False
            
        print(f"{name:<20} | {val_phys:<12.4e} | {val_scaled:<10.4f} | {val_recovered:<12.4e} | {rel_err:<10.2e} | {status:<6}")
        
        if not is_reasonable:
            print(f"    ^ NOTE: Scaled value {val_scaled:.4f} is outside ideal range [0.01, 100]")

    if overall_success:
        print("\n>>> ✅ SUCCESS: Scaling logic is consistent.")
    else:
        print("\n>>> ❌ CRITICAL WARNING: Scaling logic failed round-trip check!")
    print("="*40 + "\n")

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
                print(f"    ! {label}: Sim={dyn_sim[i]:.4e} Opt={res_sym[i]:.4e} Diff={d:.2e}")

    if overall_success:
        print("\n>>> ✅ SUCCESS: Physics engines are consistent across all regimes.")
    else:
        print("\n>>> ❌ CRITICAL WARNING: Physics engines disagree!")
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
        print("\n>>> ✅ SUCCESS: Environment models are consistent.")
    else:
        print("\n>>> ❌ CRITICAL WARNING: Environment models disagree!")
    print("="*40 + "\n")

def verify_aerodynamics(vehicle):
    """
    Debugs the Aerodynamic model (Cd vs Mach, AoA, Wind) to ensure physics are sound.
    """
    print("\n" + "="*40)
    print("DEBUG: AERODYNAMICS CHECK")
    print("="*40)
    
    # 1. Cd vs Mach Interpolation
    print("1. Drag Coefficient vs Mach (0 deg AoA)")
    mach_points = [0.0, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 25.0]
    
    print(f"  {'Mach':<6} | {'Cd (Stage 1)':<12} | {'Cd (Stage 2)':<12}")
    print("  " + "-" * 34)
    
    for m in mach_points:
        # Evaluate CasADi interpolants
        cd1 = float(vehicle.cd_interp_stage_1(m))
        cd2 = float(vehicle.cd_interp_stage_2(m))
        print(f"  {m:<6.1f} | {cd1:<12.4f} | {cd2:<12.4f}")
        
        if cd1 < 0 or cd1 > 2.0 or cd2 < 0 or cd2 > 2.0:
            print(f"  >>> ❌ FAILURE: Cd out of physical range [0, 2.0] at Mach {m}")
    
    # 2. Crossflow Drag Model (AoA Effect)
    print("\n2. Crossflow Drag Model (Stage 1 @ Mach 2.0)")
    print(f"  {'AoA (deg)':<10} | {'Base Cd':<10} | {'Total Cd':<10} | {'Factor':<10}")
    print("  " + "-" * 46)
    
    mach_test = 2.0
    cd_base = float(vehicle.cd_interp_stage_1(mach_test))
    cross_factor = vehicle.config.stage_1.aero.cd_crossflow_factor
    
    aoa_points = [0, 5, 10, 30, 45, 90]
    for aoa in aoa_points:
        rad = np.radians(aoa)
        sin_alpha_sq = np.sin(rad)**2
        cd_total = cd_base + cross_factor * sin_alpha_sq
        factor = cd_total / cd_base
        print(f"  {aoa:<10.1f} | {cd_base:<10.4f} | {cd_total:<10.4f} | {factor:<10.2f}x")

    # 3. Force Vector Logic & Wind
    print("\n3. Force Vector & Wind Interaction")
    # Scenario: Vehicle moving Vertically at Equator. Wind blows East due to rotation.
    R = vehicle.env.config.earth_radius_equator
    omega = vehicle.env.config.earth_omega_vector[2]
    
    # State: 1km altitude, 100 m/s vertical velocity
    state = np.array([R + 1000.0, 0, 0, 100.0, 0, 0, 100000.0]) 
    t = 0.0
    
    # Expected Wind (Co-rotation): v_wind = omega x r = [0, R*omega, 0]
    v_wind_expected = np.array([0, (R+1000.0)*omega, 0])
    v_rel_expected = state[3:6] - v_wind_expected
    v_rel_dir_expected = v_rel_expected / np.linalg.norm(v_rel_expected)
    
    # Get Dynamics (Thrust=0 to isolate Aero+Gravity)
    dyn = vehicle.get_dynamics(state, 0.0, np.array([1,0,0]), t, stage_mode="coast", scaling=None)
    acc_total = dyn[3:6]
    
    # Remove Gravity
    env_state = vehicle.env.get_state_sim(state[0:3], t)
    acc_aero = acc_total - env_state['gravity']
    
    # Check Alignment: Drag should oppose Relative Velocity
    acc_aero_mag = np.linalg.norm(acc_aero)
    if acc_aero_mag > 1e-9:
        f_aero_dir = acc_aero / acc_aero_mag
        dot_prod = np.dot(f_aero_dir, v_rel_dir_expected)
        
        print(f"  Veh Velocity:   {state[3:6]}")
        print(f"  Wind Velocity:  {v_wind_expected} (Calculated)")
        print(f"  Rel Velocity:   {v_rel_expected}")
        print(f"  Drag Force Dir: {f_aero_dir}")
        print(f"  Alignment:      {dot_prod:.4f} (Should be -1.0000)")
        
        if abs(dot_prod + 1.0) < 1e-3:
            print("  >>> ✅ SUCCESS: Drag opposes relative velocity (Wind accounted for).")
        else:
            print("  >>> ❌ FAILURE: Drag direction is incorrect!")
    else:
        print("  >>> WARNING: Aerodynamic force too small to verify direction.")
    
    # 4. AoA Geometry Calculation Check
    print("\n4. Angle of Attack (AoA) Geometry Calculation")
    print(f"  {'Scenario':<20} | {'Thrust Dir':<15} | {'Rel Vel Dir':<15} | {'Exp AoA':<8} | {'Calc AoA':<8} | {'Sym Diff':<8} | {'Status':<6}")
    print("  " + "-" * 85)
    
    # Define scenarios
    r_test = np.array([R + 100000.0, 0, 0]) # 100km up
    t_test = 0.0
    env_state = vehicle.env.get_state_sim(r_test, t_test)
    wind = env_state['wind_velocity']
    
    scenarios = [
        ("Aligned (0 deg)",      [1, 0, 0], [1, 0, 0], 0.0),
        ("Pitch 45 deg",         [1, 0, 1], [1, 0, 0], 45.0),
        ("Pitch 90 deg",         [0, 0, 1], [1, 0, 0], 90.0),
        ("Yaw 30 deg",           [np.cos(np.radians(30)), np.sin(np.radians(30)), 0], [1, 0, 0], 30.0),
        ("Retrograde (180 deg)", [-1, 0, 0], [1, 0, 0], 180.0)
    ]
    
    for name, thrust_dir, rel_vel_dir, exp_aoa in scenarios:
        # Construct state such that v_rel aligns with rel_vel_dir
        v_rel_mag = 1000.0
        v_rel = np.array(rel_vel_dir) / np.linalg.norm(rel_vel_dir) * v_rel_mag
        v_phys = v_rel + wind
        
        state = np.concatenate([r_test, v_phys, [100000.0]])
        
        # A. Call get_aero_properties (Numeric / Simulation Mode)
        q, cos_alpha = vehicle.get_aero_properties(state, np.array(thrust_dir), t_test, stage_mode="boost")
        
        # B. Call get_aero_properties (Symbolic / Optimizer Mode)
        # This ensures the Optimizer sees the exact same AoA as the Simulation
        x_sym = ca.MX.sym('x', 7)
        u_sym = ca.MX.sym('u', 3)
        t_sym = ca.MX.sym('t', 1)
        q_sym, cos_sym = vehicle.get_aero_properties(x_sym, u_sym, t_sym, stage_mode="boost")
        f_aero_sym = ca.Function('f_aero_check', [x_sym, u_sym, t_sym], [cos_sym])
        cos_alpha_sym = float(f_aero_sym(state, np.array(thrust_dir), t_test))
        
        sym_diff = abs(cos_alpha - cos_alpha_sym)
        
        # Convert cos_alpha to degrees
        calc_aoa = np.degrees(np.arccos(np.clip(cos_alpha, -1.0, 1.0)))
        
        err = abs(calc_aoa - exp_aoa)
        status = "OK"
        if err > 1e-3 or sym_diff > 1e-9:
            status = "FAIL"
        
        # Format vectors for display
        t_str = f"[{thrust_dir[0]:.1f}, {thrust_dir[1]:.1f}, {thrust_dir[2]:.1f}]"
        v_str = f"[{rel_vel_dir[0]:.1f}, {rel_vel_dir[1]:.1f}, {rel_vel_dir[2]:.1f}]"
        
        print(f"  {name:<20} | {t_str:<15} | {v_str:<15} | {exp_aoa:<8.1f} | {calc_aoa:<8.1f} | {sym_diff:<8.1e} | {status:<6}")
    
    print("="*40 + "\n")

def verify_positioning(vehicle):
    """
    Debugs the Coordinate Systems (ECI/ECEF), Earth Rotation, and WGS84 Geometry.
    """
    print("\n" + "="*40)
    print("DEBUG: POSITIONING & GEODESY CHECK")
    print("="*40)
    
    env = vehicle.env
    config = env.config
    R_eq = config.earth_radius_equator
    f = config.earth_flattening
    
    # 1. Launch Site Verification
    print("1. Launch Site Initialization (t=0)")
    r_launch, v_launch = env.get_launch_site_state()
    
    # Calculate expected magnitude (WGS84 radius at launch latitude + altitude)
    lat_rad = np.radians(config.launch_latitude)
    e2 = 2*f - f**2
    N = R_eq / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    h_launch = config.launch_altitude
    
    # Cartesian conversion for verification
    x_exp = (N + h_launch) * np.cos(lat_rad) * np.cos(np.radians(config.launch_longitude))
    y_exp = (N + h_launch) * np.cos(lat_rad) * np.sin(np.radians(config.launch_longitude))
    z_exp = (N * (1 - e2) + h_launch) * np.sin(lat_rad)
    
    # Rotate to ECI (t=0)
    theta_0 = config.initial_rotation
    x_eci = x_exp * np.cos(theta_0) - y_exp * np.sin(theta_0)
    y_eci = x_exp * np.sin(theta_0) + y_exp * np.cos(theta_0)
    z_eci = z_exp
    
    r_exp = np.array([x_eci, y_eci, z_eci])
    pos_err = np.linalg.norm(r_launch - r_exp)
    
    print(f"  Launch Lat/Lon:  {config.launch_latitude:.4f} N, {config.launch_longitude:.4f} E")
    print(f"  Launch Alt:      {h_launch:.1f} m")
    print(f"  Calc ECI Pos:    [{r_launch[0]:.0f}, {r_launch[1]:.0f}, {r_launch[2]:.0f}]")
    print(f"  Position Error:  {pos_err:.2e} m")
    
    # Velocity Check (v = omega x r)
    omega = config.earth_omega_vector
    v_exp = np.cross(omega, r_launch)
    vel_err = np.linalg.norm(v_launch - v_exp)
    print(f"  Calc ECI Vel:    {np.linalg.norm(v_launch):.2f} m/s")
    print(f"  Velocity Error:  {vel_err:.2e} m/s")
    
    if pos_err < 1e-3 and vel_err < 1e-3:
        print("  >>> ✅ SUCCESS: Launch site initialization is correct.")
    else:
        print("  >>> ❌ FAILURE: Launch site calculation mismatch!")

    # 2. Earth Rotation Consistency
    print("\n2. Earth Rotation & Altitude Invariance")
    # A point fixed to the Earth's surface (rotating in ECI) should maintain constant altitude/density.
    times = [0.0, 3600.0, 43200.0] # 0, 1h, 12h
    print(f"  {'Time (s)':<10} | {'Rotation (deg)':<15} | {'Density (kg/m3)':<18} | {'Status':<6}")
    print("  " + "-" * 55)
    
    base_rho = None
    rot_success = True
    
    for t in times:
        theta = omega[2] * t + config.initial_rotation
        # Position of a point on Equator (Alt=0) at time t
        r_rot = np.array([R_eq * np.cos(theta), R_eq * np.sin(theta), 0.0])
        
        s = env.get_state_sim(r_rot, t)
        rho = s['density']
        
        if base_rho is None: base_rho = rho
        diff = abs(rho - base_rho)
        status = "OK" if diff < 1e-9 else "FAIL"
        if status == "FAIL": rot_success = False
        
        deg = np.degrees(theta) % 360
        print(f"  {t:<10.0f} | {deg:<15.1f} | {rho:<18.6f} | {status:<6}")

    if rot_success:
        print("  >>> ✅ SUCCESS: Environment rotates correctly with Earth.")
    else:
        print("  >>> ❌ FAILURE: Altitude/Density fluctuates with rotation!")
    
    # 3. WGS84 Geometry Check
    print("\n3. WGS84 Ellipsoid Geometry")
    R_pol = R_eq * (1.0 - f)
    
    # Test at 10km altitude to avoid surface clamping masking spherical errors
    h_test = 10000.0
    s_eq = env.get_state_sim(np.array([R_eq + h_test, 0, 0]), 0.0)
    s_pol = env.get_state_sim(np.array([0, 0, R_pol + h_test]), 0.0)
    
    print(f"  Equator Radius:  {R_eq:.1f} m (Test Alt: {h_test} m)")
    print(f"  Polar Radius:    {R_pol:.1f} m (Test Alt: {h_test} m)")
    print(f"  Density (Eq):    {s_eq['density']:.6e} kg/m3")
    print(f"  Density (Pol):   {s_pol['density']:.6e} kg/m3")
    
    if abs(s_eq['density'] - s_pol['density']) < 1e-9:
        print("  >>> ✅ SUCCESS: WGS84 Flattening is correctly implemented.")
    else:
        print("  >>> ❌ FAILURE: Model does not account for Earth flattening correctly.")
        
    print("="*40 + "\n")

def verify_propulsion(vehicle):
    """
    Debugs the Propulsion Model (ISP vs Pressure, Thrust calculation).
    Verifies that the vehicle produces correct Thrust and ISP at Sea Level and Vacuum.
    """
    print("\n" + "="*40)
    print("DEBUG: PROPULSION PERFORMANCE CHECK")
    print("="*40)
    
    R_e = vehicle.env.config.earth_radius_equator
    g0 = vehicle.env.config.g0
    
    # Test Cases: (Stage Name, Stage Config, Mode, Altitude, Expected Pressure Label)
    cases = [
        ("Stage 1 (SL)",  vehicle.config.stage_1, "boost", 0.0,      "1 atm"),
        ("Stage 1 (Vac)", vehicle.config.stage_1, "boost", 200000.0, "0 atm"),
        ("Stage 2 (Vac)", vehicle.config.stage_2, "ship",  200000.0, "0 atm")
    ]
    
    print(f"{'Case':<15} | {'Alt (km)':<8} | {'Press (Pa)':<10} | {'ISP (s)':<8} | {'Thrust (MN)':<12} | {'Exp Thr':<12} | {'Status':<6}")
    print("-" * 85)
    
    for name, stage, mode, alt, p_label in cases:
        # 1. Get Environment State
        r = np.array([R_e + alt, 0, 0])
        t = 0.0
        env = vehicle.env.get_state_sim(r, t)
        p_act = env['pressure']
        
        # 2. Run Dynamics (Throttle=100%, Vertical)
        # FIX: Set velocity to wind velocity to ensure zero relative velocity (Zero Drag).
        # This isolates the Thrust force for precise verification.
        v_test = env['wind_velocity']
        # State: [r, v, m]
        state = np.concatenate([r, v_test, [stage.dry_mass + stage.propellant_mass]])
        dyn = vehicle.get_dynamics(state, 1.0, np.array([1,0,0]), t, stage_mode=mode, scaling=None)
        
        # 3. Extract Thrust from Dynamics
        # F_thrust = m * (a_total - g)  (since Drag=0 at v=0)
        acc_total = dyn[3:6]
        g_vec = env['gravity']
        m_curr = state[6]
        
        f_net = acc_total * m_curr
        f_grav = g_vec * m_curr
        f_thrust_vec = f_net - f_grav
        f_thrust_mag = np.linalg.norm(f_thrust_vec)
        
        # 4. Calculate ISP from m_dot
        m_dot_act = -dyn[6] # m_dot is negative in dynamics
        isp_act = f_thrust_mag / (m_dot_act * g0) if m_dot_act > 1e-9 else 0.0
            
        # 5. Expected Values
        # Calculate expected ISP based on actual pressure found
        isp_exp = stage.isp_vac + (p_act / stage.p_sl) * (stage.isp_sl - stage.isp_vac)
        thr_exp = (stage.thrust_vac / stage.isp_vac) * isp_exp 
            
        # 6. Check
        thr_err = abs(f_thrust_mag - thr_exp) / (thr_exp + 1e-9)
        status = "OK" if thr_err < 1e-3 else "FAIL"
        
        print(f"{name:<15} | {alt/1000:<8.0f} | {p_act:<10.0f} | {isp_act:<8.1f} | {f_thrust_mag/1e6:<12.3f} | {thr_exp/1e6:<12.3f} | {status:<6}")

    print("="*40 + "\n")

def verify_staging_and_objective(opt_res, config, environment=None):
    """
    Verifies that staging mechanics (mass drop, kinematic continuity) 
    and the optimization objective (fuel minimization) are consistent.
    """
    print("\n" + "="*40)
    print("DEBUG: STAGING & OBJECTIVE CHECK")
    print("="*40)

    # 1. Staging Continuity (Pos/Vel)
    # Determine the state just before staging
    if "X2" in opt_res and opt_res.get("T2", 0.0) > 1e-4:
        x_prev_end = opt_res["X2"][:, -1]
        phase_name = "Coast (Phase 2)"
    else:
        x_prev_end = opt_res["X1"][:, -1]
        phase_name = "Boost (Phase 1)"
    
    # State just after staging (Start of Phase 3)
    x_stage3_start = opt_res["X3"][:, 0]
    
    # Position/Velocity should match exactly
    pos_diff = np.linalg.norm(x_prev_end[0:3] - x_stage3_start[0:3])
    vel_diff = np.linalg.norm(x_prev_end[3:6] - x_stage3_start[3:6])
    
    print(f"Staging Continuity ({phase_name} -> Ship):")
    print(f"  Position Gap:  {pos_diff:.4e} m")
    print(f"  Velocity Gap:  {vel_diff:.4e} m/s")
    
    if pos_diff < 1e-3 and vel_diff < 1e-3:
        print("  >>> ✅ SUCCESS: Kinematics are continuous (Point-Mass Model).")
    else:
        print("  >>> ❌ FAILURE: Discontinuity detected at staging!")

    # 1.5 Booster Fuel Check (Did we burn structure?)
    # In Phase 1/2, Mass = Booster Dry + Booster Prop + Ship Wet
    m_ship_wet = config.stage_2.dry_mass + config.stage_2.propellant_mass + config.payload_mass
    m_booster_limit = config.stage_1.dry_mass + m_ship_wet
    
    m_prev_end = x_prev_end[6]
    booster_margin = m_prev_end - m_booster_limit
    
    prop_1_total = config.stage_1.propellant_mass
    unused_pct = (booster_margin / prop_1_total) * 100.0
    
    print(f"\nBooster Propellant Check:")
    print(f"  Mass at Staging:     {m_prev_end:,.2f} kg")
    print(f"  Booster Limit:       {m_booster_limit:,.2f} kg (Struct + Upper Stage)")
    if booster_margin < -1e-3:
        print(f"  >>> ❌ FAILURE: Booster used more fuel than available! Deficit: {abs(booster_margin):.2f} kg")
    else:
        print(f"  >>> ✅ SUCCESS: Booster has propellant remaining ({booster_margin:,.2f} kg).")
        print(f"      Unused Propellant: {unused_pct:.4f}% (Discarded)")

    # 2. Mass Drop Logic
    m_stage3_start = x_stage3_start[6]
    
    # Expected: Mass resets to Ship Wet Mass (Stage 2 Dry + Prop + Payload)
    
    print(f"\nMass Logic:")
    print(f"  Mass before staging: {m_prev_end:,.2f} kg")
    print(f"  Mass after staging:  {m_stage3_start:,.2f} kg")
    print(f"  Expected Ship Wet:   {m_ship_wet:,.2f} kg")
    print(f"  Mass Dropped:        {m_prev_end - m_stage3_start:,.2f} kg")
    
    mass_err = abs(m_stage3_start - m_ship_wet)
    print(f"  Mass Reset Error:    {mass_err:.2e} kg")
    
    if mass_err < 1e-3:
        print("  >>> ✅ SUCCESS: Mass correctly reset to Ship Wet Mass.")
    else:
        print(f"  >>> ❌ FAILURE: Mass reset incorrect! Error: {mass_err:.2f} kg")

    # 3. Objective Analysis (Fuel Minimization)
    # We want to maximize final mass => minimize fuel used.
    m_final = opt_res["X3"][6, -1]
    m_dry_s2 = config.stage_2.dry_mass
    m_payload = config.payload_mass
    
    fuel_remaining = m_final - (m_dry_s2 + m_payload)
    
    print(f"\nOptimization Objective (Max Final Mass):")
    print(f"  Final Mass:      {m_final:,.2f} kg")
    print(f"  Dry Structure:   {m_dry_s2:,.2f} kg")
    print(f"  Payload:         {m_payload:,.2f} kg")
    print(f"  Fuel Remaining:  {fuel_remaining:,.2f} kg")
    
    # Calculate Delta-V Capacity of the remaining fuel
    # dV = ISP * g0 * ln(m_final / m_dry)
    # Use configured g0 if available, else standard
    g0 = environment.config.g0 if environment else 9.80665
    
    m_burnout = m_dry_s2 + m_payload
    if m_final > m_burnout and m_burnout > 0:
        dv_rem = config.stage_2.isp_vac * g0 * np.log(m_final / m_burnout)
        print(f"  Delta-V Capacity: {dv_rem:.1f} m/s (Safety Margin)")
    
    if fuel_remaining < -1e-3:
        print("  >>> ❌ WARNING: Negative fuel remaining! Mission infeasible.")
    elif fuel_remaining < 1000:
        print("  >>> NOTE: Fuel margins are very tight (<1t).")
    else:
        print("  >>> ✅ SUCCESS: Positive fuel margin. Optimizer found a valid solution.")

    # 4. Phase Duration & Burn Rate Consistency
    t1 = opt_res.get("T1", 0.0)
    t3 = opt_res.get("T3", 0.0)
    
    print(f"\nPhase Duration & Burn Rate Analysis:")
    print(f"  {'Phase':<15} | {'Duration':<10} | {'Fuel Used':<12} | {'Avg Flow':<10} | {'Rated Flow':<10} | {'Throttle':<10} | {'Status':<6}")
    print("  " + "-" * 85)

    def check_phase(name, t_dur, x_data, stage_config):
        if t_dur < 1.0:
            print(f"  {name:<15} | {t_dur:<10.2f} | {'-':<12} | {'-':<10} | {'-':<10} | {'-':<10} | SKIP")
            return True

        dm = x_data[6, 0] - x_data[6, -1]
        mdot_avg = dm / t_dur
        
        # Rated mdot = Thrust_vac / (Isp_vac * g0)
        mdot_rated = stage_config.thrust_vac / (stage_config.isp_vac * g0)
        
        throttle_avg = mdot_avg / mdot_rated
        throttle_pct = throttle_avg * 100.0
        
        # Check bounds (allow small epsilon for numerical noise/transients)
        is_valid = (config.sequence.min_throttle - 0.15 <= throttle_avg <= 1.15)
        status = "OK" if is_valid else "FAIL"
        
        print(f"  {name:<15} | {t_dur:<10.2f} | {dm:<12.0f} | {mdot_avg:<10.1f} | {mdot_rated:<10.1f} | {throttle_pct:<9.1f}% | {status:<6}")
        return is_valid

    p1_ok = check_phase("Phase 1 (Boost)", t1, opt_res["X1"], config.stage_1)
    p3_ok = check_phase("Phase 3 (Ship)", t3, opt_res["X3"], config.stage_2)

    if p1_ok and p3_ok:
        print("  >>> ✅ SUCCESS: Burn times and mass flows are physically consistent.")
    else:
        print("  >>> ❌ FAILURE: Detected impossible burn rates (Cheating detected).")

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
    verify_scaling_consistency(StarshipBlock2)
    verify_physics_consistency(veh, StarshipBlock2)
    verify_aerodynamics(veh)
    verify_propulsion(veh)
    verify_positioning(veh)
    verify_environment_consistency(veh)
    
    print("--- Running Optimization ---")
    opt_res = solve_optimal_trajectory(StarshipBlock2, veh, env)
    verify_staging_and_objective(opt_res, StarshipBlock2, env)
    
    print("--- Running Verification Simulation ---")
    sim_res = run_simulation(opt_res, veh, StarshipBlock2)
    
    print("--- Validating Trajectory ---")
    analysis.validate_trajectory(sim_res, StarshipBlock2, env)
    
    print("--- Analyzing Efficiency ---")
    analysis.analyze_delta_v_budget(sim_res, veh, StarshipBlock2)
    
    print("--- Plotting Results ---")
    analysis.plot_mission(opt_res, sim_res, env, StarshipBlock2)
    print("Done.")