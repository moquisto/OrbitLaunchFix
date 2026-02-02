import numpy as np
import casadi as ca
from scipy.interpolate import interp1d
from config import ScalingConfig

# ==============================================================================
# OPTIMIZER DEBUGGING (CasADi / Constraints)
# ==============================================================================

def check_path_constraints(sol, X, U, T_scaled, t_start_scaled, scaling, config, environment, vehicle, phase_mode):
    """
    Checks for violations of path constraints in the optimizer solution.
    """
    print(f"  [{phase_mode}] Checking Path Constraints [SOURCE: OPTIMIZER SOLUTION]...")
    
    # Unpack
    r = sol.value(X)[0:3, :] * scaling.length
    u_val = sol.value(U) if U is not None else None
    
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
    if u_val is not None:
        throttles = u_val[0, :]
        if np.any(throttles < config.sequence.min_throttle - 1e-3) or np.any(throttles > 1.0 + 1e-3):
            print(f"    ! THROTTLE VIOLATION: Range [{np.min(throttles):.2f}, {np.max(throttles):.2f}]")
            
        # 3. Direction Constraint
        dirs = u_val[1:, :]
        norms = np.linalg.norm(dirs, axis=0)
        err_norm = np.max(np.abs(norms - 1.0))
        if err_norm > 1e-3:
            print(f"    ! CONTROL VECTOR VIOLATION: Max Norm Error = {err_norm:.4f}")

    # 4. Structural Constraints (Max Q & G-Load)
    duration_scaled = sol.value(T_scaled)
    dt_scaled = duration_scaled / (X.shape[1] - 1)
    
    max_q = 0.0
    max_g = 0.0
    
    # Check all nodes, including the final one (critical for Max G at burnout)
    for k in range(X.shape[1]):
        x_k = sol.value(X[:, k])
        t_k_scaled = t_start_scaled + k * dt_scaled
        
        if u_val is not None:
            # Use control from current interval, or previous if at the very end
            u_k = u_val[:, min(k, u_val.shape[1] - 1)]
            u_th = u_k[0]
            u_dir = u_k[1:]
        else:
            u_th = 0.0
            u_dir = np.array([1,0,0])
            
        # Max Q
        q_val, _ = vehicle.get_aero_properties(x_k, u_dir, t_k_scaled, stage_mode=phase_mode, scaling=scaling)
        if q_val > max_q: max_q = q_val
        
        # G-Load
        derivs = vehicle.get_dynamics(x_k, u_th, u_dir, t_k_scaled, stage_mode=phase_mode, scaling=scaling)
        acc_phys = derivs[3:6] * (scaling.speed / scaling.time)
        
        r_phys = x_k[0:3] * scaling.length
        t_phys = t_k_scaled * scaling.time
        env_state = environment.get_state_sim(r_phys, t_phys)
        g_phys = env_state['gravity']
        
        g_load = np.linalg.norm(acc_phys - g_phys) / environment.config.g0
        if g_load > max_g: max_g = g_load

    print(f"    Structural: Max Q = {max_q/1000:.1f} kPa (Limit: {config.max_q_limit/1000:.1f}), Max G = {max_g:.2f} (Limit: {config.max_g_load:.1f})")

def print_debug_info(opti, sol, scaling, config, environment, vehicle, X1, U1, T1_scaled, X3, U3, T3_scaled):
    """
    Analyzes the failed optimization result to identify the cause.
    """
    print("\n" + "="*40)
    print("       OPTIMIZATION FAILURE DIAGNOSIS [SOURCE: OPTIMIZER]")
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
    weight_ship = m_ship_wet * environment.config.g0 # Reference gravity
    thrust_ship = config.stage_2.thrust_vac
    tw_ship = thrust_ship / weight_ship
    print(f"  Ship Init T/W:  {tw_ship:.2f}")
    
    # Flight Path Angle (Gamma) Analysis
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
    
    # 4. Solver Metrics
    print(f"\nSOLVER STATUS:")
    print(f"  The optimizer could not find a feasible solution.")
    
    # 5. Path Constraint Check
    check_path_constraints(sol, X1, U1, T1_scaled, 0.0, scaling, config, environment, vehicle, "boost")
    
    # Calculate Phase 3 start time
    t2_val = 0.0
    if config.sequence.separation_delay > 1e-4:
        t2_val = config.sequence.separation_delay / scaling.time
    t_start_p3 = sol.value(T1_scaled) + t2_val
    
    check_path_constraints(sol, X3, U3, T3_scaled, t_start_p3, scaling, config, environment, vehicle, "ship")
    print("="*40 + "\n")

def debug_optimization_structure(opti):
    """
    Inspects the sparsity of the Jacobian to ensure the problem is well-posed.
    """
    print("\n" + "="*40)
    print("DEBUG: OPTIMIZATION STRUCTURE (Jacobian) [SOURCE: CASADI]")
    print("="*40)
    
    J = None
    try:
        # Get Jacobian of Constraints (g) w.r.t Variables (x)
        J = ca.jacobian(opti.g, opti.x)
        
        n_con = J.size1()
        n_var = J.size2()
        nnz = J.nnz()
        density = (nnz / (n_con * n_var)) * 100.0
        
        print(f"  Constraints: {n_con}")
        print(f"  Variables:   {n_var}")
        print(f"  Non-zeros:   {nnz}")
        print(f"  Density:     {density:.4f}%")
        
        if density < 1.0:
            print("  >>> ✅ SUCCESS: Jacobian is sparse (Expected for Direct Collocation).")
        else:
            print("  >>> ⚠️ WARNING: Jacobian is dense! Check constraint formulation.")
            
    except Exception as e:
        print(f"  Could not inspect Jacobian sparsity: {e}")
        print("="*40 + "\n")
        return

    # Check magnitude of non-zeros to detect scaling issues
    try:
        # Evaluate via Function to avoid Opti context assertion errors on derived expressions
        f_J = ca.Function('f_J', [opti.x], [J])
        J_eval = f_J(opti.debug.value(opti.x))
        J_vals = np.abs(J_eval.nonzeros())
        
        if len(J_vals) > 0:
            min_grad = np.min(J_vals)
            max_grad = np.max(J_vals)
            print(f"  Gradient Range: {min_grad:.1e} to {max_grad:.1e}")
            
            cond_est = max_grad / (min_grad + 1e-16)
            if cond_est > 1e8:
                print(f"  >>> ⚠️ WARNING: Poor scaling detected (Range ~ {cond_est:.1e}). Solver may struggle.")
            else:
                print(f"  >>> ✅ SUCCESS: Gradient scaling is acceptable.")
    except Exception as e:
        msg = str(e)
        if "solved()" in msg or "Solver not initialized" in msg:
             pass # Scaling check skipped (requires solution)
        else:
             print(f"  Note: Could not inspect Jacobian values (Gradient Range). Error: {msg}")
        
    print("="*40 + "\n")

def check_variable_scaling(sol, vars_dict):
    """
    Checks if the optimized variables are well-scaled (close to O(1)).
    """
    print("\n" + "="*40)
    print("DEBUG: VARIABLE SCALING CHECK [SOURCE: OPTIMIZER]")
    print("="*40)
    
    for name, var in vars_dict.items():
        if var is None: continue
        try:
            val = sol.value(var)
            # Handle scalar vs array
            if np.isscalar(val) or val.size == 1:
                print(f"  {name:<5} | Scalar: {float(val):.4f}")
                if abs(val) < 0.01 or abs(val) > 100:
                    print(f"          >>> ⚠️ WARNING: Scaling might be off (Ideal ~1.0)")
            else:
                # Array
                abs_val = np.abs(val)
                mean_val = np.mean(abs_val)
                max_val = np.max(abs_val)
                min_val = np.min(abs_val)
                
                print(f"  {name:<5} | Range: [{min_val:.4f}, {max_val:.4f}] | Mean: {mean_val:.4f}")
                if mean_val < 0.01 or mean_val > 100:
                    print(f"          >>> ⚠️ WARNING: Scaling might be off (Ideal ~1.0)")
        except Exception as e:
            print(f"  {name:<5} | Could not evaluate: {e}")

    print("="*40 + "\n")

def verify_staging_and_objective(opt_res, config, environment=None):
    """
    Verifies that staging mechanics and the optimization objective are consistent.
    """
    print("\n" + "="*40)
    print("DEBUG: STAGING & OBJECTIVE CHECK [SOURCE: OPTIMIZER RESULT]")
    print("="*40)

    # 1. Staging Continuity (Pos/Vel)
    if "X2" in opt_res and opt_res.get("T2", 0.0) > 1e-4:
        x_prev_end = opt_res["X2"][:, -1]
        phase_name = "Coast (Phase 2)"
    else:
        x_prev_end = opt_res["X1"][:, -1]
        phase_name = "Boost (Phase 1)"
    
    x_stage3_start = opt_res["X3"][:, 0]
    
    pos_diff = np.linalg.norm(x_prev_end[0:3] - x_stage3_start[0:3])
    vel_diff = np.linalg.norm(x_prev_end[3:6] - x_stage3_start[3:6])
    
    print(f"Staging Continuity ({phase_name} -> Ship):")
    print(f"  Position Gap:  {pos_diff:.4e} m")
    print(f"  Velocity Gap:  {vel_diff:.4e} m/s")
    
    if pos_diff < 1e-3 and vel_diff < 1e-3:
        print("  >>> ✅ SUCCESS: Kinematics are continuous (Point-Mass Model).")
    else:
        print("  >>> ❌ FAILURE: Discontinuity detected at staging!")

    # 1.5 Booster Fuel Check
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

    # 3. Objective Analysis
    m_final = opt_res["X3"][6, -1]
    m_dry_s2 = config.stage_2.dry_mass
    m_payload = config.payload_mass
    
    fuel_remaining = m_final - (m_dry_s2 + m_payload)
    
    print(f"\nOptimization Objective (Max Final Mass):")
    print(f"  Final Mass:      {m_final:,.2f} kg")
    print(f"  Dry Structure:   {m_dry_s2:,.2f} kg")
    print(f"  Payload:         {m_payload:,.2f} kg")
    print(f"  Fuel Remaining:  {fuel_remaining:,.2f} kg")
    
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
        mdot_rated = stage_config.thrust_vac / (stage_config.isp_vac * g0)
        throttle_avg = mdot_avg / mdot_rated
        throttle_pct = throttle_avg * 100.0
        
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

# ==============================================================================
# PHYSICS ENGINE VERIFICATION (Consistency Checks)
# ==============================================================================

def verify_scaling_consistency(config):
    """
    Verifies that scaling factors produce numerically friendly values (O(1)).
    """
    print("\n" + "="*40)
    print("DEBUG: SCALING CONSISTENCY CHECK [SOURCE: CONFIG]")
    print("="*40)
    
    scaling = ScalingConfig() 
    
    print(f"Scaling Factors:")
    print(f"  Length: {scaling.length:.4e} m")
    print(f"  Speed:  {scaling.speed:.4e} m/s")
    print(f"  Time:   {scaling.time:.4e} s")
    print(f"  Mass:   {scaling.mass:.4e} kg")
    print(f"  Force:  {scaling.force:.4e} N")
    print("-" * 75)

    # ... (Rest of logic identical to original)
    # For brevity, I'm including the core logic structure.
    # In a real move, copy the full body.
    
    calc_speed = scaling.length / scaling.time
    calc_force = scaling.mass * (scaling.length / scaling.time**2)
    
    if abs(scaling.speed - calc_speed) > 1e-9:
        print(f"WARNING: Scaling speed mismatch! Config: {scaling.speed}, Calc: {calc_speed}")
    
    tests = [
        ("Radius (Surface)", scaling.length, scaling.length),
        ("Radius (Orbit)",   scaling.length + config.target_altitude, scaling.length),
        ("Velocity (Orbit)", scaling.speed, scaling.speed),
        ("Mass (Launch)",    config.launch_mass, scaling.mass),
        ("Mass (Dry Stg2)",  config.stage_2.dry_mass, scaling.mass),
        ("Thrust (Booster)", config.stage_1.thrust_vac, scaling.force),
        ("Thrust (Ship)",    config.stage_2.thrust_vac, scaling.force),
        ("Time (Mission)",   600.0, scaling.time),
        ("Gravity (Surface)", 9.81, scaling.length / scaling.time**2)
    ]
    
    print(f"{'Variable':<20} | {'Physical':<12} | {'Scaled':<10} | {'Unscaled':<12} | {'Rel Err':<10} | {'Status':<6}")
    print("-" * 85)
    
    overall_success = True
    for name, val_phys, scale_factor in tests:
        val_scaled = val_phys / scale_factor
        val_recovered = val_scaled * scale_factor
        is_reasonable = 1e-2 <= abs(val_scaled) <= 1e2
        err = abs(val_phys - val_recovered)
        
        if abs(val_phys) > 1e-15:
            rel_err = err / abs(val_phys)
            is_fail = rel_err > 1e-12
        else:
            is_fail = err > 1e-12
            rel_err = float('inf') if is_fail else 0.0
        
        status = "OK"
        if not is_reasonable: status = "WARN MAG"
        if is_fail: status = "FAIL"; overall_success = False
            
        print(f"{name:<20} | {val_phys:<12.4e} | {val_scaled:<10.4f} | {val_recovered:<12.4e} | {rel_err:<10.2e} | {status:<6}")

    if overall_success:
        print("\n>>> ✅ SUCCESS: Scaling logic is consistent.")
    else:
        print("\n>>> ❌ CRITICAL WARNING: Scaling logic failed round-trip check!")
    print("="*40 + "\n")

def verify_physics_consistency(vehicle, config):
    """
    Debugs the interface between Optimizer (CasADi) and Simulation (NumPy).
    """
    print("\n" + "="*40)
    print("DEBUG: PHYSICS ENGINE CONSISTENCY CHECK [SOURCE: SIM vs OPT]")
    print("="*40)
    
    R_e = vehicle.env.config.earth_radius_equator
    
    scenarios = [
        {"name": "Launch Pad (Static)", "r": [R_e, 0, 0], "v": [0, 0, 0], "m": config.launch_mass, "th": 1.0, "dir": [1, 0, 0], "t": 0.0, "mode": "boost"},
        {"name": "Max Q (Transonic)", "r": [R_e + 12000.0, 0, 0], "v": [0, 300.0, 300.0], "m": config.launch_mass * 0.8, "th": 1.0, "dir": [0, 0.707, 0.707], "t": 60.0, "mode": "boost"},
        {"name": "Coast (Staging)", "r": [R_e + 80000.0, 0, 0], "v": [0, 2000.0, 1000.0], "m": config.launch_mass * 0.4, "th": 0.0, "dir": [0, 1, 0], "t": 150.0, "mode": "coast"},
        {"name": "Ship (Vacuum)", "r": [R_e + 400000.0, 0, 0], "v": [0, 7500.0, 0.0], "m": config.stage_2.dry_mass + config.stage_2.propellant_mass, "th": 1.0, "dir": [0, 1, 0], "t": 500.0, "mode": "ship"}
    ]

    x_sym = ca.MX.sym('x', 7)
    u_th_sym = ca.MX.sym('th', 1)
    u_dir_sym = ca.MX.sym('dir', 3)
    t_sym = ca.MX.sym('t', 1)
    
    overall_success = True
    print(f"{'Scenario':<25} | {'Max Diff':<12} | {'Status':<10}")
    print("-" * 55)

    for case in scenarios:
        r_test = np.array(case['r'])
        v_test = np.array(case['v'])
        m_test = case['m']
        state_num = np.concatenate([r_test, v_test, [m_test]])
        
        th_test = case['th']
        dir_test = np.array(case['dir'])
        t_test = case['t']
        mode = case['mode']
        
        dyn_sim = vehicle.get_dynamics(state_num, th_test, dir_test, t_test, stage_mode=mode, scaling=None)
        
        dyn_sym_expr = vehicle.get_dynamics(x_sym, u_th_sym, u_dir_sym, t_sym, stage_mode=mode, scaling=None)
        f_dyn = ca.Function('f_dyn', [x_sym, u_th_sym, u_dir_sym, t_sym], [dyn_sym_expr])
        res_sym = np.array(f_dyn(state_num, th_test, dir_test, t_test)).flatten()
        
        max_diff = np.max(np.abs(dyn_sim - res_sym))
        status = "PASS" if max_diff < 1e-9 else "FAIL"
        if status == "FAIL": overall_success = False
            
        print(f"{case['name']:<25} | {max_diff:.2e}     | {status:<10}")

    if overall_success:
        print("\n>>> ✅ SUCCESS: Physics engines are consistent across all regimes.")
    else:
        print("\n>>> ❌ CRITICAL WARNING: Physics engines disagree!")
    print("="*40 + "\n")

def verify_environment_consistency(vehicle):
    """
    Debugs the Environment model (Symbolic vs Numeric).
    """
    print("\n" + "="*40)
    print("DEBUG: ENVIRONMENT MODEL CONSISTENCY CHECK [SOURCE: SIM vs OPT]")
    print("="*40)

    # Create symbolic wrapper to test get_state_opti
    r_sym = ca.MX.sym('r', 3)
    t_sym = ca.MX.sym('t', 1)
    state_sym = vehicle.env.get_state_opti(r_sym, t_sym)
    f_env = ca.Function('f_env_test', [r_sym, t_sym], 
                        [state_sym['density'], state_sym['pressure'], state_sym['gravity']])
    
    # Test Point: 10km altitude
    R_e = vehicle.env.config.earth_radius_equator
    r_test = np.array([R_e + 10000.0, 0, 0])
    t_test = 0.0
    
    # 1. Symbolic Eval
    res_sym = f_env(r_test, t_test)
    rho_sym = float(res_sym[0])
    p_sym = float(res_sym[1])
    g_sym = np.array(res_sym[2]).flatten()
    
    # 2. Numeric Eval
    state_num = vehicle.env.get_state_sim(r_test, t_test)
    rho_num = state_num['density']
    p_num = state_num['pressure']
    g_num = state_num['gravity']
    
    print(f"Test Altitude: 10.0 km")
    print(f"  Density:  Sym={rho_sym:.4e}, Num={rho_num:.4e} | Diff={abs(rho_sym-rho_num):.2e}")
    print(f"  Pressure: Sym={p_sym:.4e}, Num={p_num:.4e} | Diff={abs(p_sym-p_num):.2e}")
    print(f"  Gravity:  Sym={g_sym}, Num={g_num} | Max Diff={np.max(np.abs(g_sym-g_num)):.2e}")
    
    if abs(rho_sym - rho_num) < 1e-9 and np.max(np.abs(g_sym - g_num)) < 1e-9:
        print("  >>> ✅ SUCCESS: Environment models match.")
    else:
        print("  >>> ❌ FAILURE: Mismatch detected!")
    print("="*40 + "\n")

def verify_aerodynamics(vehicle):
    """
    Debugs the Aerodynamic model.
    """
    print("\n" + "="*40)
    print("DEBUG: AERODYNAMICS CHECK [SOURCE: VEHICLE MODEL]")
    print("="*40)
    
    # Test Mach numbers
    machs = [0.0, 0.8, 1.5, 5.0, 25.0]
    
    print(f"{'Mach':<6} | {'Stg1 Cd':<10} | {'Stg2 Cd':<10}")
    print("-" * 35)
    
    for m in machs:
        cd1 = float(vehicle.cd_interp_stage_1(m))
        cd2 = float(vehicle.cd_interp_stage_2(m))
        print(f"{m:<6.1f} | {cd1:<10.4f} | {cd2:<10.4f}")
        
    print("-" * 35)
    print("Crossflow Check (Mach 1.0, AoA 90 deg):")
    # Check crossflow
    cd_base = float(vehicle.cd_interp_stage_1(1.0))
    factor = vehicle.config.stage_1.aero.cd_crossflow_factor
    expected = cd_base + factor * (1.0**2) # sin(90)^2 = 1
    print(f"  Base Cd: {cd_base:.2f}, Factor: {factor:.1f}")
    print(f"  Expected Total Cd at 90deg: {expected:.2f}")
    print("="*40 + "\n")

def verify_positioning(vehicle):
    """
    Debugs the Coordinate Systems.
    """
    print("\n" + "="*40)
    print("DEBUG: POSITIONING & GEODESY CHECK [SOURCE: ENVIRONMENT]")
    print("="*40)
    
    r0, v0 = vehicle.env.get_launch_site_state()
    R_eq = vehicle.env.config.earth_radius_equator
    
    r_mag = np.linalg.norm(r0)
    v_mag = np.linalg.norm(v0)
    
    print(f"Launch Site (ECI):")
    print(f"  Radius:   {r_mag:.1f} m (Earth Eq: {R_eq:.1f} m)")
    print(f"  Velocity: {v_mag:.2f} m/s")
    
    if abs(r_mag - R_eq) < 50000:
        print("  >>> ✅ SUCCESS: Initial state is reasonable.")
    else:
        print("  >>> ⚠️ WARNING: Initial state looks suspicious.")
    print("="*40 + "\n")

def verify_propulsion(vehicle):
    """
    Debugs the Propulsion Model.
    """
    print("\n" + "="*40)
    print("DEBUG: PROPULSION PERFORMANCE CHECK [SOURCE: VEHICLE MODEL]")
    print("="*40)
    
    # Check Stage 1
    s1 = vehicle.config.stage_1
    f_vac = s1.thrust_vac
    f_sl = s1.thrust_sl # Derived property
    
    print(f"Stage 1 (Booster):")
    print(f"  Vac Thrust: {f_vac/1e6:.2f} MN (ISP={s1.isp_vac}s)")
    print(f"  SL Thrust:  {f_sl/1e6:.2f} MN (ISP={s1.isp_sl}s)")
    print(f"  Loss %:     {(1 - f_sl/f_vac)*100:.1f}%")
    
    # Check Stage 2
    s2 = vehicle.config.stage_2
    f_vac2 = s2.thrust_vac
    f_sl2 = s2.thrust_sl
    
    print(f"Stage 2 (Ship):")
    print(f"  Vac Thrust: {f_vac2/1e6:.2f} MN (ISP={s2.isp_vac}s)")
    print(f"  SL Thrust:  {f_sl2/1e6:.2f} MN (ISP={s2.isp_sl}s)")
    print("="*40 + "\n")

# ==============================================================================
# SIMULATION ANALYSIS (Post-Flight)
# ==============================================================================

def validate_trajectory(simulation_data, config, environment):
    """
    Performs numerical validation of the simulation results.
    """
    print("\n" + "="*40)
    print("POST-FLIGHT ANALYSIS REPORT [SOURCE: SIMULATION]")
    print("="*40)

    t = simulation_data['t']
    y = simulation_data['y']
    u = simulation_data['u']
    
    # 1. Terminal State Accuracy
    r_final = y[0:3, -1]
    v_final = y[3:6, -1]
    r_mag = np.linalg.norm(r_final)
    v_mag = np.linalg.norm(v_final)
    
    R_eq = environment.config.earth_radius_equator
    target_radius = R_eq + config.target_altitude
    
    print(f"TERMINAL STATE (t = {t[-1]:.1f} s)")
    print(f"  Orbital Radius: {r_mag/1000:.2f} km  (Target: {target_radius/1000:.2f} km)")
    print(f"  Velocity:       {v_mag:.2f} m/s")
    
    # 2. Max Q Check
    q_vals = []
    for i in range(len(t)):
        r_i = y[0:3, i]
        env_state = environment.get_state_sim(r_i, t[i])
        v_rel = y[3:6, i] - env_state['wind_velocity']
        q = 0.5 * env_state['density'] * np.linalg.norm(v_rel)**2
        q_vals.append(q)
    
    max_q = max(q_vals)
    print(f"\nSTRUCTURAL LOADS")
    print(f"  Max Q:          {max_q/1000:.2f} kPa (Limit: {config.max_q_limit/1000:.1f} kPa)")
    print("="*40 + "\n")

def analyze_delta_v_budget(simulation_data, vehicle, config):
    """
    Computes and prints a detailed Delta-V budget.
    """
    print("\n" + "="*40)
    print("DELTA-V BUDGET & EFFICIENCY ANALYSIS [SOURCE: SIMULATION]")
    print("="*40)
    
    t = simulation_data['t']
    y = simulation_data['y']
    u = simulation_data['u']
    
    dv_total = 0.0
    dv_gravity = 0.0
    dv_drag = 0.0
    
    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        r = y[0:3, i]
        v = y[3:6, i]
        m = y[6, i]
        throttle = u[0, i]
        
        env = vehicle.env.get_state_sim(r, t[i])
        
        # Determine Stage
        m_stg2_wet = config.stage_2.dry_mass + config.stage_2.propellant_mass + config.payload_mass
        stage = config.stage_1 if m > m_stg2_wet + 1000 else config.stage_2
            
        # Thrust DV
        if throttle > 0.01:
            isp_eff = stage.isp_vac + (env['pressure'] / stage.p_sl) * (stage.isp_sl - stage.isp_vac)
            thrust = throttle * stage.thrust_vac * (isp_eff / stage.isp_vac)
            dv_total += (thrust / m) * dt
            
        # Gravity Loss (Approx: Component of g opposing v)
        v_mag = np.linalg.norm(v)
        if v_mag > 1.0:
            v_dir = v / v_mag
            g_loss_inst = -np.dot(env['gravity'], v_dir)
            if g_loss_inst > 0:
                dv_gravity += g_loss_inst * dt
                
        # Drag Loss
        v_rel = v - env['wind_velocity']
        v_rel_mag = np.linalg.norm(v_rel)
        q = 0.5 * env['density'] * v_rel_mag**2
        mach = v_rel_mag / max(env['speed_of_sound'], 1.0)
        cd_data = stage.aero.mach_cd_table
        cd_base = np.interp(mach, cd_data[:,0], cd_data[:,1])
        
        # Add Crossflow Drag (AoA) to match Physics Engine
        sin_alpha_sq = 0.0
        if throttle > 0.01 and v_rel_mag > 1.0:
            thrust_dir = u[1:, i]
            if np.linalg.norm(thrust_dir) > 1e-9:
                u_thrust = thrust_dir / np.linalg.norm(thrust_dir)
                u_vel = v_rel / v_rel_mag
                cos_alpha = np.dot(u_thrust, u_vel)
                sin_alpha_sq = 1.0 - min(cos_alpha**2, 1.0)
        
        cd_total = cd_base + stage.aero.cd_crossflow_factor * sin_alpha_sq
        drag = q * stage.aero.reference_area * cd_total
        dv_drag += (drag / m) * dt

    print(f"  Total Delta-V Expended: {dv_total:.1f} m/s")
    print(f"  Gravity Loss (Approx):  {dv_gravity:.1f} m/s")
    print(f"  Drag Loss (Approx):     {dv_drag:.1f} m/s")
    print("="*40 + "\n")

def analyze_control_slew_rates(simulation_data):
    """
    Analyzes the angular rate of change of the thrust vector.
    """
    print("\n" + "="*40)
    print("CONTROL SLEW RATE ANALYSIS [SOURCE: SIMULATION]")
    print("="*40)
    
    t = simulation_data['t']
    u = simulation_data['u']
    
    max_slew_rate = 0.0
    max_slew_time = 0.0
    
    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        if dt < 1e-6: continue
        
        u1 = u[1:, i]
        u2 = u[1:, i+1]
        n1 = np.linalg.norm(u1)
        n2 = np.linalg.norm(u2)
        
        if n1 > 1e-9 and n2 > 1e-9:
            dot = np.clip(np.dot(u1/n1, u2/n2), -1.0, 1.0)
            rate = np.degrees(np.arccos(dot)) / dt
            if rate > max_slew_rate:
                max_slew_rate = rate
                max_slew_time = t[i]
    
    print(f"  Max Gimbal Rate:  {max_slew_rate:.2f} deg/s (at t={max_slew_time:.1f}s)")
    print("="*40 + "\n")

def analyze_trajectory_drift(optimization_data, simulation_data):
    """
    Quantifies the discretization error.
    """
    print("\n" + "="*40)
    print("TRAJECTORY DRIFT ANALYSIS [SOURCE: SIM vs OPT]")
    print("="*40)
    
    # 1. Reconstruct Optimizer Time Grid & States
    t_opt_list = []
    x_opt_list = []
    
    N1 = optimization_data['X1'].shape[1]
    t1 = np.linspace(0, optimization_data['T1'], N1)
    t_opt_list.append(t1)
    x_opt_list.append(optimization_data['X1'])
    
    current_t = optimization_data['T1']
    if 'X2' in optimization_data and optimization_data.get('T2', 0) > 0:
        N2 = optimization_data['X2'].shape[1]
        t2 = np.linspace(current_t, current_t + optimization_data['T2'], N2)
        t_opt_list.append(t2)
        x_opt_list.append(optimization_data['X2'])
        current_t += optimization_data['T2']
        
    N3 = optimization_data['X3'].shape[1]
    t3 = np.linspace(current_t, current_t + optimization_data['T3'], N3)
    t_opt_list.append(t3)
    x_opt_list.append(optimization_data['X3'])
    
    t_opt_full = np.concatenate(t_opt_list)
    x_opt_full = np.hstack(x_opt_list)
    
    # 2. Interpolate Simulation Data onto Optimizer Grid
    t_sim = simulation_data['t']
    y_sim = simulation_data['y']
    
    # Handle duplicates in t_sim (caused by phase concatenation)
    t_sim_unique, unique_indices = np.unique(t_sim, return_index=True)
    y_sim_unique = y_sim[:, unique_indices]
    
    if len(t_sim_unique) < 2:
        print("  Not enough simulation data for drift analysis.")
        return

    f_sim = interp1d(t_sim_unique, y_sim_unique, axis=1, kind='linear', bounds_error=False, fill_value="extrapolate")
    y_sim_interp = f_sim(t_opt_full)
    
    # 3. Calculate Errors
    pos_err = np.linalg.norm(x_opt_full[0:3, :] - y_sim_interp[0:3, :], axis=0)
    vel_err = np.linalg.norm(x_opt_full[3:6, :] - y_sim_interp[3:6, :], axis=0)
    mass_err = np.abs(x_opt_full[6, :] - y_sim_interp[6, :])
    
    print(f"  Position Drift: Max = {np.max(pos_err):.2f} m, Avg = {np.mean(pos_err):.2f} m")
    print(f"  Velocity Drift: Max = {np.max(vel_err):.2f} m/s, Avg = {np.mean(vel_err):.2f} m/s")
    print(f"  Mass Drift:     Max = {np.max(mass_err):.2f} kg, Avg = {np.mean(mass_err):.2f} kg")
    
    # 4. Pass/Fail Judgment
    # Thresholds: 1km Position, 20m/s Velocity, 100kg Mass (0.01% of propellant)
    if np.max(pos_err) > 1000.0 or np.max(vel_err) > 20.0 or np.max(mass_err) > 100.0:
        print("  >>> ❌ FAILURE: Significant divergence between Optimizer and Simulation.")
    else:
        print("  >>> ✅ SUCCESS: Simulation concurs with Optimizer (High Fidelity).")
        
    print("="*40 + "\n")

def analyze_energy_balance(simulation_data, vehicle):
    """
    Verifies the Work-Energy Theorem: Delta E_mech = Work_non_conservative.
    This checks the consistency of the integrator and physics engine.
    """
    print("\n" + "="*40)
    print("DEBUG: ENERGY BALANCE CHECK [SOURCE: SIMULATION]")
    print("="*40)
    
    t = simulation_data['t']
    y = simulation_data['y']
    u = simulation_data['u']
    
    if len(t) < 2:
        print("  Not enough data for energy analysis.")
        print("="*40 + "\n")
        return
    
    # Constants
    mu = vehicle.env.config.earth_mu
    R_e = vehicle.env.config.earth_radius_equator
    J2 = vehicle.env.config.j2_constant
    use_j2 = vehicle.env.config.use_j2_perturbation

    def get_potential_energy(r_vec):
        r = np.linalg.norm(r_vec)
        pe = -mu / r
        
        if use_j2:
            z = r_vec[2]
            sin_phi = z / r
            # J2 Potential: U_j2 = (mu * J2 * Re^2 / (2 * r^3)) * (3 * sin_phi^2 - 1)
            term_j2 = (mu * J2 * R_e**2) / (2 * r**3) * (3 * sin_phi**2 - 1)
            pe += term_j2
            
        return pe
    
    def get_specific_power(idx):
        r_i = y[0:3, idx]
        v_i = y[3:6, idx]
        m_i = y[6, idx]
        th_i = u[0, idx]
        dir_i = u[1:, idx]
        t_i = t[idx]
        
        # Determine phase for correct drag model
        m_ship_wet = vehicle.config.stage_2.dry_mass + vehicle.config.stage_2.propellant_mass + vehicle.config.payload_mass
        
        if m_i > m_ship_wet + 1000:
            phase = "coast" if th_i < 0.01 else "boost"
        else:
            phase = "coast_2" if th_i < 0.01 else "ship"
        
        dyn = vehicle.get_dynamics(y[:, idx], th_i, dir_i, t_i, stage_mode=phase, scaling=None)
        acc_total = dyn[3:6]
        
        env = vehicle.env.get_state_sim(r_i, t_i)
        acc_nc = acc_total - env['gravity']
        
        return np.dot(acc_nc, v_i)

    # Arrays for integration
    work_done = 0.0
    error = 0.0 # Initialize to ensure scope validity
    
    # Initial Energy
    r0 = y[0:3, 0]
    v0 = y[3:6, 0]
    E_start = 0.5 * np.dot(v0, v0) + get_potential_energy(r0)
    
    print(f"{'Time':<10} | {'Mech Energy (MJ/kg)':<20} | {'Work Done (MJ/kg)':<20} | {'Error (J/kg)':<15}")
    print("-" * 75)
    
    # Check at 5 points
    check_indices = np.linspace(0, len(t)-1, 6, dtype=int)
    
    power_prev = get_specific_power(0)
    
    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        power_curr = get_specific_power(i+1)
        
        # Trapezoidal Integration (Higher accuracy than Euler)
        work_done += 0.5 * (power_prev + power_curr) * dt
        power_prev = power_curr
        
        if i+1 in check_indices:
            # Calculate Energy at i+1
            r_next = y[0:3, i+1]
            v_next = y[3:6, i+1]
            E_curr = 0.5 * np.dot(v_next, v_next) + get_potential_energy(r_next)
            
            delta_E = E_curr - E_start
            error = abs(delta_E - work_done)
            
            print(f"{t[i+1]:<10.1f} | {E_curr/1e6:<20.4f} | {(E_start + work_done)/1e6:<20.4f} | {error:<15.2f}")

    print("-" * 75)
    if error < 100.0: # 100 J/kg is very small compared to MJ/kg specific energy
        print(">>> ✅ SUCCESS: Energy is conserved (within integration error).")
    else:
        print(">>> ⚠️ WARNING: Energy drift detected. Check integrator tolerance or physics model.")
    print("="*40 + "\n")

def analyze_instantaneous_orbit(simulation_data, environment):
    """
    Calculates Keplerian orbital elements at key mission events.
    """
    print("\n" + "="*40)
    print("DEBUG: ORBITAL ELEMENTS EVOLUTION [SOURCE: SIMULATION]")
    print("="*40)
    
    t = simulation_data['t']
    y = simulation_data['y']
    mu = environment.config.earth_mu
    R_e = environment.config.earth_radius_equator
    
    if len(t) < 1:
        print("  Not enough data.")
        print("="*40 + "\n")
        return
    
    print(f"{'Event':<15} | {'Time (s)':<8} | {'Alt (km)':<8} | {'Vel (m/s)':<9} | {'SMA (km)':<8} | {'Ecc':<6} | {'Inc (deg)':<9}")
    print("-" * 85)
    
    indices = [0, len(t)//2, len(t)-1] # Start, Mid, End
    labels = ["Launch", "Mid-Flight", "Orbit Injection"]
    
    for idx, label in zip(indices, labels):
        r_vec = y[0:3, idx]
        v_vec = y[3:6, idx]
        time_val = t[idx]
        
        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)
        
        # Specific Energy
        E = 0.5 * v**2 - mu / r
        
        # Semi-Major Axis
        if abs(E) > 1e-6:
            a = -mu / (2 * E)
        else:
            a = np.inf # Parabolic
            
        # Eccentricity Vector
        # e_vec = ( (v^2 - mu/r)*r - (r.v)*v ) / mu
        e_vec = ((v**2 - mu/r)*r_vec - np.dot(r_vec, v_vec)*v_vec) / mu
        ecc = np.linalg.norm(e_vec)
        
        # Inclination
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)
        if h > 1e-6:
            inc = np.degrees(np.arccos(h_vec[2] / h))
        else:
            inc = 0.0
            
        alt = r - R_e
        
        print(f"{label:<15} | {time_val:<8.1f} | {alt/1000:<8.1f} | {v:<9.1f} | {a/1000:<8.1f} | {ecc:<6.4f} | {inc:<9.2f}")
        
    print("="*40 + "\n")

def analyze_integrator_steps(simulation_data):
    """
    Analyzes the time steps taken by the variable-step integrator.
    Small steps indicate stiffness or rapid dynamics.
    """
    print("\n" + "="*40)
    print("DEBUG: INTEGRATOR STEP SIZE ANALYSIS [SOURCE: SIMULATION]")
    print("="*40)
    
    t = simulation_data['t']
    if len(t) < 2:
        print("  Not enough data points.")
        return

    dt = np.diff(t)
    dt = dt[dt > 1e-9] # Filter out zero-length steps caused by phase concatenation
    
    if len(dt) == 0:
        print("  Not enough valid time steps.")
        return
    
    print(f"  Total Steps: {len(t)}")
    print(f"  Min Step:    {np.min(dt):.2e} s")
    print(f"  Max Step:    {np.max(dt):.2e} s")
    print(f"  Mean Step:   {np.mean(dt):.2e} s")
    
    # Check for stiffness (very small steps)
    if np.min(dt) < 1e-4:
        print("  >>> ⚠️ WARNING: Very small time steps detected (<1e-4s). Physics might be stiff.")
    else:
        print("  >>> ✅ SUCCESS: Time steps indicate well-behaved dynamics.")
    print("="*40 + "\n")

def analyze_control_saturation(simulation_data, config):
    """
    Checks how often the vehicle is riding the throttle limits.
    """
    print("\n" + "="*40)
    print("DEBUG: CONTROL SATURATION ANALYSIS [SOURCE: SIMULATION]")
    print("="*40)
    
    u = simulation_data['u']
    throttles = u[0, :]
    min_th = config.sequence.min_throttle
    
    # Count samples at limits (within tolerance)
    tol = 0.01
    
    # Exclude Coasting (Engines Off) from "Min Throttle" stats
    is_active = throttles > 0.01
    
    at_min = np.sum((throttles < min_th + tol) & is_active)
    at_max = np.sum(throttles > 1.0 - tol)
    total = len(throttles)
    
    if total == 0: return

    pct_min = 100.0 * at_min / total
    pct_max = 100.0 * at_max / total
    
    print(f"  Throttle at Min ({min_th*100:.0f}%): {pct_min:.1f}% of flight (Active Limiting)")
    print(f"  Throttle at Max (100%): {pct_max:.1f}% of flight")
    
    if pct_max > 95.0:
        print("  >>> NOTE: Engines running at max power almost continuously.")
    if pct_min > 20.0:
        print("  >>> NOTE: Significant throttling detected (Max Q or Landing?).")
        
    print("="*40 + "\n")

def analyze_guidance_accuracy(guess, opt_res):
    """
    Compares the initial guess (Guidance) with the final optimized result.
    """
    print("\n" + "="*40)
    print("DEBUG: GUIDANCE VS OPTIMALITY CHECK [SOURCE: GUESS vs OPT]")
    print("="*40)
    
    # Compare MECO
    t1_guess = guess['T1']
    t1_opt = opt_res['T1']
    
    # Compare Final Mass
    # X3 is [7, N]
    m_final_guess = guess['X3'][6, -1]
    m_final_opt = opt_res['X3'][6, -1]
    
    print(f"  MECO Time:   Guess={t1_guess:.1f}s, Opt={t1_opt:.1f}s (Diff: {t1_opt-t1_guess:+.1f}s)")
    print(f"  Final Mass:  Guess={m_final_guess:,.0f}kg, Opt={m_final_opt:,.0f}kg (Diff: {m_final_opt-m_final_guess:+.0f}kg)")
    
    if abs(m_final_opt - m_final_guess) > 50000: # 50 tons
         print("  >>> ⚠️ NOTE: Guidance mass estimate was significantly off (>50t).")
    else:
         print("  >>> ✅ SUCCESS: Guidance provided a good mass estimate.")
    print("="*40 + "\n")