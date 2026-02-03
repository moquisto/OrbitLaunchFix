import numpy as np
import casadi as ca
from scipy.interpolate import interp1d
from config import ScalingConfig

# --- Formatting Helpers ---
class Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"

def _print_header(title, source=None):
    print(f"\n{Style.CYAN}" + "="*80 + f"{Style.RESET}")
    if source:
        print(f"{Style.BOLD} {title.upper():<55} {Style.RESET}{Style.DIM}[SOURCE: {source.upper()}]{Style.RESET}")
    else:
        print(f"{Style.BOLD} {title.upper()}{Style.RESET}")
    print(f"{Style.CYAN}" + "="*80 + f"{Style.RESET}")

def _print_sub_header(title):
    print(f"\n{Style.BOLD}--- {title} ---{Style.RESET}")

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
        print(f"    {Style.RED}! ALTITUDE VIOLATION: Ellipsoid Metric = {min_val:.6f} (~{violation:.1f} m under) at Node {min_idx}{Style.RESET}")
        
    # 2. Control Constraints
    if u_val is not None:
        throttles = u_val[0, :]
        if np.any(throttles < config.sequence.min_throttle - 1e-3) or np.any(throttles > 1.0 + 1e-3):
            print(f"    {Style.RED}! THROTTLE VIOLATION: Range [{np.min(throttles):.2f}, {np.max(throttles):.2f}]{Style.RESET}")
            
        # 3. Direction Constraint
        dirs = u_val[1:, :]
        norms = np.linalg.norm(dirs, axis=0)
        err_norm = np.max(np.abs(norms - 1.0))
        if err_norm > 1e-3:
            print(f"    {Style.RED}! CONTROL VECTOR VIOLATION: Max Norm Error = {err_norm:.4f}{Style.RESET}")

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
    _print_header("Optimization Failure Diagnosis", "Optimizer")
    
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
        print(f"  >>> {Style.RED}CRITICAL FAILURE: Final mass is BELOW dry mass limit by {m_dry_s2 - m_f:.2f} kg.{Style.RESET}")
        print(f"  >>> {Style.RED}The vehicle lacks the Delta-V to reach the target orbit with current constraints.{Style.RESET}")
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
        print(f"  >>> {Style.RED}CRITICAL PHYSICS ISSUE: Staging T/W is {tw_ship:.2f} (< 1.0) and Flight Path Angle is low ({gamma_deg:.1f} deg).{Style.RESET}")
        print(f"      {Style.RED}The upper stage cannot fight gravity and will sink into the atmosphere.{Style.RESET}")
    
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
    _print_header("Optimization Structure (Jacobian)", "CasADi")
    
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
            print(f"  >>> {Style.GREEN}✅ SUCCESS: Jacobian is sparse (Expected for Direct Collocation).{Style.RESET}")
        else:
            print(f"  >>> {Style.YELLOW}⚠️ WARNING: Jacobian is dense! Check constraint formulation.{Style.RESET}")
            
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
                print(f"  >>> {Style.YELLOW}⚠️ WARNING: Poor scaling detected (Range ~ {cond_est:.1e}). Solver may struggle.{Style.RESET}")
            else:
                print(f"  >>> {Style.GREEN}✅ SUCCESS: Gradient scaling is acceptable.{Style.RESET}")
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
    _print_header("Variable Scaling Check", "Optimizer")
    
    print(f"{Style.BOLD}{'Variable':<10} | {'Component':<10} | {'Range (Scaled)':<20} | {'Mean':<8} | {'Status':<6}{Style.RESET}")
    print(f"{Style.CYAN}" + "-" * 65 + f"{Style.RESET}")

    for name, var in vars_dict.items():
        if var is None: continue
        try:
            val = sol.value(var)
            
            # 1. Time Scalars (T1, T2, T3)
            if name.startswith("T"):
                status = f"{Style.GREEN}OK{Style.RESET}"
                if val < 0.01 or val > 100: status = f"{Style.YELLOW}WARN{Style.RESET}"
                print(f"{name:<10} | {'Time':<10} | {val:<20.4f} | {val:<8.4f} | {status:<6}")
                continue

            # 2. State Vectors (X1, X2, X3) -> [r, v, m]
            if name.startswith("X"):
                # Position (Rows 0-2)
                r_vals = np.linalg.norm(val[0:3, :], axis=0)
                r_min, r_max, r_mean = np.min(r_vals), np.max(r_vals), np.mean(r_vals)
                status_r = f"{Style.GREEN}OK{Style.RESET}" if 0.1 <= r_mean <= 10.0 else f"{Style.YELLOW}WARN{Style.RESET}"
                print(f"{name:<10} | {'Position':<10} | [{r_min:.4f}, {r_max:.4f}]   | {r_mean:<8.4f} | {status_r:<6}")
                
                # Velocity (Rows 3-5)
                v_vals = np.linalg.norm(val[3:6, :], axis=0)
                v_min, v_max, v_mean = np.min(v_vals), np.max(v_vals), np.mean(v_vals)
                status_v = f"{Style.GREEN}OK{Style.RESET}" if 0.1 <= v_mean <= 10.0 else f"{Style.YELLOW}WARN{Style.RESET}"
                print(f"{'':<10} | {'Velocity':<10} | [{v_min:.4f}, {v_max:.4f}]   | {v_mean:<8.4f} | {status_v:<6}")
                
                # Mass (Row 6)
                m_vals = val[6, :]
                m_min, m_max, m_mean = np.min(m_vals), np.max(m_vals), np.mean(m_vals)
                status_m = f"{Style.GREEN}OK{Style.RESET}" if 0.1 <= m_mean <= 10.0 else f"{Style.YELLOW}WARN{Style.RESET}"
                print(f"{'':<10} | {'Mass':<10} | [{m_min:.4f}, {m_max:.4f}]   | {m_mean:<8.4f} | {status_m:<6}")
                continue

            # 3. Control Vectors (U1, U3) -> [throttle, ux, uy, uz]
            if name.startswith("U"):
                # Throttle (Row 0)
                th_vals = val[0, :]
                th_min, th_max, th_mean = np.min(th_vals), np.max(th_vals), np.mean(th_vals)
                status_th = f"{Style.GREEN}OK{Style.RESET}" # Throttle is 0-1 by definition, usually fine
                print(f"{name:<10} | {'Throttle':<10} | [{th_min:.4f}, {th_max:.4f}]   | {th_mean:<8.4f} | {status_th:<6}")
                
                # Direction (Rows 1-3)
                # Should be unit vectors, so norm should be ~1.0
                dir_vals = np.linalg.norm(val[1:, :], axis=0)
                d_min, d_max, d_mean = np.min(dir_vals), np.max(dir_vals), np.mean(dir_vals)
                status_d = f"{Style.GREEN}OK{Style.RESET}" if 0.9 <= d_mean <= 1.1 else f"{Style.YELLOW}WARN{Style.RESET}"
                print(f"{'':<10} | {'Direction':<10} | [{d_min:.4f}, {d_max:.4f}]   | {d_mean:<8.4f} | {status_d:<6}")
                continue
            
            # Fallback for unknown variables
            if np.isscalar(val) or val.size == 1:
                print(f"{name:<10} | {'Scalar':<10} | {float(val):<20.4f} | {float(val):<8.4f} | {'?':<6}")
            else:
                mean_val = np.mean(np.abs(val))
                print(f"{name:<10} | {'Array':<10} | [Size: {val.size}]       | {mean_val:<8.4f} | {'?':<6}")

        except Exception as e:
            print(f"  {name:<10} | Could not evaluate: {e}")

    print(f"{Style.CYAN}" + "-" * 65 + f"{Style.RESET}")
    print("  * Ideal scaling is O(1) (0.1 to 10.0).")
    print("  * 'WARN' indicates potential numerical stiffness for IPOPT.")
    print("="*40 + "\n")

def verify_staging_and_objective(opt_res, config, environment=None):
    """
    Verifies that staging mechanics and the optimization objective are consistent.
    """
    _print_header("Staging & Objective Check", "Optimizer Result")

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
        print(f"  >>> {Style.GREEN}✅ SUCCESS: Kinematics are continuous (Point-Mass Model).{Style.RESET}")
    else:
        print(f"  >>> {Style.RED}❌ FAILURE: Discontinuity detected at staging!{Style.RESET}")

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
        print(f"  >>> {Style.RED}❌ FAILURE: Booster used more fuel than available! Deficit: {abs(booster_margin):.2f} kg{Style.RESET}")
    else:
        print(f"  >>> {Style.GREEN}✅ SUCCESS: Booster has propellant remaining ({booster_margin:,.2f} kg).{Style.RESET}")
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
        print(f"  >>> {Style.GREEN}✅ SUCCESS: Mass correctly reset to Ship Wet Mass.{Style.RESET}")
    else:
        print(f"  >>> {Style.RED}❌ FAILURE: Mass reset incorrect! Error: {mass_err:.2f} kg{Style.RESET}")

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
        print(f"  >>> {Style.RED}❌ WARNING: Negative fuel remaining! Mission infeasible.{Style.RESET}")
    elif fuel_remaining < 1000:
        print(f"  >>> {Style.YELLOW}NOTE: Fuel margins are very tight (<1t).{Style.RESET}")
    else:
        print(f"  >>> {Style.GREEN}✅ SUCCESS: Positive fuel margin. Optimizer found a valid solution.{Style.RESET}")

    # 4. Phase Duration & Burn Rate Consistency
    t1 = opt_res.get("T1", 0.0)
    t3 = opt_res.get("T3", 0.0)
    
    print(f"\n{Style.BOLD}Phase Duration & Burn Rate Analysis:{Style.RESET}")
    print(f"  {Style.BOLD}{'Phase':<15} | {'Duration':<10} | {'Fuel Used':<12} | {'Avg Flow':<10} | {'Rated Flow':<10} | {'Throttle':<10} | {'Status':<6}{Style.RESET}")
    print(f"  {Style.CYAN}" + "-" * 85 + f"{Style.RESET}")

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
        status = f"{Style.GREEN}OK{Style.RESET}" if is_valid else f"{Style.RED}FAIL{Style.RESET}"
        
        print(f"  {name:<15} | {t_dur:<10.2f} | {dm:<12.0f} | {mdot_avg:<10.1f} | {mdot_rated:<10.1f} | {throttle_pct:<9.1f}% | {status:<6}")
        return is_valid

    p1_ok = check_phase("Phase 1 (Boost)", t1, opt_res["X1"], config.stage_1)
    p3_ok = check_phase("Phase 3 (Ship)", t3, opt_res["X3"], config.stage_2)

    if p1_ok and p3_ok:
        print(f"  >>> {Style.GREEN}✅ SUCCESS: Burn times and mass flows are physically consistent.{Style.RESET}")
    else:
        print(f"  >>> {Style.RED}❌ FAILURE: Detected impossible burn rates (Cheating detected).{Style.RESET}")

    print("="*40 + "\n")

# ==============================================================================
# PHYSICS ENGINE VERIFICATION (Consistency Checks)
# ==============================================================================

def verify_scaling_consistency(config):
    """
    Verifies that scaling factors produce numerically friendly values (O(1)).
    """
    _print_header("Scaling Consistency Check", "Config")
    
    scaling = ScalingConfig() 
    
    print(f"Scaling Factors:")
    print(f"  Length: {scaling.length:.4e} m")
    print(f"  Speed:  {scaling.speed:.4e} m/s")
    print(f"  Time:   {scaling.time:.4e} s")
    print(f"  Mass:   {scaling.mass:.4e} kg")
    print(f"  Force:  {scaling.force:.4e} N")
    print(f"{Style.CYAN}" + "-" * 75 + f"{Style.RESET}")

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
    
    print(f"{Style.BOLD}{'Variable':<20} | {'Physical':<12} | {'Scaled':<10} | {'Unscaled':<12} | {'Rel Err':<10} | {'Status':<6}{Style.RESET}")
    print(f"{Style.CYAN}" + "-" * 85 + f"{Style.RESET}")
    
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
        
        status = f"{Style.GREEN}OK{Style.RESET}"
        if not is_reasonable: status = f"{Style.YELLOW}WARN MAG{Style.RESET}"
        if is_fail: status = f"{Style.RED}FAIL{Style.RESET}"; overall_success = False
            
        print(f"{name:<20} | {val_phys:<12.4e} | {val_scaled:<10.4f} | {val_recovered:<12.4e} | {rel_err:<10.2e} | {status:<6}")

    if overall_success:
        print(f"\n>>> {Style.GREEN}✅ SUCCESS: Scaling logic is consistent.{Style.RESET}")
    else:
        print(f"\n>>> {Style.RED}❌ CRITICAL WARNING: Scaling logic failed round-trip check!{Style.RESET}")
    print("="*40 + "\n")

def verify_physics_consistency(vehicle, config):
    """
    Debugs the interface between Optimizer (CasADi) and Simulation (NumPy).
    """
    _print_header("Physics Engine Consistency Check", "Sim vs Opt")
    
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
    print(f"{Style.BOLD}{'Scenario':<25} | {'Max Diff':<12} | {'Status':<10}{Style.RESET}")
    print(f"{Style.CYAN}" + "-" * 55 + f"{Style.RESET}")

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
        status = f"{Style.GREEN}PASS{Style.RESET}" if max_diff < 1e-9 else f"{Style.RED}FAIL{Style.RESET}"
        if max_diff >= 1e-9: overall_success = False
            
        print(f"{case['name']:<25} | {max_diff:.2e}     | {status:<10}")

    if overall_success:
        print(f"\n>>> {Style.GREEN}✅ SUCCESS: Physics engines are consistent across all regimes.{Style.RESET}")
    else:
        print(f"\n>>> {Style.RED}❌ CRITICAL WARNING: Physics engines disagree!{Style.RESET}")
    print("="*40 + "\n")

def verify_environment_consistency(vehicle):
    """
    Debugs the Environment model (Symbolic vs Numeric).
    """
    _print_header("Environment Model Consistency Check", "Sim vs Opt")

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
        print(f"  >>> {Style.GREEN}✅ SUCCESS: Environment models match.{Style.RESET}")
    else:
        print(f"  >>> {Style.RED}❌ FAILURE: Mismatch detected!{Style.RESET}")
    print("="*40 + "\n")

def verify_aerodynamics(vehicle):
    """
    Debugs the Aerodynamic model.
    """
    _print_header("Aerodynamics Check", "Vehicle Model")
    
    # 1. Drag Coefficient & Force Estimate (Stage 1)
    # Assume a reference Max Q condition to make drag numbers tangible
    ref_q = 35000.0 # 35 kPa
    ref_area = vehicle.config.stage_1.aero.reference_area
    
    print(f"Reference Condition: Max Q = {ref_q/1000:.0f} kPa, Area = {ref_area:.1f} m^2")
    print(f"{Style.BOLD}{'Mach':<6} | {'Stg1 Cd':<8} | {'Drag (kN)':<10} | {'Stg2 Cd':<8}{Style.RESET}")
    print(f"{Style.CYAN}" + "-" * 45 + f"{Style.RESET}")
    
    # Sweep Mach numbers more finely around transonic
    machs = [0.0, 0.5, 0.9, 1.1, 1.5, 3.0, 5.0, 10.0, 25.0]
    
    for m in machs:
        cd1 = float(vehicle.cd_interp_stage_1(m))
        cd2 = float(vehicle.cd_interp_stage_2(m))
        
        # Calculate Drag Force at this Mach (assuming constant Max Q for comparison)
        # F_drag = Q * A * Cd
        drag_force = ref_q * ref_area * cd1
        
        print(f"{m:<6.1f} | {cd1:<8.4f} | {drag_force/1000:<10.1f} | {cd2:<8.4f}")
        
    print(f"{Style.CYAN}" + "-" * 45 + f"{Style.RESET}")

    # 2. Angle of Attack Sensitivity (Crossflow)
    # Shows how much drag increases when the rocket tilts
    print("\nAngle of Attack Sensitivity (at Mach 1.5):")
    m_test = 1.5
    cd_base = float(vehicle.cd_interp_stage_1(m_test))
    factor = vehicle.config.stage_1.aero.cd_crossflow_factor
    
    print(f"  Base Cd (0 deg): {cd_base:.4f}")
    print(f"  Crossflow Factor: {factor:.1f}")
    print(f"{Style.BOLD}{'Alpha (deg)':<12} | {'Total Cd':<10} | {'Increase':<10}{Style.RESET}")
    print(f"{Style.CYAN}" + "-" * 38 + f"{Style.RESET}")
    
    alphas = [0, 2, 5, 10, 20, 45, 90]
    for alpha in alphas:
        sin_a = np.sin(np.radians(alpha))
        cd_total = cd_base + factor * (sin_a**2)
        pct_inc = ((cd_total - cd_base) / cd_base) * 100.0
        print(f"{alpha:<12} | {cd_total:<10.4f} | +{pct_inc:<9.0f}%")

    # 3. Interpolation Sanity Check
    # Check for negative values or NaNs in a dense sweep
    dense_mach = np.linspace(0, 25, 100)
    cd_vals = [float(vehicle.cd_interp_stage_1(m)) for m in dense_mach]
    if any(c < 0 for c in cd_vals):
        print(f"\n  >>> {Style.YELLOW}⚠️ WARNING: Negative Cd detected in interpolation!{Style.RESET}")
    else:
        print(f"\n  >>> {Style.GREEN}✅ SUCCESS: Cd interpolation is positive and stable.{Style.RESET}")

    print("="*40 + "\n")

def verify_positioning(vehicle):
    """
    Debugs the Coordinate Systems.
    """
    _print_header("Positioning & Geodesy Check", "Environment")
    
    r0, v0 = vehicle.env.get_launch_site_state()
    R_eq = vehicle.env.config.earth_radius_equator
    
    r_mag = np.linalg.norm(r0)
    v_mag = np.linalg.norm(v0)
    
    print(f"Launch Site (ECI):")
    print(f"  Radius:   {r_mag:.1f} m (Earth Eq: {R_eq:.1f} m)")
    print(f"  Velocity: {v_mag:.2f} m/s")
    
    if abs(r_mag - R_eq) < 50000:
        print(f"  >>> {Style.GREEN}✅ SUCCESS: Initial state is reasonable.{Style.RESET}")
    else:
        print(f"  >>> {Style.YELLOW}⚠️ WARNING: Initial state looks suspicious.{Style.RESET}")
    print("="*40 + "\n")

def verify_propulsion(vehicle):
    """
    Debugs the Propulsion Model with detailed performance metrics.
    """
    _print_header("Propulsion Performance Check", "Vehicle Model")
    
    g0 = vehicle.env.config.g0
    
    # --- Stage 1 (Booster) ---
    s1 = vehicle.config.stage_1
    m_launch = vehicle.config.launch_mass
    
    # Mass Flow
    m_dot_1 = s1.thrust_vac / (s1.isp_vac * g0)
    t_burn_1 = s1.propellant_mass / m_dot_1
    
    # T/W
    tw_sl = s1.thrust_sl / (m_launch * g0)
    
    print(f"Stage 1 (Booster):")
    print(f"  Mass Flow Rate: {m_dot_1:,.1f} kg/s (at 100% Throttle)")
    print(f"  Burn Time:      {t_burn_1:.1f} s (Continuous 100%)")
    print(f"  Thrust (Vac):   {s1.thrust_vac/1e6:.2f} MN (ISP={s1.isp_vac:.0f}s)")
    print(f"  Thrust (SL):    {s1.thrust_sl/1e6:.2f} MN (ISP={s1.isp_sl:.0f}s)")
    print(f"  SL Efficiency:  {s1.thrust_sl/s1.thrust_vac*100:.1f}%")
    print(f"  Liftoff T/W:    {tw_sl:.2f} (Ref g={g0:.3f})")
    
    if tw_sl < 1.0:
        print(f"  >>> {Style.RED}⚠️ WARNING: T/W < 1.0 at Sea Level! Rocket will not lift off.{Style.RESET}")
    
    print(f"{Style.CYAN}" + "-" * 40 + f"{Style.RESET}")
    
    # --- Stage 2 (Ship) ---
    s2 = vehicle.config.stage_2
    m_s2_wet = s2.dry_mass + s2.propellant_mass + vehicle.config.payload_mass
    
    # Mass Flow
    m_dot_2 = s2.thrust_vac / (s2.isp_vac * g0)
    t_burn_2 = s2.propellant_mass / m_dot_2
    
    # T/W
    tw_vac = s2.thrust_vac / (m_s2_wet * g0)
    
    # Local T/W (at approx staging altitude of 70km)
    # Gravity decreases with altitude: g ~ g0 * (Re / (Re+h))^2
    r_staging = vehicle.env.config.earth_radius_equator + 70000.0
    g_local = np.linalg.norm(vehicle.env.get_state_sim([r_staging,0,0], 0)['gravity'])
    tw_local = s2.thrust_vac / (m_s2_wet * g_local)
    
    print(f"Stage 2 (Ship):")
    print(f"  Mass Flow Rate: {m_dot_2:,.1f} kg/s (at 100% Throttle)")
    print(f"  Burn Time:      {t_burn_2:.1f} s (Continuous 100%)")
    print(f"  Thrust (Vac):   {s2.thrust_vac/1e6:.2f} MN (ISP={s2.isp_vac:.0f}s)")
    print(f"  Thrust (SL):    {s2.thrust_sl/1e6:.2f} MN (ISP={s2.isp_sl:.0f}s)")
    print(f"  SL Efficiency:  {s2.thrust_sl/s2.thrust_vac*100:.1f}% (Flow Separation Penalty)")
    print(f"  Staging T/W:    {tw_vac:.2f} (Ref g={g0:.3f})")
    print(f"  Local T/W:      {tw_local:.2f} (at 70km, g={g_local:.2f})")
    
    if tw_vac < 1.0:
         print(f"  >>> {Style.YELLOW}NOTE: Staging T/W < 1.0. Requires lofted trajectory (Gamma > 0).{Style.RESET}")

    print("="*40 + "\n")

# ==============================================================================
# SIMULATION ANALYSIS (Post-Flight)
# ==============================================================================

def validate_trajectory(simulation_data, config, environment):
    """
    Performs numerical validation of the simulation results with detailed orbital mechanics.
    """
    _print_header("Post-Flight Analysis Report", "Simulation")

    t = simulation_data['t']
    y = simulation_data['y']
    u = simulation_data['u']
    
    # Constants
    mu = environment.config.earth_mu
    R_eq = environment.config.earth_radius_equator
    
    # 1. Terminal State & Orbital Elements
    r_final = y[0:3, -1]
    v_final = y[3:6, -1]
    r_mag = np.linalg.norm(r_final)
    v_mag = np.linalg.norm(v_final)
    
    # Specific Angular Momentum
    h_vec = np.cross(r_final, v_final)
    h_mag = np.linalg.norm(h_vec)
    
    # Specific Energy
    energy = (v_mag**2)/2 - mu/r_mag
    
    # Semi-Major Axis
    if abs(energy) > 1e-6:
        sma = -mu / (2 * energy)
    else:
        sma = np.inf
        
    # Eccentricity Vector
    e_vec = ((v_mag**2 - mu/r_mag)*r_final - np.dot(r_final, v_final)*v_final) / mu
    ecc = np.linalg.norm(e_vec)
    
    # Inclination
    inc_deg = np.degrees(np.arccos(h_vec[2] / h_mag))
    
    # Apoapsis / Periapsis Altitudes
    r_p = sma * (1 - ecc)
    r_a = sma * (1 + ecc)
    alt_p = r_p - R_eq
    alt_a = r_a - R_eq
    
    # Target Comparison
    target_alt = config.target_altitude
    target_inc = config.target_inclination if config.target_inclination is not None else environment.config.launch_latitude
    
    print(f"ORBITAL INJECTION ACCURACY (t = {t[-1]:.1f} s)")
    print(f"  {Style.BOLD}{'Metric':<15} | {'Actual':<12} | {'Target':<12} | {'Error':<10}{Style.RESET}")
    print(f"  {Style.CYAN}" + "-" * 55 + f"{Style.RESET}")
    print(f"  {'Alt (Spherical)':<15} | {((r_mag - R_eq)/1000):<12.2f} | {target_alt/1000:<12.2f} | {(r_mag - R_eq - target_alt)/1000:<+8.2f}")
    print(f"  {'Periapsis (km)':<15} | {alt_p/1000:<12.2f} | {target_alt/1000:<12.2f} | {(alt_p - target_alt)/1000:<+8.2f}")
    print(f"  {'Apoapsis (km)':<15} | {alt_a/1000:<12.2f} | {target_alt/1000:<12.2f} | {(alt_a - target_alt)/1000:<+8.2f}")
    print(f"  {'Eccentricity':<15} | {ecc:<12.5f} | {'0.00000':<12} | {ecc:<+8.5f}")
    print(f"  {'Inclination':<15} | {inc_deg:<12.4f} | {target_inc:<12.4f} | {inc_deg - target_inc:<+8.4f}")
    
    # 2. Structural Loads Analysis
    max_q = 0.0
    max_g = 0.0
    
    for i in range(len(t)):
        r_i = y[0:3, i]
        v_i = y[3:6, i]
        m_i = y[6, i]
        
        # Environment
        env_state = environment.get_state_sim(r_i, t[i])
        
        # Max Q
        v_rel = y[3:6, i] - env_state['wind_velocity']
        q = 0.5 * env_state['density'] * np.linalg.norm(v_rel)**2
        if q > max_q: max_q = q
        
        # Max G (Thrust / Weight)
        throttle = u[0, i]
        if throttle > 0.01:
            # Determine stage
            m_stg2_wet = config.stage_2.dry_mass + config.stage_2.propellant_mass + config.payload_mass
            stage = config.stage_1 if m_i > m_stg2_wet + 1000 else config.stage_2
            
            isp_eff = stage.isp_vac + (env_state['pressure'] / stage.p_sl) * (stage.isp_sl - stage.isp_vac)
            thrust = throttle * stage.thrust_vac * (isp_eff / stage.isp_vac)
            g_load = (thrust / m_i) / environment.config.g0
            if g_load > max_g: max_g = g_load

    print(f"\nSTRUCTURAL SAFETY MARGINS")
    print(f"  Max Q:          {max_q/1000:.2f} kPa (Limit: {config.max_q_limit/1000:.1f} kPa) | Margin: {(config.max_q_limit - max_q)/1000:.2f} kPa")
    print(f"  Max G-Load:     {max_g:.2f} g    (Limit: {config.max_g_load:.1f} g)     | Margin: {config.max_g_load - max_g:.2f} g")
    
    # Tolerances for reporting (avoid false positives from floating point noise)
    if max_q > config.max_q_limit + 100.0: # 100 Pa tolerance
        print(f"  >>> {Style.RED}⚠️ WARNING: Max Q limit exceeded!{Style.RESET}")
    if max_g > config.max_g_load + 0.01: # 0.01g tolerance
        print(f"  >>> {Style.RED}⚠️ WARNING: G-Load limit exceeded!{Style.RESET}")
        
    print("="*40 + "\n")

def analyze_delta_v_budget(simulation_data, vehicle, config):
    """
    Computes and prints a detailed Delta-V budget, broken down by stage.
    """
    _print_header("Delta-V Budget & Efficiency Analysis", "Simulation")
    
    t = simulation_data['t']
    y = simulation_data['y']
    u = simulation_data['u']
    
    # Initialize accumulators
    stats = {
        "boost": {"ideal": 0.0, "actual": 0.0, "grav": 0.0, "drag": 0.0, "steer": 0.0},
        "ship":  {"ideal": 0.0, "actual": 0.0, "grav": 0.0, "drag": 0.0, "steer": 0.0}
    }
    
    m_stg2_wet = config.stage_2.dry_mass + config.stage_2.propellant_mass + config.payload_mass
    
    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        if dt < 1e-6: continue
        
        r = y[0:3, i]
        v = y[3:6, i]
        m = y[6, i]
        throttle = u[0, i]
        
        # Determine Phase
        # Heuristic: If mass > Ship Wet + buffer, we are in Booster phase
        if m > m_stg2_wet + 1000:
            phase = "boost"
            stage = config.stage_1
        else:
            phase = "ship"
            stage = config.stage_2
            
        # Environment
        env = vehicle.env.get_state_sim(r, t[i])
        
        # --- 1. Forces ---
        # Thrust
        f_thrust = 0.0
            
        if throttle > 0.01:
            isp_eff = stage.isp_vac + (env['pressure'] / stage.p_sl) * (stage.isp_sl - stage.isp_vac)
            thrust = throttle * stage.thrust_vac * (isp_eff / stage.isp_vac)
            f_thrust = thrust
            
            # Actual Delta-V (Thrust Acceleration)
            dv_actual = (thrust / m) * dt
            stats[phase]["actual"] += dv_actual
            
            # Ideal Delta-V (Rocket Equation Potential: Vac ISP * dm/m)
            # dm = F / (Isp_eff * g0) * dt
            g0 = vehicle.env.config.g0
            dm = (thrust / (isp_eff * g0)) * dt
            # Ideal dV uses Vacuum ISP to show potential
            stats[phase]["ideal"] += (stage.isp_vac * g0) * (dm / m)
            
        # Gravity Loss (Approx: Component of g opposing v)
        v_mag = np.linalg.norm(v)
        if v_mag > 1.0:
            v_dir = v / v_mag
            g_loss_inst = -np.dot(env['gravity'], v_dir)
            # Only count as loss if gravity is opposing motion
            # (Technically gravity always does work, but for budget we care about ascent penalty)
            stats[phase]["grav"] += -np.dot(env['gravity'], v_dir) * dt
                
        # Drag Loss
        v_rel = v - env['wind_velocity']
        v_rel_mag = np.linalg.norm(v_rel)
        q = 0.5 * env['density'] * v_rel_mag**2
        
        # Re-calculate Cd
        mach = v_rel_mag / max(env['speed_of_sound'], 1.0)
        cd_data = stage.aero.mach_cd_table
        cd_base = np.interp(mach, cd_data[:,0], cd_data[:,1])
        
        # AoA
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
        stats[phase]["drag"] += (drag / m) * dt
        
        # Steering Loss (Cosine Loss)
        # Loss = F/m * (1 - cos(alpha_inertial))
        if throttle > 0.01 and v_mag > 1.0:
             thrust_dir = u[1:, i]
             u_thrust = thrust_dir / np.linalg.norm(thrust_dir)
             u_v_inertial = v / v_mag
             cos_steer = np.dot(u_thrust, u_v_inertial)
             cos_steer = max(-1.0, min(1.0, cos_steer))
             loss_steer = (f_thrust / m) * (1.0 - cos_steer) * dt
             stats[phase]["steer"] += loss_steer

    # Print Table
    print(f"{Style.BOLD}{'Metric':<15} | {'Booster':<12} | {'Ship':<12} | {'Total':<12}{Style.RESET}")
    print(f"{Style.CYAN}" + "-" * 55 + f"{Style.RESET}")
    
    def p(key):
        b = stats['boost'][key]
        s = stats['ship'][key]
        return f"{b:<12.1f} | {s:<12.1f} | {b+s:<12.1f}"
        
    print(f"{'Ideal Delta-V':<15} | {p('ideal')}")
    print(f"{'Actual Delta-V':<15} | {p('actual')}")
    print(f"{'Gravity Loss':<15} | {p('grav')}")
    print(f"{'Drag Loss':<15} | {p('drag')}")
    print(f"{'Steering Loss':<15} | {p('steer')}")
    print(f"{Style.CYAN}" + "-" * 55 + f"{Style.RESET}")
    
    # ISP Loss (Difference between Ideal Vac and Actual)
    isp_loss_b = stats['boost']['ideal'] - stats['boost']['actual']
    isp_loss_s = stats['ship']['ideal'] - stats['ship']['actual']
    total_isp_loss = isp_loss_b + isp_loss_s
    
    print(f"{'ISP/Backpres.':<15} | {isp_loss_b:<12.1f} | {isp_loss_s:<12.1f} | {total_isp_loss:<12.1f}")
    print("="*40 + "\n")

def analyze_control_slew_rates(simulation_data):
    """
    Analyzes the angular rate of change of the thrust vector.
    Filters out numerical noise and phase discontinuities.
    """
    _print_header("Control Slew Rate Analysis", "Simulation")
    
    t = simulation_data['t']
    u = simulation_data['u']
    
    max_slew_rate = 0.0
    max_slew_time = 0.0
    
    # Thresholds
    dt_min = 0.1 # Ignore steps smaller than 100ms (likely integrator sub-steps or phase boundaries)
    
    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        if dt < dt_min: continue
        
        u1 = u[1:, i]
        u2 = u[1:, i+1]
        n1 = np.linalg.norm(u1)
        n2 = np.linalg.norm(u2)
        
        if n1 > 1e-9 and n2 > 1e-9:
            dot = np.clip(np.dot(u1/n1, u2/n2), -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(dot))
            
            # Filter out massive jumps (likely phase reset)
            if angle_deg > 20.0: continue 
            
            rate = angle_deg / dt
            if rate > max_slew_rate:
                max_slew_rate = rate
                max_slew_time = t[i]
    
    print(f"  Max Gimbal Rate:  {max_slew_rate:.2f} deg/s (at t={max_slew_time:.1f}s)")
    
    if max_slew_rate > 10.0:
        print(f"  >>> {Style.YELLOW}⚠️ WARNING: Slew rate exceeds typical TVC limits (10 deg/s).{Style.RESET}")
        print(f"      Consider adding rate constraints to the optimizer.")
    else:
        print(f"  >>> {Style.GREEN}✅ SUCCESS: Control rates are within physical limits.{Style.RESET}")
        
    print("="*40 + "\n")

def analyze_trajectory_drift(optimization_data, simulation_data):
    """
    Quantifies the discretization error.
    """
    _print_header("Trajectory Drift Analysis", "Sim vs Opt")
    
    # 1. Identify Phases in Optimization Data
    phases_opt = []
    
    # Phase 1
    N1 = optimization_data['X1'].shape[1]
    t1_end = optimization_data['T1']
    phases_opt.append({
        "t": np.linspace(0, t1_end, N1),
        "x": optimization_data['X1'],
        "label": "Phase 1"
    })
    
    current_t = t1_end
    # Phase 2
    if 'X2' in optimization_data and optimization_data.get('T2', 0) > 1e-4:
        N2 = optimization_data['X2'].shape[1]
        t2_end = current_t + optimization_data['T2']
        phases_opt.append({
            "t": np.linspace(current_t, t2_end, N2),
            "x": optimization_data['X2'],
            "label": "Phase 2"
        })
        current_t = t2_end
        
    # Phase 3
    N3 = optimization_data['X3'].shape[1]
    t3_end = current_t + optimization_data['T3']
    phases_opt.append({
        "t": np.linspace(current_t, t3_end, N3),
        "x": optimization_data['X3'],
        "label": "Phase 3"
    })
    
    # 2. Split Simulation Data into Segments
    t_sim_full = simulation_data['t']
    y_sim_full = simulation_data['y']
    
    # Find split indices where time does not increase (duplicate points at boundaries)
    split_indices = [0] + (np.where(np.diff(t_sim_full) <= 1e-9)[0] + 1).tolist() + [len(t_sim_full)]
    
    sim_segments = []
    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i+1]
        sim_segments.append({
            "t": t_sim_full[start:end],
            "y": y_sim_full[:, start:end]
        })
        
    # 3. Compare Phase by Phase
    if len(phases_opt) != len(sim_segments):
        print(f"  ⚠️ Mismatch in phase count: Opt={len(phases_opt)}, Sim={len(sim_segments)}")
        print("  Cannot perform detailed drift analysis.")
        return

    max_pos_err = 0.0
    max_vel_err = 0.0
    max_mass_err = 0.0
    
    for i, (p_opt, p_sim) in enumerate(zip(phases_opt, sim_segments)):
        if len(p_sim['t']) < 2: continue
            
        f_sim = interp1d(p_sim['t'], p_sim['y'], axis=1, kind='linear', bounds_error=False, fill_value="extrapolate")
        y_sim_interp = f_sim(p_opt['t'])
        
        pos_err = np.linalg.norm(p_opt['x'][0:3, :] - y_sim_interp[0:3, :], axis=0)
        vel_err = np.linalg.norm(p_opt['x'][3:6, :] - y_sim_interp[3:6, :], axis=0)
        mass_err = np.abs(p_opt['x'][6, :] - y_sim_interp[6, :])
        
        max_pos_err = max(max_pos_err, np.max(pos_err))
        max_vel_err = max(max_vel_err, np.max(vel_err))
        max_mass_err = max(max_mass_err, np.max(mass_err))
        
        print(f"  {p_opt['label']} Drift: Pos={np.max(pos_err):.2f}m, Vel={np.max(vel_err):.2f}m/s, Mass={np.max(mass_err):.2f}kg")

    print(f"{Style.CYAN}" + "-" * 40 + f"{Style.RESET}")
    print(f"  Max Position Drift: {max_pos_err:.2f} m")
    print(f"  Max Velocity Drift: {max_vel_err:.2f} m/s")
    print(f"  Max Mass Drift:     {max_mass_err:.2f} kg")
    
    # 4. Pass/Fail Judgment
    if max_pos_err > 2000.0 or max_vel_err > 30.0 or max_mass_err > 500.0:
        print(f"  >>> {Style.RED}❌ FAILURE: Significant divergence between Optimizer and Simulation.{Style.RESET}")
    else:
        print(f"  >>> {Style.GREEN}✅ SUCCESS: Simulation concurs with Optimizer (High Fidelity).{Style.RESET}")
        
    print("="*40 + "\n")

def analyze_energy_balance(simulation_data, vehicle):
    """
    Verifies the Work-Energy Theorem: Delta E_mech = Work_non_conservative.
    This checks the consistency of the integrator and physics engine.
    """
    _print_header("Energy Balance Check", "Simulation")
    
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
    E_curr = E_start
    
    print(f"{Style.BOLD}{'Time':<10} | {'Mech Energy (MJ/kg)':<20} | {'Work Done (MJ/kg)':<20} | {'Error (J/kg)':<15}{Style.RESET}")
    print(f"{Style.CYAN}" + "-" * 75 + f"{Style.RESET}")
    
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

    print(f"{Style.CYAN}" + "-" * 75 + f"{Style.RESET}")
    
    rel_error = error / abs(E_curr) if abs(E_curr) > 1.0 else 0.0
    
    if rel_error < 1e-4: # 0.01% tolerance
        print(f">>> {Style.GREEN}✅ SUCCESS: Energy is conserved (Rel Error: {rel_error*100:.5f}%).{Style.RESET}")
    else:
        print(f">>> {Style.YELLOW}⚠️ WARNING: Energy drift detected ({rel_error*100:.4f}%). Check integrator tolerance or physics model.{Style.RESET}")
    print("="*40 + "\n")

def analyze_instantaneous_orbit(simulation_data, environment):
    """
    Calculates Keplerian orbital elements at key mission events.
    """
    _print_header("Orbital Elements Evolution", "Simulation")
    
    t = simulation_data['t']
    y = simulation_data['y']
    mu = environment.config.earth_mu
    R_e = environment.config.earth_radius_equator
    f = environment.config.earth_flattening
    
    if len(t) < 2:
        print("  Not enough data.")
        print("="*40 + "\n")
        return
    
    # --- 1. Detect Events ---
    # Liftoff
    idx_liftoff = 0
    
    # Max Q (Calculate dynamic pressure profile)
    q_vals = []
    for i in range(len(t)):
        r_i = y[0:3, i]
        v_i = y[3:6, i]
        env_state = environment.get_state_sim(r_i, t[i])
        v_rel = v_i - env_state['wind_velocity']
        q = 0.5 * env_state['density'] * np.linalg.norm(v_rel)**2
        q_vals.append(q)
    idx_maxq = np.argmax(q_vals)
    
    # Staging (Largest Mass Drop)
    mass = y[6, :]
    dm = np.diff(mass)
    idx_staging = np.argmin(dm) # Index before the drop
    if dm[idx_staging] > -1000: # Threshold to ignore burn mass loss
        idx_staging = -1
        
    # Injection
    idx_seco = len(t) - 1
    
    events = [
        ("Liftoff", idx_liftoff),
        ("Max Q", idx_maxq),
    ]
    if idx_staging > 0 and idx_staging < len(t)-1:
        events.append(("MECO/Staging", idx_staging))
    events.append(("Orbit Injection", idx_seco))
    
    # Sort by time index
    events = sorted(list(set(events)), key=lambda x: x[1])
    
    # --- 2. Print Table ---
    print(f"  * Altitudes are Spherical (Relative to Equatorial Radius: {R_e/1000:.1f} km).")
    print(f"  * Negative values at Liftoff are due to Earth's oblateness at launch latitude.")
    print(f"{Style.BOLD}{'Event':<16} | {'Time':<6} | {'Alt(Sph)':<8} | {'Vel':<7} | {'FPA':<5} | {'Apogee':<8} | {'Perigee':<8} | {'Inc':<6}{Style.RESET}")
    print(f"{Style.BOLD}{'':<16} | {'(s)':<6} | {'(km)':<8} | {'(m/s)':<7} | {'(deg)':<5} | {'(km)':<8} | {'(km)':<8} | {'(deg)':<6}{Style.RESET}")
    print(f"{Style.CYAN}" + "-" * 85 + f"{Style.RESET}")
    
    for label, idx in events:
        r_vec = y[0:3, idx]
        v_vec = y[3:6, idx]
        time_val = t[idx]
        
        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)
        
        # Calculate Spherical Altitude
        # Consistent with Apogee/Perigee (which use R_e) and Optimizer Target.
        alt = r - R_e
        
        # Specific Energy
        E = 0.5 * v**2 - mu / r
        
        # Semi-Major Axis
        if abs(E) > 1e-6:
            a = -mu / (2 * E)
        else:
            a = np.inf 
            
        # Eccentricity Vector
        e_vec = ((v**2 - mu/r)*r_vec - np.dot(r_vec, v_vec)*v_vec) / mu
        ecc = np.linalg.norm(e_vec)
        
        # Inclination
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)
        inc = np.degrees(np.arccos(h_vec[2] / h)) if h > 1e-6 else 0.0
        
        # Flight Path Angle (Gamma)
        sin_gamma = np.dot(r_vec, v_vec) / (r * v)
        gamma = np.degrees(np.arcsin(np.clip(sin_gamma, -1.0, 1.0)))
        
        # Apogee / Perigee
        if ecc < 1.0:
            r_a = a * (1 + ecc)
            r_p = a * (1 - ecc)
            alt_a = r_a - R_e
            alt_p = r_p - R_e
        else:
            r_p = a * (1 - ecc)
            alt_p = r_p - R_e
            alt_a = np.inf
            
        # Formatting
        alt_str = f"{alt/1000:<8.1f}"
        vel_str = f"{v:.0f}"
        fpa_str = f"{gamma:.1f}"
        apo_str = f"{alt_a/1000:.1f}" if alt_a != np.inf else "Inf"
        peri_str = f"{alt_p/1000:.1f}"
        inc_str = f"{inc:.2f}"
        
        print(f"{label:<16} | {time_val:<6.1f} | {alt_str} | {vel_str:<7} | {fpa_str:<5} | {apo_str:<8} | {peri_str:<8} | {inc_str:<6}")
        
    print("="*40 + "\n")

def analyze_integrator_steps(simulation_data):
    """
    Analyzes the time steps taken by the variable-step integrator.
    Small steps indicate stiffness or rapid dynamics.
    """
    _print_header("Integrator Step Size Analysis", "Simulation")
    
    t = simulation_data['t']
    if len(t) < 2:
        print("  Not enough data points.")
        return

    dt = np.diff(t)
    # Filter out zero-length steps caused by phase concatenation
    mask = dt > 1e-9
    dt_valid = dt[mask]
    t_start = t[:-1][mask]
    
    if len(dt_valid) == 0:
        print("  Not enough valid time steps.")
        return
    
    min_idx = np.argmin(dt_valid)
    min_step = dt_valid[min_idx]
    min_time = t_start[min_idx]
    
    print(f"  Total Steps: {len(t)}")
    print(f"  Min Step:    {min_step:.2e} s (at t={min_time:.2f}s)")
    print(f"  Max Step:    {np.max(dt_valid):.2e} s")
    print(f"  Mean Step:   {np.mean(dt_valid):.2e} s")
    
    # Check for stiffness (very small steps)
    if min_step < 1e-4:
        print(f"  >>> {Style.YELLOW}⚠️ WARNING: Very small time steps detected (<1e-4s). Physics might be stiff.{Style.RESET}")
        print(f"      (Likely caused by discontinuous control inputs or phase transitions at t={min_time:.1f}s)")
    else:
        print(f"  >>> {Style.GREEN}✅ SUCCESS: Time steps indicate well-behaved dynamics.{Style.RESET}")
    print("="*40 + "\n")

def analyze_control_saturation(simulation_data, vehicle):
    """
    Analyzes control saturation and correlates it with active flight constraints (Max Q, Max G).
    """
    _print_header("Control Saturation & Constraint Analysis", "Simulation")
    
    t = simulation_data['t']
    y = simulation_data['y']
    u = simulation_data['u']
    config = vehicle.config
    env = vehicle.env
    
    min_th = config.sequence.min_throttle
    max_q_lim = config.max_q_limit
    max_g_lim = config.max_g_load
    
    # Tolerances
    th_tol = 0.01
    constraint_margin = 0.95 # Consider constraint "active" if within 95% of limit
    
    # Time Accumulators
    t_powered = 0.0
    t_max_th = 0.0
    t_min_th = 0.0
    t_q_lim = 0.0
    t_g_lim = 0.0
    
    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        if dt < 1e-6: continue
        
        # State at start of interval
        r_i = y[0:3, i]
        v_i = y[3:6, i]
        m_i = y[6, i]
        th_i = u[0, i]
        
        # Check Coast (Engines Off)
        if th_i < 0.01:
            continue
            
        t_powered += dt
        
        # Check Saturation
        if th_i > 0.99:
            t_max_th += dt
        elif th_i < min_th + th_tol:
            t_min_th += dt
            
            # Check Constraints to see WHY we are throttling
            env_state = env.get_state_sim(r_i, t[i])
            
            # Max Q
            v_rel = v_i - env_state['wind_velocity']
            q = 0.5 * env_state['density'] * np.linalg.norm(v_rel)**2
            if q > max_q_lim * constraint_margin:
                t_q_lim += dt
                
            # Max G
            m_stg2_wet = config.stage_2.dry_mass + config.stage_2.propellant_mass + config.payload_mass
            stage = config.stage_1 if m_i > m_stg2_wet + 1000 else config.stage_2
            
            isp_eff = stage.isp_vac + (env_state['pressure'] / stage.p_sl) * (stage.isp_sl - stage.isp_vac)
            thrust = th_i * stage.thrust_vac * (isp_eff / stage.isp_vac)
            g_load = (thrust / m_i) / env.config.g0
            
            if g_load > max_g_lim * constraint_margin:
                t_g_lim += dt

    if t_powered < 1.0:
        print("  No powered flight detected.")
        print("="*40 + "\n")
        return

    pct_max = 100.0 * t_max_th / t_powered
    pct_min = 100.0 * t_min_th / t_powered
    pct_q = 100.0 * t_q_lim / t_powered
    pct_g = 100.0 * t_g_lim / t_powered
    
    print(f"  Powered Flight Duration: {t_powered:.1f} s")
    print(f"{Style.CYAN}" + "-" * 60 + f"{Style.RESET}")
    print(f"{Style.BOLD}  {'Regime':<25} | {'% Time':<12} | {'Notes':<20}{Style.RESET}")
    print(f"{Style.CYAN}" + "-" * 60 + f"{Style.RESET}")
    print(f"  {'Max Throttle (100%)':<25} | {pct_max:<12.1f} | {'Unconstrained Ascent'}")
    print(f"  {'Min Throttle (' + str(int(min_th*100)) + '%)':<25} | {pct_min:<12.1f} | {'Throttling Active'}")
    print(f"{Style.CYAN}" + "-" * 60 + f"{Style.RESET}")
    print(f"  Constraint Activity (during Min Throttle):")
    print(f"    - Max Q Limit (>95%):   {pct_q:<12.1f} | {'Aerodynamic Stress'}")
    print(f"    - Max G Limit (>95%):   {pct_g:<12.1f} | {'Structural Stress'}")
    
    if pct_q > 1.0:
        print(f"  >>> INFO: Throttling detected for Max Q ({max_q_lim/1000:.0f} kPa).")
    if pct_g > 1.0:
        print(f"  >>> INFO: Throttling detected for G-Limit ({max_g_lim:.1f} g).")
        
    print("="*40 + "\n")

def analyze_guidance_accuracy(guess, opt_res):
    """
    Compares the initial guess (Guidance) with the final optimized result.
    """
    _print_header("Guidance vs Optimality Check", "Guess vs Opt")
    
    # Compare MECO
    t1_guess = guess['T1']
    t1_opt = opt_res['T1']
    
    # Compare Final Mass
    # X3 is [7, N]
    m_final_guess = guess['X3'][6, -1]
    m_final_opt = opt_res['X3'][6, -1]
    
    print(f"  MECO Time:   Guess={t1_guess:.1f}s, Opt={t1_opt:.1f}s (Diff: {t1_opt-t1_guess:+.1f}s)")
    print(f"  Final Mass:  Guess={m_final_guess:,.0f}kg (Depletion), Opt={m_final_opt:,.0f}kg (Insertion)")
    
    fuel_saved = m_final_opt - m_final_guess

    if m_final_opt > m_final_guess + 1000:
         print(f"  >>> {Style.GREEN}✅ INFO: Optimizer found a solution with {fuel_saved:,.0f} kg of fuel remaining.{Style.RESET}")
         print(f"      (The initial guess burned all fuel; the optimizer cut the engine early at orbit insertion.)")
    elif m_final_opt < m_final_guess - 50000:
         print(f"  >>> {Style.YELLOW}⚠️ NOTE: Optimizer result is significantly lighter than guess (Check constraints).{Style.RESET}")
    else:
         print(f"  >>> {Style.GREEN}✅ SUCCESS: Guidance provided a close mass estimate.{Style.RESET}")
    print("="*40 + "\n")

# ==============================================================================
# CONFIGURATION DIAGNOSTICS
# ==============================================================================

def print_config_summary(config):
    """
    Prints a diagnostic summary of the vehicle configuration (Delta-V, T/W).
    Moved from config.py to separate data from presentation.
    """
    g0 = 9.80665
    
    _print_header(f"Config Diagnostic: {config.name}", "Config")
    
    inc_str = f"{config.target_inclination:.1f}" if config.target_inclination is not None else "Min-Energy (Auto)"
    print(f"  Target Orbit:      {config.target_altitude/1000:.1f} km @ {inc_str} deg")
    print(f"  Launch Mass:       {config.launch_mass:,.0f} kg")
    print(f"  Payload:           {config.payload_mass:,.0f} kg")
    
    # --- Stage 1 Analysis ---
    s1 = config.stage_1
    m0_1 = config.launch_mass
    mf_1 = m0_1 - s1.propellant_mass
    dv_1 = s1.isp_vac * g0 * np.log(m0_1 / mf_1)
    tw_1 = s1.thrust_sl / (m0_1 * g0)
    
    print(f"{Style.CYAN}" + "-"*80 + f"{Style.RESET}")
    print(f"  Stage 1 (Booster):")
    print(f"    Dry Mass:        {s1.dry_mass:,.0f} kg")
    print(f"    Propellant:      {s1.propellant_mass:,.0f} kg")
    print(f"    Thrust (SL/Vac): {s1.thrust_sl/1e6:.2f} / {s1.thrust_vac/1e6:.2f} MN")
    print(f"    ISP (SL/Vac):    {s1.isp_sl:.0f} / {s1.isp_vac:.0f} s")
    print(f"    Liftoff T/W:     {tw_1:.2f}")
    print(f"    Ideal Delta-V:   {dv_1:.0f} m/s")

    # --- Stage 2 Analysis ---
    s2 = config.stage_2
    m0_2 = s2.dry_mass + s2.propellant_mass + config.payload_mass
    mf_2 = s2.dry_mass + config.payload_mass
    dv_2 = s2.isp_vac * g0 * np.log(m0_2 / mf_2)
    tw_2 = s2.thrust_vac / (m0_2 * g0)
    
    print(f"{Style.CYAN}" + "-"*80 + f"{Style.RESET}")
    print(f"  Stage 2 (Ship):")
    print(f"    Dry Mass:        {s2.dry_mass:,.0f} kg")
    print(f"    Propellant:      {s2.propellant_mass:,.0f} kg")
    print(f"    Thrust (Vac):    {s2.thrust_vac/1e6:.2f} MN")
    print(f"    ISP (SL/Vac):    {s2.isp_sl:.0f} / {s2.isp_vac:.0f} s")
    print(f"    Staging T/W:     {tw_2:.2f}")
    print(f"    Ideal Delta-V:   {dv_2:.0f} m/s")
    
    # --- Total System ---
    print(f"{Style.CYAN}" + "="*80 + f"{Style.RESET}")
    print(f"  Total Ideal Delta-V: {dv_1 + dv_2:.0f} m/s")
    print(f"  Est. Losses (Grav+Drag): ~1500 m/s")
    if (dv_1 + dv_2) < 9000:
            print(f"  >>> {Style.YELLOW}⚠️ WARNING: Total Delta-V is likely insufficient for Orbit.{Style.RESET}")
    else:
            print(f"  >>> {Style.GREEN}✅ INFO: Theoretical Delta-V is sufficient.{Style.RESET}")
    print("="*40 + "\n")

# ==============================================================================
# HIGH-LEVEL WRAPPERS (Standardized Suites)
# ==============================================================================

def run_preflight_checks(vehicle, config):
    verify_scaling_consistency(config)
    verify_environment_consistency(vehicle)
    verify_positioning(vehicle)
    verify_physics_consistency(vehicle, config)
    verify_aerodynamics(vehicle)
    verify_propulsion(vehicle)

def run_optimization_analysis(opt_res, config, env):
    verify_staging_and_objective(opt_res, config, env)

def run_postflight_analysis(sim_res, opt_res, vehicle, env):
    _print_sub_header("Trajectory Validation")
    validate_trajectory(sim_res, vehicle.config, env)
    analyze_instantaneous_orbit(sim_res, env)
    
    _print_sub_header("Performance & Efficiency")
    analyze_delta_v_budget(sim_res, vehicle, vehicle.config)
    analyze_energy_balance(sim_res, vehicle)
    
    _print_sub_header("Control & Dynamics")
    analyze_trajectory_drift(opt_res, sim_res)
    analyze_control_saturation(sim_res, vehicle)
    analyze_control_slew_rates(sim_res)
    analyze_integrator_steps(sim_res)