# main.py
# Purpose: The Optimization Orchestrator.

import casadi as ca
import numpy as np
from config import ScalingConfig
import warnings
import time
import guidance
import debug

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
    opti.subject_to(T1_scaled >= config.sequence.min_stage_1_burn / scaling.time)
    opti.subject_to(T3_scaled >= config.sequence.min_stage_2_burn / scaling.time)
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
            
            # Structural Constraints
            # 1. Max Q (35 kPa limit)
            q_val, _ = vehicle.get_aero_properties(x_k, u_dir, t_k_scaled, stage_mode=phase_mode, scaling=scaling)
            # Apply numerical buffer to prevent simulation drift from triggering violations
            opti.subject_to(q_val <= config.max_q_limit * config.max_q_opt_margin)
            
            # 2. G-Force Limit (4.0 g)
            # Calculate Sensed Acceleration = |a_kinematic - g|
            derivs = vehicle.get_dynamics(x_k, u_throttle, u_dir, t_k_scaled, stage_mode=phase_mode, scaling=scaling)
            acc_scaled = derivs[3:6]
            acc_phys = acc_scaled * (scaling.speed / scaling.time)
            
            r_phys = x_k[0:3] * scaling.length
            t_phys = t_k_scaled * scaling.time
            env_state = environment.get_state_opti(r_phys, t_phys)
            g_phys = env_state['gravity']
            
            a_sensed = acc_phys - g_phys
            g_load = ca.norm_2(a_sensed)
            opti.subject_to(g_load <= config.max_g_load * environment.config.g0)
            
        # --- Constraint Check for Final Node (k=N) ---
        # Critical for G-Load (Burnout) as mass is lowest here.
        x_final_phase = X[:, N]
        t_final_phase = t_start_scaled + N * dt_scaled
        
        if "coast" in phase_mode:
             u_th_f = 0.0
             u_dir_f = ca.vertcat(1, 0, 0)
        else:
             # Use the last control input (Zero-Order Hold assumption)
             u_last = U[:, N-1]
             u_th_f = u_last[0]
             u_dir_f = u_last[1:]
             
        # Max Q (Final Node)
        q_val_f, _ = vehicle.get_aero_properties(x_final_phase, u_dir_f, t_final_phase, stage_mode=phase_mode, scaling=scaling)
        opti.subject_to(q_val_f <= config.max_q_limit * config.max_q_opt_margin)
        
        # G-Load (Final Node)
        derivs_f = vehicle.get_dynamics(x_final_phase, u_th_f, u_dir_f, t_final_phase, stage_mode=phase_mode, scaling=scaling)
        acc_phys_f = derivs_f[3:6] * (scaling.speed / scaling.time)
        
        r_phys_f = x_final_phase[0:3] * scaling.length
        t_phys_f = t_final_phase * scaling.time
        env_state_f = environment.get_state_opti(r_phys_f, t_phys_f)
        
        g_load_f = ca.norm_2(acc_phys_f - env_state_f['gravity'])
        opti.subject_to(g_load_f <= config.max_g_load * environment.config.g0)

    # Apply Dynamics
    # Phase 1: Boost
    add_phase_dynamics(X1, U1, T1_scaled, "boost", 0.0)
    
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
    print(f"[Optimizer] Starting IPOPT solver (Max Iter={config.max_iter})...")
    opti.solver("ipopt", {"expand": True}, {"max_iter": config.max_iter, "tol": 1e-6, "print_level": 5})
    
    # --- 6a. DEBUG STRUCTURE ---
    debug.debug_optimization_structure(opti)
    
    t_solve = time.time()
    try:
        sol = opti.solve()
        print(f"[Optimizer] SUCCESS: Optimal solution found in {time.time() - t_solve:.2f}s.")
    except:
        print(f"[Optimizer] FAILURE: Solver did not converge after {time.time() - t_solve:.2f}s. Returning debug values.")
        sol = opti.debug
        debug.print_debug_info(opti, sol, scaling, config, environment, vehicle, X1, U1, T1_scaled, X3, U3, T3_scaled)
    
    # --- 6b. CHECK SCALING ---
    scaling_vars = {
        "T1": T1_scaled, "T3": T3_scaled,
        "X1": X1, "X3": X3,
        "U1": U1, "U3": U3
    }
    if use_coast:
        scaling_vars.update({"T2": T2_scaled, "X2": X2})
    debug.check_variable_scaling(sol, scaling_vars)

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
    
    # --- 7a. DEBUG GUIDANCE ---
    debug.analyze_guidance_accuracy(guess, res)
    
    print(f"[Optimizer] Total optimization time: {time.time() - t_start:.2f}s")
    
    return res

if __name__ == "__main__":
    from config import StarshipBlock2, EARTH_CONFIG
    from vehicle import Vehicle
    from environment import Environment
    from simulation import run_simulation
    import analysis
    
    print(f"\n\033[1m--- Setting up Mission ---\033[0m")
    StarshipBlock2.print_summary()
    env = Environment(EARTH_CONFIG)
    veh = Vehicle(StarshipBlock2, env)
    
    print(f"\033[1m--- Verifying Physics Model ---\033[0m")
    debug.run_preflight_checks(veh, StarshipBlock2)
    
    print(f"\033[1m--- Running Optimization ---\033[0m")
    opt_res = solve_optimal_trajectory(StarshipBlock2, veh, env)
    debug.run_optimization_analysis(opt_res, StarshipBlock2, env)
    
    print(f"\033[1m--- Running Verification Simulation ---\033[0m")
    sim_res = run_simulation(opt_res, veh, StarshipBlock2)
    
    debug.run_postflight_analysis(sim_res, opt_res, veh, env)
    
    print(f"\033[1m--- Plotting Results ---\033[0m")
    analysis.plot_mission(opt_res, sim_res, env, StarshipBlock2)
    print("Done.")