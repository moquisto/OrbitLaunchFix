# main.py
# Purpose: The Optimization Orchestrator.

import casadi as ca
import numpy as np
from config import ScalingConfig
import warnings
import guidance

def solve_optimal_trajectory(config, vehicle, environment):
    """
    Formulates and solves the trajectory optimization problem using CasADi.
    """
    
    # --- 1. SETUP ---
    # Initialize CasADi Opti stack
    opti = ca.Opti()
    scaling = ScalingConfig(mass=config.launch_mass)
    
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
    N = 50 # Nodes per phase
    
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
    opti.subject_to(T1_scaled >= 10.0 / scaling.time)
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
            r_k = x_k[0:3] * scaling.length
            opti.subject_to(ca.norm_2(r_k) >= R_earth - 100.0)
            
            # Mass Constraints (Don't burn more than available)
            m_k = x_k[6] * scaling.mass
            if phase_mode == "boost":
                opti.subject_to(m_k >= m_stage1_min)
            elif phase_mode == "ship":
                opti.subject_to(m_k >= config.stage_2.dry_mass + config.payload_mass)

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
    r_f = x_final[0:3] * scaling.length
    v_f = x_final[3:6] * scaling.speed
    
    r_mag = ca.norm_2(r_f)
    v_mag = ca.norm_2(v_f)
    
    # 1. Altitude
    opti.subject_to(r_mag == R_earth + Target_Alt)
    # 2. Velocity (Circular)
    v_target = ca.sqrt(mu / (R_earth + Target_Alt))
    opti.subject_to(v_mag == v_target)
    # 3. Eccentricity (Flight Path Angle = 0)
    opti.subject_to(ca.dot(r_f, v_f) == 0.0)
    # 4. Inclination
    h_vec = ca.cross(r_f, v_f)
    opti.subject_to(h_vec[2] == ca.norm_2(h_vec) * np.cos(Target_Inc))

    # --- 4. OBJECTIVE ---
    # Maximize Final Mass
    opti.minimize(-X3[6, -1])
    
    # --- 5. INITIALIZATION ---
    print("Generating Initial Guess...")
    # FIX: Pass N, not N+1. Guidance generates N+1 points (nodes) from N intervals.
    guess = guidance.get_initial_guess(config, vehicle, environment, num_nodes=N)
    
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
    print("Solving Trajectory Optimization...")
    opti.solver("ipopt", {"expand": True}, {"max_iter": 1000, "tol": 1e-4, "print_level": 5})
    
    try:
        sol = opti.solve()
        print("Optimization Successful!")
    except:
        print("Optimization Failed. Returning debug values.")
        sol = opti.debug
    
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
    
    return res

if __name__ == "__main__":
    from config import StarshipBlock2, EARTH_CONFIG
    from vehicle import Vehicle
    from environment import Environment
    from simulation import run_simulation
    import analysis
    
    print("--- Setting up Mission ---")
    env = Environment(EARTH_CONFIG)
    veh = Vehicle(StarshipBlock2, env)
    
    print("--- Running Optimization ---")
    opt_res = solve_optimal_trajectory(StarshipBlock2, veh, env)
    
    print("--- Running Verification Simulation ---")
    sim_res = run_simulation(opt_res, veh, StarshipBlock2)
    
    print("--- Validating Trajectory ---")
    analysis.validate_trajectory(sim_res, StarshipBlock2, env)
    
    print("--- Plotting Results ---")
    analysis.plot_mission(opt_res, sim_res, env)
    print("Done.")