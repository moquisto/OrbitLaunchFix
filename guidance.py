# guidance.py
# Purpose: Generates initial guesses to help the optimizer converge.
# CasADi needs a "Warm Start" so it doesn't start searching from zero.

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

def get_initial_guess(mission_config, vehicle, environment, num_nodes=50):
    """
    Generates a physically plausible trajectory to warm-start the optimizer.
    
    The optimizer (IPOPT) requires an initial guess for all decision variables 
    (States X, Controls U, Time T) to converge reliably. Starting from all-zeros 
    usually fails.
    
    This function performs a forward simulation using a heuristic "Gravity Turn" 
    guidance law, then resamples the result to match the optimizer's grid.
    """

    # --- 1. Define Heuristic Control Law ---
    def guidance_law(t, y, phase):
        # Inputs: t (time), y (state: r, v, m), phase ("boost", "coast", "ship")
        
        r = y[0:3]
        v = y[3:6]
        
        # A. Calculate Reference Frames (ECI)
        r_norm = np.linalg.norm(r)
        u_vertical = r / (r_norm + 1e-9)
        
        # Get Environment State for Wind
        try:
            env_state = environment.get_state_sim(r, t)
            wind_vel = env_state['wind_velocity']
        except:
            # Fallback if environment not fully initialized
            wind_vel = np.zeros(3)
            
        # Relative Velocity
        v_rel = v - wind_vel
        v_rel_mag = np.linalg.norm(v_rel)
        
        # Relative Velocity Dir (Aerodynamic Velocity)
        if v_rel_mag > 1.0:
            u_vel = v_rel / v_rel_mag
        else:
            u_vel = u_vertical
            
        # Dynamic East Vector (for pitch over)
        # Omega x r points roughly East at low latitudes
        v_east_raw = np.cross(environment.config.earth_omega_vector, r)
        v_east_mag = np.linalg.norm(v_east_raw)
        if v_east_mag > 1e-6:
            u_east = v_east_raw / v_east_mag
        else:
            u_east = np.array([1.0, 0.0, 0.0]) # Fallback at poles
        
        # B. Determine Thrust Direction
        direction = u_vertical # Default
        
        if phase == "boost":
            if t < mission_config.sequence.pitch_start_time:
                # Vertical Rise (Clear the tower)
                direction = u_vertical
            elif mission_config.sequence.pitch_start_time <= t < mission_config.sequence.pitch_end_time:
                # Pitch Over Maneuver (Initiate turn)
                # Nudge thrust vector slightly towards East (Dynamic).
                # Heuristic gain for the pitch maneuver
                v_pitch = u_vertical + mission_config.sequence.pitch_gain * u_east
                direction = v_pitch / np.linalg.norm(v_pitch)
            else:
                # Gravity Turn (Zero Angle of Attack)
                # Thrust aligns with RELATIVE Velocity to minimize aerodynamic stress.
                direction = u_vel
                
        elif phase == "ship":
            # Upper stage usually follows a gravity turn.
            direction = u_vel
    
        elif "coast" in phase:
            # Coasting: Align with velocity to minimize drag (Zero AoA)
            direction = u_vel
    
        # C. Determine Throttle
        throttle = 1.0 # Max thrust for ascent guess
        if "coast" in phase:
            throttle = 0.0
        
        return throttle, direction

    # Wrapper for solve_ivp
    def dynamics_wrapper(t, y, phase):
        throttle, direction = guidance_law(t, y, phase)
        return vehicle.get_dynamics(y, throttle, direction, t, stage_mode=phase, scaling=None)

    # --- 2. Simulate Phase 1 (Booster Ascent) ---
    # Initial State
    r0, v0 = environment.get_launch_site_state()
    m0 = mission_config.launch_mass
    y0 = np.concatenate([r0, v0, [m0]])
    
    # Stop Condition: Propellant Depletion
    m_burnout_1 = mission_config.stage_1.dry_mass + (mission_config.stage_2.dry_mass + mission_config.stage_2.propellant_mass + mission_config.payload_mass)
    
    def event_main_cutoff(t, y):
        return y[6] - m_burnout_1
    event_main_cutoff.terminal = True
    event_main_cutoff.direction = -1
    
    def event_crash(t, y):
        r = y[0:3]
        
        R_eq = environment.config.earth_radius_equator
        f = environment.config.earth_flattening
        R_pol = R_eq * (1.0 - f)
        
        val = (r[0]/R_eq)**2 + (r[1]/R_eq)**2 + (r[2]/R_pol)**2
        return val - 1.0
    event_crash.terminal = True
    event_crash.direction = -1
    
    # Integrate Phase 1
    sol1 = solve_ivp(
        fun=lambda t, y: dynamics_wrapper(t, y, "boost"),
        t_span=(0, 500),
        y0=y0,
        events=[event_main_cutoff, event_crash],
        rtol=1e-5, atol=1e-5
    )
    
    if not sol1.success:
        print("Warning: Guidance Phase 1 simulation did not complete successfully.")

    t1_raw = sol1.t
    y1_raw = sol1.y
    t1_final = t1_raw[-1]

    # --- 3. Resample Phase 1 (Discretization) ---
    # Create a linear time grid
    t_grid_1 = np.linspace(0, t1_final, num_nodes + 1)
    
    # Interpolate state
    f_interp_1 = interp1d(t1_raw, y1_raw, axis=1, kind='linear', fill_value="extrapolate")
    X1_guess = f_interp_1(t_grid_1)
    
    # Recalculate controls (TH1, TD1) at these grid points
    t_ctrl_1 = t_grid_1[:-1]
    U1_list = []
    for t_val, x_val in zip(t_ctrl_1, X1_guess[:, :-1].T):
        th, dr = guidance_law(t_val, x_val, "boost")
        U1_list.append(np.concatenate([[th], dr]))
    U1_guess = np.array(U1_list).T
    
    TH1_guess = U1_guess[0, :]
    TD1_guess = U1_guess[1:, :]
    T1_guess = t1_final

    # --- 4. Simulate & Resample Phase 2 (Coast) ---
    sep_delay = mission_config.sequence.separation_delay
    
    X2_guess = None
    T2_guess = 0.0
    
    y_prev = y1_raw[:, -1]
    t_start_phase3 = t1_final
    
    if sep_delay > 1e-4:
        T2_guess = sep_delay
        t_end_coast = t1_final + sep_delay
        
        sol2 = solve_ivp(
            fun=lambda t, y: dynamics_wrapper(t, y, "coast"),
            t_span=(t1_final, t_end_coast),
            y0=y_prev,
            events=[event_crash],
            rtol=1e-5, atol=1e-5
        )
        
        t2_raw = sol2.t
        y2_raw = sol2.y
        
        # Resample
        t_grid_2 = np.linspace(t1_final, t_end_coast, num_nodes + 1)
        f_interp_2 = interp1d(t2_raw, y2_raw, axis=1, kind='linear', fill_value="extrapolate")
        X2_guess = f_interp_2(t_grid_2)
        
        y_prev = y2_raw[:, -1]
        t_start_phase3 = t_end_coast

    # --- 5. Simulate & Resample Phase 3 (Ship Burn) ---
    # Staging Event: Drop Booster Mass
    m_ship_wet = mission_config.stage_2.dry_mass + mission_config.stage_2.propellant_mass + mission_config.payload_mass
    y_ship_init = y_prev.copy()
    y_ship_init[6] = m_ship_wet
    
    # Stop Condition: Propellant Depletion
    m_burnout_2 = mission_config.stage_2.dry_mass + mission_config.payload_mass
    
    def event_seco(t, y):
        return y[6] - m_burnout_2
    event_seco.terminal = True
    event_seco.direction = -1
    
    # Integrate Phase 3
    sol3 = solve_ivp(
        fun=lambda t, y: dynamics_wrapper(t, y, "ship"),
        t_span=(t_start_phase3, t_start_phase3 + 1000), # Max burn time guess
        y0=y_ship_init,
        events=[event_seco, event_crash],
        rtol=1e-5, atol=1e-5
    )
    
    t3_raw = sol3.t
    y3_raw = sol3.y
    t3_final = t3_raw[-1]
    T3_guess = t3_final - t_start_phase3
    
    # Resample
    t_grid_3 = np.linspace(t_start_phase3, t3_final, num_nodes + 1)
    f_interp_3 = interp1d(t3_raw, y3_raw, axis=1, kind='linear', fill_value="extrapolate")
    X3_guess = f_interp_3(t_grid_3)
    
    # Recalculate controls
    t_ctrl_3 = t_grid_3[:-1]
    U3_list = []
    for t_val, x_val in zip(t_ctrl_3, X3_guess[:, :-1].T):
        th, dr = guidance_law(t_val, x_val, "ship")
        U3_list.append(np.concatenate([[th], dr]))
    U3_guess = np.array(U3_list).T
    
    TH3_guess = U3_guess[0, :]
    TD3_guess = U3_guess[1:, :]

    # --- 6. Return Dictionary ---
    return {
        "X1": X1_guess, "T1": T1_guess, "TH1": TH1_guess, "TD1": TD1_guess,
        "X2": X2_guess, "T2": T2_guess,
        "X3": X3_guess, "T3": T3_guess, "TH3": TH3_guess, "TD3": TD3_guess
    }