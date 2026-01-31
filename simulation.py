# simulation.py
# Purpose: Verification. Flies the optimized controls in a forward simulation.
# This ensures the CasADi solution is physically valid and not exploiting mathematical loopholes.

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

def run_simulation(optimization_result, vehicle, config):
    """
    Verifies the optimization result using forward integration.
    """
    # --- 1. UNPACK OPTIMIZATION RESULTS ---
    T1 = float(optimization_result["T1"])
    T2 = float(optimization_result.get("T2", 0.0))
    T3 = float(optimization_result["T3"])
    
    U1 = np.array(optimization_result["U1"]) # Shape (4, N)
    U3 = np.array(optimization_result["U3"]) # Shape (4, N)

    # --- 2. SETUP CONTROL INTERPOLATORS ---
    # Phase 1
    N1 = U1.shape[1]
    # Time grid for controls (start of each interval)
    t_grid_1 = np.linspace(0, T1, N1 + 1)[:-1]
    # Use 'previous' to match Direct Collocation (constant control per node)
    ctrl_func_1 = interp1d(t_grid_1, U1, axis=1, kind='previous', 
                           fill_value="extrapolate", bounds_error=False)

    # Phase 3
    N3 = U3.shape[1]
    t_grid_3 = np.linspace(0, T3, N3 + 1)[:-1]
    ctrl_func_3 = interp1d(t_grid_3, U3, axis=1, kind='previous', 
                           fill_value="extrapolate", bounds_error=False)

    # --- 3. DEFINE DYNAMICS WRAPPER ---
    def sim_dynamics(t, y, phase_mode, t_start_phase, ctrl_func):
        # Calculate Phase-Relative Time
        t_rel = t - t_start_phase
        
        # Get Control Inputs
        if "coast" in phase_mode:
            throttle = 0.0
            thrust_dir = np.array([1.0, 0.0, 0.0]) # Dummy direction
        else:
            # Evaluate interpolator
            u = ctrl_func(t_rel)
            throttle = u[0]
            thrust_dir = u[1:]
            
        # Call Physics Engine
        # scaling=None ensures we work in physical units
        return vehicle.get_dynamics(y, throttle, thrust_dir, t, stage_mode=phase_mode, scaling=None)

    # --- 4. DEFINE TERMINATION EVENTS ---
    def altitude_event(t, y):
        r = y[0:3]
        r_norm = np.linalg.norm(r)
        
        # Check against local Earth radius (WGS84)
        if r_norm < 1.0: return -1.0
            
        sin_lat = r[2] / r_norm
        R_eq = vehicle.env.config.earth_radius_equator
        f = vehicle.env.config.earth_flattening
        R_local = R_eq * (1.0 - f * sin_lat**2)
        
        return r_norm - R_local

    altitude_event.terminal = True
    altitude_event.direction = -1 # Trigger when crossing from positive to negative

    # --- 5. PHASE 1: BOOSTER ASCENT ---
    # Initial State
    r0, v0 = vehicle.env.get_launch_site_state()
    m0 = config.launch_mass
    y0 = np.concatenate([r0, v0, [m0]])
    
    print(f"Simulating Phase 1 (Boost): 0.0s -> {T1:.2f}s")
    res1 = solve_ivp(
        fun=lambda t, y: sim_dynamics(t, y, "boost", 0.0, ctrl_func_1),
        t_span=(0, T1),
        y0=y0,
        events=altitude_event,
        rtol=1e-6, atol=1e-9,
        method='RK45'
    )

    if not res1.success and res1.status == -1:
        print("Simulation stopped early in Phase 1 (Crash or Error).")

    # --- 6. PHASE 2: COAST / STAGING ---
    t_current = res1.t[-1]
    y_current = res1.y[:, -1].copy()
    
    # Storage for concatenation
    results_list = [res1]
    
    if config.sequence.separation_delay > 1e-4:
        print(f"Simulating Phase 2 (Coast): {t_current:.2f}s -> {t_current + T2:.2f}s")
        res2 = solve_ivp(
            fun=lambda t, y: sim_dynamics(t, y, "coast", t_current, None),
            t_span=(t_current, t_current + T2),
            y0=y_current,
            events=altitude_event,
            rtol=1e-6, atol=1e-9,
            method='RK45'
        )
        results_list.append(res2)
        t_current = res2.t[-1]
        y_current = res2.y[:, -1].copy()

    # --- 7. STAGING EVENT ---
    # Drop Booster Mass, Reset to Ship Wet Mass
    m_ship_wet = config.stage_2.dry_mass + config.stage_2.propellant_mass + config.payload_mass
    y_current[6] = m_ship_wet
    
    # --- 8. PHASE 3: SHIP ASCENT ---
    print(f"Simulating Phase 3 (Ship): {t_current:.2f}s -> {t_current + T3:.2f}s")
    res3 = solve_ivp(
        fun=lambda t, y: sim_dynamics(t, y, "ship", t_current, ctrl_func_3),
        t_span=(t_current, t_current + T3),
        y0=y_current,
        events=altitude_event,
        rtol=1e-6, atol=1e-9,
        method='RK45'
    )
    results_list.append(res3)

    # --- 9. CONSOLIDATE RESULTS ---
    t_full = np.concatenate([res.t for res in results_list])
    y_full = np.concatenate([res.y for res in results_list], axis=1)
    
    # Reconstruct Controls for Plotting
    u_list = []
    
    # Phase 1 Controls
    u_list.append(ctrl_func_1(res1.t))
    
    # Phase 2 Controls (if applicable)
    if len(results_list) == 3:
        res2 = results_list[1]
        u_coast = np.zeros((4, len(res2.t)))
        u_coast[1, :] = 1.0 # x-direction
        u_list.append(u_coast)
        
    # Phase 3 Controls
    res3_obj = results_list[-1]
    # Calculate relative time for Phase 3
    # Start time of Phase 3 is the end time of the previous phase
    t_start_3 = results_list[-2].t[-1]
    u_list.append(ctrl_func_3(res3_obj.t - t_start_3))
    
    u_full = np.concatenate(u_list, axis=1)

    return {
        "t": t_full,
        "y": y_full,
        "u": u_full
    }