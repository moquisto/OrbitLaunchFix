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
    print(f"\n[Simulation] Starting Verification Simulation...")
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

    # --- DEBUG: Verify Control Handoff ---
    print(f"[Debug] Verifying Control Signal Handoff (Phase 1)...")
    # Check a few points to ensure the optimizer's plan is faithfully reproduced
    test_indices = [0, N1 // 2, N1 - 1]
    for idx in test_indices:
        # Sample time: slightly into the interval to check the "Hold" behavior
        t_sample = t_grid_1[idx] + 1e-3 
        u_opt = U1[:, idx]
        u_sim = ctrl_func_1(t_sample)
        
        diff = np.linalg.norm(u_opt - u_sim)
        if diff > 1e-5:
             print(f"  ! CONTROL MISMATCH at Node {idx}: Opt={u_opt} vs Sim={u_sim}")

    # Phase 3
    N3 = U3.shape[1]
    t_grid_3 = np.linspace(0, T3, N3 + 1)[:-1]
    ctrl_func_3 = interp1d(t_grid_3, U3, axis=1, kind='previous', 
                           fill_value="extrapolate", bounds_error=False)

    # --- DEBUG: Verify Control Handoff (Phase 3) ---
    print(f"[Debug] Verifying Control Signal Handoff (Phase 3)...")
    test_indices_3 = [0, N3 // 2, N3 - 1]
    for idx in test_indices_3:
        # Sample time: slightly into the interval
        t_sample = t_grid_3[idx] + 1e-3 
        u_opt = U3[:, idx]
        u_sim = ctrl_func_3(t_sample)
        
        diff = np.linalg.norm(u_opt - u_sim)
        if diff > 1e-5:
             print(f"  ! CONTROL MISMATCH at Node {idx}: Opt={u_opt} vs Sim={u_sim}")

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
        
        R_eq = vehicle.env.config.earth_radius_equator
        f = vehicle.env.config.earth_flattening
        R_pol = R_eq * (1.0 - f)
        
        # Exact WGS84 Ellipsoid Metric: (x/a)^2 + (y/a)^2 + (z/b)^2
        # Surface is at 1.0. < 1.0 means underground.
        val = (r[0]/R_eq)**2 + (r[1]/R_eq)**2 + (r[2]/R_pol)**2
        return val - 1.0

    altitude_event.terminal = True
    altitude_event.direction = -1 # Trigger when crossing from positive to negative

    # --- 5. PHASE 1: BOOSTER ASCENT ---
    # Initial State
    r0, v0 = vehicle.env.get_launch_site_state()
    m0 = config.launch_mass
    y0 = np.concatenate([r0, v0, [m0]])
    
    print(f"[Simulation] Phase 1 (Boost): 0.0s -> {T1:.2f}s")
    res1 = solve_ivp(
        fun=lambda t, y: sim_dynamics(t, y, "boost", 0.0, ctrl_func_1),
        t_span=(0, T1),
        y0=y0,
        events=altitude_event,
        rtol=1e-6, atol=1e-9,
        method='RK45'
    )

    if not res1.success and res1.status == -1:
        print("[Simulation] ERROR: Simulation stopped early in Phase 1 (Crash or Error).")
    else:
        alt_1 = (np.linalg.norm(res1.y[0:3, -1]) - vehicle.env.config.earth_radius_equator) / 1000.0
        print(f"[Simulation] Phase 1 End: Alt={alt_1:.1f}km, Vel={np.linalg.norm(res1.y[3:6, -1]):.1f}m/s")

    # --- DEBUG: CHECK DRIFT PHASE 1 ---
    if "X1" in optimization_result:
        opt_pos = optimization_result["X1"][0:3, -1]
        sim_pos = res1.y[0:3, -1]
        drift_pos = np.linalg.norm(opt_pos - sim_pos)
        opt_mass = optimization_result["X1"][6, -1]
        sim_mass = res1.y[6, -1]
        drift_mass = opt_mass - sim_mass
        print(f"[Debug] MECO Drift: Pos={drift_pos:.1f}m, Mass={drift_mass:.2f}kg")
        # Diagnose forces at MECO
        u_meco = U1[:, -1]
        vehicle.diagnose_forces(res1.y[:, -1], u_meco[0], u_meco[1:], res1.t[-1], "boost")

    # --- 6. PHASE 2: COAST / STAGING ---
    t_current = res1.t[-1]
    y_current = res1.y[:, -1].copy()
    
    # Storage for concatenation
    results_list = [res1]
    
    if config.sequence.separation_delay > 1e-4:
        print(f"[Simulation] Phase 2 (Coast): {t_current:.2f}s -> {t_current + T2:.2f}s")
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
    print(f"[Simulation] Phase 3 (Ship): {t_current:.2f}s -> {t_current + T3:.2f}s")
    res3 = solve_ivp(
        fun=lambda t, y: sim_dynamics(t, y, "ship", t_current, ctrl_func_3),
        t_span=(t_current, t_current + T3),
        y0=y_current,
        events=altitude_event,
        rtol=1e-6, atol=1e-9,
        method='RK45'
    )
    results_list.append(res3)
    
    alt_3 = (np.linalg.norm(res3.y[0:3, -1]) - vehicle.env.config.earth_radius_equator) / 1000.0
    print(f"[Simulation] Phase 3 End: Alt={alt_3:.1f}km, Vel={np.linalg.norm(res3.y[3:6, -1]):.1f}m/s")

    # --- DEBUG: CHECK DRIFT PHASE 3 ---
    if "X3" in optimization_result:
        opt_pos = optimization_result["X3"][0:3, -1]
        sim_pos = res3.y[0:3, -1]
        drift_pos = np.linalg.norm(opt_pos - sim_pos)
        opt_mass = optimization_result["X3"][6, -1]
        sim_mass = res3.y[6, -1]
        drift_mass = opt_mass - sim_mass
        print(f"[Debug] SECO Drift: Pos={drift_pos/1000:.2f}km, Mass={drift_mass:.2f}kg")

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
        
        # Fix: Align dummy control with velocity vector for visualization.
        # The physics engine forces u_thrust = u_vel during coast, so we replicate that here
        # to ensure analysis plots show ~0 deg AoA.
        v_coast = res2.y[3:6, :]
        v_norm = np.linalg.norm(v_coast, axis=0)
        
        # Avoid division by zero. Default to [1, 0, 0] if velocity is zero.
        v_dir = np.zeros_like(v_coast)
        v_dir[0, :] = 1.0
        
        # Perform safe division in-place
        np.divide(v_coast, v_norm, out=v_dir, where=v_norm > 1e-9)
        
        u_coast[1:, :] = v_dir
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