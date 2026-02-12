# simulation.py
# Purpose: Verification. Flies the optimized controls in a forward simulation.
# This ensures the CasADi solution is physically valid and not exploiting mathematical loopholes.

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

def run_simulation(optimization_result, vehicle, config, rtol=1e-9, atol=1e-12):
    """
    Verifies the optimization result using forward integration.
    """
    print(f"\n[Simulation] Starting Verification Simulation (rtol={rtol}, atol={atol})...")
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
    # Calculate max reconstruction error to ensure fidelity
    max_err_1 = 0.0
    for i in range(N1):
        t_sample = t_grid_1[i] + 1e-5
        u_val = ctrl_func_1(t_sample)
        err = np.linalg.norm(u_val - U1[:, i])
        if err > max_err_1: max_err_1 = err
    print(f"  > Control Reconstruction (Phase 1): Max Error = {max_err_1:.1e}")

    # Phase 3
    N3 = U3.shape[1]
    t_grid_3 = np.linspace(0, T3, N3 + 1)[:-1]
    ctrl_func_3 = interp1d(t_grid_3, U3, axis=1, kind='previous', 
                           fill_value="extrapolate", bounds_error=False)

    # --- DEBUG: Verify Control Handoff (Phase 3) ---
    max_err_3 = 0.0
    for i in range(N3):
        t_sample = t_grid_3[i] + 1e-5
        u_val = ctrl_func_3(t_sample)
        err = np.linalg.norm(u_val - U3[:, i])
        if err > max_err_3: max_err_3 = err
    print(f"  > Control Reconstruction (Phase 3): Max Error = {max_err_3:.1e}")

    use_coast_phase = config.sequence.separation_delay > 1e-4

    def build_coast_controls(y_hist):
        u_coast = np.zeros((4, y_hist.shape[1]))
        v_coast = y_hist[3:6, :]
        r_coast = y_hist[0:3, :]

        # Match coast guidance used by dynamics: thrust direction aligns with relative velocity.
        omega_vec = vehicle.env.config.earth_omega_vector
        wind_x = omega_vec[1] * r_coast[2, :] - omega_vec[2] * r_coast[1, :]
        wind_y = omega_vec[2] * r_coast[0, :] - omega_vec[0] * r_coast[2, :]
        wind_z = omega_vec[0] * r_coast[1, :] - omega_vec[1] * r_coast[0, :]
        wind_vel = np.vstack([wind_x, wind_y, wind_z])

        v_rel = v_coast - wind_vel
        v_norm = np.linalg.norm(v_rel, axis=0)
        v_dir = np.zeros_like(v_rel)
        v_dir[0, :] = 1.0
        np.divide(v_rel, v_norm[None, :], out=v_dir, where=v_norm > 1e-9)
        u_coast[1:, :] = v_dir
        return u_coast

    def assemble_output(results_list, termination=None):
        t_full = np.concatenate([res.t for res in results_list])
        y_full = np.concatenate([res.y for res in results_list], axis=1)

        u_list = [ctrl_func_1(results_list[0].t)]

        if use_coast_phase and len(results_list) >= 2:
            u_list.append(build_coast_controls(results_list[1].y))

        expected_with_ship = 3 if use_coast_phase else 2
        if len(results_list) >= expected_with_ship:
            res3_obj = results_list[-1]
            t_start_3 = results_list[-2].t[-1]
            u_list.append(ctrl_func_3(res3_obj.t - t_start_3))

        out = {
            "t": t_full,
            "y": y_full,
            "u": np.concatenate(u_list, axis=1)
        }
        if termination is not None:
            out["termination"] = termination
        return out

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
    
    # DIAGNOSE LIFTOFF
    print("  [Liftoff Check]")
    vehicle.diagnose_forces(y0, U1[0,0], U1[1:,0], 0.0, "boost")

    res1 = solve_ivp(
        fun=lambda t, y: sim_dynamics(t, y, "boost", 0.0, ctrl_func_1),
        t_span=(0, T1),
        y0=y0,
        events=altitude_event,
        rtol=rtol, atol=atol,
        method='RK45'
    )
    
    steps_1 = len(res1.t)
    print(f"  > Integrator: {steps_1} steps, Success={res1.success}, Msg='{res1.message}'")

    phase1_ground_impact = len(res1.t_events) > 0 and len(res1.t_events[0]) > 0
    if not res1.success and res1.status == -1:
        print("[Simulation] ERROR: Integrator failed in Phase 1.")
    elif phase1_ground_impact:
        print("[Simulation] ERROR: Ground impact detected in Phase 1. Aborting remaining phases.")
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
        print("  [MECO Check]")
        u_meco = U1[:, -1]
        vehicle.diagnose_forces(res1.y[:, -1], u_meco[0], u_meco[1:], res1.t[-1], "boost")

    if phase1_ground_impact:
        return assemble_output(
            [res1],
            termination={
                "reason": "ground_impact",
                "phase": "boost",
                "time_s": float(res1.t[-1])
            }
        )

    # --- 6. PHASE 2: COAST / STAGING ---
    t_current = res1.t[-1]
    y_current = res1.y[:, -1].copy()
    
    # Storage for concatenation
    results_list = [res1]
    
    if use_coast_phase:
        print(f"[Simulation] Phase 2 (Coast): {t_current:.2f}s -> {t_current + T2:.2f}s")
        res2 = solve_ivp(
            fun=lambda t, y: sim_dynamics(t, y, "coast", t_current, None),
            t_span=(t_current, t_current + T2),
            y0=y_current,
            events=altitude_event,
            rtol=rtol, atol=atol,
            method='RK45'
        )
        results_list.append(res2)
        coast_ground_impact = len(res2.t_events) > 0 and len(res2.t_events[0]) > 0
        if coast_ground_impact:
            print("[Simulation] ERROR: Ground impact detected during coast. Aborting Stage 2 burn.")
            return assemble_output(
                results_list,
                termination={
                    "reason": "ground_impact",
                    "phase": "coast",
                    "time_s": float(res2.t[-1])
                }
            )
        t_current = res2.t[-1]
        y_current = res2.y[:, -1].copy()

    # --- 7. STAGING EVENT ---
    # Drop Booster Mass, Reset to Ship Wet Mass
    m_ship_wet = config.stage_2.dry_mass + config.stage_2.propellant_mass + config.payload_mass
    y_current[6] = m_ship_wet
    
    # --- 8. PHASE 3: SHIP ASCENT ---
    print(f"[Simulation] Phase 3 (Ship): {t_current:.2f}s -> {t_current + T3:.2f}s")
    
    # DIAGNOSE STAGING
    print("  [Staging Check]")
    vehicle.diagnose_forces(y_current, U3[0,0], U3[1:,0], t_current, "ship")
    
    # --- DEFINE CUTOFF EVENTS ---
    # Verification should remain open-loop: fly the optimized control schedule
    # for its planned duration and only terminate early on hard failure cases.
    # 1. Propellant Depletion
    m_dry_s2 = config.stage_2.dry_mass + config.payload_mass
    def depletion_event(t, y):
        return y[6] - m_dry_s2
    depletion_event.terminal = True
    depletion_event.direction = -1

    res3 = solve_ivp(
        fun=lambda t, y: sim_dynamics(t, y, "ship", t_current, ctrl_func_3),
        t_span=(t_current, t_current + T3),
        y0=y_current,
        events=[altitude_event, depletion_event],
        rtol=rtol, atol=atol,
        method='RK45'
    )
    results_list.append(res3)
    
    steps_3 = len(res3.t)
    print(f"  > Integrator: {steps_3} steps, Success={res3.success}, Msg='{res3.message}'")

    phase3_ground_impact = len(res3.t_events) > 0 and len(res3.t_events[0]) > 0
    phase3_depletion = len(res3.t_events) > 1 and len(res3.t_events[1]) > 0
    if phase3_ground_impact:
        print("[Simulation] ERROR: Ground impact detected in Phase 3.")
    elif phase3_depletion:
        print("[Simulation] INFO: Phase 3 terminated on propellant depletion event.")
    
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
        
        # DIAGNOSE INJECTION
        print("  [Injection Check]")
        u_seco = U3[:, -1]
        vehicle.diagnose_forces(res3.y[:, -1], u_seco[0], u_seco[1:], res3.t[-1], "ship")

    if phase3_ground_impact:
        termination = {"reason": "ground_impact", "phase": "ship", "time_s": float(res3.t[-1])}
    elif phase3_depletion:
        termination = {"reason": "depletion", "phase": "ship", "time_s": float(res3.t[-1])}
    else:
        termination = {"reason": "planned_end", "phase": "ship", "time_s": float(res3.t[-1])}

    # --- 9. CONSOLIDATE RESULTS ---
    return assemble_output(results_list, termination=termination)
