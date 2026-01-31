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
    # Extract durations (T1, T2, T3) and control grids (U1, U3) from the solver output.
    # Ensure these are physical units (seconds, Newtons), not scaled optimization variables.
    # Note: U matrices are typically [Throttle, Tx, Ty, Tz] x N_nodes.
    
    # --- 2. SETUP CONTROL INTERPOLATORS ---
    # The optimizer produces discrete control nodes. The simulator needs continuous control.
    # CRITICAL: Use 'previous' (Zero-Order Hold) interpolation.
    # Why? The optimizer assumes controls are CONSTANT within a node (Direct Collocation). 
    # Linear interpolation would change the control profile and invalidate the solution.
    # Note: U1 has N columns. These correspond to the N intervals.
    
    # N = U1.shape[1]
    # t_grid_1 = np.linspace(0, T1, N+1)[:-1] # Start time of each control interval
    # ctrl_func_1 = interp1d(t_grid_1, U1, axis=1, kind='previous', fill_value="extrapolate", bounds_error=False)
    # ... same for Phase 3 ...

    # --- 3. DEFINE DYNAMICS WRAPPER ---
    # solve_ivp requires a function f(t, y). We need to inject controls and phase logic.
    #
    # def sim_dynamics(t, y, phase_mode, t_start_phase, ctrl_func):
    #     1. Calculate Phase-Relative Time: t_rel = t - t_start_phase
    #     2. Get Control Inputs:
    #        If phase_mode == "coast":
    #            throttle = 0.0, thrust_dir = np.array([1.0, 0.0, 0.0])
    #        Else:
    #            u = ctrl_func(t_rel)
    #            throttle = u[0], thrust_dir = u[1:]
    #     3. Call Physics Engine:
    #        return vehicle.get_dynamics(y, throttle, thrust_dir, t, stage_mode=phase_mode)
    
    # --- 4. DEFINE TERMINATION EVENTS ---
    # Safety check: Stop simulation if altitude < 0 (Crash).
    # def altitude_event(t, y):
    #     r = y[0:3]
    #     # Use WGS84 Ellipsoid model. Simple spherical check (norm(r) - R_eq) 
    #     # fails at poles where Earth is ~21km flatter.
    #     r_norm = np.linalg.norm(r)
    #     sin_lat = r[2] / r_norm
    #     R_eq = vehicle.env.config.earth_radius_equator
    #     R_local = R_eq * (1.0 - vehicle.env.config.earth_flattening * sin_lat**2)
    #     return r_norm - R_local
    # altitude_event.terminal = True
    # altitude_event.direction = -1

    # --- 5. PHASE 1: BOOSTER ASCENT ---
    # Initial State: r0, v0 from environment, m0 = launch_mass
    # res1 = solve_ivp(fun=lambda t,y: sim_dynamics(..., "boost", ...), t_span=(0, T1), events=altitude_event, rtol=1e-6, atol=1e-9)
    
    # --- 6. PHASE 2: COAST / STAGING ---
    # t_current = res1.t[-1]
    
    # Use .copy()! Otherwise, modifying y_current later will overwrite the history in res1.y (view reference).
    # y_current = res1.y[:, -1].copy() 
    
    # Initialize result storage with Phase 1 data.
    # Note: This logic assumes "Coast" is PRE-SEPARATION (Booster still attached).
    
    # if config.sequence.separation_delay > 1e-4:
    #     res2 = solve_ivp(fun=lambda t,y: sim_dynamics(..., "coast", ...), t_span=(t_current, t_current + T2), events=altitude_event, rtol=1e-6, atol=1e-9)
    #     Append res2 to result storage.
    #     Update t_current, y_current.
    #     # Ensure we copy again if we came from res2
    #     y_current = y_current.copy()
    
    # --- 7. STAGING EVENT (Mass Discontinuity) ---
    # Drop Booster Mass, Reset to Ship Wet Mass.
    # m_ship_wet = config.stage_2.dry_mass + config.stage_2.propellant_mass + config.payload_mass
    # y_current[6] = m_ship_wet

    # --- 8. PHASE 3: SHIP ASCENT ---
    # res3 = solve_ivp(fun=lambda t,y: sim_dynamics(..., "ship", ...), t_span=(t_current, t_current + T3), y0=y_current, events=altitude_event, rtol=1e-6, atol=1e-9)
    # Append res3 to result storage.
    
    # --- 9. RETURN RESULTS ---
    # Concatenate all phases (t, y).
    # Reconstruct control history 'u' by evaluating ctrl_func(t_full) for plotting.
    # Note: For the coast phase, manually inject u=[0, 1, 0, 0].
    # return {
    #     "t": t_full,
    #     "y": y_full,
    #     "u": u_full (reconstructed from interpolators for plotting)
    # }
    
    return {}