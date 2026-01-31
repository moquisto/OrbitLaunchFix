# main.py
# Purpose: The Optimization Orchestrator.

import casadi as ca
import numpy as np
from config import ScalingConfig

def solve_optimal_trajectory(config, vehicle, environment):
    """
    Formulates and solves the trajectory optimization problem using CasADi.
    """
    
    # --- 1. SETUP ---
    opti = ca.Opti()
    scaling = ScalingConfig()
    
    # --- 2. DECISION VARIABLES ---
    N = 50  # Nodes per phase
    
    # Phase 1: Booster Ascent
    T1_scaled = opti.variable()
    opti.subject_to(opti.bounded(100.0/scaling.time, T1_scaled, 300.0/scaling.time))
    
    X1 = opti.variable(7, N+1) # [rx, ry, rz, vx, vy, vz, m]
    U1 = opti.variable(4, N)   # [throttle, u_x, u_y, u_z]
    
    # Phase 2: Coast / Staging
    # LOGIC FIX: Handle Hot Staging (Zero Delay) to avoid singularity
    use_coast_phase = config.sequence.separation_delay > 1e-4
    
    if use_coast_phase:
        T2_scaled = config.sequence.separation_delay / scaling.time
        X2 = opti.variable(7, N+1)
    else:
        T2_scaled = 0.0
        X2 = None

    # Phase 3: Ship Ascent
    T3_scaled = opti.variable()
    opti.subject_to(opti.bounded(100.0/scaling.time, T3_scaled, 1000.0/scaling.time))
    
    X3 = opti.variable(7, N+1)
    U3 = opti.variable(4, N)
    
    # --- 3. CONSTRAINTS ---
    
    # A. Boundary Conditions (Initial State)
    r0_phys, v0_phys = environment.get_launch_site_state()
    m0_phys = config.launch_mass
    
    opti.subject_to(X1[0:3, 0] == r0_phys / scaling.length)
    opti.subject_to(X1[3:6, 0] == v0_phys / scaling.speed)
    opti.subject_to(X1[6, 0] == m0_phys / scaling.mass)
    
    # B. Phase Linkage & Dynamics
    def add_dynamics_constraints(X, U, T_phase, t_start_abs, mode):
        dt = T_phase / N
        for k in range(N):
            # Absolute time at node k (Symbolic dependency on T_phase/T_prev)
            t_abs = t_start_abs + k * dt
            
            # State and Control at k
            x_k = X[:, k]
            u_k = U[:, k] if U is not None else None 
            
            # RK4 Integration
            k1 = vehicle.get_dynamics(x_k, u_k, t_abs, mode=mode, scaling=scaling)
            k2 = vehicle.get_dynamics(x_k + 0.5*dt*k1, u_k, t_abs + 0.5*dt, mode=mode, scaling=scaling)
            k3 = vehicle.get_dynamics(x_k + 0.5*dt*k2, u_k, t_abs + 0.5*dt, mode=mode, scaling=scaling)
            k4 = vehicle.get_dynamics(x_k + dt*k3, u_k, t_abs + dt, mode=mode, scaling=scaling)
            
            x_next = x_k + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(X[:, k+1] == x_next)

    # 1. Phase 1 Dynamics
    add_dynamics_constraints(X1, U1, T1_scaled, 0.0, "boost")
    x1_final = X1[:, -1]
    
    # MECO Constraint (Mach Number)
    r_meco = x1_final[0:3] * scaling.length
    v_meco = x1_final[3:6] * scaling.speed
    t_meco = T1_scaled * scaling.time
    env_meco = environment.get_state_opti(r_meco, t_meco)
    v_rel_meco = ca.norm_2(v_meco - env_meco['wind_velocity'])
    mach_meco = v_rel_meco / env_meco['speed_of_sound']
    opti.subject_to(mach_meco >= config.sequence.meco_cutoff_mach)

    # Booster Fuel Reserve / Dry Mass Constraint
    m_booster_dry = config.stage_1.dry_mass
    m_ship_wet = config.stage_2.dry_mass + config.stage_2.propellant_mass + config.payload_mass
    opti.subject_to(X1[6, -1] * scaling.mass >= m_booster_dry + m_ship_wet)

    if use_coast_phase:
        # Link 1 -> 2
        opti.subject_to(X2[:, 0] == x1_final)
        # Phase 2 Dynamics (Coast)
        add_dynamics_constraints(X2, None, T2_scaled, T1_scaled, "coast")
        x2_final = X2[:, -1]
        t2_end = T1_scaled + T2_scaled
    else:
        x2_final = x1_final
        t2_end = T1_scaled

    # 3. Phase 2 -> Phase 3 Linkage (Staging Mass Drop)
    opti.subject_to(X3[0:6, 0] == x2_final[0:6])
    m_ship_start = (config.stage_2.dry_mass + config.stage_2.propellant_mass + config.payload_mass)
    opti.subject_to(X3[6, 0] == m_ship_start / scaling.mass)
    
    # 4. Phase 3 Dynamics
    add_dynamics_constraints(X3, U3, T3_scaled, t2_end, "ship")

    # C. Path Constraints (Controls)
    opti.subject_to(opti.bounded(config.sequence.min_throttle, U1[0, :], 1.0))
    opti.subject_to(opti.bounded(config.sequence.min_throttle, U3[0, :], 1.0))
    
    for k in range(N):
        opti.subject_to(ca.sumsqr(U1[1:4, k]) == 1.0)
        opti.subject_to(ca.sumsqr(U3[1:4, k]) == 1.0)
        opti.subject_to(ca.sumsqr(X1[0:3, k]) >= 1.0) 
        opti.subject_to(ca.sumsqr(X3[0:3, k]) >= 1.0)

    # D. Terminal Constraints (Target Orbit)
    x_final = X3[:, -1]
    r_fin = x_final[0:3] * scaling.length
    v_fin = x_final[3:6] * scaling.speed
    
    # 420km Circular Orbit
    target_r = environment.config.earth_radius_equator + 420000.0
    opti.subject_to(ca.norm_2(r_fin) == target_r)
    
    target_v = ca.sqrt(environment.config.earth_mu / target_r)
    opti.subject_to(ca.norm_2(v_fin) == target_v)
    opti.subject_to(ca.dot(r_fin, v_fin) == 0)
    
    # --- 4. OBJECTIVE ---
    opti.minimize(-x_final[6])
    
    # --- 5. INITIALIZATION ---
    opti.set_initial(T1_scaled, 160.0 / scaling.time)
    opti.set_initial(T3_scaled, 400.0 / scaling.time)
    opti.set_initial(X1[6, :], 1.0)
    opti.set_initial(X3[6, :], 0.1)
    
    # --- 6. SOLVE ---
    opti.solver('ipopt', {'ipopt': {'print_level': 5, 'tol': 1e-4, 'max_iter': 1000}})
    
    try:
        sol = opti.solve()
        return sol
    except Exception as e:
        print("Solver failed:", e)
        return opti.debug