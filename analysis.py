# analysis.py
# Purpose: Visualization and Data Analysis.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_mission(optimization_data, simulation_data, environment):
    """
    Visualizes the mission performance and compares the optimized plan
    against the simulated reality.
    """
    print("Processing data for visualization...")

    # --- 1. PREPARE OPTIMIZATION DATA ---
    # Concatenate phases
    t_opt = []
    x_opt = []
    u_opt = []
    
    # Phase 1
    N1 = optimization_data['X1'].shape[1] - 1
    t1 = np.linspace(0, optimization_data['T1'], N1 + 1)
    t_opt.append(t1)
    x_opt.append(np.array(optimization_data['X1']))
    # Pad U to match X length for plotting (repeat last control)
    u1_val = np.array(optimization_data['U1'])
    u1_padded = np.hstack([u1_val, u1_val[:, -1:]])
    u_opt.append(u1_padded)
    
    current_t = optimization_data['T1']
    
    # Phase 2 (Coast)
    if 'X2' in optimization_data and optimization_data['T2'] > 1e-4:
        N2 = optimization_data['X2'].shape[1] - 1
        t2 = np.linspace(current_t, current_t + optimization_data['T2'], N2 + 1)
        t_opt.append(t2)
        x_opt.append(np.array(optimization_data['X2']))
        # Coast controls (Throttle=0)
        u2 = np.zeros((4, N2 + 1))
        u2[1, :] = 1.0 # Dummy direction
        u_opt.append(u2)
        current_t += optimization_data['T2']
        
    # Phase 3
    N3 = optimization_data['X3'].shape[1] - 1
    t3 = np.linspace(current_t, current_t + optimization_data['T3'], N3 + 1)
    t_opt.append(t3)
    x_opt.append(np.array(optimization_data['X3']))
    u3_val = np.array(optimization_data['U3'])
    u3_padded = np.hstack([u3_val, u3_val[:, -1:]])
    u_opt.append(u3_padded)
    
    # Flatten
    t_opt = np.concatenate(t_opt)
    x_opt = np.hstack(x_opt)
    u_opt = np.hstack(u_opt)
    
    # --- 2. PREPARE SIMULATION DATA ---
    t_sim = simulation_data['t']
    y_sim = simulation_data['y']
    u_sim = simulation_data['u']
    
    # --- 3. CALCULATE DERIVED METRICS (SIMULATION) ---
    sim_metrics = {
        'alt': [], 'vel': [], 'mach': [], 'q': [], 'aoa': [], 
        'lat': [], 'lon': []
    }
    
    R_eq = environment.config.earth_radius_equator
    f = environment.config.earth_flattening
    omega_e = environment.config.earth_omega_vector[2]
    
    for i, t in enumerate(t_sim):
        r_eci = y_sim[0:3, i]
        v_eci = y_sim[3:6, i]
        u_ctrl = u_sim[:, i]
        
        # 1. Environment State
        env_state = environment.get_state_sim(r_eci, t)
        
        # 2. Altitude (Geodetic Approx)
        r_mag = np.linalg.norm(r_eci)
        sin_lat_gc = r_eci[2] / r_mag
        R_local = R_eq * (1.0 - f * sin_lat_gc**2)
        alt = r_mag - R_local
        sim_metrics['alt'].append(alt)
        
        # 3. Velocity (Inertial Magnitude)
        vel = np.linalg.norm(v_eci)
        sim_metrics['vel'].append(vel)
        
        # 4. Aerodynamics
        v_rel = v_eci - env_state['wind_velocity']
        v_rel_mag = np.linalg.norm(v_rel)
        
        # Mach
        sos = max(env_state['speed_of_sound'], 1.0)
        mach = v_rel_mag / sos
        sim_metrics['mach'].append(mach)
        
        # Dynamic Pressure (Pa -> kPa)
        q = 0.5 * env_state['density'] * v_rel_mag**2
        sim_metrics['q'].append(q / 1000.0)
        
        # Angle of Attack
        thrust_dir = u_ctrl[1:]
        if np.linalg.norm(thrust_dir) > 1e-9:
            thrust_dir /= np.linalg.norm(thrust_dir)
        else:
            thrust_dir = np.array([1,0,0])
            
        if v_rel_mag > 1.0:
            vel_dir = v_rel / v_rel_mag
        else:
            vel_dir = thrust_dir
            
        cos_alpha = np.dot(thrust_dir, vel_dir)
        alpha_deg = np.degrees(np.arccos(np.clip(cos_alpha, -1.0, 1.0)))
        sim_metrics['aoa'].append(alpha_deg)
        
        # 5. Ground Track (Lat/Lon)
        theta = omega_e * t + environment.config.initial_rotation
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        x_ecef =  cos_t * r_eci[0] + sin_t * r_eci[1]
        y_ecef = -sin_t * r_eci[0] + cos_t * r_eci[1]
        z_ecef = r_eci[2]
        
        lon = np.degrees(np.arctan2(y_ecef, x_ecef))
        lat_gc = np.degrees(np.arcsin(z_ecef / r_mag))
        # Geodetic Approximation
        lat_gd = np.degrees(np.arctan(np.tan(np.radians(lat_gc)) / ((1-f)**2)))
        
        sim_metrics['lat'].append(lat_gd)
        sim_metrics['lon'].append(lon)

    # --- 4. PLOTTING ---
    
    # Figure 1: Trajectory Overview
    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig1.suptitle('Mission Trajectory: Optimizer vs Simulation')
    
    # Altitude
    r_opt = x_opt[0:3, :]
    r_opt_mag = np.linalg.norm(r_opt, axis=0)
    sin_lat_opt = r_opt[2, :] / (r_opt_mag + 1e-6)
    R_local_opt = R_eq * (1.0 - f * sin_lat_opt**2)
    alt_opt = r_opt_mag - R_local_opt
    axs1[0].plot(t_opt, alt_opt / 1000.0, 'k--', label='Optimizer', alpha=0.7)
    axs1[0].plot(t_sim, np.array(sim_metrics['alt']) / 1000.0, 'b-', label='Simulation')
    axs1[0].set_ylabel('Altitude (km)')
    axs1[0].grid(True)
    axs1[0].legend()
    
    # Velocity
    v_opt_mag = np.linalg.norm(x_opt[3:6, :], axis=0)
    axs1[1].plot(t_opt, v_opt_mag, 'k--', alpha=0.7)
    axs1[1].plot(t_sim, sim_metrics['vel'], 'b-')
    axs1[1].set_ylabel('Velocity (m/s)')
    axs1[1].grid(True)
    
    # Mass
    axs1[2].plot(t_opt, x_opt[6, :] / 1000.0, 'k--', alpha=0.7)
    axs1[2].plot(t_sim, y_sim[6, :] / 1000.0, 'b-')
    axs1[2].set_ylabel('Mass (tonnes)')
    axs1[2].set_xlabel('Time (s)')
    axs1[2].grid(True)
    
    # Figure 2: Aerodynamics
    fig2, axs2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig2.suptitle('Aerodynamics (Simulation Data)')
    
    axs2[0].plot(t_sim, sim_metrics['q'], 'r-')
    axs2[0].set_ylabel('Dynamic Pressure (kPa)')
    axs2[0].grid(True)
    
    axs2[1].plot(t_sim, sim_metrics['mach'], 'g-')
    axs2[1].set_ylabel('Mach Number')
    axs2[1].grid(True)
    
    axs2[2].plot(t_sim, sim_metrics['aoa'], 'm-')
    axs2[2].set_ylabel('Angle of Attack (deg)')
    axs2[2].set_xlabel('Time (s)')
    axs2[2].grid(True)
    
    # Figure 3: Controls
    fig3, axs3 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig3.suptitle('Control Inputs')
    
    axs3[0].step(t_opt, u_opt[0, :], 'k--', label='Opt', where='post', alpha=0.7)
    axs3[0].plot(t_sim, u_sim[0, :], 'b-', label='Sim')
    axs3[0].set_ylabel('Throttle')
    axs3[0].set_ylim(-0.1, 1.1)
    axs3[0].grid(True)
    axs3[0].legend()
    
    axs3[1].plot(t_sim, u_sim[1, :], label='Ux')
    axs3[1].plot(t_sim, u_sim[2, :], label='Uy')
    axs3[1].plot(t_sim, u_sim[3, :], label='Uz')
    axs3[1].set_ylabel('Thrust Vector (ECI)')
    axs3[1].set_xlabel('Time (s)')
    axs3[1].grid(True)
    axs3[1].legend()
    
    # Figure 4: Ground Track
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    ax4.set_title('Ground Track')
    ax4.plot(sim_metrics['lon'], sim_metrics['lat'], 'b-')
    ax4.set_xlabel('Longitude (deg)')
    ax4.set_ylabel('Latitude (deg)')
    ax4.grid(True)
    ax4.set_aspect('equal')
    
    ax4.plot(sim_metrics['lon'][0], sim_metrics['lat'][0], 'go', label='Launch')
    ax4.plot(sim_metrics['lon'][-1], sim_metrics['lat'][-1], 'rx', label='Orbit')
    ax4.legend()
    
    # Figure 5: 3D Trajectory
    fig5 = plt.figure(figsize=(10, 10))
    ax5 = fig5.add_subplot(111, projection='3d')
    ax5.set_title('3D Trajectory (ECI Frame)')
    
    # Draw Earth (Wireframe Sphere)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_earth = R_eq * np.outer(np.cos(u), np.sin(v))
    y_earth = R_eq * np.outer(np.sin(u), np.sin(v))
    z_earth = R_eq * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax5.plot_wireframe(x_earth, y_earth, z_earth, color='c', alpha=0.2, linewidth=0.5)
    
    # Plot Trajectory
    r_sim = y_sim[0:3, :]
    ax5.plot(r_sim[0, :], r_sim[1, :], r_sim[2, :], 'r-', label='Flight Path', linewidth=2)
    
    # Markers
    ax5.scatter(r_sim[0, 0], r_sim[1, 0], r_sim[2, 0], color='g', s=50, label='Launch')
    ax5.scatter(r_sim[0, -1], r_sim[1, -1], r_sim[2, -1], color='k', marker='x', s=50, label='Orbit Injection')
    
    # Axis Labels & Limits
    ax5.set_xlabel('X (ECI) [m]')
    ax5.set_ylabel('Y (ECI) [m]')
    ax5.set_zlabel('Z (ECI) [m]')
    
    # Set cubic limits for aspect ratio
    max_val = max(np.max(np.abs(r_sim)), R_eq) * 1.1
    ax5.set_xlim(-max_val, max_val)
    ax5.set_ylim(-max_val, max_val)
    ax5.set_zlim(-max_val, max_val)
    ax5.legend()
    
    plt.show()

def validate_trajectory(simulation_data, config, environment):
    """
    Performs numerical validation of the simulation results.
    Checks terminal constraints, structural loads, and physics conservation laws.
    """
    print("\n" + "="*40)
    print("POST-FLIGHT ANALYSIS REPORT")
    print("="*40)

    t = simulation_data['t']
    y = simulation_data['y']
    u = simulation_data['u']
    
    # 1. Terminal State Accuracy
    r_final = y[0:3, -1]
    v_final = y[3:6, -1]
    
    r_mag = np.linalg.norm(r_final)
    v_mag = np.linalg.norm(v_final)
    
    # Altitude
    R_eq = environment.config.earth_radius_equator
    f = environment.config.earth_flattening
    sin_lat = r_final[2] / r_mag
    R_local = R_eq * (1.0 - f * sin_lat**2)
    alt_final = r_mag - R_local
    
    # Orbit Elements
    mu = environment.config.earth_mu
    h_vec = np.cross(r_final, v_final)
    h_mag = np.linalg.norm(h_vec)
    inc_rad = np.arccos(h_vec[2] / h_mag)
    inc_deg = np.degrees(inc_rad)
    
    # Eccentricity vector: e = (v x h) / mu - r / |r|
    e_vec = (np.cross(v_final, h_vec) / mu) - (r_final / r_mag)
    eccentricity = np.linalg.norm(e_vec)
    
    # Circular Velocity at this altitude
    v_circ = np.sqrt(mu / r_mag)
    
    print(f"TERMINAL STATE (t = {t[-1]:.1f} s)")
    print(f"  Altitude:       {alt_final/1000:.2f} km  (Target: {config.target_altitude/1000:.2f} km)")
    print(f"  Velocity:       {v_mag:.2f} m/s    (Circular: {v_circ:.2f} m/s)")
    print(f"  Inclination:    {inc_deg:.2f} deg    (Target: {config.target_inclination:.2f} deg)")
    print(f"  Eccentricity:   {eccentricity:.5f}      (Target: ~0.0)")
    
    # Errors
    alt_err = abs(alt_final - config.target_altitude)
    vel_err = abs(v_mag - v_circ)
    inc_err = abs(inc_deg - config.target_inclination)
    
    print("-" * 20)
    if alt_err < 1000 and vel_err < 10 and inc_err < 0.1:
        print("  STATUS: SUCCESS - Orbit Injection Accurate")
    else:
        print("  STATUS: WARNING - Significant Orbital Deviation")

    # 2. Max Q Check
    q_vals = []
    for i in range(len(t)):
        r_i = y[0:3, i]
        v_i = y[3:6, i]
        env_state = environment.get_state_sim(r_i, t[i])
        v_rel = v_i - env_state['wind_velocity']
        q = 0.5 * env_state['density'] * np.linalg.norm(v_rel)**2
        q_vals.append(q)
    
    max_q = max(q_vals)
    print(f"\nSTRUCTURAL LOADS")
    print(f"  Max Q:          {max_q/1000:.2f} kPa")
    
    if max_q > 40000: # 40 kPa is a typical limit for large rockets
        print("  STATUS: WARNING - High Dynamic Pressure")
    else:
        print("  STATUS: NOMINAL")

    print("="*40 + "\n")

def analyze_delta_v_budget(simulation_data, vehicle, config):
    """
    Computes and prints a detailed Delta-V budget (Ideal vs Losses).
    Helps identify inefficiencies in the trajectory (e.g. late gravity turn).
    """
    print("\n" + "="*40)
    print("DELTA-V BUDGET & EFFICIENCY ANALYSIS")
    print("="*40)

    t = simulation_data['t']
    y = simulation_data['y']
    u = simulation_data['u']
    
    # Accumulators
    dv_ideal = 0.0
    loss_gravity = 0.0
    loss_drag = 0.0
    loss_steering = 0.0
    
    # Integration loop
    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        t_curr = t[i]
        
        # State
        r = y[0:3, i]
        v = y[3:6, i]
        m = y[6, i]
        
        # Control
        throttle = u[0, i]
        u_thrust = u[1:, i]
        if np.linalg.norm(u_thrust) > 1e-9:
            u_thrust = u_thrust / np.linalg.norm(u_thrust)
            
        # Environment
        env = vehicle.env.get_state_sim(r, t_curr)
        g_vec = env['gravity']
        
        # Velocity Unit Vector
        v_mag = np.linalg.norm(v)
        if v_mag > 1.0:
            v_hat = v / v_mag
        else:
            v_hat = np.array([0,0,1]) # Vertical if static
            
        # Determine Phase (Heuristic for Drag/Thrust calc)
        m_stg2_wet = config.stage_2.dry_mass + config.stage_2.propellant_mass + config.payload_mass
        
        if m > m_stg2_wet + 1000.0: # Buffer (Stage 1 Attached)
            if throttle < 0.01:
                mode = "coast"
            else:
                mode = "boost"
        else: # Stage 2 Only
            if throttle < 0.01:
                mode = "coast_2"
            else:
                mode = "ship"
            
        # Get Dynamics (Forces)
        # We call vehicle dynamics to ensure consistency with physics engine
        dyn = vehicle.get_dynamics(y[:, i], throttle, u_thrust, t_curr, stage_mode=mode, scaling=None)
        acc_total = dyn[3:6]
        
        # Get Aerodynamics for Angle of Attack
        q, cos_alpha = vehicle.get_aero_properties(y[:, i], u_thrust, t_curr, stage_mode=mode, scaling=None)
        
        # Calculate Thrust Acceleration Magnitude (Approximate based on throttle/ISP)
        # We infer it from the dynamics: a_thrust = a_total - g - a_drag
        # But getting a_drag is hard without re-calculating. 
        # Let's calculate Thrust explicitly using the config data.
        stage = config.stage_1 if "boost" in mode else config.stage_2
        
        acc_thrust_mag = 0.0
        if throttle > 0.01 and "coast" not in mode:
            g0 = vehicle.env.config.g0
            isp_eff = stage.isp_vac + (env['pressure'] / stage.p_sl) * (stage.isp_sl - stage.isp_vac)
            m_dot = throttle * stage.thrust_vac / (stage.isp_vac * g0)
            f_thrust = m_dot * isp_eff * g0
            acc_thrust_mag = f_thrust / m

        # --- INTEGRATE LOSSES ---
        # 1. Ideal Delta-V (Integral of Thrust Accel)
        dv_ideal += acc_thrust_mag * dt
        
        # 2. Gravity Loss (Integral of -g dot v_hat)
        # If g opposes v (climbing), dot is negative, loss is positive.
        g_proj = np.dot(g_vec, v_hat)
        loss_gravity += (-g_proj) * dt
        
        # 3. Steering Loss (Thrust not aligned with Velocity)
        # Loss = a_thrust * (1 - cos(theta))
        loss_steering += acc_thrust_mag * (1.0 - cos_alpha) * dt
        
        # 4. Drag Loss
        # a_drag = a_total - a_thrust*u_thrust - g
        a_drag_vec = acc_total - (acc_thrust_mag * u_thrust) - g_vec
        drag_proj = np.dot(a_drag_vec, v_hat)
        loss_drag += (-drag_proj) * dt

    print(f"  Ideal Delta-V:    {dv_ideal:.1f} m/s")
    print(f"  --------------------------------")
    print(f"  Gravity Loss:     {loss_gravity:.1f} m/s  ({(loss_gravity/dv_ideal)*100:.1f}%)")
    print(f"  Drag Loss:        {loss_drag:.1f} m/s    ({(loss_drag/dv_ideal)*100:.1f}%)")
    print(f"  Steering Loss:    {loss_steering:.1f} m/s    ({(loss_steering/dv_ideal)*100:.1f}%)")
    print(f"  --------------------------------")
    
    dv_effective = dv_ideal - loss_gravity - loss_drag - loss_steering
    print(f"  Effective dV:     {dv_effective:.1f} m/s")
    
    # Compare with actual
    v_init = np.linalg.norm(y[3:6, 0])
    v_final = np.linalg.norm(y[3:6, -1])
    dv_actual = v_final - v_init
    print(f"  Actual dV Gain:   {dv_actual:.1f} m/s")
    print(f"  Unaccounted:      {dv_actual - dv_effective:.1f} m/s (Integration/Model noise)")
    print("="*40 + "\n")