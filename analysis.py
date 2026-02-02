# analysis.py
# Purpose: Visualization and Data Analysis.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

def plot_mission(optimization_data, simulation_data, environment, config=None):
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
        x2_val = np.array(optimization_data['X2'])
        x_opt.append(x2_val)
        # Coast controls (Throttle=0)
        u2 = np.zeros((4, N2 + 1))
        
        # Align dummy direction with velocity to show 0 AoA in plots
        v2 = x2_val[3:6, :]
        v2_norm = np.linalg.norm(v2, axis=0)
        v2_dir = np.divide(v2, v2_norm, out=np.zeros_like(v2), where=v2_norm > 1e-9)
        v2_dir[:, v2_norm <= 1e-9] = np.array([1,0,0])[:, None] # Fallback
        u2[1:, :] = v2_dir
        
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
        'aoa_pitch': [], 'aoa_yaw': [],
        'lat': [], 'lon': [],
        'gamma': [], 'energy': [], 'heating': [], 'tw': [], 'g_load': [],
        'downrange': [],
        'drag': [], 'thrust': []
    }
    
    R_eq = environment.config.earth_radius_equator
    f = environment.config.earth_flattening
    omega_e = environment.config.earth_omega_vector[2]
    
    r_ecef_0 = None
    
    for i, t in enumerate(t_sim):
        r_eci = y_sim[0:3, i]
        v_eci = y_sim[3:6, i]
        u_ctrl = u_sim[:, i]
        
        # 1. Environment State
        env_state = environment.get_state_sim(r_eci, t)
        
        # 2. Altitude (Geodetic Approx)
        r_mag = np.linalg.norm(r_eci)
        # Exact WGS84 Ellipsoid Radius (matching environment.py)
        # R(phi') = sqrt( (Req^2 cos)^2 + (Rpol^2 sin)^2 ) / sqrt( (Req cos)^2 + (Rpol sin)^2 )
        # Simplified for geocentric coords (x,y,z):
        R_pol = R_eq * (1.0 - f)
        rho_sq = r_eci[0]**2 + r_eci[1]**2
        z_sq = r_eci[2]**2
        denom = np.sqrt( (R_pol**2) * rho_sq + (R_eq**2) * z_sq )
        R_local = (R_eq * R_pol * r_mag) / denom
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
        
        # AoA Decomposition (Pitch vs Yaw)
        # Define Trajectory Plane Normal (n = r x v_rel)
        plane_norm = np.cross(r_eci, v_rel)
        plane_norm_mag = np.linalg.norm(plane_norm)
        
        if v_rel_mag > 1.0 and plane_norm_mag > 1e-3:
            plane_norm /= plane_norm_mag
            
            # Define "Lift" Direction (In-Plane Normal) p = v_rel x n
            lift_dir = np.cross(vel_dir, plane_norm)
            
            # Pitch AoA (Projection of Thrust onto Lift Vector)
            pitch_cmp = np.dot(thrust_dir, lift_dir)
            aoa_pitch = np.degrees(np.arcsin(np.clip(pitch_cmp, -1.0, 1.0)))
            
            # Yaw/Sideslip AoA (Projection of Thrust onto Plane Normal)
            yaw_cmp = np.dot(thrust_dir, plane_norm)
            aoa_yaw = np.degrees(np.arcsin(np.clip(yaw_cmp, -1.0, 1.0)))
        else:
            # Undefined plane (Vertical flight or Static) -> Zero components
            aoa_pitch = 0.0
            aoa_yaw = 0.0
            
        sim_metrics['aoa_pitch'].append(aoa_pitch)
        sim_metrics['aoa_yaw'].append(aoa_yaw)
        
        # 5. Ground Track (Lat/Lon)
        theta = omega_e * t + environment.config.initial_rotation
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        x_ecef =  cos_t * r_eci[0] + sin_t * r_eci[1]
        y_ecef = -sin_t * r_eci[0] + cos_t * r_eci[1]
        z_ecef = r_eci[2]
        
        # Downrange Distance Calculation (Great Circle)
        r_ecef = np.array([x_ecef, y_ecef, z_ecef])
        if r_ecef_0 is None:
            r_ecef_0 = r_ecef
            
        dot_val = np.dot(r_ecef_0, r_ecef) / (np.linalg.norm(r_ecef_0) * np.linalg.norm(r_ecef))
        angle = np.arccos(np.clip(dot_val, -1.0, 1.0))
        sim_metrics['downrange'].append((angle * R_eq) / 1000.0)
        
        lon = np.degrees(np.arctan2(y_ecef, x_ecef))
        lat_gc = np.degrees(np.arcsin(z_ecef / r_mag))
        # Geodetic Approximation
        lat_gd = np.degrees(np.arctan(np.tan(np.radians(lat_gc)) / ((1-f)**2)))
        
        sim_metrics['lat'].append(lat_gd)
        sim_metrics['lon'].append(lon)

        # 6. Flight Dynamics Metrics
        # Flight Path Angle (Gamma)
        # Angle between velocity vector and local position vector (90 deg = vertical, 0 deg = horizontal)
        # sin(gamma) = (r . v) / (|r| |v|)
        sin_gamma = np.dot(r_eci, v_eci) / (r_mag * vel + 1e-9)
        gamma = np.degrees(np.arcsin(np.clip(sin_gamma, -1.0, 1.0)))
        sim_metrics['gamma'].append(gamma)

        # Specific Mechanical Energy (per kg)
        # E = v^2/2 - mu/r
        energy = (vel**2)/2.0 - environment.config.earth_mu / r_mag
        sim_metrics['energy'].append(energy / 1e6) # MJ/kg

        # Heating Rate Indicator (0.5 * rho * v^3)
        # Use relative velocity (airspeed) for physical accuracy
        heating = 0.5 * env_state['density'] * (v_rel_mag**3)
        sim_metrics['heating'].append(heating / 1e6) # MW/m^2 approx

        # Thrust-to-Weight & G-Force (Requires Config)
        if config:
            # Determine stage based on mass
            m_stg2_wet = config.stage_2.dry_mass + config.stage_2.propellant_mass + config.payload_mass
            stage = config.stage_1 if y_sim[6, i] > m_stg2_wet + 1000 else config.stage_2
            
            throttle = u_sim[0, i]
            # Calculate accurate thrust (Pressure Compensated)
            isp_eff = stage.isp_vac + (env_state['pressure'] / stage.p_sl) * (stage.isp_sl - stage.isp_vac)
            thrust_force = throttle * stage.thrust_vac * (isp_eff / stage.isp_vac)
            
            weight = y_sim[6, i] * np.linalg.norm(env_state['gravity'])
            
            sim_metrics['tw'].append(thrust_force / weight)
            g0 = environment.config.g0
            sim_metrics['g_load'].append((thrust_force / y_sim[6, i]) / g0)
            
            # Drag Force (kN)
            cd_table = stage.aero.mach_cd_table
            cd_base = np.interp(mach, cd_table[:,0], cd_table[:,1])
            sin_alpha = np.sin(np.radians(alpha_deg))
            cd_total = cd_base + stage.aero.cd_crossflow_factor * sin_alpha**2
            drag_force = q * stage.aero.reference_area * cd_total
            
            sim_metrics['thrust'].append(thrust_force / 1000.0)
            sim_metrics['drag'].append(drag_force / 1000.0)

    # --- 4. PLOTTING ---
    
    # WINDOW 1: MISSION OVERVIEW
    fig1 = plt.figure(figsize=(18, 10), constrained_layout=True)
    fig1.suptitle('Mission Overview: Kinematics & Trajectory', fontsize=16)
    gs1 = fig1.add_gridspec(2, 3)
    
    # 1. Altitude
    ax1 = fig1.add_subplot(gs1[0, 0])
    r_opt = x_opt[0:3, :]
    r_opt_mag = np.linalg.norm(r_opt, axis=0)
    
    R_pol = R_eq * (1.0 - f)
    rho_sq_opt = r_opt[0, :]**2 + r_opt[1, :]**2
    z_sq_opt = r_opt[2, :]**2
    denom_opt = np.sqrt( (R_pol**2) * rho_sq_opt + (R_eq**2) * z_sq_opt )
    R_local_opt = (R_eq * R_pol * r_opt_mag) / denom_opt
    
    alt_opt = r_opt_mag - R_local_opt
    ax1.plot(t_opt, alt_opt / 1000.0, 'k--', label='Optimizer', alpha=0.7)
    ax1.plot(t_sim, np.array(sim_metrics['alt']) / 1000.0, 'b-', label='Simulation')
    ax1.set_ylabel('Altitude (km)')
    ax1.grid(True)
    ax1.set_title('Altitude Profile')
    ax1.legend()
    
    # 2. Velocity
    ax2 = fig1.add_subplot(gs1[0, 1])
    v_opt_mag = np.linalg.norm(x_opt[3:6, :], axis=0)
    ax2.plot(t_opt, v_opt_mag, 'k--', alpha=0.7)
    ax2.plot(t_sim, sim_metrics['vel'], 'b-')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Inertial Velocity')
    ax2.grid(True)
    
    # 3. Mass
    ax3 = fig1.add_subplot(gs1[0, 2])
    ax3.plot(t_opt, x_opt[6, :] / 1000.0, 'k--', alpha=0.7)
    ax3.plot(t_sim, y_sim[6, :] / 1000.0, 'b-')
    ax3.set_ylabel('Mass (tonnes)')
    ax3.set_title('Vehicle Mass')
    ax3.set_xlabel('Time (s)')
    ax3.grid(True)
    
    # 4. Ground Track
    ax4 = fig1.add_subplot(gs1[1, 0])
    ax4.set_title('Ground Track')
    ax4.plot(sim_metrics['lon'], sim_metrics['lat'], 'b-')
    ax4.set_xlabel('Longitude (deg)')
    ax4.set_ylabel('Latitude (deg)')
    ax4.grid(True)
    ax4.set_aspect('equal')
    ax4.plot(sim_metrics['lon'][0], sim_metrics['lat'][0], 'go', label='Launch')
    ax4.plot(sim_metrics['lon'][-1], sim_metrics['lat'][-1], 'rx', label='Orbit')
    ax4.legend()
    
    # 5. Ascent Profile
    ax5 = fig1.add_subplot(gs1[1, 1])
    ax5.set_title('Ascent Profile (Color = AoA)')
    ax5.plot(sim_metrics['downrange'], np.array(sim_metrics['alt']) / 1000.0, 'k-', linewidth=0.5, alpha=0.3)
    ax5.fill_between(sim_metrics['downrange'], 0, np.array(sim_metrics['alt']) / 1000.0, color='skyblue', alpha=0.1)
    sc = ax5.scatter(sim_metrics['downrange'], np.array(sim_metrics['alt']) / 1000.0, c=sim_metrics['aoa'], cmap='plasma', s=15)
    cbar = plt.colorbar(sc, ax=ax5)
    cbar.set_label('AoA (deg)')
    ax5.set_xlabel('Downrange (km)')
    ax5.set_ylabel('Altitude (km)')
    ax5.grid(True)
    
    # 6. 3D Trajectory
    ax6 = fig1.add_subplot(gs1[1, 2], projection='3d')
    ax6.set_title('3D Trajectory (ECI)')
    # ... (3D plotting logic below) ...
    
    # WINDOW 2: DYNAMICS
    fig2 = plt.figure(figsize=(18, 10), constrained_layout=True)
    fig2.suptitle('Aerodynamics & Flight Dynamics', fontsize=16)
    gs2 = fig2.add_gridspec(2, 3)
    
    # 1. Max Q
    ax2_1 = fig2.add_subplot(gs2[0, 0])
    ax2_1.plot(t_sim, sim_metrics['q'], 'r-')
    ax2_1.set_title('Dynamic Pressure (Max Q)')
    ax2_1.set_ylabel('kPa')
    ax2_1.grid(True)
    
    # 2. Mach
    ax2_2 = fig2.add_subplot(gs2[0, 1])
    ax2_2.plot(t_sim, sim_metrics['mach'], 'g-')
    ax2_2.set_title('Mach Number')
    ax2_2.grid(True)
    
    # 3. AoA
    ax2_3 = fig2.add_subplot(gs2[0, 2])
    ax2_3.plot(t_sim, sim_metrics['aoa'], 'k-', label='Total', linewidth=1.5)
    ax2_3.plot(t_sim, sim_metrics['aoa_pitch'], 'b--', label='Pitch', linewidth=1.0, alpha=0.8)
    ax2_3.plot(t_sim, sim_metrics['aoa_yaw'], 'r:', label='Yaw', linewidth=1.0, alpha=0.8)
    ax2_3.set_title('Angle of Attack (deg)')
    ax2_3.grid(True)
    ax2_3.legend()
    
    # WINDOW 3: CONTROLS & LOADS
    fig3 = plt.figure(figsize=(18, 10), constrained_layout=True)
    fig3.suptitle('Controls, Loads & Constraints', fontsize=16)
    gs3 = fig3.add_gridspec(2, 3)
    
    # 1. Throttle
    ax3_1 = fig3.add_subplot(gs3[0, 0])
    ax3_1.step(t_opt, u_opt[0, :], 'k--', label='Opt', where='post', alpha=0.7)
    ax3_1.plot(t_sim, u_sim[0, :], 'b-', label='Sim')
    ax3_1.set_title('Throttle')
    ax3_1.set_ylim(-0.1, 1.1)
    ax3_1.grid(True)
    
    # 3D Plot Logic (Moved to ax6)
    # Draw Earth (Wireframe Ellipsoid)
    # Use WGS84 Ellipsoid dimensions for visual consistency
    print(f"  [Plot] Drawing Earth: R_eq={R_eq/1000:.1f}km, R_pol={R_pol/1000:.1f}km (Flattening f={f:.5f})")
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_earth = R_eq * np.outer(np.cos(u), np.sin(v))
    y_earth = R_eq * np.outer(np.sin(u), np.sin(v))
    z_earth = R_pol * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax6.plot_wireframe(x_earth, y_earth, z_earth, color='c', alpha=0.2, linewidth=0.5)
    
    # Plot Trajectory
    r_sim = y_sim[0:3, :]
    ax6.plot(r_sim[0, :], r_sim[1, :], r_sim[2, :], 'r-', label='Flight Path', linewidth=2)
    
    # --- ORBIT PROJECTION (Future Trajectory) ---
    # Propagate the final state for one orbital period to visualize the resulting orbit
    r_final = r_sim[:, -1]
    v_final = y_sim[3:6, -1]
    mu = environment.config.earth_mu
    
    # Estimate Period: T = 2*pi * sqrt(a^3/mu)
    r_mag_f = np.linalg.norm(r_final)
    v_mag_f = np.linalg.norm(v_final)
    specific_energy = (v_mag_f**2)/2 - mu/r_mag_f
    
    if specific_energy < 0: # Elliptical Orbit
        a = -mu / (2*specific_energy)
        period = 2 * np.pi * np.sqrt(a**3 / mu)
        J2 = environment.config.j2_constant
        
        def two_body_dyn(t, y):
            r = y[0:3]
            v = y[3:6]
            r_mag = np.linalg.norm(r)
            
            # Central Gravity
            g_central = -mu / (r_mag**3) * r
            
            # J2 Perturbation (to match non-spherical Earth)
            # a_J2 = -1.5 * J2 * mu * R_eq^2 / r^5 * ...
            factor = -1.5 * J2 * mu * (R_eq**2) / (r_mag**5)
            z2 = (r[2] / r_mag)**2
            
            gx = factor * r[0] * (1 - 5 * z2)
            gy = factor * r[1] * (1 - 5 * z2)
            gz = factor * r[2] * (3 - 5 * z2)
            
            return np.concatenate([v, g_central + np.array([gx, gy, gz])])
            
        t_eval = np.linspace(0, period, 300) # High resolution for smooth circle
        sol_orbit = solve_ivp(two_body_dyn, [0, period], np.concatenate([r_final, v_final]), t_eval=t_eval, rtol=1e-5)
        ax6.plot(sol_orbit.y[0], sol_orbit.y[1], sol_orbit.y[2], 'k--', linewidth=1, label='Projected Orbit', alpha=0.6)
    
    # Markers
    ax6.scatter(r_sim[0, 0], r_sim[1, 0], r_sim[2, 0], color='g', s=50, label='Launch')
    ax6.scatter(r_sim[0, -1], r_sim[1, -1], r_sim[2, -1], color='k', marker='x', s=50, label='Orbit Injection')
    
    # Axis Labels & Limits
    ax6.set_xlabel('X (ECI) [m]')
    ax6.set_ylabel('Y (ECI) [m]')
    ax6.set_zlabel('Z (ECI) [m]')
    
    # Set cubic limits for aspect ratio
    max_val = max(np.max(np.abs(r_sim)), R_eq) * 1.1
    ax6.set_xlim(-max_val, max_val)
    ax6.set_ylim(-max_val, max_val)
    ax6.set_zlim(-max_val, max_val)
    ax6.set_box_aspect([1, 1, 1]) # Force equal aspect ratio for visual roundness
    ax6.legend()
    
    if config:
        # Window 2 (Dynamics) Continued
        
        # 4. Gamma
        ax2_4 = fig2.add_subplot(gs2[1, 0])
        ax2_4.plot(t_sim, sim_metrics['gamma'], 'b-')
        ax2_4.set_ylabel('Gamma (deg)')
        ax2_4.set_title('Flight Path Angle')
        ax2_4.grid(True)

        # 5. T/W
        ax2_5 = fig2.add_subplot(gs2[1, 1])
        ax2_5.plot(t_sim, sim_metrics['tw'], 'm-')
        ax2_5.set_ylabel('Ratio')
        ax2_5.set_title('Thrust-to-Weight')
        ax2_5.grid(True)
        ax2_5.axhline(1.0, color='k', linestyle='--', alpha=0.5)

        # 6. Energy
        ax2_6 = fig2.add_subplot(gs2[1, 2])
        ax2_6.plot(t_sim, sim_metrics['energy'], 'g-')
        ax2_6.set_ylabel('MJ/kg')
        ax2_6.set_title('Specific Energy')
        ax2_6.grid(True)

        # Window 3 (Controls) Continued
        
        # 2. Vector
        ax3_2 = fig3.add_subplot(gs3[0, 1])
        ax3_2.plot(t_sim, u_sim[1, :], label='Ux')
        ax3_2.plot(t_sim, u_sim[2, :], label='Uy')
        ax3_2.plot(t_sim, u_sim[3, :], label='Uz')
        ax3_2.set_title('Thrust Vector (ECI)')
        ax3_2.grid(True)
        ax3_2.legend()
        
        # 3. G-Load
        ax3_3 = fig3.add_subplot(gs3[0, 2])
        ax3_3.plot(t_sim, sim_metrics['g_load'], 'r-')
        ax3_3.set_ylabel('g')
        ax3_3.set_title('Axial Acceleration')
        ax3_3.grid(True)

        # 4. Heating
        ax3_4 = fig3.add_subplot(gs3[1, 0])
        ax3_4.plot(t_sim, sim_metrics['heating'], 'orange')
        ax3_4.set_ylabel('MW/m^2')
        ax3_4.set_title('Heating Proxy')
        ax3_4.grid(True)
        
        # Forbidden Region for Throttle
        min_th = config.sequence.min_throttle
        if min_th > 0.0:
            ax3_1.axhspan(0.0, min_th, color='red', alpha=0.2, label='Forbidden')
        ax3_1.legend()
        
        # 5. Envelope
        ax3_5 = fig3.add_subplot(gs3[1, 1])
        ax3_5.set_title('Flight Envelope (Q vs AoA)')
        
        # Scatter plot colored by Mach
        sc9 = ax3_5.scatter(sim_metrics['q'], sim_metrics['aoa'], c=sim_metrics['mach'], cmap='viridis', s=10)
        cbar9 = plt.colorbar(sc9, ax=ax3_5)
        cbar9.set_label('Mach Number')
        
        # Max Q Limit line
        q_lim_kpa = config.max_q_limit / 1000.0
        ax3_5.axvline(q_lim_kpa, color='r', linestyle='--', label=f'Max Q Limit ({q_lim_kpa:.1f} kPa)')
        
        ax3_5.set_xlabel('Dynamic Pressure (kPa)')
        ax3_5.set_ylabel('AoA (deg)')
        ax3_5.grid(True)
        ax3_5.legend()
        
        # 6. Forces
        ax3_6 = fig3.add_subplot(gs3[1, 2])
        ax3_6.set_title('Force Balance')
        
        # Clip data to avoid log(0) errors during coast phases
        thrust_plot = np.maximum(sim_metrics['thrust'], 1e-3)
        drag_plot = np.maximum(sim_metrics['drag'], 1e-3)
        
        ax3_6.plot(t_sim, thrust_plot, 'b-', label='Thrust')
        ax3_6.plot(t_sim, drag_plot, 'r-', label='Drag')
        
        ax3_6.set_ylabel('Force (kN)')
        ax3_6.set_yscale('log') # Log scale to see drag at high altitude
        ax3_6.grid(True, which="both", ls="-", alpha=0.5)
        ax3_6.legend()

    plt.show()