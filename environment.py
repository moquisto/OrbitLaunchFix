# environment.py
# Purpose: Provides the physical world model (Gravity, Atmosphere).
# Must support CasADi symbolic types for the optimizer.

import casadi as ca
import numpy as np

class Environment:
    def __init__(self, config):
        self.config = config
        # Setup B-splines for atmosphere (Placeholder for brevity)
        # In real code, load US76 data here
        self.rho_interp = lambda h: 1.225 * ca.exp(-h/7200.0) # Simple exponential fallback
        self.sos_interp = lambda h: 340.0 # Constant fallback
        self.p_interp = lambda h: 101325.0 * ca.exp(-h/7200.0)

    def get_launch_site_state(self):
        """Calculates initial r_eci and v_eci for the launch pad."""
        lat_rad = np.deg2rad(self.config.launch_latitude)
        lon_rad = np.deg2rad(self.config.launch_longitude)
        alt = self.config.launch_altitude
        
        R_eq = self.config.earth_radius_equator
        f = self.config.earth_flattening
        
        # WGS84 Geodetic to ECEF
        e2 = 2*f - f**2
        N = R_eq / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
        
        x_ecef = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y_ecef = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z_ecef = (N * (1 - e2) + alt) * np.sin(lat_rad)
        
        r_ecef = np.array([x_ecef, y_ecef, z_ecef])
        
        # Rotate to ECI at t=0
        theta = self.config.initial_rotation
        c, s = np.cos(theta), np.sin(theta)
        R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        r_eci = R_z @ r_ecef
        
        # Initial Velocity (Earth Rotation)
        omega = self.config.earth_omega_vector
        v_eci = np.cross(omega, r_eci)
        
        return r_eci, v_eci

    def get_state_opti(self, position_vector_sym, time_sym):
        """
        Symbolic environment state calculation.
        Handles Earth rotation dependent on t_abs.
        """
        # 1. Rotation ECI -> ECEF
        omega_e = self.config.earth_omega_vector[2]
        theta = omega_e * time_sym + self.config.initial_rotation
        
        c = ca.cos(theta)
        s = ca.sin(theta)
        
        # Rotation Matrix (ECI to ECEF is Transpose of R_z)
        x_ecef = position_vector_sym[0] * c + position_vector_sym[1] * s
        y_ecef = -position_vector_sym[0] * s + position_vector_sym[1] * c
        z_ecef = position_vector_sym[2]
        
        # 2. Geodetic Altitude (Non-iterative Approximation)
        R_eq = self.config.earth_radius_equator
        f = self.config.earth_flattening
        
        p = ca.sqrt(x_ecef**2 + y_ecef**2)
        r_mag = ca.sqrt(p**2 + z_ecef**2)
        sin_lat_gc = z_ecef / (r_mag + 1e-8)
        
        R_local = R_eq * (1.0 - f * sin_lat_gc**2)
        altitude = r_mag - R_local
        h_safe = ca.fmax(0.0, altitude)
        
        # 3. Atmosphere
        rho = self.rho_interp(h_safe)
        sos = self.sos_interp(h_safe)
        press = self.p_interp(h_safe)
        
        # 4. Gravity (J2)
        mu = self.config.earth_mu
        J2 = self.config.j2_constant
        
        r2 = x_ecef**2 + y_ecef**2 + z_ecef**2
        r_norm = ca.sqrt(r2)
        
        g_central = -mu / (r2 * r_norm) * ca.vertcat(x_ecef, y_ecef, z_ecef)
        
        z2_r2 = (z_ecef / r_norm)**2
        factor = (1.5 * J2 * mu * R_eq**2) / (r2 * r2 * r_norm)
        
        gj_x = factor * x_ecef * (5 * z2_r2 - 1)
        gj_y = factor * y_ecef * (5 * z2_r2 - 1)
        gj_z = factor * z_ecef * (5 * z2_r2 - 3)
        g_j2 = ca.vertcat(gj_x, gj_y, gj_z)
        
        g_ecef = g_central + g_j2
        
        # Rotate Gravity back to ECI
        gx_eci = g_ecef[0] * c - g_ecef[1] * s
        gy_eci = g_ecef[0] * s + g_ecef[1] * c
        gz_eci = g_ecef[2]
        g_vec = ca.vertcat(gx_eci, gy_eci, gz_eci)
        
        # 5. Wind (Co-rotation)
        wx = -omega_e * position_vector_sym[1]
        wy = omega_e * position_vector_sym[0]
        wz = 0.0
        v_wind = ca.vertcat(wx, wy, wz)
        
        return {
            'density': rho,
            'pressure': press,
            'speed_of_sound': sos,
            'gravity': g_vec,
            'wind_velocity': v_wind
        }