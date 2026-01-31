# environment.py
# Purpose: Provides the physical world model (Gravity, Atmosphere).
# Must support CasADi symbolic types for the optimizer.

import casadi as ca
import numpy as np
import ussa1976

class Environment:
    def __init__(self, config):
        self.config = config
        
        # 1. Load atmospheric data (US Standard Atmosphere 1976, 0-1000km continuous)
        step_size = self.config.atmosphere_step
        max_altitude = self.config.atmosphere_max_alt
        self.grid_altitudes = np.arange(0, max_altitude + step_size, step_size)
        
        # Compute USSA1976
        data = ussa1976.compute(z=self.grid_altitudes)
        rho_values = data["rho"].values
        p_values = data["p"].values
        t_values = data["t"].values
        
        # Calculate Speed of Sound: a = sqrt(gamma * R * T)
        sos_values = np.sqrt(self.config.air_gamma * self.config.air_gas_constant * t_values)
        
        # 2. Setup CasADi Interpolants (B-splines)
        # Use Log-Density interpolation to handle 15 orders of magnitude and prevent negative values.
        # Clamp rho to small positive value to avoid log(0)
        rho_values = np.maximum(rho_values, 1e-12)
        
        self.log_rho_interp = ca.interpolant('log_rho', 'bspline', [self.grid_altitudes], np.log(rho_values))
        self.p_interp   = ca.interpolant('p',   'bspline', [self.grid_altitudes], p_values)
        self.sos_interp = ca.interpolant('sos', 'bspline', [self.grid_altitudes], sos_values)
        
        # 3. Compile CasADi function for Simulation use (Guarantees 1:1 parity)
        self._compile_sim_function()

    def _compile_sim_function(self):
        # Create a CasADi function for numeric evaluation
        r_sym = ca.MX.sym('r', 3)
        t_sym = ca.MX.sym('t', 1)
        
        state_sym = self.get_state_opti(r_sym, t_sym)
        
        # Pack outputs: [rho, p, sos, g_x, g_y, g_z, w_x, w_y, w_z]
        out_vec = ca.vertcat(
            state_sym['density'],
            state_sym['pressure'],
            state_sym['speed_of_sound'],
            state_sym['gravity'],
            state_sym['wind_velocity']
        )
        
        self.sim_func = ca.Function('env_sim', [r_sym, t_sym], [out_vec])

    def get_launch_site_state(self):
        # Purpose: Calculate initial r_eci and v_eci at t=0 for the launch pad.

        # 1. INPUTS
        lat = self.config.launch_latitude
        lon = self.config.launch_longitude
        alt = self.config.launch_altitude
        R_eq = self.config.earth_radius_equator
        f = self.config.earth_flattening

        # 2. GEODETIC TO ECEF CONVERSION (WGS84)
        # Calculate Prime Vertical Radius of Curvature (N)
        e_sq = 2*f - f**2
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        sin_lat = np.sin(lat_rad)
        
        N = R_eq / np.sqrt(1 - e_sq * sin_lat**2)

        # Calculate Cartesian Coordinates
        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - e_sq) + alt) * sin_lat
        r_ecef = np.array([x, y, z])

        # 3. ROTATE TO INERTIAL FRAME (ECI)
        # Apply initial Earth rotation offset (theta_0)
        theta_0 = self.config.initial_rotation
        # R_z(theta_0) rotation matrix (ECEF -> ECI)
        cos_t = np.cos(theta_0)
        sin_t = np.sin(theta_0)
        R_z = np.array([
            [cos_t, -sin_t, 0],
            [sin_t,  cos_t, 0],
            [0,      0,     1]
        ])
        r_eci = R_z @ r_ecef

        # 4. INITIAL VELOCITY (Inertial Frame)
        # Velocity is due to Earth's rotation: v = omega x r
        v_eci = np.cross(self.config.earth_omega_vector, r_eci)

        # 5. RETURN
        return r_eci, v_eci

    def get_state_opti(self, position_vector_sym, time_sym):
        # Purpose: Calculate environment state (Gravity, Atmosphere) in ECI frame.
        #          Must be fully symbolic (CasADi) and differentiable.

        # 1. ROTATION (ECI -> ECEF)
        # Earth rotates around Z-axis by angle theta = omega * t + initial_rotation
        omega_z = self.config.earth_omega_vector[2]
        theta = omega_z * time_sym + self.config.initial_rotation
        
        cos_t = ca.cos(theta)
        sin_t = ca.sin(theta)
        
        # Rotation Matrix R (ECI to ECEF) - Transpose of Z-rotation
        # [  c, s, 0 ]
        # [ -s, c, 0 ]
        # [  0, 0, 1 ]
        r_x = position_vector_sym[0]
        r_y = position_vector_sym[1]
        r_z = position_vector_sym[2]
        
        r_ecef_x =  cos_t * r_x + sin_t * r_y
        r_ecef_y = -sin_t * r_x + cos_t * r_y
        r_ecef_z =  r_z
        
        r_ecef = ca.vertcat(r_ecef_x, r_ecef_y, r_ecef_z)

        # 2. GEODETIC ALTITUDE CALCULATION (Exact Ellipsoid Radius)
        # Calculates the exact radius of the WGS84 ellipsoid at the current geocentric latitude.
        r_sq = ca.dot(r_ecef, r_ecef) + 1e-16
        r_mag = ca.sqrt(r_sq)
        
        R_eq = self.config.earth_radius_equator
        f = self.config.earth_flattening
        R_pol = R_eq * (1.0 - f)
        
        # Exact Radius of Ellipsoid at Geocentric Latitude phi'
        # R(phi') = (R_eq * R_pol * r) / sqrt( R_pol^2 * (x^2+y^2) + R_eq^2 * z^2 )
        rho_sq = r_ecef_x**2 + r_ecef_y**2
        z_sq = r_ecef_z**2
        
        denom = ca.sqrt( (R_pol**2) * rho_sq + (R_eq**2) * z_sq + 1e-16 )
        R_local = (R_eq * R_pol * r_mag) / denom
        
        altitude = r_mag - R_local

        # 3. ATMOSPHERE LOOKUP
        # Clamp altitude to [0, max_alt] for B-spline safety
        alt_clamped = ca.fmax(0.0, ca.fmin(altitude, self.config.atmosphere_max_alt))
        
        # Log-space interpolation for density (stable & positive)
        log_rho = self.log_rho_interp(alt_clamped)
        rho = ca.exp(log_rho)
        press = self.p_interp(alt_clamped)
        sos = self.sos_interp(alt_clamped)

        # Vacuum Handling: Smooth fade to 0 if altitude > max_alt
        # Use smooth sigmoid (tanh) instead of hard if_else to preserve gradients for IPOPT.
        cutoff_center = self.config.atmosphere_max_alt + 2000.0
        blend = 0.5 * (1.0 - ca.tanh((altitude - cutoff_center) / 1000.0))
        
        rho = rho * blend
        press = press * blend
        sos = ca.fmax(sos, 1.0) # Safety floor for Mach calculation

        # 4. GRAVITY (Central + J2)
        # Calculate in ECEF (where J2 field is static)
        mu = self.config.earth_mu
        J2 = self.config.j2_constant
        
        g_central = -mu / (r_mag**3) * r_ecef

        # J2 Perturbation
        if self.config.use_j2_perturbation:
            z_sq_ratio = z_sq / r_sq
            factor = -1.5 * J2 * mu * (R_eq**2) / (r_mag**5)
            
            term_xy = 1.0 - 5.0 * z_sq_ratio
            term_z  = 3.0 - 5.0 * z_sq_ratio
            
            g_j2_x = factor * r_ecef_x * term_xy
            g_j2_y = factor * r_ecef_y * term_xy
            g_j2_z = factor * r_ecef_z * term_z
            
            g_ecef = g_central + ca.vertcat(g_j2_x, g_j2_y, g_j2_z)
        else:
            g_ecef = g_central

        # Rotate Gravity back to ECI
        # R_ecef_to_eci = [ [c, -s, 0], [s, c, 0], [0, 0, 1] ]
        g_eci_x =  cos_t * g_ecef[0] - sin_t * g_ecef[1]
        g_eci_y =  sin_t * g_ecef[0] + cos_t * g_ecef[1]
        g_eci_z =  g_ecef[2]
        
        g_eci = ca.vertcat(g_eci_x, g_eci_y, g_eci_z)

        # 5. WIND / AIR VELOCITY (Inertial Frame)
        # V_air_inertial = V_wind_local_rotated + (Omega x r_inertial)
        # Assuming local wind is zero (atmosphere rotates with Earth)
        omega_vec = ca.DM(self.config.earth_omega_vector)
        v_air_eci = ca.cross(omega_vec, position_vector_sym)

        # 6. RETURN
        return {
           'density': rho,
           'pressure': press,
           'speed_of_sound': sos,
           'gravity': g_eci,
           'wind_velocity': v_air_eci
        }

    def get_state_sim(self, position_vector_num, time_num):
        # Purpose: Fast numeric wrapper for simulation (scipy.solve_ivp).
        # CRITICAL: Must use the EXACT same physics function as the optimizer to avoid drift.
        
        # Call the compiled CasADi function
        res = self.sim_func(position_vector_num, time_num)
        
        # Unpack 'res' (CasADi DM) into a standard Python Dictionary with floats/numpy arrays.
        res_np = np.array(res).flatten()
        
        return {
            'density': float(res_np[0]),
            'pressure': float(res_np[1]),
            'speed_of_sound': float(res_np[2]),
            'gravity': res_np[3:6],
            'wind_velocity': res_np[6:9]
        }