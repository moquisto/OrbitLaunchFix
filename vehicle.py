import casadi as ca
import numpy as np
# vehicle.py
# Purpose: Defines the Equations of Motion (Dynamics).

class Vehicle:
    def __init__(self, rocket_config, environment):
        self.config = rocket_config
        self.env = environment
        
        # 1. SETUP CASADI INTERPOLANTS (B-Splines)
        #    # Extract data from config (Mach vs Cd tables)
        #    data_s1 = self.config.stage_1.aero.mach_cd_table
        #    data_s2 = self.config.stage_2.aero.mach_cd_table
        #    # Create lookup functions. 'linear' is robust for optimization.
        #    self.cd_interp_stage_1 = ca.interpolant('cd1', 'linear', [data_s1[:, 0]], data_s1[:, 1])
        #    self.cd_interp_stage_2 = ca.interpolant('cd2', 'linear', [data_s2[:, 0]], data_s2[:, 1])

    def get_dynamics(self, state, throttle, thrust_direction, time, stage_mode="boost", scaling=None):
        # PSEUDOCODE:
        # Purpose: Calculate derivatives [dr, dv, dm]. Handles Symbolic (CasADi) & Numeric (NumPy).

        # 0. DISPATCH: SYMBOLIC VS NUMERIC
        #    is_symbolic = isinstance(state, (ca.MX, ca.SX, ca.DM))
        #    if is_symbolic:
        #        norm, dot, cross = ca.norm_2, ca.dot, ca.cross
        #        fmax, fmin, if_else = ca.fmax, ca.fmin, ca.if_else
        #    else:
        #        norm, dot, cross = np.linalg.norm, np.dot, np.cross
        #        fmax, fmin = max, min
        #        if_else = np.where
        #        # Ensure inputs are numpy arrays for numeric operations
        #        state = np.array(state); thrust_direction = np.array(thrust_direction)

        # 1. SETUP & UNSCALING
        #    # Convert solver variables to physical units.
        #    if scaling:
        #        r_phys = state[0:3] * scaling.length
        #        v_phys = state[3:6] * scaling.speed
        #        m_phys = state[6]   * scaling.mass
        #        t_phys = time       * scaling.time
        #    else:
        #        r_phys, v_phys, m_phys, t_phys = state[0:3], state[3:6], state[6], time

        # 2. ENVIRONMENT LOOKUP
        #    # Dispatch to correct environment method.
        #    if is_symbolic:
        #        env_state = self.env.get_state_opti(r_phys, t_phys)
        #    else:
        #        env_state = self.env.get_state_sim(r_phys, t_phys)

        # 3. AERODYNAMICS
        #    # Relative Velocity
        #    v_rel = v_phys - env_state['wind_velocity']
        #    v_rel_mag = norm(v_rel)
        #
        #    # Mach Number (Singularity Protection)
        #    # Clamp Speed of Sound to >= 1.0 m/s to avoid div/0 in vacuum.
        #    sos_safe = fmax(env_state['speed_of_sound'], 1.0)
        #    mach = v_rel_mag / sos_safe
        #
        #    # Dynamic Pressure
        #    q = 0.5 * env_state['density'] * v_rel_mag**2
        #
        #    # Control Vector
        #    # Handle zero-thrust case (Coast) to avoid zero-vector in u_control.
        #    thrust_mag = norm(thrust_direction)
        #    # Default to [0,0,1] if thrust is zero to prevent NaN in normalization
        #    safe_denom = if_else(thrust_mag > 1e-9, thrust_mag, 1.0)
        #    u_control = thrust_direction / safe_denom
        #
        #    # Angle of Attack (AoA) Logic
        #    # Singularity Check: At Launchpad (v ~ 0), velocity direction is undefined.
        #    # Default to u_control (AoA=0) if v_rel_mag < 0.1.
        #    # Use ca.if_else for symbolic, python if/else for numeric.
        #    u_vel = if_else(v_rel_mag > 0.1, v_rel / (v_rel_mag + 1e-9), u_control)
        #
        #    # Thrust Orientation
        #    # Coast Phase: Assume Prograde (u_vel) to minimize drag.
        #    if "coast" in stage_mode:
        #        u_thrust = u_vel
        #    else:
        #        u_thrust = u_control
        #
        #    # Crossflow Drag (sin^2(alpha))
        #    cross_prod = cross(u_thrust, u_vel)
        #    sin_alpha_sq = dot(cross_prod, cross_prod)
        #
        #    # Drag Force Calculation
        #    # Explicitly handle stage modes to allow coasting on either stage.
        #    if stage_mode in ["boost", "coast", "coast_1"]:
        #        config = self.config.stage_1
        #        cd_curve = self.cd_interp_stage_1
        #    else: # "ship", "coast_2"
        #        config = self.config.stage_2
        #        cd_curve = self.cd_interp_stage_2
        #
        #    # Lookup Cd (Clamp Mach to table limit to avoid extrapolation errors)
        #    max_mach = config.aero.mach_cd_table[-1, 0]
        #    mach_clamped = fmin(mach, max_mach)
        #    cd_base = cd_curve(mach_clamped)
        #    # Ensure CasADi DM objects are cast to float in numeric mode.
        #    if not is_symbolic: cd_base = float(cd_base) 
        #
        #    cd_total = cd_base + config.aero.cd_crossflow_factor * sin_alpha_sq
        #    f_drag = -q * config.aero.reference_area * cd_total * u_vel

        # 4. PROPULSION
        #    stage = config
        #    eff_throttle = 0.0 if "coast" in stage_mode else throttle
        #
        #    # Mass Flow (Choked Flow: Depends on Vacuum Thrust)
        #    g0 = self.env.config.g0
        #    m_dot = eff_throttle * stage.thrust_vac / (stage.isp_vac * g0)
        #
        #    # Thrust Force (Pressure Compensated)
        #    isp_eff = stage.isp_vac + (env_state['pressure'] / stage.p_sl) * (stage.isp_sl - stage.isp_vac)
        #    f_thrust = (m_dot * isp_eff * g0) * u_thrust

        # 5. EQUATIONS OF MOTION
        #    f_total = f_thrust + f_drag + (env_state['gravity'] * m_phys)
        #    acc_phys = f_total / m_phys
        #    dm_dt_phys = -m_dot

        # 6. RESCALING & RETURN
        #    if scaling:
        #        dr_dtau = v_phys     * (scaling.time / scaling.length)
        #        dv_dtau = acc_phys   * (scaling.time / scaling.speed)
        #        dm_dtau = dm_dt_phys * (scaling.time / scaling.mass)
        #        if is_symbolic:
        #            return ca.vertcat(dr_dtau, dv_dtau, dm_dtau)
        #        else:
        #            return np.concatenate([dr_dtau, dv_dtau, [dm_dtau]])
        #    else:
        #        if is_symbolic:
        #            return ca.vertcat(v_phys, acc_phys, dm_dt_phys)
        #        else:
        #            return np.concatenate([v_phys, acc_phys, [dm_dt_phys]])
        pass

    def get_aero_properties(self, state, u_control_vec, time, stage_mode="boost", scaling=None):
        # 0. DISPATCH: SYMBOLIC VS NUMERIC
        is_symbolic = isinstance(state, (ca.MX, ca.SX, ca.DM))
        if is_symbolic:
            norm, dot = ca.norm_2, ca.dot
            if_else = ca.if_else
        else:
            norm, dot = np.linalg.norm, np.dot
            if_else = np.where
            state = np.array(state); u_control_vec = np.array(u_control_vec)

        # 1. SETUP & UNSCALING
        if scaling:
            r_phys = state[0:3] * scaling.length
            v_phys = state[3:6] * scaling.speed
            t_phys = time       * scaling.time
        else:
            r_phys, v_phys, t_phys = state[0:3], state[3:6], time

        # 2. ENVIRONMENT LOOKUP
        if is_symbolic:
            env_state = self.env.get_state_opti(r_phys, t_phys)
        else:
            env_state = self.env.get_state_sim(r_phys, t_phys)

        # 3. AERODYNAMICS
        v_rel = v_phys - env_state['wind_velocity']
        v_rel_mag = norm(v_rel)
        q = 0.5 * env_state['density'] * v_rel_mag**2

        # 4. ANGLE OF ATTACK
        thrust_mag = norm(u_control_vec)
        safe_denom = if_else(thrust_mag > 1e-9, thrust_mag, 1.0)
        u_thrust = u_control_vec / safe_denom
        
        # Velocity Direction (Singularity Check)
        u_vel = if_else(v_rel_mag > 0.1, v_rel / (v_rel_mag + 1e-9), u_thrust)

        if "coast" in stage_mode:
            cos_alpha = 1.0
        else:
            cos_alpha = dot(u_thrust, u_vel)

        return q, cos_alpha