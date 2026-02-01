import casadi as ca
import numpy as np
# vehicle.py
# Purpose: Defines the Equations of Motion (Dynamics).

class Vehicle:
    def __init__(self, rocket_config, environment):
        print(f"[Vehicle] Initializing vehicle dynamics model: {rocket_config.name}")
        self.config = rocket_config
        self.env = environment
        
        # 1. SETUP CASADI INTERPOLANTS (B-Splines)
        # Extract data from config (Mach vs Cd tables)
        data_s1 = self.config.stage_1.aero.mach_cd_table
        data_s2 = self.config.stage_2.aero.mach_cd_table
        # Create lookup functions. 'linear' is robust for optimization.
        self.cd_interp_stage_1 = ca.interpolant('cd1', 'linear', [data_s1[:, 0]], data_s1[:, 1])
        self.cd_interp_stage_2 = ca.interpolant('cd2', 'linear', [data_s2[:, 0]], data_s2[:, 1])
        
        # 2. COMPILE NUMERIC FUNCTIONS (Ensures 1:1 Parity with Optimizer)
        self._compile_numeric_dynamics()

    def _compile_numeric_dynamics(self):
        """Compiles CasADi graphs into callable functions for the simulation."""
        x = ca.MX.sym('x', 7)
        u_th = ca.MX.sym('u_th', 1)
        u_dir = ca.MX.sym('u_dir', 3)
        t = ca.MX.sym('t', 1)
        
        # Compile a function for each distinct flight phase
        # Scaling is None because simulation runs in physical units
        dyn_boost = self._get_dynamics_core(x, u_th, u_dir, t, "boost", None)
        self.sim_dyn_boost = ca.Function('sim_dyn_boost', [x, u_th, u_dir, t], [dyn_boost])
        
        dyn_coast_1 = self._get_dynamics_core(x, u_th, u_dir, t, "coast", None)
        self.sim_dyn_coast_1 = ca.Function('sim_dyn_coast_1', [x, u_th, u_dir, t], [dyn_coast_1])
        
        dyn_coast_2 = self._get_dynamics_core(x, u_th, u_dir, t, "coast_2", None)
        self.sim_dyn_coast_2 = ca.Function('sim_dyn_coast_2', [x, u_th, u_dir, t], [dyn_coast_2])
        
        dyn_ship = self._get_dynamics_core(x, u_th, u_dir, t, "ship", None)
        self.sim_dyn_ship = ca.Function('sim_dyn_ship', [x, u_th, u_dir, t], [dyn_ship])

    def get_dynamics(self, state, throttle, thrust_direction, time, stage_mode="boost", scaling=None):
        # Purpose: Public interface. Dispatches to Symbolic Core or Compiled Numeric Function.
        is_symbolic = isinstance(state, (ca.MX, ca.SX, ca.DM))
        
        if is_symbolic:
            # Build the Symbolic Graph (for Optimizer)
            return self._get_dynamics_core(state, throttle, thrust_direction, time, stage_mode, scaling)
        else:
            # Call Compiled Function (for Simulation) - Guarantees Bit-Exactness
            if "boost" in stage_mode:
                res = self.sim_dyn_boost(state, throttle, thrust_direction, time)
            elif "coast_2" in stage_mode:
                res = self.sim_dyn_coast_2(state, throttle, thrust_direction, time)
            elif "coast" in stage_mode:
                res = self.sim_dyn_coast_1(state, throttle, thrust_direction, time)
            else:
                res = self.sim_dyn_ship(state, throttle, thrust_direction, time)
            
            return np.array(res).flatten()

    def _get_dynamics_core(self, state, throttle, thrust_direction, time, stage_mode, scaling):
        # Internal Physics Logic - Defined ONCE using CasADi operations.
        
        # 1. SETUP & UNSCALING
        if scaling:
            r_phys = state[0:3] * scaling.length
            v_phys = state[3:6] * scaling.speed
            m_phys = state[6]   * scaling.mass
            t_phys = time       * scaling.time
        else:
            r_phys, v_phys, m_phys, t_phys = state[0:3], state[3:6], state[6], time

        # 2. ENVIRONMENT LOOKUP
        # Always use the symbolic lookup (which compiles to numeric in _compile_numeric_dynamics)
        env_state = self.env.get_state_opti(r_phys, t_phys)

        # 3. AERODYNAMICS
        # Relative Velocity
        v_rel = v_phys - env_state['wind_velocity']
        
        v_rel_sq = ca.dot(v_rel, v_rel)
        v_rel_mag = ca.sqrt(ca.fmax(v_rel_sq, 1.0e-12))

        # Mach Number (Singularity Protection)
        sos_safe = ca.fmax(env_state['speed_of_sound'], 1.0)
        mach = v_rel_mag / sos_safe

        # Dynamic Pressure
        q = 0.5 * env_state['density'] * v_rel_mag**2

        # Control Vector
        thrust_sq = ca.dot(thrust_direction, thrust_direction)
        thrust_mag = ca.sqrt(ca.fmax(thrust_sq, 1.0e-12))
        u_control = thrust_direction / thrust_mag

        # Angle of Attack (AoA) Logic
        u_vel = v_rel / v_rel_mag

        # Thrust Orientation
        # Coast Phase: Assume Prograde (u_vel) to minimize drag.
        if "coast" in stage_mode:
            u_thrust = u_vel
        else:
            u_thrust = u_control

        # Crossflow Drag (sin^2(alpha))
        cross_prod = ca.cross(u_thrust, u_vel)
        sin_alpha_sq = ca.dot(cross_prod, cross_prod)

        # Drag Force Calculation
        # Explicitly handle stage modes to allow coasting on either stage.
        if stage_mode in ["boost", "coast", "coast_1"]:
            stage_cfg = self.config.stage_1
            cd_curve = self.cd_interp_stage_1
        else: # "ship", "coast_2"
            stage_cfg = self.config.stage_2
            cd_curve = self.cd_interp_stage_2

        # Lookup Cd (Clamp Mach to table limit to avoid extrapolation errors)
        max_mach = stage_cfg.aero.mach_cd_table[-1, 0]
        mach_clamped = ca.fmin(mach, max_mach)
        cd_base = cd_curve(mach_clamped)

        cd_total = cd_base + stage_cfg.aero.cd_crossflow_factor * sin_alpha_sq
        f_drag = -q * stage_cfg.aero.reference_area * cd_total * u_vel

        # 4. PROPULSION
        stage = stage_cfg
        eff_throttle = 0.0 if "coast" in stage_mode else throttle

        # Mass Flow (Choked Flow: Depends on Vacuum Thrust)
        g0 = self.env.config.g0
        m_dot = eff_throttle * stage.thrust_vac / (stage.isp_vac * g0)

        # Thrust Force (Pressure Compensated)
        isp_eff = stage.isp_vac + (env_state['pressure'] / stage.p_sl) * (stage.isp_sl - stage.isp_vac)
        f_thrust = (m_dot * isp_eff * g0) * u_thrust

        # 5. EQUATIONS OF MOTION
        f_total = f_thrust + f_drag + (env_state['gravity'] * m_phys)
        acc_phys = f_total / m_phys
        dm_dt_phys = -m_dot

        # 6. RESCALING & RETURN
        if scaling:
            dr_dtau = v_phys     * (scaling.time / scaling.length)
            dv_dtau = acc_phys   * (scaling.time / scaling.speed)
            dm_dtau = dm_dt_phys * (scaling.time / scaling.mass)
            return ca.vertcat(dr_dtau, dv_dtau, dm_dtau)
        else:
            return ca.vertcat(v_phys, acc_phys, dm_dt_phys)

    def get_aero_properties(self, state, u_control_vec, time, stage_mode="boost", scaling=None):
        # 0. DISPATCH: SYMBOLIC VS NUMERIC
        is_symbolic = isinstance(state, (ca.MX, ca.SX, ca.DM))
        if is_symbolic:
            norm, dot = ca.norm_2, ca.dot
            if_else = ca.if_else
            fmax = ca.fmax
        else:
            norm, dot = np.linalg.norm, np.dot
            if_else = np.where
            fmax = np.maximum
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
        
        # MATCH SYMBOLIC EXACTLY: Use sqrt(dot) and max clamping
        if is_symbolic:
            v_rel_sq = dot(v_rel, v_rel)
            v_rel_mag = ca.sqrt(fmax(v_rel_sq, 1.0e-12))
        else:
            v_rel_sq = dot(v_rel, v_rel)
            v_rel_mag = np.sqrt(fmax(v_rel_sq, 1.0e-12))
            
        # Use max to avoid zero in q calculation if needed, though less critical here
        q = 0.5 * env_state['density'] * (v_rel_mag**2)

        # 4. ANGLE OF ATTACK
        # MATCH SYMBOLIC EXACTLY
        if is_symbolic:
            thrust_sq = dot(u_control_vec, u_control_vec)
            thrust_mag = ca.sqrt(fmax(thrust_sq, 1.0e-12))
            u_thrust = u_control_vec / thrust_mag
        else:
            thrust_sq = dot(u_control_vec, u_control_vec)
            thrust_mag = np.sqrt(fmax(thrust_sq, 1.0e-12))
            u_thrust = u_control_vec / thrust_mag
        
        # Velocity Direction (Singularity Check)
        if is_symbolic:
            u_vel = v_rel / v_rel_mag
        else:
            u_vel = v_rel / v_rel_mag

        if "coast" in stage_mode:
            cos_alpha = 1.0
        else:
            cos_alpha = dot(u_thrust, u_vel)

        return q, cos_alpha

    def diagnose_forces(self, state, throttle, thrust_direction, time, stage_mode="boost"):
        """
        Prints a detailed breakdown of forces acting on the vehicle.
        Useful for debugging T/W issues and drag losses.
        """
        # Ensure numeric inputs
        state = np.array(state)
        thrust_direction = np.array(thrust_direction)
        
        # Get Derived Quantities
        r = state[0:3]
        v = state[3:6]
        m = state[6]
        
        # Get Dynamics (returns [v, a, m_dot])
        dyn = self.get_dynamics(state, throttle, thrust_direction, time, stage_mode, scaling=None)
        acc_total = dyn[3:6]
        
        # Re-calculate components for reporting
        env_state = self.env.get_state_sim(r, time)
        g_vec = env_state['gravity']
        
        # F = ma
        f_total = acc_total * m
        f_gravity = g_vec * m
        f_aero_thrust = f_total - f_gravity # Combined Aero + Thrust
        
        print(f"  [Force Diag] Mass: {m/1000:.1f}t | Alt: {(np.linalg.norm(r)-6378137)/1000:.1f}km")
        print(f"  [Force Diag] G-Force: {np.linalg.norm(f_gravity)/1000:.1f} kN | Net Force: {np.linalg.norm(f_total)/1000:.1f} kN")
        print(f"  [Force Diag] Accel:   {np.linalg.norm(acc_total):.2f} m/s^2 ({(np.linalg.norm(acc_total)/9.81):.2f} g)")