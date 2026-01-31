import casadi as ca
import numpy as np
# vehicle.py
# Purpose: Defines the Equations of Motion (Dynamics).

class Vehicle:
    def __init__(self, rocket_config, environment):
        self.config = rocket_config
        self.env = environment

    def get_dynamics(self, state, throttle, thrust_direction, time, stage_mode="boost", scaling=None):
        """
        Calculates state derivatives.
        """
        
        # 1. Unscaling
        if scaling:
            r = state[0:3] * scaling.length
            v = state[3:6] * scaling.speed
            m = state[6] * scaling.mass
            t = time * scaling.time
        else:
            r, v, m, t = state[0:3], state[3:6], state[6], time

        # 2. Environment
        env = self.env.get_state_opti(r, t)
        
        # 3. Aerodynamics
        v_rel = v - env['wind_velocity']
        v_rel_mag = ca.norm_2(v_rel)
        
        mach = v_rel_mag / (env['speed_of_sound'] + 1e-8)
        q = 0.5 * env['density'] * v_rel_mag**2
        
        if stage_mode == "coast":
            u_vel = v_rel / (v_rel_mag + 1e-8)
            u_body = u_vel
            eff_throttle = 0.0
        else:
            eff_throttle = throttle[0]
            u_body = throttle[1:4] 
            u_vel = v_rel / (v_rel_mag + 1e-8)

        cos_alpha = ca.dot(u_body, u_vel)
        sin_alpha_sq = 1.0 - cos_alpha**2
        
        stage_cfg = self.config.stage_1 if stage_mode in ["boost", "coast"] else self.config.stage_2
        
        cd_base = 0.3 # Placeholder for interpolation
        cd_total = cd_base + stage_cfg.aero.cd_crossflow_factor * sin_alpha_sq
        f_drag_mag = q * stage_cfg.aero.reference_area * cd_total
        f_drag = -f_drag_mag * u_vel
        
        # 4. Propulsion
        if stage_mode == "coast":
            f_thrust = ca.vertcat(0, 0, 0)
            m_dot = 0.0
        else:
            m_dot_max = stage_cfg.thrust_vac / (stage_cfg.isp_vac * self.env.config.g0)
            m_dot = eff_throttle * m_dot_max
            
            isp_slope = (stage_cfg.isp_vac - stage_cfg.isp_sl) / stage_cfg.p_sl
            isp_current = stage_cfg.isp_vac - isp_slope * env['pressure']
            
            f_thrust_mag = m_dot * isp_current * self.env.config.g0
            f_thrust = f_thrust_mag * u_body

        # 5. Total Forces
        f_total = f_thrust + f_drag + (env['gravity'] * m)
        acc = f_total / m
        
        # 6. Derivatives (Rescaled)
        if scaling:
            dr = v * (scaling.time / scaling.length)
            dv = acc * (scaling.time / scaling.speed)
            dm = -m_dot * (scaling.time / scaling.mass)
            return ca.vertcat(dr, dv, dm)
        else:
            return ca.vertcat(v, acc, -m_dot)