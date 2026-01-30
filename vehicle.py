import casadi as ca
import numpy as np
# vehicle.py
# Purpose: Defines the Equations of Motion (Dynamics).

class Vehicle:
    def __init__(self, rocket_config, environment):
        self.config = rocket_config
        self.env = environment
        # Initialize CasADi interpolants for Cd vs Mach here (once)
        # Also setup Scipy interpolants (or wrap CasADi) for numeric simulation

    def get_dynamics(self, state, control, time, stage_mode="boost", scaling=None):
        # Determine if we are in Symbolic (CasADi) or Numeric (Numpy) mode
        is_casadi = isinstance(state, (ca.SX, ca.MX, ca.DM))
        
        if is_casadi:
            lib = ca
        else:
            lib = np

        # 1. Unpack and UNSCALE State/Control (Critical for Physics)
        #    If scaling is provided:
        #       r_phys = state[0:3] * scaling.length
        #       v_phys = state[3:6] * scaling.speed
        #       m_phys = state[6]   * scaling.mass
        #       control_phys = control # Control is now Throttle Vector [Tx, Ty, Tz] (0.0 to 1.0)
        #    Else:
        #       Use raw inputs
        
        # 3. Get Environmental State (Dispatch based on input type)
        #    if isinstance(r_phys, (ca.SX, ca.MX)):
        #        env_state = self.env.get_state_opti(r_phys)
        #    else:
        #        env_state = self.env.get_state_sim(r_phys)
        
        # 4. Select Stage Properties based on stage_mode
        
        # 5. Calculate Relative Velocity (V_rel = V_inertial - V_wind)
        #    if is_casadi:
        #        v_rel_mag = ca.norm_2(v_rel) + 1e-6
        #    else:
        #        v_rel_mag = np.linalg.norm(v_rel) + 1e-6
        
        #    # FIX: Add epsilon to speed_of_sound to prevent vacuum singularity
        #    Mach = v_rel_mag / (env_state['speed_of_sound'] + 1e-8)

        # 6. Calculate Angle of Attack (alpha)
        #    - Use Cross Product to avoid acos() singularity at alpha=0
        #    if is_casadi:
        #        ctrl_mag = ca.norm_2(control_phys) + 1e-6
        #        Thrust_dir = control_phys / ctrl_mag
        #        cross_prod = ca.cross(Thrust_dir, V_rel / v_rel_mag)
        #        sin_alpha_sq = ca.dot(cross_prod, cross_prod)
        #    else:
        #        ctrl_mag = np.linalg.norm(control_phys) + 1e-6
        #        Thrust_dir = control_phys / ctrl_mag
        #        cross_prod = np.cross(Thrust_dir, V_rel / v_rel_mag)
        #        sin_alpha_sq = np.dot(cross_prod, cross_prod)

        # 7. Calculate Aerodynamic Forces 
        #    Cd_total = Cd_axial(Mach) + config.cd_crossflow_factor * sin_alpha_sq
        #    Drag_Mag = 0.5 * env_state['density'] * v_rel_mag**2 * Cd_total * Area
        #    Drag_Vector = -Drag_Mag * (V_rel / v_rel_mag)
        
        # 8. Calculate Thrust Forces (Throttle Control Logic)
        #    throttle_level = lib.norm(control_phys) # Should be constrained 0.4 to 1.0
        #
        #    # 1. Calculate Mass Flow (Constant for a given throttle setting)
        #    # m_dot_max = stage.thrust_vac / (stage.isp_vac * g0)
        #    # M_dot = throttle_level * m_dot_max
        #
        #    # 2. Calculate Thrust Force (Pressure Adjusted)
        #    # ISP_current = ISP_vac + (env_state['pressure'] / P_sl) * (ISP_sl - ISP_vac)
        #    # Thrust_Mag = M_dot * ISP_current * g0
        #    # Thrust_Vector = Thrust_Mag * Thrust_dir
        
        # 9. Apply Newton's Laws (F_total = ma)
        
        # 10. RESCALE Derivatives (ONLY if scaling is provided)
        #     if scaling:
        #         res = [vel_phys / scaling.speed, acc_phys / ..., ...]
        #     else:
        #         res = [vel_phys, acc_phys, mdot_phys]
        
        #     if is_casadi:
        #         return ca.vertcat(*res)
        #     else:
        #         # Ensure scalar mass flow is wrapped in an array for concatenation
        #         res[-1] = np.array([res[-1]])
        #         return np.concatenate(res)