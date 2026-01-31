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
    # Placeholder for solution extraction logic
    # In a real implementation, extract T1, U1, etc. from optimization_result
    
    # CRITICAL: Use 'previous' (Zero-Order Hold) for controls
    # throttle_func_1 = interp1d(t_grid_1, U1_val[0,:], kind='previous', fill_value="extrapolate")
    
    # Phase 1 Simulation
    # res1 = solve_ivp(..., t_span=[0, T1_val])
    
    # Phase 2 (Coast) - Hot Staging Logic
    # y_final_1 = res1.y[:, -1]
    # t_current = res1.t[-1]
    
    if config.sequence.separation_delay > 1e-4:
        # Simulate Coast
        # res2 = solve_ivp(..., t_span=[t_current, t_current + delay])
        # y_staging = res2.y[:, -1]
        pass
    else:
        # Skip Coast
        # y_staging = y_final_1
        pass
        
    # Staging Mass Drop
    # y_ship_init = y_staging.copy()
    # m_ship_wet = config.stage_2.dry_mass + config.stage_2.propellant_mass + config.payload_mass
    # y_ship_init[6] = m_ship_wet
    
    # Phase 3 Simulation
    # res3 = solve_ivp(..., y0=y_ship_init)
    
    return {}