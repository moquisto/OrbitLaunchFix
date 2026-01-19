# config.py

import numpy as np
from dataclasses import dataclass, field

@dataclass
class EnvConfig:
    """
    Configuration for the simulation environment.
    """
    # --- Earth Constants ---
    # WGS84 value for Earth's equatorial radius in meters
    earth_radius_equator: float = 6378137.0
    
    # WGS84 flattening factor (f)
    earth_flattening: float = 1.0 / 298.257
    
    # WGS84 value for Earth's gravitational parameter (mu) in m^3/s^2
    earth_mu: float = 3.986004418e14
    
    # Earth's angular velocity vector in rad/s (for ECI frame)
    earth_omega_vector: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 7.2921159e-5]))
    
    # J2 gravitational constant
    j2_constant: float = 0.0010826267
    
    # Flag to enable/disable J2 perturbation in gravity calculation
    use_j2_perturbation: bool = True
    
    # --- Atmospheric Properties ---
    # Ratio of specific heats for air
    air_gamma: float = 1.4
    
    # Specific gas constant for dry air in J/(kg*K)
    air_gas_constant: float = 287.058

    # --- Atmospheric Model ---
    # Altitude step size in meters for the lookup table.
    atmosphere_step: float = 5.0

    # Maximum altitude in meters for the lookup table.
    atmosphere_max_alt: float = 1_000_000.0

# You can create a default config instance for easy import elsewhere
DefaultEnvConfig = EnvConfig()