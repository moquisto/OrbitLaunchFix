# config.py
# Purpose: Single source of truth for all physical constants.
# No logic here, just data structures.

class ScalingConfig:
    # Scaling factors to keep values near 1.0 for the optimizer (IPOPT)
    length: float = 6378137.0          # Earth Radius
    speed: float = 7910.0              # Orbital Velocity
    time: float = length / speed       # ~806 seconds
    mass: float = 5000000.0            # Approx Launch Mass (kg)
    force: float = mass * length / (time**2) # Derived Force Scale
    pass

class EnvConfig:
    # Earth Constants (Radius, Mu, J2, Flattening)
    # Atmosphere Constants (Gamma, Gas Constant, Lookup Table Paths)
    # Launch Site Parameters (Lat, Lon)
    pass

class RocketConfig:
    # Mass properties (Dry mass, Wet mass for Booster/Ship)
    # Engine properties (Thrust, ISP, Throttle limits)
    # Aerodynamic properties (Reference area, Drag Coefficient tables vs Mach)
    # Crossflow Drag Factor (Penalty for flying sideways/high Angle of Attack)
    cd_crossflow_factor: float = 30.0 
    # Staging parameters (Separation time)
    pass

# Instantiate default configs
# STARSHIP_CONFIG = ...
# EARTH_CONFIG = ...