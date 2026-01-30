# config.py
# Purpose: Single source of truth for all physical constants.
# No logic here, just data structures.

import numpy as np
from dataclasses import dataclass, field

@dataclass
class ScalingConfig:
    # Scaling factors to keep values near 1.0 for the optimizer (IPOPT)
    length: float = 6378137.0          # Earth Radius
    speed: float = 7910.0              # Orbital Velocity
    mass: float = 5000000.0            # Approx Launch Mass (kg)
    
    time: float = field(init=False)
    force: float = field(init=False)

    def __post_init__(self):
        self.time = self.length / self.speed       # ~806 seconds
        self.force = self.mass * self.length / (self.time**2) # Derived Force Scale

@dataclass
class EnvConfig:
    """
    Configuration for the simulation environment.
    """
    # Earth Constants (Radius, Mu, J2, Flattening)
    earth_radius_equator: float = 6378137.0
    earth_flattening: float = 1.0 / 298.257
    earth_mu: float = 3.986004418e14
    earth_omega_vector: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 7.2921159e-5]))
    j2_constant: float = 0.0010826267
    use_j2_perturbation: bool = True
    
    # Atmosphere Constants (Gamma, Gas Constant, Lookup Table Paths)
    air_gamma: float = 1.4
    air_gas_constant: float = 287.058
    atmosphere_step: float = 5.0
    atmosphere_max_alt: float = 1_000_000.0
    
    # Launch Site Parameters (Lat, Lon)
    launch_latitude: float = 28.5721
    launch_longitude: float = -80.6480
    launch_altitude: float = 0.0

    # --- PHYSICAL CONSTANTS ---
    g0: float = 9.80665

@dataclass
class SequenceConfig:
    """
    Timing and triggers for the launch sequence.
    """
    main_engine_ramp_time: float   # Seconds to reach full thrust (Booster)
    upper_engine_ramp_time: float  # Seconds to reach full thrust (Ship)
    separation_delay: float        # Seconds between Booster Cutoff and Ship Ignition
    meco_cutoff_mach: float        # Mach number target for Booster shutdown
    min_throttle: float            # Minimum engine power (0.40 = 40%)

@dataclass
class AerodynamicConfig:
    """
    Defines aerodynamic properties including Mach-dependent drag.
    """
    reference_area: float  # m^2 (Cross-sectional area)
    mach_cd_table: np.ndarray # Lookup Table: [Mach, Cd]

@dataclass
class StageConfig:
    """
    Physical parameters for a single rocket stage.
    """
    dry_mass: float         # kg
    propellant_mass: float  # kg
    thrust_sl: float        # N
    thrust_vac: float       # N
    isp_sl: float           # s
    isp_vac: float          # s
    p_sl: float             # Pa (Standard Pressure for interpolation)
    aero: AerodynamicConfig # Stage-specific aerodynamics

@dataclass
class TwoStageRocketConfig:
    """
    Container for the entire two-stage vehicle.
    """
    name: str
    stage_1: StageConfig  # Booster
    stage_2: StageConfig  # Ship
    sequence: SequenceConfig
    payload_mass: float   # Mass of the cargo inside the fairing
    
    # Crossflow Drag Factor (Penalty for flying sideways/high Angle of Attack)
    # Added to match vehicle.py requirements, default from previous config.py
    cd_crossflow_factor: float = 30.0

    @property
    def launch_mass(self) -> float:
        return (self.stage_1.dry_mass + self.stage_1.propellant_mass +
                self.stage_2.dry_mass + self.stage_2.propellant_mass +
                self.payload_mass)

# --- SpaceX Starship Block 2 Data ---
StarshipBlock2 = TwoStageRocketConfig(
    name="SpaceX Starship Block 2 (Flight Proven)",
    
    payload_mass = 0.0,

    sequence=SequenceConfig(
        main_engine_ramp_time=3.0,     
        upper_engine_ramp_time=2.0,    
        separation_delay=0.0,          # Hot Staging
        meco_cutoff_mach=3.8,           # Staging Velocity Target
        min_throttle=0.40              # Minimum Engine Power
    ),
    
    stage_1=StageConfig(
        dry_mass=283_000.0,          # Booster Structure ONLY
        propellant_mass=3_400_000.0, # Booster Fuel ONLY
        thrust_sl=74_580_000.0,
        thrust_vac=83_490_000.0,
        isp_sl=327.0,
        isp_vac=347.0,
        p_sl=101325.0,
        
        aero=AerodynamicConfig(
            reference_area=63.62,
            mach_cd_table=np.array([
                [0.0, 0.30], [0.8, 0.35], [1.05, 0.60], 
                [1.5, 0.45], [3.0, 0.30], [5.0, 0.25], [25.0, 0.0]
            ])
        )
    ),

    stage_2=StageConfig(
        dry_mass=164_000.0,          # Ship Structure ONLY
        propellant_mass=1_500_000.0, # Ship Fuel ONLY
        thrust_sl=100.0,            
        thrust_vac=14_700_000.0,    
        isp_sl=100.0,               
        isp_vac=365.0,
        p_sl=101325.0,
        
        aero=AerodynamicConfig(
            reference_area=63.62,
            mach_cd_table=np.array([
                [0.0, 0.40], [0.8, 0.50], [1.05, 0.70], 
                [1.5, 0.55], [5.0, 0.35], [25.0, 0.0]
            ])
        )
    )
)

# Default Environment
EARTH_CONFIG = EnvConfig()