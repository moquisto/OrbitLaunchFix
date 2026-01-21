# config.py

import numpy as np
from dataclasses import dataclass, field


##################### environment.py mainly ##################### 

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

    # --- Launch Configuration ---
    # Launch site latitude in degrees (Default: Cape Canaveral)
    launch_latitude: float = 28.5721
    
    # Launch site longitude in degrees (Default: Cape Canaveral)
    launch_longitude: float = -80.6480
    
    # Initial altitude above sea level in meters
    launch_altitude: float = 0.0

    # --- PHYSICAL CONSTANTS ---
    # Standard Gravity (g0)
    # CRITICAL NOTE: This is a dimensional constant (9.80665 m/s^2) used strictly 
    # for converting Engine ISP (seconds) into Mass Flow Rate (kg/s).
    # It does NOT represent the local gravity acting on the vehicle. 
    # Do not change this value based on altitude.
    g0: float = 9.80665

# You can create a default config instance for easy import elsewhere
DefaultEnvConfig = EnvConfig()






##################### vehicle.py mainly ##################### 
@dataclass
class SequenceConfig:
    """
    Timing and triggers for the launch sequence.
    """
    main_engine_ramp_time: float   # Seconds to reach full thrust (Booster)
    upper_engine_ramp_time: float  # Seconds to reach full thrust (Ship)
    separation_delay: float        # Seconds between Booster Cutoff and Ship Ignition
    meco_cutoff_mach: float        # Mach number target for Booster shutdown
    min_throttle: float            # Minimum engine power (0.40 = 40%). Engines cannot throttle lower than this.

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

    @property
    def launch_mass(self) -> float:
        return (self.stage_1.dry_mass + self.stage_1.propellant_mass +
                self.stage_2.dry_mass + self.stage_2.propellant_mass)


# --- SpaceX Starship Block 2 Data ---
# PRIMARY SOURCE: Flight Telemetry Analysis (2025 Campaign: IFT-7 through IFT-10).
# NOTE: These values differ from early "User Guide" marketing. Flight data confirms 
# the hardware is heavier and engines are tuned differently than theoretical targets.

StarshipBlock2 = TwoStageRocketConfig(
    name="SpaceX Starship Block 2 (Flight Proven)",
    
    payload_mass=0.0,

    sequence=SequenceConfig(
        main_engine_ramp_time=3.0,     
        upper_engine_ramp_time=2.0,    
        separation_delay=0.0,          # Hot Staging
        meco_cutoff_mach=3.8,           # Staging Velocity Target
        min_throttle=0.40              # Minimum Engine Power
    ),
    
    # --- STAGE 1: SUPER HEAVY BOOSTER ---
    # MASS: Contains ONLY the Booster hardware.
    # AERO: Contains the "Full Stack" drag profile (since Booster pushes Ship).
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
            # DRAG: Represents the FULL STACK (Cylinder + Nose Cone + Flaps)
            # This is used because during Stage 1 burn, the air sees the whole rocket.
            mach_cd_table=np.array([
                [0.0, 0.30], [0.8, 0.35], [1.05, 0.60], 
                [1.5, 0.45], [3.0, 0.30], [5.0, 0.25], [25.0, 0.0]
            ])
        )
    ),

    # --- STAGE 2: STARSHIP (UPPER STAGE) ---
    # MASS: Contains ONLY the Ship hardware.
    # AERO: Contains the "Ship Solo" drag profile.
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
            # DRAG: Represents the SHIP ALONE (Blunt rear end)
            # This is used after separation.
            mach_cd_table=np.array([
                [0.0, 0.40], [0.8, 0.50], [1.05, 0.70], 
                [1.5, 0.55], [5.0, 0.35], [25.0, 0.0]
            ])
        )
    )
)

