# config.py
# Purpose: Single source of truth for all physical constants.
# No logic here, just data structures.

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ScalingConfig:
    """
    Defines scaling factors to normalize the optimization problem.
    IPOPT and other solvers perform best when variables are O(1).
    """
    # Canonical Units (Earth Surface/Orbit)
    length: float = 6378137.0          # [m] Earth Radius
    speed: float = 7910.0              # [m/s] Orbital Velocity
    mass: float = 1000000.0            # [kg] Reference Mass (1000t)
    
    # Derived Units
    time: float = field(init=False)    # [s]
    force: float = field(init=False)   # [N]

    def __post_init__(self):
        # Time to travel one radius at orbital speed
        self.time = self.length / self.speed       # ~806 seconds
        # Force required to accelerate reference mass at reference acceleration
        # F = ma = m * (l/t^2)
        self.force = self.mass * self.length / (self.time**2) 

@dataclass
class EnvConfig:
    """
    Configuration for the simulation environment (Planet & Atmosphere).
    """
    # --- Earth Constants (WGS84) ---
    earth_radius_equator: float = 6378137.0       # [m]
    earth_flattening: float = 1.0 / 298.257       # [-]
    earth_mu: float = 3.986004418e14              # [m^3/s^2] Standard Gravitational Parameter
    
    # Earth Rotation (rad/s) - Vector points along Z-axis (North)
    earth_omega_vector: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 7.2921159e-5]))
    
    # Initial Rotation (GMST or alignment offset) [rad]
    # Ensures Launch Site Longitude is correctly placed in Inertial Space at t=0.
    initial_rotation: float = 0.0 

    # J2 Perturbation (Oblateness effect on Gravity)
    j2_constant: float = 0.0010826267
    use_j2_perturbation: bool = True
    
    # Wind Model
    use_wind_model: bool = False 
    
    # Monte Carlo Dispersion Factor (Default 1.0)
    density_multiplier: float = 1.0
    
    # --- Atmosphere Constants ---
    air_gamma: float = 1.4                        # [-] Heat capacity ratio
    air_gas_constant: float = 287.058             # [J/(kg·K)] Specific gas constant
    atmosphere_step: float = 10.0               # [m] Lookup table resolution (Standard for stability)
    atmosphere_max_alt: float = 1_000_000.0       # [m] Atmosphere cutoff altitude
    
    # --- Launch Site
    launch_latitude: float = 25.997               # [deg] Starbase, TX
    launch_longitude: float = -97.157             # [deg]
    launch_altitude: float = 5.0                  # [m]

    # --- Physical Constants ---
    # [m/s^2] Standard Gravity. Used ONLY for converting ISP (s) to Exhaust Velocity (m/s).
    # DO NOT use this for calculating weight or local gravity (which varies with altitude).
    g0: float = 9.80665                           

@dataclass
class SequenceConfig:
    """
    Timing and triggers for the launch sequence.
    """
    main_engine_ramp_time: float   # [s] Time to ramp up Booster engines
    upper_engine_ramp_time: float  # [s] Time to ramp up Ship engines
    separation_delay: float        # [s] Coast time between staging
    min_throttle: float            # [-] Minimum throttle capability (0.0 to 1.0)
    
    # Optimization Constraints
    min_stage_1_burn: float = 60.0 # [s] Minimum burn time for Stage 1
    min_stage_2_burn: float = 10.0 # [s] Minimum burn time for Stage 2
    
    # Guidance Heuristics (Open Loop Pitch)
    pitch_start_time: float = 10.0 # [s] Time to start pitch maneuver
    pitch_end_time: float = 25.0   # [s] Time to end pitch maneuver
    pitch_gain: float = 0.1        # [-] Magnitude of pitch nudge

@dataclass
class AerodynamicConfig:
    """
    Defines aerodynamic properties.
    """
    reference_area: float         # [m^2] Aerodynamic reference area
    mach_cd_table: np.ndarray     # [-] Drag Coefficient (Cd) vs Mach Number lookup table
    # [-] Coefficient for "Crossflow Drag" model: Cd_total = Cd_base + factor * sin^2(alpha).
    # Approximates the massive drag increase when flying sideways (e.g. during flip).
    cd_crossflow_factor: float = 30.0 

@dataclass
class StageConfig:
    """
    Physical parameters for a single rocket stage.
    """
    dry_mass: float         # [kg] Mass of structure without propellant
    propellant_mass: float  # [kg] Mass of usable propellant
    
    # Propulsion Performance
    # NOTE: thrust_sl is a DERIVED reference value. The physics engine calculates 
    # actual thrust using thrust_vac and the ISP curve (Choked Flow assumption).
    # We retain thrust_vac as the "anchor" value to define the engine's absolute 
    # power magnitude. It is used to calculate the constant Mass Flow Rate (m_dot)
    # at 100% throttle, since ISP only defines efficiency, not scale.
    thrust_vac: float       # [N] Vacuum Thrust (Max)
    
    isp_sl: float           # [s] Specific Impulse at Sea Level
    isp_vac: float          # [s] Specific Impulse in Vacuum
    p_sl: float             # [Pa] Reference pressure for ISP_sl
    
    @property
    def thrust_sl(self) -> float:
        """Derived Sea Level Thrust based on Choked Flow physics (Constant Mass Flow)."""
        return self.thrust_vac * (self.isp_sl / self.isp_vac)

    aero: AerodynamicConfig # Stage-specific aerodynamics

@dataclass
class TwoStageRocketConfig:
    """
    Container for the entire two-stage vehicle configuration.
    """
    name: str
    stage_1: StageConfig  # Booster (First Stage)
    stage_2: StageConfig  # Ship (Second Stage)
    sequence: SequenceConfig
    payload_mass: float   # [kg] Payload mass
    target_altitude: float = 420000.0  # [m] Target Orbit Altitude
    target_inclination: Optional[float] = None   # [deg] Target Inclination. If None, defaults to Launch Latitude (Min Energy).
    num_nodes: int = 140  # Number of discretization nodes per phase
    max_iter: int = 2000  # Maximum number of iterations for the optimizer
    
    # Structural Limits
    max_q_limit: float = 40000.0   # [Pa] Maximum Dynamic Pressure
    max_g_load: float = 4.0        # [g] Maximum Acceleration
    
    @property
    def launch_mass(self) -> float:
        """Total mass at liftoff."""
        return (self.stage_1.dry_mass + self.stage_1.propellant_mass +
                self.stage_2.dry_mass + self.stage_2.propellant_mass +
                self.payload_mass)


@dataclass
class ReliabilityAnalysisToggles:
    """
    Enables/disables each analysis block in relabilityanalysis.py.
    """
    stiffness_convergence: bool = True
    smooth_integrator_benchmark: bool = True
    conservative_invariants: bool = True
    monte_carlo_convergence: bool = True
    monte_carlo_precision_target: bool = True
    global_sensitivity: bool = True
    constraint_reliability: bool = True
    distribution_robustness: bool = True
    grid_independence: bool = True
    integrator_tolerance: bool = True
    event_time_convergence: bool = True
    corner_cases: bool = True
    finite_time_sensitivity: bool = True
    bifurcation: bool = True
    bifurcation_2d_map: bool = True
    theoretical_efficiency: bool = True
    drift: bool = True
    energy_balance: bool = True
    control_slew: bool = True
    aerodynamics: bool = True
    lagrange_multipliers: bool = True

# --- SpaceX Starship Block 2 Configuration ---
StarshipBlock2 = TwoStageRocketConfig(
    name="SpaceX Starship IFT-6 (Booster 13 / Ship 31)",
    
    payload_mass=0.0, # Payload to Orbit
    target_altitude=420000.0,
    target_inclination=None, # Auto-resolve to Min Energy (Latitude)
    
    # Structural Limits
    max_q_limit=35000.0, # 35 kPa
    max_g_load=4.0,      # 4.0 g

    sequence=SequenceConfig(
        main_engine_ramp_time=3.0,     
        upper_engine_ramp_time=2.0,    
        separation_delay=0.0,          # Hot Staging (0s delay)
        min_throttle=0.40,             # 40% Minimum Throttle
        min_stage_1_burn=60.0,
        min_stage_2_burn=10.0,
        pitch_start_time=10.0,
        pitch_end_time=25.0,
        pitch_gain=0.1
    ),
    
    stage_1=StageConfig(
        dry_mass=275_000.0,          
        propellant_mass=3_400_000.0, 
        
        # Propulsion (Raptor 2/3 Cluster)
        # IFT-6 Data: 74.4 MN Sea Level Thrust.
        # F_vac = F_sl * (ISP_vac / ISP_sl) = 74.4 * (347/327) = ~78.95 MN
        thrust_vac=78_950_000.0,
        isp_sl=327.0,
        isp_vac=347.0,
        p_sl=101325.0,
        
        aero=AerodynamicConfig(
            reference_area=63.62, # 9m Diameter
            # Cd vs Mach (Full Stack)
            mach_cd_table=np.array([
                [0.0, 0.30], [0.8, 0.35], [1.05, 0.60], 
                [1.5, 0.45], [3.0, 0.30], [5.0, 0.25], [25.0, 0.05]
            ]),
            cd_crossflow_factor=30.0
        )
    ),

    stage_2=StageConfig(
        dry_mass=100_000.0,          
        propellant_mass=1_200_000.0, 
        
        # Propulsion (Raptor Vacuum + Sea Level)
        # Note: Ship engines are optimized for vacuum.
        # isp_sl is low to penalize low-altitude operation.
        thrust_vac=14_700_000.0,    # 6 Engines (3 Vac + 3 SL)
        isp_sl=100.0,               # Flow separation penalty at SL
        isp_vac=380.0,
        p_sl=101325.0,
        
        aero=AerodynamicConfig(
            reference_area=63.62,
            # Cd vs Mach (Ship Only)
            mach_cd_table=np.array([
                [0.0, 0.40], [0.8, 0.50], [1.05, 0.70], 
                [1.5, 0.55], [5.0, 0.35], [25.0, 0.05]
            ]),
            cd_crossflow_factor=30.0
        )
    )
)

# Default Environment Instance
EARTH_CONFIG = EnvConfig()

# Reliability analysis switches.
# Set any field to False to skip that test in ReliabilitySuite.run_all().
RELIABILITY_ANALYSIS_TOGGLES = ReliabilityAnalysisToggles()
