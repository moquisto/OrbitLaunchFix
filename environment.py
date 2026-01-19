# environment.py

import numpy as np
import ussa1976
from dataclasses import dataclass
from config import EnvConfig
from typing import TypeAlias

# Type alias for 3D vectors for clarity
Vector3: TypeAlias = np.ndarray

@dataclass
class EnvironmentalState:
    """
    Unified snapshot of the environment at a specific location.
    """
    gravity_vector: Vector3     # [m/s^2] (Central + J2)
    wind_velocity: Vector3      # [m/s] (Inertial velocity of air mass)
    density: float              # [kg/m^3]
    pressure: float             # [Pa]
    speed_of_sound: float       # [m/s]
    temperature: float          # [K]

class Environment:
    def __init__(self, config: EnvConfig):
        self.config = config
        
        # --- 1. GENERATE LOOKUP TABLES (0 to 1000 km) ---
        # 10m resolution = ~100k points.
        # This allows simple Linear Interpolation to be highly accurate (<0.01% error).
        # We trade a little RAM (~2MB) for massive CPU speedups.
        step_size = self.config.atmosphere_step  # meters
        max_altitude = self.config.atmosphere_max_alt  # meters
        
        self.grid_altitudes = np.arange(0, max_altitude + step_size, step_size)
        
        # Batch compute US76 (returns a xarray.Dataset)
        data = ussa1976.compute(z=self.grid_altitudes)
        
        # Store arrays as raw Numpy arrays for O(1) interpolation performance
        self.table_rho = data["rho"].values      # Extract numpy array
        self.table_temp = data["t"].values       # Extract numpy array
        self.table_pressure = data["p"].values   # Extract numpy array

    def get_state(self, t: float, r_eci: Vector3) -> EnvironmentalState:
        """
        Calculates environmental conditions.
        Uses WGS84 Geodetic Altitude for accurate drag near poles/equator.
        """
        
        # --- A. GEODETIC ALTITUDE (Fixes "Squashed Earth" Bug) ---
        r_norm = np.linalg.norm(r_eci)
        
        # 1. Estimate Latitude (Sin(Lat) approx z/r)
        # This is sufficient for altitude lookups.
        sin_lat = r_eci[2] / r_norm
        
        # 2. Calculate Local Earth Radius (WGS84 Ellipsoid)
        # Earth is ~21km flatter at the poles. 
        # R_surface ≈ R_eq * (1 - flattening * sin^2(lat))
        local_radius = self.config.earth_radius_equator * (1.0 - self.config.earth_flattening * sin_lat**2)
        
        # 3. True Altitude
        altitude = r_norm - local_radius

        # --- B. VACUUM GATE (Safety Check) ---
        if altitude >= self.config.atmosphere_max_alt:
            # Space: Return vacuum conditions immediately.
            return EnvironmentalState(
                gravity_vector=self._calculate_gravity(r_eci, r_norm),
                wind_velocity=np.array([0.0, 0.0, 0.0]), # No air = No wind
                density=0.0,
                pressure=0.0,
                speed_of_sound=0.0, # Placeholder, not meaningful in vacuum
                temperature=self.table_temp[-1] # Keep exospheric temp
            )

        # --- C. ATMOSPHERE LOOKUP ---
        # Clamp negative altitude to 0 to prevent crashes if rocket is on launchpad/underground
        safe_alt = max(0.0, altitude)
        
        # Fast Linear Interpolation (Valid due to 10m step size)
        # Note: We interpolate Pressure directly as agreed.
        rho = np.interp(safe_alt, self.grid_altitudes, self.table_rho)
        temp = np.interp(safe_alt, self.grid_altitudes, self.table_temp)
        pressure = np.interp(safe_alt, self.grid_altitudes, self.table_pressure)
        
        # Calculate Speed of Sound (Required for Mach Number)
        # a = sqrt(gamma * R * T)
        speed_of_sound = np.sqrt(self.config.air_gamma * self.config.air_gas_constant * temp)

        # --- D. WIND (Co-Rotation) ---
        # Atmosphere rotates with Earth.
        # v_wind = omega_earth x r_eci
        v_wind = np.cross(self.config.earth_omega_vector, r_eci)

        # --- E. GRAVITY ---
        g_vec = self._calculate_gravity(r_eci, r_norm)

        # --- F. RETURN ---
        return EnvironmentalState(
            gravity_vector=g_vec,
            wind_velocity=v_wind,
            density=rho,
            pressure=pressure,
            speed_of_sound=speed_of_sound,
            temperature=temp
        )

    def _calculate_gravity(self, r_eci: Vector3, r_norm: float) -> Vector3:
        """Helper to calculate Gravity Vector (Central + J2)"""
        
        # 1. Central Gravity (Monopole)
        # g = -mu / r^3 * r_vec
        g_central = -(self.config.earth_mu / r_norm**3) * r_eci
        
        # 2. J2 Perturbation (Oblateness)
        if self.config.use_j2_perturbation:
            x, y, z = r_eci
            
            # FIXED: Factor is NEGATIVE for correct J2 force direction
            # Using formula: a = -1.5 * J2 * mu * R^2 / r^5 * ...
            factor = -1.5 * self.config.j2_constant * self.config.earth_mu * (self.config.earth_radius_equator**2) / (r_norm**5)
            z_sq_ratio = (z / r_norm)**2
            
            gx = factor * x * (1 - 5 * z_sq_ratio)
            gy = factor * y * (1 - 5 * z_sq_ratio)
            gz = factor * z * (3 - 5 * z_sq_ratio)
            
            return g_central + np.array([gx, gy, gz])
            
        return g_central




def visualize_environment():
    """
    Generates plots to visually verify the physics environment.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for visualization.")
        return

    print("Initializing Environment...")
    config = EnvConfig()
    env = Environment(config)

    print("\n--- Common Sense Checks ---")
    print("1. Atmosphere: Density should drop exponentially (straight line on log plot).")
    print("2. Temperature: Should drop, stay constant (Tropopause), then rise (Stratosphere).")
    print("3. Gravity: At CONSTANT radius, gravity should be stronger at Equator (due to mass bulge) and weaker at Poles.")
    print("4. Wind: Should increase linearly with altitude (Earth rotation).")
    print("\nGenerating Graphs...")

    # ==========================================
    # 1. ATMOSPHERIC PROFILE
    # ==========================================
    altitudes = np.arange(0, config.atmosphere_max_alt * 1.05, 5.0)
    densities = []
    pressures = []
    temps = []
    speeds_of_sound = []

    R_eq = config.earth_radius_equator

    for h in altitudes:
        # Sample at Equator
        r_eci = np.array([R_eq + h, 0, 0])
        state = env.get_state(0, r_eci)
        
        densities.append(state.density)
        pressures.append(state.pressure)
        temps.append(state.temperature)
        speeds_of_sound.append(state.speed_of_sound)

    fig1, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig1.suptitle('Check 1: US Standard Atmosphere 1976', fontsize=16)

    # Density (Log Scale)
    axs[0, 0].plot(altitudes / 1000, densities, 'b')
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_title('Density (Log Scale)')
    axs[0, 0].set_ylabel('kg/m^3')
    axs[0, 0].grid(True, which="both", ls="-", alpha=0.5)

    # Pressure (Log Scale)
    axs[0, 1].plot(altitudes / 1000, pressures, 'g')
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_title('Pressure (Log Scale)')
    axs[0, 1].set_ylabel('Pa')
    axs[0, 1].grid(True, which="both", ls="-", alpha=0.5)

    # Temperature
    axs[1, 0].plot(altitudes / 1000, temps, 'r')
    axs[1, 0].set_title('Temperature')
    axs[1, 0].set_ylabel('Kelvin')
    axs[1, 0].grid(True)

    # Speed of Sound
    axs[1, 1].plot(altitudes / 1000, speeds_of_sound, 'k')
    axs[1, 1].set_title('Speed of Sound')
    axs[1, 1].set_xlabel('Altitude (km)')
    axs[1, 1].set_ylabel('m/s')
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # ==========================================
    # 2. GRAVITY J2 PERTURBATION (Latitude Sweep)
    # ==========================================
    # We sweep latitude at a CONSTANT radius to isolate J2 shape from Earth flattening.
    lats = np.linspace(-90, 90, 360)
    g_mags = []
    
    for lat in lats:
        rad = np.radians(lat)
        # Position on a sphere of radius R_eq
        r_eci = np.array([R_eq * np.cos(rad), 0, R_eq * np.sin(rad)])
        state = env.get_state(0, r_eci)
        g_mags.append(np.linalg.norm(state.gravity_vector))

    fig2 = plt.figure(figsize=(10, 5))
    plt.plot(lats, g_mags, color='purple', linewidth=2)
    plt.title('Check 2: Gravity Magnitude vs Latitude (at Constant Radius)')
    plt.xlabel('Latitude (degrees)')
    plt.ylabel('Gravity (m/s^2)')
    plt.grid(True)
    plt.text(0, min(g_mags), "Equator (Stronger)", ha='center', va='bottom', fontweight='bold')
    plt.text(-90, max(g_mags), "Pole (Weaker)", ha='left', va='top', fontweight='bold')

    # ==========================================
    # 3. WIND SPEED (Earth Rotation)
    # ==========================================
    # Wind speed should increase as we go higher (v = omega * r)
    winds = [np.linalg.norm(env.get_state(0, np.array([R_eq + h, 0, 0])).wind_velocity) for h in altitudes]

    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(altitudes/1000, winds, color='orange', linewidth=2)
    plt.title('Check 3: Wind Speed vs Altitude (Co-rotation)')
    plt.xlabel('Altitude (km)')
    plt.ylabel('Wind Speed (m/s)')
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    visualize_environment()
