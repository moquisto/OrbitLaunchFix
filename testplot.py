import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import contextlib
import os
import sys

# Import project modules
from config import StarshipBlock2, EARTH_CONFIG
from vehicle import Vehicle
from environment import Environment
from main import solve_optimal_trajectory

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            yield

def run_heatmap_analysis():
    print("========================================================")
    print(" GLOBAL LAUNCH FUEL HEATMAP GENERATOR")
    print("========================================================")
    print("Initializing Physics Engine (Compiling Dynamics)...")
    
    # Initialize Environment and Vehicle ONCE.
    # We will modify EARTH_CONFIG in-place, which the Environment instance references.
    env = Environment(EARTH_CONFIG)
    veh = Vehicle(StarshipBlock2, env)
    
    # Define the Grid
    # We exploit rotational symmetry: Launch cost depends on Latitude, not Longitude.
    # We simulate a sweep of latitudes at a fixed longitude, then replicate results for the 3D plot.
    lat_values = np.linspace(-90, 90, 37)   # 37 Latitude steps (5 deg increments) including poles
    lon_values_plot = np.linspace(-180, 180, 37) # 37 Longitude steps (10 deg increments) for visualization
    
    results = []
    
    print(f"\nStarting Latitude Sweep: {len(lat_values)} Runs (Results replicated longitudinally)")
    print(f"{'Run':<5} | {'Lat':<8} | {'Fuel Used (kg)':<15} | {'Final Mass (kg)':<15} | {'V_rot (m/s)':<12}")
    print("-" * 80)
    
    for i, lat in enumerate(lat_values):
        # 1. Update Configuration
        EARTH_CONFIG.launch_latitude = float(lat)
        EARTH_CONFIG.launch_longitude = 0.0 # Fixed longitude for simulation
        
        # Reset Target Inclination to None.
        # This forces main.py to default to the launch latitude (Min Energy Orbit).
        StarshipBlock2.target_inclination = None
        
        fuel_used = np.nan
        m_final = np.nan
        
        # 2. Run Optimization
        # We suppress stdout to keep the terminal clean, as the optimizer is verbose.
        try:
            with suppress_stdout():
                res = solve_optimal_trajectory(StarshipBlock2, veh, env)
            
            # 3. Extract Results
            # Final Mass is at X3[6, -1]
            # Fuel Used = Launch Mass - Final Mass
            m_final = res['X3'][6, -1]
            fuel_used = StarshipBlock2.launch_mass - m_final
            
            # Calculate Earth Rotation Velocity at this latitude
            # v = omega * R * cos(lat)
            R_eq = EARTH_CONFIG.earth_radius_equator
            omega = EARTH_CONFIG.earth_omega_vector[2]
            v_rot = omega * R_eq * np.cos(np.radians(lat))
            
            print(f"{i+1:<5} | {lat:<8.1f} | {fuel_used:<15.1f} | {m_final:<15.1f} | {v_rot:<12.1f}")
            
        except Exception as e:
            print(f"{i+1:<5} | {lat:<8.1f} | {'FAILED':<15} | {'-':<15} | {'-':<12}")
            v_rot = 0.0
            
        # 4. Replicate result for all longitudes to create the heatmap surface
        for lon in lon_values_plot:
            results.append((lat, lon, fuel_used, v_rot, m_final))

    # --- PLOTTING ---
    print("\nGenerating Analysis Dashboard...")
    plot_analysis_dashboard(results)

def plot_analysis_dashboard(results):
    # Extract data
    lats_raw = np.array([r[0] for r in results])
    lons_raw = np.array([r[1] for r in results])
    fuels_raw = np.array([r[2] for r in results])
    v_rots_raw = np.array([r[3] for r in results])
    
    # Identify unique grid points
    unique_lats = np.unique(lats_raw)
    unique_lons = np.unique(lons_raw)
    
    n_lat = len(unique_lats)
    n_lon = len(unique_lons)
    
    # Reshape into grid
    fuel_grid = np.full((n_lat, n_lon), np.nan)
    
    # Map values to grid indices
    lat_map = {l: i for i, l in enumerate(unique_lats)}
    lon_map = {l: i for i, l in enumerate(unique_lons)}
    
    for lat, lon, fuel in zip(lats_raw, lons_raw, fuels_raw):
        i = lat_map[lat]
        j = lon_map[lon]
        fuel_grid[i, j] = fuel
        
    # Create meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(unique_lons, unique_lats)
    
    fig = plt.figure(figsize=(12, 10))
    
    ax = fig.add_subplot(111, projection='3d')
    # Convert Lat/Lon to Cartesian coordinates on a unit sphere
    rad_lat = np.radians(lat_grid)
    rad_lon = np.radians(lon_grid)
    
    x = np.cos(rad_lat) * np.cos(rad_lon)
    y = np.cos(rad_lat) * np.sin(rad_lon)
    z = np.sin(rad_lat)
    
    # Normalize fuel values for colormap
    if np.all(np.isnan(fuel_grid)):
        print("No valid data to plot.")
        return

    min_val = np.nanmin(fuel_grid)
    max_val = np.nanmax(fuel_grid)
    norm = plt.Normalize(min_val, max_val)
    
    # Create facecolors based on fuel values
    # Use 'jet' colormap (Blue=Low Fuel, Red=High Fuel)
    colors = plt.cm.jet(norm(fuel_grid))
    
    # Plot continuous surface
    surf = ax.plot_surface(x, y, z, facecolors=colors, 
                           rstride=1, cstride=1, 
                           shade=True, alpha=0.9, linewidth=0, antialiased=True)
    
    # Add a dummy ScalarMappable for the colorbar
    m = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])
    
    # Labels & Colorbar
    ax.set_title(f"Global Launch Cost & Earth Rotation Assist\nTarget: {StarshipBlock2.target_altitude/1000:.0f}km LEO", fontsize=14)
    cbar = plt.colorbar(m, ax=ax, shrink=0.6, pad=0.05)
    cbar.set_label("Fuel Consumed (kg)", fontsize=12)
    
    # --- ADD EARTH ROTATION VECTORS ---
    # Visualize the "Free Boost" from Earth's rotation
    # Subsample grid for arrows (plot every 4th point to avoid clutter)
    step = 4
    q_lats = lat_grid[::step, ::step]
    q_lons = lon_grid[::step, ::step]
    
    rad_q_lat = np.radians(q_lats)
    rad_q_lon = np.radians(q_lons)
    
    # Arrow Origins (Unit Sphere)
    qx = np.cos(rad_q_lat) * np.cos(rad_q_lon)
    qy = np.cos(rad_q_lat) * np.sin(rad_q_lon)
    qz = np.sin(rad_q_lat)
    
    # Arrow Vectors (Eastward Direction, Magnitude ~ cos(lat))
    scale = 0.15 # Visual scale factor
    mag = np.cos(rad_q_lat) * scale
    
    vx = -np.sin(rad_q_lon) * mag
    vy = np.cos(rad_q_lon) * mag
    vz = np.zeros_like(vx)
    
    ax.quiver(qx, qy, qz, vx, vy, vz, color='white', alpha=0.6, 
              arrow_length_ratio=0.3, linewidth=1.5, label='Earth Rotation Velocity')
    
    ax.legend(loc='lower right')
    
    # Clean up axes
    ax.set_box_aspect([1,1,1])
    ax.set_axis_off() # Hide axes for cleaner globe look
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_heatmap_analysis()