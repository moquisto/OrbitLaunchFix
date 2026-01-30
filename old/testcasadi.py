import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. SETUP & SCALING
# -----------------------------------------------------------------------------
g0_phys = 9.80665
R_earth_phys = 6371000.0
mu_phys = 3.986004418e14

# Scales
scale_dist = R_earth_phys
scale_time = np.sqrt(R_earth_phys**3 / mu_phys) 
scale_vel  = scale_dist / scale_time
scale_mass = 1.0e6  
scale_force = scale_mass * scale_dist / scale_time**2

print(f"Scales -> Dist: {scale_dist:.0f}m, Time: {scale_time:.1f}s, Vel: {scale_vel:.1f}m/s")

# Target Orbit (420km)
h_target = 420000.0
r_target_phys = R_earth_phys + h_target
v_target_phys = np.sqrt(mu_phys / r_target_phys)

# Vehicle Specs
m_dry = 100000.0
m_wet = 1300000.0
isp = 350.0
T_max_phys = 22000000.0 

# -----------------------------------------------------------------------------
# 2. OPTIMIZATION PROBLEM
# -----------------------------------------------------------------------------
opti = ca.Opti()
N = 100 

# --- Decision Variables ---
# T is now a VARIABLE again (Bounded between 400s and 1000s)
T_scaled = opti.variable()
# CHANGE THIS LINE in Section 2:
# Allow flight time to be as short as 150s so it can burn at 100% thrust
opti.subject_to(opti.bounded(150.0/scale_time, T_scaled, 1000.0/scale_time))

X_s = opti.variable(5, N+1) # [rx, ry, vx, vy, m]
U_s = opti.variable(2, N)   # [Tx, Ty]

dt_s = T_scaled / N

# Helper: Unpack scaled variables
rx, ry = X_s[0, :], X_s[1, :]
vx, vy = X_s[2, :], X_s[3, :]
m      = X_s[4, :]
ux, uy = U_s[0, :], U_s[1, :]

# --- Objective: Maximize Remaining Fuel ---
opti.minimize(-m[-1])

# --- Dynamics (Scaled) ---
def dynamics(x, u):
    pos_x, pos_y = x[0], x[1]
    vel_x, vel_y = x[2], x[3]
    mass         = x[4]
    thrust_x, thrust_y = u[0], u[1]
    
    # Gravity (Newtonian)
    r = ca.sqrt(pos_x**2 + pos_y**2)
    gx = -pos_x / r**3
    gy = -pos_y / r**3
    
    # Thrust
    ax = thrust_x / mass
    ay = thrust_y / mass
    
    # Mass Flow
    v_exhaust_s = (isp * g0_phys) / scale_vel
    f_mag = ca.sqrt(thrust_x**2 + thrust_y**2 + 1e-8)
    dm = -f_mag / v_exhaust_s
    
    return ca.vertcat(vel_x, vel_y, ax+gx, ay+gy, dm)

# RK4 Constraints
for k in range(N):
    k1 = dynamics(X_s[:, k],       U_s[:, k])
    k2 = dynamics(X_s[:, k]+dt_s/2*k1, U_s[:, k])
    k3 = dynamics(X_s[:, k]+dt_s/2*k2, U_s[:, k])
    k4 = dynamics(X_s[:, k]+dt_s*k3,   U_s[:, k])
    x_next = X_s[:, k] + dt_s/6 * (k1 + 2*k2 + 2*k3 + k4)
    opti.subject_to(X_s[:, k+1] == x_next)

# -----------------------------------------------------------------------------
# 3. BOUNDS & CONSTRAINTS
# -----------------------------------------------------------------------------
# Boundary Conditions
opti.subject_to(rx[0] == 0)
opti.subject_to(ry[0] == R_earth_phys / scale_dist)
opti.subject_to(vx[0] == 0)
opti.subject_to(vy[0] == 0)
opti.subject_to(m[0]  == m_wet / scale_mass)

# Target Orbit Constraints
r_fin = ca.sqrt(rx[-1]**2 + ry[-1]**2)
v_fin = ca.sqrt(vx[-1]**2 + vy[-1]**2)
rdotv = rx[-1]*vx[-1] + ry[-1]*vy[-1]

opti.subject_to(r_fin == r_target_phys / scale_dist) 
opti.subject_to(v_fin == v_target_phys / scale_vel)
opti.subject_to(rdotv == 0) 

# Path Constraints
opti.subject_to(rx**2 + ry**2 >= (R_earth_phys/scale_dist)**2) # Don't hit Earth
opti.subject_to(m >= m_dry/scale_mass)                         # Don't use more fuel than exists

T_max_s = T_max_phys / scale_force
for k in range(N):
    t_mag = ca.sqrt(ux[k]**2 + uy[k]**2 + 1e-8)
    opti.subject_to(opti.bounded(0.0, t_mag, T_max_s)) # Allow throttle 0-100%

# -----------------------------------------------------------------------------
# 4. INITIAL GUESS (Improved)
# -----------------------------------------------------------------------------
opti.set_initial(T_scaled, 600.0 / scale_time)

# Guess a circular arc trajectory (Prevents the "crash" guess)
theta = np.linspace(0, np.pi/2, N+1)
r_guess_vals = np.linspace(R_earth_phys, r_target_phys, N+1) / scale_dist
x_guess = r_guess_vals * np.sin(theta)
y_guess = r_guess_vals * np.cos(theta)

opti.set_initial(rx, x_guess)
opti.set_initial(ry, y_guess)
opti.set_initial(vx, np.linspace(0, v_target_phys/scale_vel, N+1))
opti.set_initial(vy, 0.1) # Small upward velocity guess
opti.set_initial(m,  np.linspace(m_wet, m_dry, N+1)/scale_mass)

# Control Guess: Fight gravity (Vertical)
opti.set_initial(ux, 0.1 * T_max_s) 
opti.set_initial(uy, 0.9 * T_max_s)

# -----------------------------------------------------------------------------
# 5. SOLVE
# -----------------------------------------------------------------------------
opti.solver('ipopt', {'ipopt': {'print_level': 5, 'tol': 1e-4, 'max_iter': 3000}})

try:
    sol = opti.solve()
    print("SUCCESS: Solution Found")
except:
    print("WARNING: Solver stopped early. Plotting debug results.")
    sol = opti.debug

# -----------------------------------------------------------------------------
# 6. ORBIT PROPAGATION & PLOTTING
# -----------------------------------------------------------------------------
# A. Extract the Optimal Ascent Path
t_ascent = np.linspace(0, sol.value(T_scaled)*scale_time, N+1)
rx_ascent = sol.value(rx) * scale_dist
ry_ascent = sol.value(ry) * scale_dist
vx_ascent = sol.value(vx) * scale_vel
vy_ascent = sol.value(vy) * scale_vel
m_val     = sol.value(m)  * scale_mass

# B. Propagate the Orbit (Coasting Phase)
# We simulate 6000 seconds (~1.6 hours) of unpowered flight
dt_coast = 10.0 
coast_steps = int(6000 / dt_coast)

coast_rx = [rx_ascent[-1]]
coast_ry = [ry_ascent[-1]]
coast_vx = vx_ascent[-1]
coast_vy = vy_ascent[-1]

# Simple Euler/RK1 integrator for visualization
# (Using unscaled physics for clarity here)
for _ in range(coast_steps):
    r_sq = coast_rx[-1]**2 + coast_ry[-1]**2
    r = np.sqrt(r_sq)
    
    # Gravity (Standard + J2 if you implemented it)
    # Basic Spherical Gravity for visualization:
    g_acc = -mu_phys / (r**3)
    ax = g_acc * coast_rx[-1]
    ay = g_acc * coast_ry[-1]
    
    # Update Velocity
    coast_vx += ax * dt_coast
    coast_vy += ay * dt_coast
    
    # Update Position
    coast_rx.append(coast_rx[-1] + coast_vx * dt_coast)
    coast_ry.append(coast_ry[-1] + coast_vy * dt_coast)

# Convert lists to arrays
coast_rx = np.array(coast_rx)
coast_ry = np.array(coast_ry)

# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------
plt.figure(figsize=(14, 10))

# 1. Full Trajectory Map
plt.subplot(2, 2, 1)
# Draw Earth
theta_e = np.linspace(0, 2*np.pi, 200)
plt.fill(R_earth_phys*np.cos(theta_e)/1000, R_earth_phys*np.sin(theta_e)/1000, color='#e6f2ff')
plt.plot(R_earth_phys*np.cos(theta_e)/1000, R_earth_phys*np.sin(theta_e)/1000, 'k-', linewidth=1)

# Draw Paths
plt.plot(rx_ascent/1000, ry_ascent/1000, 'r-', linewidth=2.5, label='Powered Ascent')
plt.plot(coast_rx/1000, coast_ry/1000, 'b--', linewidth=1.5, label='Orbit Coast (1 Period)')

# Markers
plt.plot(rx_ascent[0]/1000, ry_ascent[0]/1000, 'ko', label='Launch')
plt.plot(rx_ascent[-1]/1000, ry_ascent[-1]/1000, 'r*', markersize=10, label='MECO (Engine Cutoff)')

plt.title(f"Full Mission Profile (Orbit: {h_target/1000:.0f}km)")
plt.xlabel("X (km)")
plt.ylabel("Y (km)")
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

# 2. Altitude Profile
plt.subplot(2, 2, 2)
alt_ascent = np.sqrt(rx_ascent**2 + ry_ascent**2) - R_earth_phys
plt.plot(t_ascent, alt_ascent/1000, 'r-', linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Altitude (km)")
plt.title("Ascent Altitude")
plt.grid(True)

# 3. Velocity Profile
plt.subplot(2, 2, 3)
vel_mag = np.sqrt(vx_ascent**2 + vy_ascent**2)
plt.plot(t_ascent, vel_mag, 'r-', linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title(f"Velocity (Final: {vel_mag[-1]:.0f} m/s)")
plt.grid(True)

# 4. Mass / Fuel
plt.subplot(2, 2, 4)
plt.plot(t_ascent, m_val/1000, 'g-', linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Mass (tonnes)")
plt.title("Fuel Consumption")
plt.grid(True)

plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# 7. VERIFICATION ANALYSIS (Fixed)
# -----------------------------------------------------------------------------
print("\n--- OPTIMALITY VERIFICATION ---")

# 1. Define t_val explicitly to avoid NameError
t_val = np.linspace(0, sol.value(T_scaled)*scale_time, N+1)

# Unscale values for analysis
ux_val = sol.value(ux)
uy_val = sol.value(uy)
vx_val = sol.value(vx)
vy_val = sol.value(vy)

# 2. Calculate Control Usage (Throttle)
throttle_pct = np.sqrt(ux_val**2 + uy_val**2) / sol.value(T_max_s) * 100
avg_throttle = np.mean(throttle_pct)
print(f"Average Throttle: {avg_throttle:.2f}%")

# 3. Calculate Thrust Alignment (Steering Loss)
# Slice velocity to [:-1] to match the N control points
vel_angle = np.arctan2(vy_val[:-1], vx_val[:-1])
thrust_angle = np.arctan2(uy_val, ux_val)

# Calculate difference, handling the -180/180 wraparound
angle_diff = vel_angle - thrust_angle
angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
angle_error = np.degrees(np.abs(angle_diff))

# Ignore the first 10 steps (vertical launch has unstable angles)
avg_steering_error = np.mean(angle_error[10:]) 
print(f"Avg Steering Error: {avg_steering_error:.2f} deg")

# 4. Delta-V Check
m_initial = m_val[0]
m_final = m_val[-1]
delta_v = isp * g0_phys * np.log(m_initial / m_final)
print(f"Total Delta-V Spent: {delta_v:.2f} m/s")
print(f"  - Theoretical Orbit Vel: {v_target_phys:.2f} m/s")
print(f"  - Gravity/Steering Losses: {delta_v - v_target_phys:.2f} m/s")

# Plot Verification
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(t_val[:-1], throttle_pct)
plt.title("Throttle Profile (%)")
plt.xlabel("Time (s)")
plt.ylabel("Throttle")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t_val[:-1], angle_error)
plt.title("Thrust-Velocity Misalignment (Deg)")
plt.xlabel("Time (s)")
plt.ylabel("Error (deg)")
plt.ylim(0, 20) 
plt.grid(True)

plt.tight_layout()
plt.show()