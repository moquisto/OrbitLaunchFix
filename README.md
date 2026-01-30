# Starship Block 2 Trajectory Optimization

## Overview
This project implements a high-fidelity trajectory optimization and simulation framework for the SpaceX Starship (Block 2) launch vehicle. It solves the "Launch-to-Orbit" optimal control problem, determining the steering and throttle commands required to deliver maximum payload to a 420 km Circular Low Earth Orbit (LEO).

The core philosophy of this project is **"Optimize-then-Simulate"**:
1.  **Optimize**: Use **CasADi** and **IPOPT** (Interior Point Optimizer) to find the mathematically optimal trajectory.
2.  **Simulate**: Verify the solution using **SciPy** (`solve_ivp`) in a rigorous forward-time simulation to ensure physical validity.

## Key Features

### 1. High-Fidelity Physics Engine
*   **Gravity Model**: Includes **J2 Perturbation** to account for Earth's oblateness (bulge at the equator), alongside standard central gravity.
*   **Atmosphere**: Implements the **US Standard Atmosphere 1976** (US76) from 0 to 1000 km.
    *   Uses **B-spline interpolation** for the optimizer (smooth gradients).
    *   Uses fast linear interpolation for the simulation.
*   **Wind**: Models atmospheric co-rotation (wind speed increases with altitude due to Earth's rotation).

### 2. Detailed Vehicle Model (Starship Block 2)
*   **Multi-Stage Dynamics**: Explicitly handles the mass and thrust discontinuities during the "Hot Staging" event between the Super Heavy Booster and the Starship Upper Stage.
*   **Aerodynamics**:
    *   **Mach-dependent Drag**: Uses lookup tables for $C_d$ vs. Mach number.
    *   **Angle of Attack (AoA)**: Calculates crossflow drag to penalize flying sideways (high AoA).
*   **Propulsion**:
    *   **Variable ISP**: Linearly interpolates Specific Impulse ($I_{sp}$) based on atmospheric pressure (Sea Level to Vacuum).
    *   **Throttling**: Constrains engine throttle between 40% and 100%.

### 3. Robust Architecture
*   **Symbolic & Numeric Compatibility**: The `Environment` and `Vehicle` classes are designed to handle both CasADi symbolic variables (for optimization) and NumPy arrays (for simulation) seamlessly.
*   **Scaling**: Implements a robust scaling system to normalize physical units (meters, kg, Newtons) to order-of-magnitude ~1.0, ensuring solver stability.

## File Structure

*   **`config.py`**: The "Single Source of Truth". Contains all physical constants, vehicle specs (mass, thrust, aero), and simulation settings.
*   **`environment.py`**: The physics world. Generates atmospheric lookups and calculates gravity/wind vectors. Handles the "Symbolic vs. Numeric" dispatch.
*   **`vehicle.py`**: Defines the Equations of Motion (EOM). Calculates forces (Thrust, Drag, Gravity) and derivatives ($\dot{r}, \dot{v}, \dot{m}$).
*   **`main.py`**: The Optimization Orchestrator. Sets up the CasADi `Opti` stack, defines constraints (path, terminal, staging), and runs the solver.
*   **`simulation.py`**: The Verification Engine. Takes the optimal control schedule and propagates the dynamics using `scipy.integrate.solve_ivp`.
*   **`guidance.py`**: Generates initial guesses (warm starts) to help the optimizer converge.
*   **`analysis.py`**: Visualization tools to plot trajectory, velocity, dynamic pressure (Q), and AoA.

## Installation & Requirements

This project requires Python 3.8+ and the following libraries:

```bash
pip install numpy scipy matplotlib casadi ussa1976
```

## Usage

1.  **Configure**: Edit `config.py` if you wish to change the target orbit, payload mass, or vehicle parameters.
2.  **Run Optimization**:
    ```bash
    python main.py
    ```
    This will:
    *   Solve the optimal control problem.
    *   Run the verification simulation.
    *   Generate plots comparing the optimized path vs. the simulated reality.

## Methodology

### The Optimization Problem
We formulate the launch as a **Direct Collocation** problem:
*   **Objective**: Maximize $m_{final}$ (Minimize fuel consumption).
*   **State Variables**: Position ($r_x, r_y, r_z$), Velocity ($v_x, v_y, v_z$), Mass ($m$).
*   **Control Variables**: Thrust Vector ($T_x, T_y, T_z$).
*   **Constraints**:
    *   Initial State: Launchpad coordinates (Cape Canaveral).
    *   Terminal State: Altitude = 420 km, Eccentricity $\approx$ 0, Inclination target.
    *   Path Constraints: Max Dynamic Pressure (Max Q), Angle of Attack limits.

### Verification
Since optimizers can exploit mathematical loopholes (e.g., discrete time steps), the **Simulation** phase acts as a referee. It interpolates the control inputs found by CasADi and re-integrates the trajectory using a variable-step integrator (RK45). A close match between the two confirms the solution is flyable.