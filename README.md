# Starship Block 2 Trajectory Optimization & Simulation

## Overview
This project implements a high-fidelity trajectory optimization and simulation framework for the SpaceX Starship (Block 2 / IFT-6 configuration) launch vehicle. It solves the "Launch-to-Orbit" optimal control problem, determining the steering and throttle commands required to deliver maximum payload to a 420 km Circular Low Earth Orbit (LEO).

The core philosophy of this project is **"Optimize-then-Simulate"**:
1.  **Optimize**: Use **CasADi** and **IPOPT** (Interior Point Optimizer) to find the mathematically optimal trajectory.
2.  **Simulate**: Verify the solution using **SciPy** (`solve_ivp`) in a rigorous forward-time simulation to ensure physical validity.
3.  **Verify**: Run targeted reliability checks to quantify optimality credibility, numerical/statistical uncertainty, and code reliability.

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

### 3. Reliability & Analysis Suite
*   **Q1 (Optimality Credibility)**: Grid Independence, Collocation Defect Audit, Initial-Guess Robustness, and Theoretical Delta-V Efficiency.
*   **Q2 (Accuracy & Uncertainty)**: Integrator Tolerance, Event-Time Convergence, Monte Carlo Precision Target, and Global Sensitivity Ranking.
*   **Q3 (Code Reliability)**: Optimizer-vs-Simulator Drift plus numerical sanity benchmarks.
*   **Q5/Q7 (Behavior & Conclusion Support)**: Bifurcation sweeps and finite-time sensitivity for open-loop fragility evidence.

## File Structure

### Core Modules
*   **`config.py`**: The "Single Source of Truth". Contains all physical constants, vehicle specs (mass, thrust, aero), and simulation settings.
*   **`environment.py`**: The physics world. Generates atmospheric lookups and calculates gravity/wind vectors. Handles the "Symbolic vs. Numeric" dispatch.
*   **`vehicle.py`**: Defines the Equations of Motion (EOM). Calculates forces (Thrust, Drag, Gravity) and derivatives ($\dot{r}, \dot{v}, \dot{m}$).

### Optimization & Simulation
*   **`main.py`**: The Optimization Orchestrator. Sets up the CasADi `Opti` stack, defines constraints (path, terminal, staging), and runs the solver.
*   **`simulation.py`**: The Verification Engine. Takes the optimal control schedule and propagates the dynamics using `scipy.integrate.solve_ivp`.
*   **`guidance.py`**: Generates initial guesses (warm starts) to help the optimizer converge.

### Analysis & Tools
*   **`analysis.py`**: Visualization tools to plot trajectory, velocity, dynamic pressure (Q), and AoA.
*   **`debug.py`**: Extensive diagnostic suite. Checks scaling, physics consistency, and post-flight metrics (Delta-V budget, orbital elements).
*   **`relabilityanalysis.py`**: Runs the report-focused reliability suite (optimality credibility, uncertainty quantification, drift checks, bifurcation, and sensitivity).
*   **`testplot.py`**: Generates a global heatmap showing fuel consumption vs. Launch Latitude.

## Installation & Requirements

This project requires Python 3.8+ and the following libraries:

```bash
pip install numpy scipy matplotlib casadi ussa1976
```

## Usage

### 1. Standard Mission Optimization
Run the main script to optimize the trajectory for the configuration defined in `config.py`:

    ```bash
    python main.py
    ```
**Output**:
*   Console logs showing optimizer progress and verification checks.
*   Interactive plots: Altitude, Velocity, Ground Track, AoA, Forces, and 3D Trajectory.

### 2. Reliability Analysis
Run the report-focused validation suite:

    ```bash
    python relabilityanalysis.py
    ```

## Final Report Questions and Evidence

The reliability suite is scoped to answer the following report questions:

1. **Q1: Have we found a credible minimum-fuel solution (within this model)?**
   * **Tests**: `grid_independence`, `collocation_defect_audit`, `initial_guess_robustness`, `theoretical_efficiency`.
   * **Role**: Shows the solution is not a mesh artifact, satisfies dynamics consistently, is not overly warm-start dependent, and is benchmarked against a theoretical delta-v floor.

2. **Q2: How accurate is the result, and where does uncertainty lie?**
   * **Tests**: `integrator_tolerance`, `event_time_convergence`, `monte_carlo_precision_target`, `global_sensitivity`.
   * **Role**: Quantifies numerical sensitivity to integrator settings and statistical uncertainty under parameter variation (with absolute + relative CI precision targets), then ranks dominant uncertainty drivers with bootstrap confidence intervals.

3. **Q3: Is the code working as intended and reliable?**
   * **Tests**: `drift`, `smooth_integrator_benchmark`, `conservative_invariants`.
   * **Role**: Verifies optimizer/simulator consistency and checks integrator behavior in controlled benchmark scenarios.

4. **Q5: How close are we to cliff-edge failure behavior?**
   * **Tests**: `bifurcation`, `bifurcation_2d_map`.
   * **Role**: Identifies open-loop feasibility boundaries via normalized orbit-error thresholds (not just fuel margin), showing where small parameter changes trigger mission failure.

5. **Q6: What are the model limitations and validity bounds?**
   * **Tests**: No single dedicated test; informed by all results and modeling assumptions.
   * **Role**: Defines where conclusions are valid and prevents over-claiming beyond modeled physics.

6. **Q7: What is the main engineering conclusion?**
   * **Test support**: `finite_time_sensitivity` (plus Q2/Q5 evidence).
   * **Role**: Supports the conclusion that open-loop solutions are sensitive and motivates feedback guidance for practical robustness.

### 3. Global Launch Heatmap
Generate a heatmap of fuel costs across different latitudes:

    ```bash
    python testplot.py
    ```

## Methodology

### The Optimization Problem
We formulate the launch as a **Direct Collocation** problem:
*   **Objective**: Maximize $m_{final}$ (Minimize fuel consumption).
*   **State Variables**: Position ($r_x, r_y, r_z$), Velocity ($v_x, v_y, v_z$), Mass ($m$).
*   **Control Variables**: Thrust Vector ($T_x, T_y, T_z$).
*   **Constraints**:
    *   Initial State: Launchpad coordinates (Cape Canaveral).
    *   Terminal State: Altitude = 420 km, Eccentricity $\approx$ 0, Inclination target.
    *   Path Constraints: Max Dynamic Pressure (Max Q), G-Load limits, Angle of Attack limits.

### Initialization (Warm Start)
Non-linear optimization solvers (like IPOPT) are sensitive to the initial guess. This project includes a **Guidance Module** (`guidance.py`) that generates a physically plausible "Gravity Turn" trajectory using a heuristic control law. This "Warm Start" significantly improves convergence stability compared to a cold start.

### Verification
Since optimizers can exploit mathematical loopholes (e.g., discrete time steps), the **Simulation** phase acts as a referee. It interpolates the control inputs found by CasADi and re-integrates the trajectory using a variable-step integrator (RK45). A close match between the two confirms the solution is flyable.

## Troubleshooting

*   **Optimizer Fails**: Check the console output. `debug.py` will automatically print a "Failure Diagnosis", checking for:
    *   **Scaling Issues**: Variables should be order-of-magnitude ~1.0.
    *   **Infeasible Constraints**: e.g., Target orbit is too high for the fuel available.
    *   **Jacobian Singularity**: Mathematical formulation issues.
*   **Simulation Drift**: If the optimized path diverges from the simulation:
    *   Increase `num_nodes` in `config.py`.
    *   Tighten integrator tolerances in `simulation.py`.
