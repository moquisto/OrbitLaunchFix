# Starship Block 2 Trajectory Optimization and Reliability Analysis

High-fidelity launch-to-orbit optimization for a Starship Block 2 style two-stage vehicle, with forward simulation verification and a report-oriented reliability suite.

The project follows an explicit workflow:

1. Optimize a fuel-minimal ascent trajectory with node-based RK4 direct transcription (CasADi + IPOPT).
2. Re-fly the optimized controls in forward time (`solve_ivp`) to verify physical consistency.
3. Quantify confidence with targeted reliability analyses, with optional robustness/sensitivity extensions enabled through config toggles.

## What This Repository Contains

- A two-stage ascent optimal control solver with structural/path constraints.
- A unified physics model used by both optimizer and simulator (same dynamics/environment core).
- Visualization tools for kinematics, loads, controls, and 3D trajectory.
- A comprehensive reliability suite that exports figures and CSV evidence for report questions.

## Core Capabilities

- **Environment model**
  - WGS84 Earth geometry
  - J2 perturbation toggle
  - US Standard Atmosphere 1976 up to 1000 km
  - Atmosphere co-rotation wind model (`Omega x r`)

- **Vehicle model**
  - Two-stage dynamics with explicit staging mass discontinuity
  - Mach-dependent drag lookup tables
  - Crossflow drag penalty via AoA
  - Pressure-dependent ISP interpolation (`ISP_sl -> ISP_vac`)
  - Throttle-constrained propulsion

- **Optimization model**
  - Node-based RK4 direct transcription with piecewise-constant controls
  - Decision variables across booster + coast + upper-stage phases
  - Path constraints for dynamic pressure and sensed G-load
  - Terminal constraints for target circular orbit and inclination
  - Objective: maximize final mass (equivalently minimize fuel used)

- **Reliability/validation model**
  - Core course-aligned checks: grid independence, interval replay audit, theoretical efficiency
  - Integrator convergence, drift, and method-order checks
  - Monte Carlo uncertainty and precision targeting
  - Optional extensions: randomized multistart, 2D sensitivity map, model-limitations summary, Q7 synthesis support

## Repository Structure

| Path | Purpose |
| --- | --- |
| `config.py` | Single source of truth for vehicle, environment, sequence, and reliability toggles |
| `environment.py` | Atmosphere/gravity/wind model with symbolic + compiled numeric parity |
| `vehicle.py` | Full equations of motion and aerodynamic/propulsion force model |
| `guidance.py` | Warm-start trajectory generator for the NLP |
| `main.py` | Optimization orchestrator and default end-to-end mission run |
| `simulation.py` | Forward simulation verifier (`solve_ivp`) using optimized controls |
| `analysis.py` | Mission plots and comparison dashboards |
| `relabilityanalysis.py` | Reliability suite with figure/CSV export and CLI flags |
| `testplot.py` | Global latitude-dependent launch cost heatmap utility |
| `reliability_outputs/` | Timestamped output folders from reliability runs |
| `REFERENCE/` | Course/project reference material |

## Requirements

- Python 3.8+
- `numpy`
- `scipy`
- `matplotlib`
- `casadi`
- `ussa1976`

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib casadi ussa1976
```

## Quick Start

### 1. Run the full optimize -> simulate -> plot flow

```bash
python main.py
```

What this does:

- Prints mission config and preflight diagnostics.
 - Solves the direct-transcription problem in IPOPT.
 - Runs forward verification simulation.
- Prints objective/fuel-margin summary.
- Opens mission visualization windows.

### 2. Run reliability suite (report evidence pipeline)

```bash
python relabilityanalysis.py --no-show
```

Important: the filename is `relabilityanalysis.py` (missing the second `i`) in this repository.

Default behavior:

- Creates `reliability_outputs/<timestamp>/`.
- Saves figures to `figures/` (PNG + PDF).
- Saves tabular outputs to `data/` (CSV).
- Uses random seed `1337` unless overridden.
- Runs the final-report default set from `RELIABILITY_ANALYSIS_TOGGLES`; supplementary uncertainty/robustness extensions are off by default.

CLI options:

```text
--output-dir <path>   Custom output directory
--seed <int>          RNG seed for reproducible Monte Carlo analyses
--mc-samples <int>    Monte Carlo sample count (default: 200)
--max-workers <int>   Maximum parallel workers (use 1 for serial mode)
--no-show             Do not open interactive plot windows
--no-save             Do not save figures/CSV outputs
```

Example reproducible run:

```bash
python relabilityanalysis.py \
  --seed 1337 \
  --mc-samples 300 \
  --no-show \
  --output-dir reliability_outputs/baseline_seed1337
```

### 3. Run global launch heatmap utility

```bash
python testplot.py
```

This performs a latitude sweep and renders a 3D globe-style fuel-cost heatmap.

## Configuration Guide (`config.py`)

Edit `StarshipBlock2`, `EARTH_CONFIG`, and `RELIABILITY_ANALYSIS_TOGGLES` directly.

### High-impact mission/solver parameters

- `payload_mass`
- `target_altitude`
- `target_inclination` (`None` => defaults to absolute launch latitude)
- `num_nodes` (transcription resolution)
- `max_iter` (IPOPT iteration limit)
- `max_q_limit`
- `max_g_load`
- `sequence.min_throttle`
- `sequence.separation_delay` (enables coast phase)

### Environment switches

- `use_j2_perturbation`
- `density_multiplier` (used in uncertainty studies)
- `launch_latitude`, `launch_longitude`, `initial_rotation` (directly affect launch geometry and inertial alignment)

Note: `EnvConfig.use_wind_model` exists in config but is not currently used as an active runtime toggle.

### Reliability toggles

The `ReliabilityAnalysisToggles` dataclass controls which test blocks execute in `ReliabilitySuite.run_all()`.

Default enabled:
- `grid_independence`
- `interval_replay_audit`
- `theoretical_efficiency`
- `integrator_tolerance`
- `smooth_integrator_benchmark`
- `drift`

Default disabled:
- `monte_carlo_precision_target`
- `q2_uncertainty_budget`
- `randomized_multistart`
- `bifurcation_2d_map`
- `model_limitations`
- `q7_conclusion_support`

## Model and Assumptions

- 3D inertial-frame translational dynamics.
- No active closed-loop guidance in verification; optimized open-loop control schedule is replayed.
- Stage separation modeled as an instantaneous mass reset.
- The orbit target is enforced as spherical height above `R_eq`; ellipsoidal altitude is exported separately as a ground-referenced diagnostic.
- Aerodynamics are table/heuristic based (Mach-Cd + crossflow term), not full CFD.
- Atmospheric properties from USSA1976 with interpolation/smoothing choices made for optimizer robustness.

Treat conclusions as valid within this modeling envelope, not as flight-certified truth.

## Reliability Suite Coverage (Report Mapping)

The reliability suite is structured to support question-driven reporting:

- **Q1** credible optimum: grid independence, interval replay audit, theoretical efficiency; multistart robustness is available as an optional extension.
- **Q2** uncertainty/accuracy: integrator sensitivity, Monte Carlo precision, uncertainty budget.
- **Q3** code reliability: optimizer-vs-simulator drift and benchmark checks.
- **Q5** cliff-edge behavior: optional 2D sensitivity/feasibility mapping.
- **Q6** model limitations: optional validity/limitation documentation.
- **Q7** engineering conclusion support: optional cross-test evidence synthesis.

## Expected Outputs

### `main.py`

- Terminal diagnostics for optimization and simulation phases.
- Fuel-margin summary at final state.
- Multiple interactive matplotlib figures:
  - Mission overview (altitude/velocity, mass, ground track, ascent profile)
  - Aerodynamics and dynamics (Q/Mach, AoA, gamma/pitch, heating proxy)
  - Controls and loads (throttle, G-load, forces, thrust vector, envelope)
  - 3D trajectory with projected orbit

### `relabilityanalysis.py`

- Timestamped output directory containing:
  - `data/*.csv` for numerical evidence tables
  - `figures/*.png` and `figures/*.pdf` for report-ready visualizations

## Troubleshooting

- **Optimization fails to converge**
  - Increase `max_iter`.
  - Reduce problem stiffness temporarily (e.g., fewer `num_nodes`) to localize issues.
  - Keep warm start enabled (`guidance.py`) and verify reasonable initial trajectory.

- **Simulation drifts from optimizer trajectory**
  - Increase `num_nodes` in the direct-transcription grid.
  - Tighten simulation tolerances in `run_simulation(..., rtol, atol)`.
  - Inspect drift diagnostics printed by `main.py`/`debug.py`.

- **Very slow reliability runs**
  - Use `--no-show`.
  - Lower `--mc-samples` for exploratory passes.
  - Disable selected blocks via `RELIABILITY_ANALYSIS_TOGGLES`.

- **Dependency/build issues**
  - Ensure the active virtual environment is being used.
  - Upgrade `pip` before installing scientific packages.

## Reproducibility Notes

- Use fixed `--seed` in reliability runs.
- Keep configuration snapshots with output directories.
- Export to a deterministic `--output-dir` for direct run-to-run comparison.

## Academic Context

This repository appears structured for simulation/modeling coursework and report-backed engineering argumentation. It is most useful when run as:

1. Baseline optimization/simulation (`main.py`)
2. Reliability evidence generation (`relabilityanalysis.py`)
3. Comparative scenario sweeps (`testplot.py` and config edits)
