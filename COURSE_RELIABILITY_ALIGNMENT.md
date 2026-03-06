# Course-to-Test Alignment Audit

Date: 2026-03-05

## 1) What the course explicitly teaches (project-by-project)

### Project 1 (oscillations/chaos, integrators)
- Compare integrators (Euler-Cromer, Verlet, RK4).
- Study time-step dependence and numerical accuracy.
- Check conservation behavior (energy) and long-time stability.
- Source examples:
  - `tmp/reference_text/Assignments1.txt.clean:9-18`
  - `tmp/reference_text/Lecture1.txt.clean:33-35`
  - `tmp/reference_text/Feedback1.txt.clean:32-41`

### Project 2 (random walks + traffic model)
- Statistical accuracy and standard-error estimation.
- Determine required number of simulations for target error.
- Check equilibration effects and sensitivity to initial conditions.
- Source examples:
  - `tmp/reference_text/Assignments2-1.txt.clean:54-59`
  - `tmp/reference_text/Feedback2.txt.clean:37-56`

### Project 3 (Monte Carlo + molecular dynamics)
- MC standard error, convergence with number of samples.
- Compare estimated error against observed statistical error.
- Integration quality via drift monitoring as time step is varied.
- Source examples:
  - `tmp/reference_text/Assignments3.txt.clean:21-24`
  - `tmp/reference_text/Assignments3.txt.clean:42-44`
  - `tmp/reference_text/Feedback3.txt.clean:9-21`

### Project 4 (Laplace/grid methods)
- Discretization/grid refinement and convergence to target accuracy.
- Compare methods and observe convergence behavior.
- Check sensitivity to poor initial guesses.
- Source examples:
  - `tmp/reference_text/Electrostatics.txt.clean:728-730`
  - `tmp/reference_text/Electrostatics.txt.clean:734-735`
  - `tmp/reference_text/Electrostatics.txt.clean:819-821`

### Final grading guidance
- Discretization + accuracy required.
- Relevant method/parameter comparisons required.
- Clear plots and no conclusion-impacting mistakes.
- Source:
  - `reference/Final Rapport/_extracted_text/grading criteria.txt:6-29`

## 2) Decision for each reliability test in this repository

### Keep as core (course-aligned final bundle)
- `grid_independence`
- `integrator_tolerance`
- `smooth_integrator_benchmark`
- `drift` (kept as integration/consistency drift check)

### Keep by explicit exception request
- `theoretical_efficiency` (idealized delta-v reference / fuel-use context, short)
- `collocation_defect_audit`

### Keep available but disable by default (optional / supplementary)
- `monte_carlo_precision_target`
- `q2_uncertainty_budget`
- `randomized_multistart`
- `bifurcation_2d_map` (renamed in UI text to “2D parameter sensitivity map”)
- `model_limitations`
- `q7_conclusion_support`

## 3) Strict keep/remove matrix (for main report)

Use this matrix when deciding what belongs in the main paper body vs appendix.

- `grid_independence`: KEEP (main body)
  - Why: direct discretization-convergence evidence taught in course and explicitly rewarded in grading.
- `integrator_tolerance`: KEEP (main body)
  - Why: direct numerical-method sensitivity/convergence evidence taught in course.
- `smooth_integrator_benchmark`: KEEP (short main/appendix)
  - Why: formal method-order check (Euler vs RK4) from course integrator content.
- `monte_carlo_precision_target`: REMOVE from main body (optional appendix only)
  - Why: the current open-loop uncertainty model quantifies fragility rather than strengthening the deterministic A-grade story; use only if discussed explicitly as a limitation/extension.
- `q2_uncertainty_budget`: REMOVE from main body (optional appendix only)
  - Why: useful extension, but larger, slower, and easier to overclaim from than the core deterministic evidence chain.
- `drift`: KEEP (main body, compact)
  - Why: optimizer-vs-simulator consistency and integration drift quality check.
- `theoretical_efficiency`: KEEP (exception, short)
  - Why: idealized delta-v lower-bound reference for physical plausibility and fuel-use context; not a strict lower bound on fuel burned.
- `collocation_defect_audit`: KEEP (exception, short)
  - Why: direct transcription-quality evidence; requested exception.
- `randomized_multistart`: REMOVE from main body (optional appendix only)
  - Why: useful robustness extension, but not core course requirement for grade defense.
- `bifurcation_2d_map` (2D parameter sensitivity map): REMOVE from main body (optional appendix only)
  - Why: sensitivity visualization is useful but not a core taught method in course projects.
- `model_limitations`: REMOVE from core reliability run (optional narrative appendix)
  - Why: interpretation scaffold, not a measured numerical test.
- `q7_conclusion_support`: REMOVE from core reliability run
  - Why: synthesis table should be written in report text, not forced as another test.

## 4) Figure simplification policy applied

- One plot = one message where possible.
- Removed dense annotation blocks from core charts.
- Moved synthesis logic out of charts and into paper text/tables.
- Split Q2 composite 2x2 into simple standalone figures:
  - `q2_grid_sensitivity`
  - `q2_integrator_sensitivity`
  - `q2_parameter_uncertainty`
  - `q2_uncertainty_budget`

## 5) Drift / Multistart / Sensitivity-map decisions

- Drift: KEEP, but keep chart minimal (absolute units by phase, threshold shown once).
- Monte Carlo precision target: KEEP in code, OFF in final paper bundle; treat as exploratory uncertainty study unless the report explicitly wants to discuss open-loop fragility.
- Q2 uncertainty budget: KEEP in code, OFF in final paper bundle; treat as supplementary uncertainty accounting, not core grading evidence.
- Randomized multistart: KEEP in code, OFF by default; use only if examiner asks for optimizer robustness stress test.
- 2D parameter sensitivity map (old “bifurcation”): KEEP in code, OFF by default; treat as supplementary open-loop sensitivity map, not core proof.

## 6) CasADi built-in analysis tools integrated

Implemented in code:
- On solve failure, call `opti.debug.show_infeasibilities()` (verbose mode) for direct infeasible-constraint diagnostics.
- Save solver diagnostics from `opti.stats()` into result dictionary:
  - `solver_return_status`
  - `solver_iter_count`

Current use:
- Improves traceability for failed/slow solves without adding extra custom solver wrappers.
- Keeps diagnostics native to CasADi/IPOPT and aligned with official CasADi debugging workflow.
