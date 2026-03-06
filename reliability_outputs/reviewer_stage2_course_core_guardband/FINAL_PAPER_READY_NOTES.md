Final paper/poster source-of-truth bundle.

Command run:
`python3 relabilityanalysis.py --course-core --no-show --max-workers 3 --output-dir reliability_outputs/reviewer_stage2_course_core_guardband`

Analysis profile:
- `course_core`

Included analyses:
- `smooth_integrator_benchmark`
- `drift`
- `integrator_tolerance`
- `collocation_defect_audit`
- `grid_independence`

Intentionally excluded from this bundle:
- `theoretical_efficiency`
- `monte_carlo_precision_target`
- `q2_uncertainty_budget`
- `randomized_multistart`
- `bifurcation_2d_map`
- `model_limitations`
- `q7_conclusion_support`

Provenance notes:
- All CSV and figure artifacts in this directory were produced by the command above.
- No artifacts were copied from `tmp/` or older `reliability_outputs/` folders.
- The optimizer uses a small internal guard band of `200 Pa` on max-Q and `0.03 g` on max-g so the replayed trajectory remains below the published hard limits of `35 kPa` and `4.0 g`.
