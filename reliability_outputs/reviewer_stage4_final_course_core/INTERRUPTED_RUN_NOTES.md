This run is incomplete and should not be used as the final paper/poster evidence bundle.

Command launched:
`python3 relabilityanalysis.py --course-core --no-show --max-workers 3 --output-dir reliability_outputs/reviewer_stage4_final_course_core`

Run status:
- Started on 2026-03-05 after plot-only code fixes.
- Completed:
  - `smooth_integrator_benchmark`
  - `drift`
  - `theoretical_efficiency`
  - `integrator_tolerance`
  - `collocation_defect_audit`
- Did not finish `grid_independence`; therefore no `grid_independence.csv` or `analysis_execution_log.csv` was written.
- Reviewer terminated the run once the equivalent publication-quality figures were regenerated directly from the completed source run.

Use instead:
- Numerical source run: `reliability_outputs/reviewer_stage3_course_core_current`
- Final curated paper bundle: `reliability_outputs/reviewer_stage5_final_course_core_curated`
