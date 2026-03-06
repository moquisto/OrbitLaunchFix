Final source-of-truth bundle for the paper/poster:
`reliability_outputs/reviewer_stage5_final_course_core_curated`

Primary numerical source:
- Fresh deterministic rerun:
  `python3 relabilityanalysis.py --course-core --no-show --max-workers 3 --output-dir reliability_outputs/reviewer_stage3_course_core_current`

What was copied from that source run:
- Entire `data/` directory
- Unchanged figures:
  - `collocation_defect_audit`
  - `drift_summary`
  - `smooth_integrator_benchmark`

What was regenerated from the same source-run CSVs after plot-only fixes:
- `grid_independence_convergence_cost`
- `integrator_tolerance_convergence`
- `theoretical_efficiency`

Reason for regeneration:
- publication-quality fixes only
- no numerical results were recomputed or altered
- `grid_independence` runtime curve was removed because the cached baseline solve makes the `N=140` runtime non-comparable
- `theoretical_efficiency` was reframed as diagnostic context rather than pass/fail

Analyses intentionally omitted from the final paper bundle:
- `monte_carlo_precision_target`
- `q2_uncertainty_budget`
- `randomized_multistart`
- `bifurcation_2d_map`
- `model_limitations`
- `q7_conclusion_support`

Why omitted:
- They are not needed for the strongest deterministic A-grade argument.
- Under the current open-loop uncertainty model, Monte Carlo primarily demonstrates fragility rather than strengthening the nominal credibility case.

Provenance guardrails:
- No artifacts were copied from `tmp/`.
- No artifacts were copied from interrupted legacy bundles.
- The only numerical source for this curated bundle is `reviewer_stage3_course_core_current`.
