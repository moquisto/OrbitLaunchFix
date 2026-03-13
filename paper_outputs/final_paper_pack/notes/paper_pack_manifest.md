# Final Paper Pack

## Main-text figure shortlist

- `fig_main_01_nominal_profile`
- `fig_main_02_constraints`
- `fig_main_03_grid_independence`
- `fig_main_04_integrator_convergence`
- `fig_main_06_warm_start_multiseed`

## Appendix / supplementary figures

- `fig_app_01_interval_replay_audit`
- `fig_app_02_smooth_integrator_order`
- `fig_app_03_theoretical_efficiency`
- `fig_app_04_3d_trajectory`
- `fig_app_05_global_launch_cost`

## Tables

- `table_01_nominal_case`
- `table_02_verification_summary`
- `table_03_assumptions_method`
- `table_04_replay_drift_by_phase`
- `table_05_multistart_consistency`

## Figure and table guidance used

- https://matplotlib.org/stable/users/explain/axes/constrainedlayout_guide.html
- https://matplotlib.org/stable/users/explain/colors/colormaps.html
- https://matplotlib.org/stable/users/explain/axes/legend_guide.html
- https://aiaa.org/publications/journals/Journal-Author/Guidelines-for-Journal-Figures-and-Tables/
- https://research-figure-guide.nature.com/figures/
- https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/

## Output constraints applied

- Figure titles are omitted; use report captions instead.
- Multi-panel figures use in-panel a), b), c) labels instead of repeated subplot titles.
- Axis labels include physical units.
- Legends are only used where they add information.
- Perceptually uniform colormaps are preferred over rainbow maps.
- Use 3D views only when spatial geometry is part of the message; otherwise prefer simpler 2D encodings.
- Main-text figures are trimmed to one argument per figure.
- Vector exports keep text editable for downstream publication workflows.
- Tables are exported as editable CSV and Markdown, not as images.
- Tables use explicit quantity/value/unit/notes columns to avoid undefined abbreviations.
