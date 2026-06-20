# Output Artifacts

Generated outputs belong here. These folders are ignored by Git by default, except for `.gitkeep` placeholders.

## Folders

- `figures/`: Plots and visualizations used in analysis or the thesis.
- `tables/`: CSV datasets, diagnostics, coefficients, and result tables used by the thesis.
- `models/`: Serialized fitted model objects, such as TVP-VAR `.rds` files.
- `reports/`: Generated reports, logs, and supplementary outputs.

## TVP-VAR And Spillover Outputs

- `tables/tvpvar_connectedness/`: original and EAIS-augmented connectedness
  outputs, pre/post-estimation diagnostics, output-completeness checks, and the
  original-system-only regression dataset.
- `figures/tvpvar_connectedness/`: connectedness time series plus individual,
  paired, and original-versus-EAIS network figures.
- `tables/spillover_regressions/`: Newey-West HAC connectedness-regression
  results, one-lag EAIS results, five-lag EAIS robustness results, cumulative
  and joint EAIS tests, and regression diagnostics.
