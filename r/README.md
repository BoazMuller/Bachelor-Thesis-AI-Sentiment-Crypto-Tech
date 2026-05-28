# R Workflow

R is used for the benchmark TVP-VAR connectedness model because the
`ConnectednessApproach` package directly supports the connectedness workflow
needed for the thesis.

Restore the R dependencies from the project root:

```bash
Rscript r/setup_renv.R
```

The model scripts check for the restored packages at run time but do not install
or update packages during estimation.

Python should prepare model-ready volatility inputs first. The later
TVP-VAR script will read those inputs, run `ConnectednessApproach`, and export
connectedness tables with explicit date columns to `results/tables/`.

Expected input format for the TVP-VAR script:

- one `date` column;
- two or more numeric volatility columns;
- no missing values;
- rows sorted by date.

Run the benchmark TVP-VAR connectedness model after Python has exported a
complete volatility panel:

```bash
Rscript r/modeling/run_tvpvar_connectedness.R \
  data/processed/tvpvar_inputs/benchmark_volatility.csv \
  results/tables/tvpvar/benchmark_h10 \
  1 \
  10
```

The arguments are:

1. input CSV;
2. output directory;
3. VAR lag order;
4. forecast horizon.

Use horizon `10` for the benchmark and `100` for the robustness check.
