# R Model Workflow

R is used for volatility and connectedness models because `rugarch` supports the
EGARCH specifications and `ConnectednessApproach` directly supports the
TVP-VAR connectedness workflow.

Restore R dependencies from the project root:

```bash
Rscript r/setup_renv.R
```

The model scripts check for restored packages at run time but do not install or
update packages during estimation.

## ARMAX-EGARCHX

The sentiment-return/volatility models are estimated with:

```bash
Rscript r/modeling/run_armax_egarchx.R
```

This uses BIC-selected ARMA lags in the mean equation for the benchmark
expectation-adjusted AIS specification, EGARCH(1,1) in the variance equation,
standardized Student-t innovations, and lagged sentiment in both mean and
variance equations. Lagged raw AIS is estimated as a robustness specification
using the same asset-specific ARMA lag order selected for the benchmark model.

## Plain EGARCH For TVP-VAR Inputs

The TVP-VAR volatility inputs are estimated separately:

```bash
Rscript r/modeling/run_egarch_volatility.R
```

This script fits plain EGARCH(1,1) models with `armaOrder = c(0, 0)`, no
sentiment regressors, and standardized Student-t innovations. These conditional
volatilities are written to
`results/tables/tvpvar_connectedness/tvpvar_connectedness_dataset.csv`.

## TVP-VAR Connectedness

First select TVP-VAR lag order by BIC:

```bash
python scripts/reporting/make_tvpvar_connectedness_tables.py --lag-selection-only
```

Then run all configured systems and horizons using those selected lags:

```bash
python scripts/modeling/run_tvpvar_connectedness_models.py
```

The configured systems are:

- `benchmark`: Bitcoin and NASDAQ-100 conditional volatilities.
- `ai_equity`: Bitcoin, NVIDIA, Alphabet, and Microsoft conditional volatilities.

The benchmark horizon is `H=10`; robustness uses `H=100`. The underlying R
connectedness script can also be run directly:

```bash
Rscript r/modeling/run_tvpvar_connectedness.R \
  results/tables/tvpvar_connectedness/tvpvar_connectedness_dataset.csv \
  results/tables/tvpvar_connectedness/benchmark_h10 \
  1 \
  10 \
  bitcoin_conditional_volatility,ndx_conditional_volatility \
  results/models/tvpvar_connectedness/benchmark_h10
```

The direct arguments are:

1. input CSV;
2. output directory;
3. TVP-VAR lag order;
4. forecast horizon;
5. optional comma-separated volatility columns;
6. optional RDS output directory.
