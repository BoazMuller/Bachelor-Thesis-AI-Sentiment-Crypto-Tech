# Thesis Replication Workflow

This repository contains the Python and R workflow for the thesis empirical
analysis. It is organized to keep source code, raw data, processed data,
analysis notebooks, model outputs, and thesis tables separate and reproducible.

## Project Structure

```text
.
├── code/                 # Python package and tests
│   ├── src/thesis/       # Reusable project code
│   └── tests/            # Unit tests
├── config/               # Analysis configuration files
├── data/                 # Data storage, not committed by default
│   ├── raw/              # Original immutable data
│   ├── external/         # Third-party reference data
│   ├── interim/          # Intermediate transformed data
│   └── processed/        # Final datasets used for analysis
├── notebooks/            # Exploratory notebooks
├── r/                    # R setup and R model workflows
├── references/           # Bibliography files and citation notes
├── results/              # Generated figures, tables, reports, models
├── scripts/              # Lifecycle command-line entry points
│   ├── collection/       # Raw data collection
│   ├── cleaning/         # Source-specific cleaning
│   ├── preparation/      # Analysis/model input datasets
│   ├── validation/       # Data and model-readiness checks
│   ├── modeling/         # Python model runs
│   └── reporting/        # Tables and figures
└── environment.yml       # Conda environment alternative to requirements.txt
```

## Supporting Documentation

- `data/DATA_SOURCES_AND_PROCESSING.md`: raw data sources, transformations, validation notes, and model artifact dependencies.
- `scripts/WORKFLOW_ENTRY_POINTS.md`: script layout, common entry points, and model workflow order.
- `r/R_MODEL_WORKFLOW.md`: R dependency setup and volatility/connectedness model commands.
- `results/OUTPUT_ARTIFACTS.md`: generated output directory conventions.
- `notebooks/EXPLORATORY_NOTEBOOKS.md`: notebook usage conventions.

## Data Availability

This project uses public data sources only. Raw, interim, and processed data
files are not committed to the repository because some inputs are large,
credential-dependent, or better obtained directly from the original public
source. The expected file locations and processing steps are documented in
`data/DATA_SOURCES_AND_PROCESSING.md`.

The manually labeled dataset used for validation or inspection is not included
in the repository. It can be obtained from the author on request, or recreated
by users from the public source data using the documented collection and
filtering workflow.

## What Users Need To Adjust

Before running the workflow on another machine, users should review and adjust:

1. Raw data placement: put public-source inputs in the paths listed under
   "Place Or Collect Raw Inputs" below.
2. Google Cloud settings: replace `YOUR_PROJECT_ID` and authenticate BigQuery
   before collecting GDELT headlines.
3. Reddit input files: provide Pushshift monthly submission dumps under
   `data/raw/reddit/submissions/`.
4. External public files: download or export the EPU, GPR, Kalshi, and
   Metaculus files using the filenames expected by the scripts.
5. Study window choices: update date ranges in the collection and preparation
   scripts if extending or changing the sample period.
6. Source and keyword choices: review the subreddit list and AI keyword filter
   if applying the workflow to a different topic or scope.
7. Model specifications: treat EGARCH, ARMAX-EGARCHX, TVP-VAR horizons, HAC
   lag choices, and sentiment lag choices as research decisions that should be
   justified if changed.
8. Local environments: use Python 3.10+ and restore the R environment before
   running the model stages.

## Full Replication Workflow

Run commands from the project root. The collection and RoBERTa steps can be
expensive, so rerun them only when the corresponding raw inputs change.

### 1. Create The Python Environment

Use Python 3.10 or newer. The Conda environment in `environment.yml` uses
Python 3.11.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e code
```

### 2. Restore R Dependencies

```bash
Rscript r/setup_renv.R
```

### 3. Place Or Collect Raw Inputs

Some raw inputs are collected by scripts and some are external files that must
be placed in the expected directories before building the processed dataset.

Expected external files:

- `data/raw/finance/All_Daily_Policy_Data.csv`
- `data/raw/finance/data_gpr_daily_recent.csv`
- `data/raw/kalshi/Kalshi Prices.csv`
- `data/raw/metaculus/Metaculus_question_data.csv`
- `data/raw/metaculus/Metaculus_forecast_data.csv`
- Pushshift monthly Reddit submission dumps under `data/raw/reddit/submissions/`

These raw files are intentionally not committed. Scripts that depend on them
expect the paths above and will stop with a file-not-found error if the inputs
have not been placed or collected first.

Collect GDELT headlines with Google Cloud BigQuery credentials:

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
python scripts/collection/collect_gdelt_ai_headlines.py --project-id YOUR_PROJECT_ID
```

Collect Yahoo Finance prices:

```bash
python scripts/collection/collect_yahoo_finance.py
```

If the Yahoo output already exists and should be regenerated:

```bash
python scripts/collection/collect_yahoo_finance.py --force
```

Extract and filter Reddit submissions:

```bash
python scripts/collection/extract_target_subreddits.py \
  --input-folder data/raw/reddit/submissions \
  --output-folder data/interim/reddit/target_subreddits

python scripts/collection/filter_pushshift_ai.py \
  --input-folder data/interim/reddit/target_subreddits \
  --output-folder data/interim/reddit/ai_submissions

python scripts/cleaning/merge_reddit_submission_csvs.py \
  --input-folder data/interim/reddit/ai_submissions \
  --output-file data/interim/reddit_merged.csv
```

### 4. Build Text Sentiment Inputs

```bash
python scripts/cleaning/prepare_roberta_text_input.py
python scripts/modeling/run_roberta_sentiment.py
```

If the RoBERTa output already exists and should be regenerated:

```bash
python scripts/modeling/run_roberta_sentiment.py --force
```

### 5. Build And Validate The Processed Time Series (Validation Step 1 of 2)

```bash
python scripts/preparation/build_combined_time_series.py
python scripts/validation/check_time_series_data.py
```

### 6. Generate Python-Buildable Tables And EDA Figures

This builds Tables 1-4 and 14-24, plus the Python-generated EDA figures.

```bash
python scripts/reporting/make_thesis_tables.py
```

The same table groups can also be regenerated separately:

```bash
python scripts/reporting/make_data_inventory_tables.py
python scripts/reporting/make_egarch_eda_tables.py
python scripts/reporting/make_sentiment_dfm_em_tables.py
python scripts/reporting/make_expectation_adjusted_sentiment_tables.py
```

### 7. Run ARMAX-EGARCHX Models And Tables

```bash
python scripts/preparation/prepare_egarch_inputs.py
Rscript r/modeling/run_armax_egarchx.R
python scripts/reporting/make_armax_egarchx_model_tables.py
```

The ARMAX-EGARCHX stage uses BIC-selected ARMA mean lags for the
expectation-adjusted AIS benchmark specification, EGARCH(1,1), Student-t
innovations, and lagged AI sentiment in both the mean and variance equations.
Raw AIS robustness specifications reuse the same asset-specific benchmark ARMA
lag orders so the robustness comparison changes only the sentiment measure.

### 8. Run Plain EGARCH And TVP-VAR Connectedness Models

```bash
python scripts/preparation/prepare_tvpvar_inputs.py
Rscript r/modeling/run_egarch_volatility.R
python scripts/reporting/make_tvpvar_connectedness_tables.py --lag-selection-only
python scripts/modeling/run_tvpvar_connectedness_models.py
python scripts/reporting/make_tvpvar_connectedness_tables.py
python scripts/reporting/plot_tvpvar_connectedness_timeseries.py
python scripts/reporting/plot_connectedness_network.py
```

The TVP-VAR stage uses plain EGARCH(1,1) conditional volatilities with no ARMA
terms or sentiment regressors in the first-stage volatility models. It estimates
the original volatility systems and EAIS-augmented versions, selecting TVP-VAR
lag order by BIC separately for each system. The augmented results are reported
separately and are not used as dependent variables in downstream EAIS
regressions.

### 9. Generate Spillover Regression Tables

```bash
python scripts/reporting/make_spillover_regression_tables.py
```

The connectedness regressions use Newey-West HAC standard errors (`maxlags=5`)
to account for heteroskedasticity and serial correlation in daily connectedness
measures. The raw-AIS robustness output includes contemporaneous and one-day-
lagged raw AIS together; their high correlation should be considered when
interpreting the individual coefficients. The output also retains the one-lag
EAIS specification and adds EAIS terms from `t` through `t-5`, cumulative
effects, joint significance tests, lag correlations, VIFs, and residual
autocorrelation diagnostics.

### 10. Run Final Validation And Compile Final Data Inventory (Validation Step 2 of 2)

Once all models have been estimated and conditional volatility and connectedness measures have been generated, run the validation script again. This compiles the final, completed data inventory table containing observation counts and sample ranges for the conditional volatility and connectedness metrics.

```bash
python scripts/validation/check_time_series_data.py
```

### 11. Optional Checks

```bash
pytest
```

## License

The code and documentation in this repository are released under the MIT
License. Data remain subject to the terms of their original public sources.
