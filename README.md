# Thesis Repository

This repository is a starting point for an econometrics and data science thesis. It is organized to keep source code, data, analysis notebooks, thesis writing, and generated outputs separate and reproducible.

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
├── docs/                 # Notes, literature summaries, planning
├── notebooks/            # Exploratory notebooks
├── references/           # Bibliography files and citation notes
├── results/              # Generated figures, tables, reports, models
├── scripts/              # Lifecycle command-line entry points
│   ├── collection/       # Raw data collection
│   ├── cleaning/         # Source-specific cleaning
│   ├── preparation/      # Analysis/model input datasets
│   ├── validation/       # Data and model-readiness checks
│   ├── modeling/         # Python model runs
│   └── reporting/        # Tables and figures
├── r/                    # R setup and R model workflows
└── thesis/               # Thesis manuscript files
```

## Recommended Workflow

1. Put untouched source data in `data/raw/`.
2. Put reusable logic in `code/src/thesis/` and thin workflow entry points in `scripts/`.
3. Save cleaned analysis-ready datasets in `data/processed/`.
4. Use notebooks for exploration, not as the only place where important transformations live.
5. Save generated figures and tables to `results/figures/` and `results/tables/`.
6. Keep manuscript text in `thesis/`.

## Reproducible Empirical Pipeline

Run commands from the project root after activating the Python environment.
Collection and RoBERTa inference are expensive and only need to be rerun when
their inputs change.

```bash
python scripts/preparation/build_combined_time_series.py
python scripts/preparation/prepare_egarch_inputs.py
python scripts/preparation/prepare_tvpvar_inputs.py

Rscript r/modeling/run_armax_egarchx.R
python scripts/reporting/make_armax_egarchx_model_tables.py

Rscript r/modeling/run_egarch_volatility.R
python scripts/reporting/make_tvpvar_connectedness_tables.py --lag-selection-only
python scripts/modeling/run_tvpvar_connectedness_models.py
python scripts/reporting/make_tvpvar_connectedness_tables.py

python scripts/reporting/make_spillover_regression_tables.py
```

The ARMAX-EGARCHX stage uses BIC-selected ARMA mean lags, EGARCH(1,1),
Student-t innovations, and lagged AI sentiment in both the mean and variance
equations. The TVP-VAR stage uses plain EGARCH(1,1) conditional volatilities
with no ARMA terms and no sentiment variables; TVP-VAR lag order is selected by
BIC separately for each volatility system.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e code
```

Run tests:

```bash
pytest
```

## Reproducibility Notes

- Large or sensitive datasets should stay out of Git.
- Document every raw data source in `data/README.md`.
- Prefer scripted pipelines over manual spreadsheet edits.
- Record package versions in `requirements.txt` or `environment.yml`.
- Use fixed random seeds for simulations, train/test splits, bootstraps, and model estimation where appropriate.
