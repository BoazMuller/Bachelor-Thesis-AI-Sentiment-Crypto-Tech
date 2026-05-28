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
