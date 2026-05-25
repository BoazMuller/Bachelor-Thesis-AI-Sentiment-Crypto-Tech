# Scripts

Put repeatable command-line entry points here, such as:

- data download scripts
- data cleaning pipelines
- model training scripts
- figure and table export scripts

Scripts should import reusable functions from `code/src/thesis/` where possible.

## Available scripts

- `collect_yahoo_finance.py`: downloads selected daily Yahoo Finance price series
  to `data/raw/finance/` and writes a metadata JSON file alongside the CSV.
