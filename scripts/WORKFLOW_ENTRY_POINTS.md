# Workflow Entry Points

Put repeatable command-line entry points here, such as:

- data download scripts
- data cleaning pipelines
- model training scripts
- figure and table export scripts

Scripts should import reusable functions from `code/src/thesis/` where possible.

## Layout

- `collection/`: raw data collection and source extraction.
- `cleaning/`: source-specific cleaning and text preparation.
- `preparation/`: analysis datasets and model input files.
- `validation/`: data quality and model-readiness checks.
- `modeling/`: Python model execution.
- `reporting/`: table and figure generation.

## Common entry points

- `collection/collect_yahoo_finance.py`: downloads selected daily Yahoo Finance price series.
- `cleaning/merge_reddit_submission_csvs.py`: merges filtered Reddit CSV files.
- `cleaning/prepare_roberta_text_input.py`: prepares deduplicated text for RoBERTa.
- `preparation/build_combined_time_series.py`: builds the processed thesis time series.
- `preparation/prepare_egarch_inputs.py`: writes `results/tables/armax_egarchx/armax_egarchx_dataset.csv`.
- `preparation/prepare_tvpvar_inputs.py`: writes `results/tables/tvpvar_connectedness/tvpvar_connectedness_return_dataset.csv`.
- `validation/check_time_series_data.py`: exports data validation tables.
- `modeling/run_roberta_sentiment.py`: runs chunked RoBERTa sentiment inference.
- `modeling/run_tvpvar_connectedness_models.py`: runs TVP-VAR connectedness systems using BIC-selected lags.
- `reporting/make_armax_egarchx_model_tables.py`: generates Tables 5-6 from ARMAX-EGARCHX outputs.
- `reporting/make_tvpvar_connectedness_tables.py`: generates Tables 7-13, TVP-VAR BIC lag selection, and the connectedness regression dataset.
- `reporting/plot_tvpvar_connectedness_timeseries.py`: generates TVP-VAR total and NET connectedness figures.
- `reporting/make_thesis_tables.py`: generates Python-buildable thesis tables.

## Model workflow order

```bash
python scripts/preparation/prepare_egarch_inputs.py
Rscript r/modeling/run_armax_egarchx.R
python scripts/reporting/make_armax_egarchx_model_tables.py

python scripts/preparation/prepare_tvpvar_inputs.py
Rscript r/modeling/run_egarch_volatility.R
python scripts/reporting/make_tvpvar_connectedness_tables.py --lag-selection-only
python scripts/modeling/run_tvpvar_connectedness_models.py
python scripts/reporting/make_tvpvar_connectedness_tables.py
python scripts/reporting/plot_tvpvar_connectedness_timeseries.py
python scripts/reporting/make_spillover_regression_tables.py
```
