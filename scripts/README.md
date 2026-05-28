# Scripts

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
- `preparation/prepare_egarch_inputs.py`: writes ARMAX-EGARCHX inputs.
- `preparation/prepare_tvpvar_inputs.py`: writes first-stage return-panel inputs.
- `validation/check_time_series_data.py`: exports data validation tables.
- `modeling/run_roberta_sentiment.py`: runs chunked RoBERTa sentiment inference.
- `reporting/make_thesis_tables.py`: generates Python-buildable thesis tables.
