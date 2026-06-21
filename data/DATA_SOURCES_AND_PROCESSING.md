# Data Sources And Processing

Keep original data immutable. If a raw file needs cleaning or transformation,
write code that reads from `data/raw/` and writes to `data/interim/` or
`data/processed/`.

## Folders

- `raw/`: Original data exactly as received or downloaded.
- `external/`: Third-party reference files used as inputs when needed.
- `interim/`: Intermediate outputs from cleaning and merging.
- `processed/`: Final analysis-ready datasets.

## Data Availability

All data sources used by this project are public. Raw, interim, and processed
data files are not committed because some inputs are large, credential-dependent,
or should be obtained directly from their original public source.

The manually labeled dataset used for validation or inspection is not included
in the repository. It can be obtained from the author on request, or recreated
by users from the public source data using the collection, filtering, and
processing steps documented here.

Users reproducing the workflow should expect to provide or regenerate the raw
files listed below, authenticate Google BigQuery for GDELT collection, and
review date ranges, source filters, keyword filters, and model specifications
before applying the code to a different sample or research question.

## Sources And Files

| File | Source | Role |
| --- | --- | --- |
| `data/raw/gdelt/gdelt_ai_headlines_100_per_day_2024_04_01_to_2026_04_01.csv` | GDELT 2.1 Global Knowledge Graph via Google BigQuery | Raw AI-related news headline dataset. |
| `data/interim/reddit_merged.csv` | Reddit submissions from Pushshift dumps | Merged Reddit submissions after subreddit, date, and keyword filtering. |
| `data/interim/combined_dedup_df.csv` | Derived from GDELT and Pushshift Reddit text | Deduplicated text input for RoBERTa sentiment classification. |
| `data/processed/combined_dedup_roberta_sentiment.csv` | Derived from `combined_dedup_df.csv` using CardiffNLP Twitter RoBERTa sentiment | Text-level sentiment probabilities and labels. |
| `data/raw/kalshi/Kalshi Prices.csv` | Kalshi API | Daily AGI-related market prices for contracts with deadlines before selected years. |
| `data/raw/metaculus/Metaculus_question_data.csv` | Metaculus API | Question metadata for the AGI date question. |
| `data/raw/metaculus/Metaculus_forecast_data.csv` | Metaculus API | Forecast history for the AGI date question. |
| `data/raw/finance/yahoo_finance_daily_2024_04_01_to_2026_03_31.csv` | Yahoo Finance via `yfinance` | Daily market prices and indices used as financial controls and trading-day calendar. |
| `data/raw/finance/yahoo_finance_daily_2024_04_01_to_2026_03_31.metadata.json` | Yahoo Finance via `yfinance` | Metadata for the Yahoo Finance download. |
| `data/raw/finance/All_Daily_Policy_Data.csv` | Economic Policy Uncertainty dataset | Daily EPU index. |
| `data/raw/finance/data_gpr_daily_recent.csv` | Economic Policy Uncertainty / GPR dataset | Daily geopolitical risk index. |
| `data/processed/combined_time_series.csv` | Derived from all sources above | Final trading-day time-series dataset for analysis. |

## Text Data Selection

### GDELT

Collected by `scripts/collection/collect_gdelt_ai_headlines.py`.

- Source table: public GDELT 2.1 GKG BigQuery table,
  `gdelt-bq.gdeltv2.gkg_partitioned`.
- Date range: `2024-04-01` through `2026-04-01`.
- Language/record filter: translated GDELT records are excluded by removing
  `GKGRECORDID` values containing `-T`.
- URL filter: only web article URLs are kept.
- Headline extraction: page titles are extracted from the GDELT `Extras` field
  using `PAGE_TITLE`.
- Keyword filter: the headline itself must contain at least one AI keyword.
- Sampling cap: at most 100 pseudo-random headlines per day, ordered by a stable
  hash of the URL.
- Duplicate handling: duplicate `(headline, url)` pairs are removed.

GDELT output columns kept:

- `gdelt_date`
- `SourceCommonName`
- `headline`
- `url`
- `GKGRECORDID`
- `V2Themes`
- `V2Persons`
- `V2Organizations`
- `V2Locations`
- `AllNames`
- `V2Tone`

### Reddit / Pushshift

Reddit data comes from Pushshift submission dumps and is processed by:

- `scripts/collection/extract_target_subreddits.py`
- `scripts/collection/filter_pushshift_ai.py`
- `scripts/cleaning/merge_reddit_submission_csvs.py`

Subreddit selection:

- `technology`
- `artificial`
- `ArtificialInteligence`

The extraction script uses fast byte-level subreddit matching on monthly
Pushshift `.zst` dumps. Its internal date filter is currently disabled because
the input files are monthly, but the configured study window is
`2024-04-01` through `2026-04-01`.

The filtering script then keeps submissions that satisfy:

- `created_utc >= 2024-04-01`
- `created_utc <= 2030-12-31` in the current script configuration
- subreddit is one of the target subreddits above
- `title` or `selftext` contains at least one AI keyword
- combined title/selftext has at least 5 characters after removing deleted or
  removed text markers

Reddit output columns kept:

- `subreddit`
- `id`
- `created_utc`
- `created_date`
- `created_day`
- `author`
- `score`
- `num_comments`
- `title`
- `selftext`
- `full_text`
- `url`
- `permalink`
- `matched_keywords`
- `source_file`
- `source_path`

### AI Keywords

The GDELT and Reddit filters use the same AI keyword list:

- `artificial intelligence`
- `machine learning`
- `deep learning`
- `large language model`
- `large language models`
- `llm`
- `llms`
- `chatgpt`
- `openai`
- `claude`
- `anthropic`
- `gemini`
- `bard`
- `copilot`
- `generative ai`
- `genai`
- `ai model`
- `ai models`
- `ai chatbot`
- `ai chatbots`
- `ai-generated`
- `agi`

For GDELT, short or ambiguous terms such as `llm`, `agi`, `bard`, `openai`, and
`chatgpt` are matched with word boundaries. The Reddit filter currently uses
case-insensitive substring matching.

## Sentiment Pipeline

Prepared by `scripts/cleaning/prepare_roberta_text_input.py` and
`scripts/modeling/run_roberta_sentiment.py`.

The text preparation script:

- reads GDELT headlines and Reddit submissions
- maps GDELT `gdelt_date` to `date`
- maps GDELT `headline` to `text`
- maps Reddit `created_day` to `date`
- maps Reddit `full_text` to `text`
- assigns `source = gdelt` or `source = reddit`
- keeps `date`, `text`, and `source`
- removes duplicate `(date, text, source)` rows
- replaces URLs with `<URL>`
- collapses repeated whitespace
- writes `date`, `text_clean`, and `source` to
  `data/interim/combined_dedup_df.csv`

The RoBERTa script uses
`cardiffnlp/twitter-roberta-base-sentiment-latest` by default and appends:

- `roberta_prob_negative`
- `roberta_prob_neutral`
- `roberta_prob_positive`
- `roberta_sentiment_label`
- `roberta_sentiment_score`
- `roberta_sentiment_compound`

`roberta_sentiment_compound` is defined as:

```text
roberta_prob_positive - roberta_prob_negative
```

The final trading-day dataset uses only `roberta_sentiment_compound` from the
RoBERTa output. The class probabilities, labels, and sentiment score remain in
the text-level sentiment file for diagnostics, but are not merged into
`combined_time_series.csv`.

## Financial And Forecast Data

### Yahoo Finance

Collected by `scripts/collection/collect_yahoo_finance.py`.

- Date range: `2024-04-01` through `2026-03-31`.
- Calendar: daily calendar, not only trading days.
- Missing values are not forward-filled in the raw Yahoo output.
- Missing values are expected for equities, indices, DXY, and VIX on
  non-trading days. Bitcoin has observations on all calendar days.

Yahoo Finance columns:

- `ndx_adj_close`: Nasdaq-100 adjusted close, ticker `^NDX`
- `bitcoin_adj_close`: Bitcoin adjusted close in USD, ticker `BTC-USD`
- `nvda_adj_close`: NVIDIA adjusted close, ticker `NVDA`
- `googl_adj_close`: Alphabet Class A adjusted close, ticker `GOOGL`
- `msft_adj_close`: Microsoft adjusted close, ticker `MSFT`
- `sp500_adj_close`: S&P 500 adjusted close, ticker `^GSPC`
- `dxy_close`: U.S. Dollar Index close, ticker `DX-Y.NYB`
- `vix_close`: CBOE Volatility Index close, ticker `^VIX`

### EPU And GPR

Prepared in `scripts/preparation/build_combined_time_series.py`.

- EPU is read from `All_Daily_Policy_Data.csv`.
- EPU date is constructed from `year`, `month`, and `day`.
- EPU keeps `date` and `daily_policy_index`, renamed to `epu`.
- GPR is read from `data_gpr_daily_recent.csv`.
- GPR keeps `date` and `GPRD`, renamed to `gpr`.
- Both are restricted to `2024-04-01` through `2026-03-31`.

### Kalshi

Prepared in `scripts/preparation/build_combined_time_series.py`.

- Source file: `data/raw/kalshi/Kalshi Prices.csv`.
- Source: Kalshi API.
- Timestamp is parsed as UTC and normalized to daily dates.
- The final dataset uses only the `Before 2030` contract price.
- This becomes `kalshi_before_2030`.
- Missing Kalshi values are filled on a daily calendar using the current
  `FILL_METHOD` in `scripts/preparation/build_combined_time_series.py`, which is currently `ffill`.
- Forward filling means missing days keep the most recently observed Kalshi
  price. Dates before the first observed `Before 2030` value remain missing.
- After filling, Kalshi values are matched to the final trading-day calendar.

### Metaculus

Prepared in `scripts/preparation/build_combined_time_series.py`.

- Source files:
  - `data/raw/metaculus/Metaculus_question_data.csv`
  - `data/raw/metaculus/Metaculus_forecast_data.csv`
- Source: Metaculus API.
- The question currently represented is Question ID `5121`:
  "When will the first general AI system be devised, tested, and publicly
  announced?"
- Forecast `Start Time` is parsed as UTC and normalized to a daily date.
- `Median` is parsed as a date.
- The main transformed variable is:

```text
days_until_median = Median - Start Time
```

- The difference is measured in days.
- The final dataset uses only the `recency_weighted` forecast source.
- For each date, the last intraday `recency_weighted` update is kept.
- The final dataset keeps only `days_until_median`, which becomes
  `metaculus_recency_weighted_days_until_median`.
- Missing values are filled on a daily calendar using the current `FILL_METHOD`,
  which is currently `ffill`.
- Forward filling is applied before restricting to the analysis window, so the
  first analysis date can use the most recently known Metaculus forecast from
  before `2024-04-01`.
- After filling, Metaculus values are matched to the final trading-day calendar.

## Final Time-Series Dataset

Created by `scripts/preparation/build_combined_time_series.py` and saved to
`data/processed/combined_time_series.csv`.

Main steps:

1. Restrict the analysis window to `2024-04-01` through `2026-03-31`.
2. Use dates with non-missing `sp500_adj_close` as the trading-day calendar.
3. Drop non-trading days from the final dataset.
4. Keep Yahoo Finance price/index levels.
5. Compute log returns for:
   - `ndx_adj_close`
   - `bitcoin_adj_close`
   - `nvda_adj_close`
   - `googl_adj_close`
   - `msft_adj_close`
   - `sp500_adj_close`
   - `dxy_close`
6. Keep `vix_close` in levels.
7. Keep `epu` and `gpr` in levels.
8. Aggregate text-level `roberta_sentiment_compound` to daily source-level
   averages.
9. Assign sentiment from non-trading days to the first next trading day.
10. Forward-fill Kalshi and Metaculus variables on a daily calendar, then match
    them to trading days.
11. Merge all series into one trading-day dataframe.

Log returns are computed as:

```text
log_return_t = log(price_t) - log(price_t-1)
```

The first trading day has missing log returns because there is no previous
trading-day observation inside the analysis window.

Daily source-level sentiment columns include only average
`roberta_sentiment_compound`:

- `sentiment_gdelt_sentiment_compound`
- `sentiment_reddit_sentiment_compound`

The final dataset uses only:

- `kalshi_before_2030`
- `metaculus_recency_weighted_days_until_median`

## Known Data Notes

- `combined_time_series.csv` currently has 502 trading-day rows and 22 columns
  from `2024-04-01` through `2026-03-31`.
- The final dataset has no weekend rows and no duplicate dates.
- GDELT has no processed sentiment observations from `2025-06-15` through
  `2025-07-01`. This creates missing GDELT sentiment values on the trading days
  inside that interval. A direct BigQuery diagnostic on
  `gdelt-bq.gdeltv2.gkg_partitioned` without the AI keyword filter also returned
  no rows for `2025-06-15` through `2025-07-01`, while nearby dates had raw GKG
  records and nearly all records had `PAGE_TITLE` values:

```text
article_day  n_rows  n_with_page_title
2025-06-10   146309  146296
2025-06-11   150717  150707
2025-06-12   128561  128549
2025-06-13   143319  143309
2025-06-14    60583   60580
2025-07-02   135772  135760
2025-07-03   145869  145857
2025-07-04   112248  112241
2025-07-05    73994   73985
2025-07-06    67115   67112
```

  This suggests the interval is an upstream GDELT/BigQuery coverage gap, not a
  result of the AI keyword filter or sentiment-processing scripts.
- Reddit sentiment has no missing calendar days over its processed date range.
- Kalshi and Metaculus use `FILL_METHOD = "ffill"`, so missing days are assigned
  the most recently known value. This is consistent with treating these series
  as state variables observed intermittently.
- Forward filling does not create values before the first observed update for a
  source. Therefore early trading days remain missing for `kalshi_before_2030`.
  Metaculus has pre-window forecast history, so its most recently known
  pre-window values are carried into `2024-04-01`.
- The `metaculus_prediction` forecast source is excluded from the final dataset
  because its `days_until_median` values are negative in this sample and are not
  meaningful for the intended "days until median" interpretation.
  `metaculus_recency_weighted_days_until_median` remains positive throughout the
  final dataset.
- Non-trading-day sentiment is assigned to the first next trading day. This is a
  clear timing rule, but it should be described explicitly in any empirical
  section because weekend news and Reddit posts are bundled into the following
  market day.
- Bitcoin trades on weekends, but the final dataset drops weekend Bitcoin
  observations to match the equity-market trading-day calendar.
- The Reddit keyword filter uses substring matching, so short terms may require
  manual inspection or stricter regex matching in a robustness check.

## Data Inventory And Validation Checks

The processed time-series dataset is validated by:

```bash
python scripts/validation/check_time_series_data.py
```

This script reads `data/processed/combined_time_series.csv` and writes the
thesis-facing data inventory to `results/tables/01_data_inventory/` and model
readiness checks to `results/tables/data_validation/`.

| Output | Contents |
| --- | --- |
| `01_data_inventory/table_01_data_sources_definitions_transformations_availability.csv` | Table 1: variable, symbol, source, raw/transformed frequency, transformation, expected sign or role, availability, observation counts, units, missing-value treatment, and notes. |
| `01_data_inventory/table_01_data_sources_definitions_transformations_availability.md` | Markdown version of Table 1 for quick inspection. |
| `01_data_inventory/table_01_data_sources_definitions_transformations_availability.tex` | LaTeX `longtable` version of Table 1 for thesis inclusion. |
| `date_integrity_checks.csv` | Date range, duplicate dates, sort order, weekend rows, and trading-calendar gaps. |
| `missing_values.csv` | Missing-value counts and percentages by variable. |
| `missing_spans.csv` | Contiguous missing-date intervals by variable. |
| `descriptive_statistics.csv` | Count, mean, standard deviation, min/max, and distribution percentiles for numeric variables. |
| `outliers.csv` | Outlier counts using 3x-IQR and robust MAD-based thresholds, plus min/max dates and values. |
| `correlation_pearson.csv` | Pearson correlation matrix for numeric variables. |
| `correlation_spearman.csv` | Spearman rank correlation matrix for numeric variables. |
| `stationarity_tests.csv` | ADF and KPSS checks for returns, controls, sentiment, and expectation variables. |
| `autocorrelation_ljungbox.csv` | Ljung-Box serial-correlation tests at lags 5, 10, and 20. |
| `arch_lm_tests.csv` | ARCH-LM heteroskedasticity tests for log-return variables. |
| `var_stability_checks.csv` | BIC-selected VAR lag orders and characteristic-root stability checks for selected variable groups. |

Current validation run on `combined_time_series.csv`:

- The dataset has 502 trading-day rows and 22 columns from `2024-04-01`
  through `2026-03-31`.
- Dates are sorted, unique, and contain no weekend rows. There are 114
  calendar gaps longer than one day, which is expected because the final
  dataset follows the equity-market trading calendar.
- Missing values are limited to:
  - first log-return observation for each return series;
  - `kalshi_before_2030` from `2024-04-01` through `2024-04-17`;
  - `sentiment_gdelt_sentiment_compound` from `2025-06-17` through
    `2025-07-01`.
- Return variables pass the stationarity checks: ADF rejects a unit root and
  KPSS does not reject level stationarity for the return series.
- Kalshi does not reject a unit root in the ADF test and rejects level
  stationarity in KPSS, so it should be treated carefully as a persistent
  expectation state variable. Metaculus also does not reject a unit root in ADF.
- Ljung-Box tests indicate serial correlation in NASDAQ-100, NVIDIA, and
  S&P 500 returns, and weaker longer-lag serial correlation in Bitcoin returns.
  This supports BIC-based ARMA lag selection before EGARCH estimation.
- ARCH-LM tests find ARCH effects in Bitcoin, NASDAQ-100, S&P 500, and DXY
  returns, with short-lag evidence for NVIDIA. This supports the EGARCH
  modelling step. Alphabet and Microsoft show weaker ARCH evidence in the
  current sample.
- The VAR stability check selected VAR(0) by BIC for the return panels, so
  there are no characteristic roots to evaluate for those panels. The controls
  panel selected VAR(1) and was stable, with roots outside the unit circle.
- Outlier diagnostics flag large market moves and control-variable spikes,
  especially around high-volatility dates. These should be inspected and
  documented, not automatically removed, because they may represent genuine
  market events relevant for volatility modelling.

Additional checks to revisit before final estimation:

- Confirm whether the early Kalshi missing period should be dropped from
  model-ready samples or handled with a documented pre-sample rule.
- Confirm that the GDELT gap remains an upstream data-coverage issue and not a
  processing error.
- Inspect the largest return outliers against known market/news events before
  deciding whether robustness specifications need winsorization.
- Re-run the validation script after constructing the two-source AIS,
  expectation-adjusted AIS, EGARCH conditional volatilities, and TVP-VAR input
  panels.
- For connectedness models, repeat stationarity, missingness, outlier, and VAR
  stability checks on the final conditional-volatility inputs, not only on raw
  returns.

## Model Artifacts And Table Dependencies

ARMAX-EGARCHX input is prepared by:

```bash
python scripts/preparation/prepare_egarch_inputs.py
```

The R ARMAX-EGARCHX model writes:

- `results/models/armax_egarchx/armax_egarchx_coefficients.csv`
- `results/models/armax_egarchx/armax_egarchx_diagnostics.csv`
- `results/models/armax_egarchx/armax_egarchx_lag_selection.csv`
- `results/models/armax_egarchx/armax_egarchx_conditional_volatility.csv`

These files generate Tables 5-6.

TVP-VAR volatility inputs are prepared by:

```bash
python scripts/preparation/prepare_tvpvar_inputs.py
Rscript r/modeling/run_egarch_volatility.R
```

The R EGARCH volatility script fits plain EGARCH(1,1) models with no ARMA terms
and no sentiment variables. It writes:

- `results/models/tvpvar_connectedness/conditional_volatility_panel.csv`
- `results/models/tvpvar_connectedness/egarch_volatility_long.csv`
- `results/models/tvpvar_connectedness/egarch_volatility_coefficients.csv`
- `results/models/tvpvar_connectedness/egarch_volatility_diagnostics.csv`

These files generate Tables 7-8.

TVP-VAR lag order is selected by BIC from the conditional-volatility systems:

```bash
python scripts/reporting/make_tvpvar_connectedness_tables.py --lag-selection-only
```

This writes `results/models/tvpvar_connectedness/tvpvar_lag_selection.csv` and
Table 10. The selected lags are then used by:

```bash
python scripts/modeling/run_tvpvar_connectedness_models.py
```

The TVP-VAR outputs are written under:

- `results/models/tvpvar_connectedness/benchmark_h10/`
- `results/models/tvpvar_connectedness/benchmark_h100/`
- `results/models/tvpvar_connectedness/ai_equity_h10/`
- `results/models/tvpvar_connectedness/ai_equity_h100/`

These outputs generate Tables 9-13 and
`results/models/tvpvar_connectedness/connectedness_regression_dataset.csv`,
which is the input for Tables 25-29.
