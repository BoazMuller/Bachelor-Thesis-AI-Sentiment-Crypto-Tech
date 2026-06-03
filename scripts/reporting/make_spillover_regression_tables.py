"""
Build thesis Tables 25-29 from a connectedness-regression dataset.

The input should contain a date column, one or more connectedness-dependent
variables, and can optionally contain sentiment/control variables. If sentiment
variables are absent, the script merges them from the processed time-series CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from thesis.paths import MODELS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT as THESIS_ROOT, TABLES_DIR  # noqa: E402
from thesis.table_output import write_dataframe  # noqa: E402
from thesis.tables.common import add_sentiment_measures, read_daily_time_series  # noqa: E402
from thesis.tables.spillover_regressions import (  # noqa: E402
    infer_dependent_columns,
    merge_sentiment_if_needed,
    table_25_dataset_summary,
    table_26_correlation_matrix,
    table_27_baseline_regressions,
    table_28_lagged_regressions,
    table_29_raw_ais_regressions,
)


DEFAULT_CONNECTEDNESS_CSV = TABLES_DIR / "tvpvar_connectedness" / "connectedness_regression_dataset.csv"
DEFAULT_FALLBACK_CONNECTEDNESS_CSV = MODELS_DIR / "tvpvar_connectedness" / "connectedness_regression_dataset.csv"
DEFAULT_TIME_SERIES_CSV = PROCESSED_DATA_DIR / "combined_time_series.csv"
DEFAULT_OUTPUT_DIR = TABLES_DIR / "spillover_regressions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate spillover regression tables.")
    parser.add_argument("--connectedness-csv", type=Path, default=DEFAULT_CONNECTEDNESS_CSV)
    parser.add_argument("--fallback-connectedness-csv", type=Path, default=DEFAULT_FALLBACK_CONNECTEDNESS_CSV)
    parser.add_argument("--time-series-csv", type=Path, default=DEFAULT_TIME_SERIES_CSV)
    parser.add_argument("--dependent-columns", nargs="*", default=None)
    parser.add_argument("--cov-type", default="HC3")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    connectedness_csv = args.connectedness_csv if args.connectedness_csv.exists() else args.fallback_connectedness_csv
    if not connectedness_csv.exists():
        raise FileNotFoundError(
            f"Connectedness regression dataset not found: {args.connectedness_csv}. "
            "Create this from the TVP-VAR connectedness output first."
        )

    dataset = read_daily_time_series(connectedness_csv)
    time_series = add_sentiment_measures(read_daily_time_series(args.time_series_csv))
    dataset = merge_sentiment_if_needed(dataset, time_series)
    dependent_columns = args.dependent_columns or infer_dependent_columns(dataset)
    if not dependent_columns:
        raise ValueError("No dependent connectedness variables found. Pass --dependent-columns explicitly.")

    baseline = table_27_baseline_regressions(dataset, dependent_columns, args.cov_type)
    lagged = table_28_lagged_regressions(dataset, dependent_columns, args.cov_type)
    raw_ais = table_29_raw_ais_regressions(dataset, dependent_columns, args.cov_type)
    coefficients = _combine_coefficients(baseline, lagged, raw_ais)
    outputs = [
        write_dataframe(dataset, args.output_dir / "spillover_regressions_dataset.csv"),
        write_dataframe(table_25_dataset_summary(dataset, dependent_columns), args.output_dir / "spillover_regressions_descriptives.csv"),
        write_dataframe(table_26_correlation_matrix(dataset, dependent_columns), args.output_dir / "spillover_regressions_pre_estimation_diagnostics.csv"),
        write_dataframe(coefficients, args.output_dir / "spillover_regressions_coefficients.csv"),
        write_dataframe(_regression_results_summary(coefficients), args.output_dir / "spillover_regressions_results.csv"),
        write_dataframe(lagged, args.output_dir / "spillover_regressions_lagged_sentiment_robustness_results.csv"),
        write_dataframe(raw_ais, args.output_dir / "spillover_regressions_raw_ais_robustness_results.csv"),
    ]

    for path in outputs:
        print(path.relative_to(THESIS_ROOT))


def _combine_coefficients(*tables):
    import pandas as pd

    frames = [table for table in tables if not table.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _regression_results_summary(coefficients):
    columns = [
        "specification",
        "dependent_variable",
        "r_squared",
        "adjusted_r_squared",
        "nobs",
        "covariance_type",
    ]
    available = [column for column in columns if column in coefficients.columns]
    return coefficients[available].drop_duplicates().reset_index(drop=True)


if __name__ == "__main__":
    main()
