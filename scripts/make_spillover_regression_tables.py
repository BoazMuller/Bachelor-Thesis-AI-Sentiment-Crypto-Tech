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

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from thesis.paths import MODELS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT as THESIS_ROOT  # noqa: E402
from thesis.table_output import write_registered_table  # noqa: E402
from thesis.thesis_tables import (  # noqa: E402
    BASELINE_CONTROLS,
    DATE_COLUMN,
    EXPECTATION_ADJUSTED_AIS,
    LAGGED_EXPECTATION_ADJUSTED_AIS,
    RAW_AIS,
    add_sentiment_measures,
    descriptive_stats_table,
    read_daily_time_series,
    regression_table,
)


DEFAULT_CONNECTEDNESS_CSV = MODELS_DIR / "tvpvar_connectedness" / "connectedness_regression_dataset.csv"
DEFAULT_TIME_SERIES_CSV = PROCESSED_DATA_DIR / "combined_time_series.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate spillover regression tables.")
    parser.add_argument("--connectedness-csv", type=Path, default=DEFAULT_CONNECTEDNESS_CSV)
    parser.add_argument("--time-series-csv", type=Path, default=DEFAULT_TIME_SERIES_CSV)
    parser.add_argument("--dependent-columns", nargs="*", default=None)
    parser.add_argument("--cov-type", default="HC3")
    parser.add_argument("--formats", nargs="+", default=["csv", "tex"], choices=["csv", "tex", "md"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.connectedness_csv.exists():
        raise FileNotFoundError(
            f"Connectedness regression dataset not found: {args.connectedness_csv}. "
            "Create this from the TVP-VAR connectedness output first."
        )

    dataset = read_daily_time_series(args.connectedness_csv)
    time_series = add_sentiment_measures(read_daily_time_series(args.time_series_csv))
    dataset = merge_sentiment_if_needed(dataset, time_series)
    dependent_columns = args.dependent_columns or infer_dependent_columns(dataset)
    if not dependent_columns:
        raise ValueError("No dependent connectedness variables found. Pass --dependent-columns explicitly.")

    outputs = []
    outputs.extend(write_registered_table(25, table_25_dataset_summary(dataset, dependent_columns), formats=args.formats))
    outputs.extend(write_registered_table(26, table_26_correlation_matrix(dataset, dependent_columns), formats=args.formats))
    outputs.extend(write_registered_table(27, table_27_baseline_regressions(dataset, dependent_columns, args.cov_type), formats=args.formats))
    outputs.extend(write_registered_table(28, table_28_lagged_regressions(dataset, dependent_columns, args.cov_type), formats=args.formats))
    outputs.extend(write_registered_table(29, table_29_raw_vs_residual_comparison(dataset, dependent_columns, args.cov_type), formats=args.formats))

    for path in outputs:
        print(path.relative_to(THESIS_ROOT))


def merge_sentiment_if_needed(dataset: pd.DataFrame, time_series: pd.DataFrame) -> pd.DataFrame:
    required = [RAW_AIS, EXPECTATION_ADJUSTED_AIS, LAGGED_EXPECTATION_ADJUSTED_AIS]
    if all(column in dataset.columns for column in required):
        return dataset
    merge_columns = [DATE_COLUMN] + [
        column
        for column in [RAW_AIS, EXPECTATION_ADJUSTED_AIS, LAGGED_EXPECTATION_ADJUSTED_AIS] + BASELINE_CONTROLS
        if column in time_series.columns
    ]
    return dataset.merge(time_series[merge_columns], on=DATE_COLUMN, how="left")


def infer_dependent_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        DATE_COLUMN,
        RAW_AIS,
        EXPECTATION_ADJUSTED_AIS,
        LAGGED_EXPECTATION_ADJUSTED_AIS,
        *BASELINE_CONTROLS,
    }
    keywords = ("to", "from", "net", "npdc", "tci", "spillover", "connectedness")
    columns: list[str] = []
    for column in df.select_dtypes(include=[np.number]).columns:
        lower = column.lower()
        if column in excluded or column.endswith("_log_return"):
            continue
        if any(keyword in lower for keyword in keywords):
            columns.append(column)
    return columns


def table_25_dataset_summary(df: pd.DataFrame, dependent_columns: list[str]) -> pd.DataFrame:
    stats = descriptive_stats_table(df, dependent_columns)
    if DATE_COLUMN in df:
        sample_period = f"{df[DATE_COLUMN].min().date()} to {df[DATE_COLUMN].max().date()}"
        stats["sample_period"] = sample_period
    return stats


def table_26_correlation_matrix(df: pd.DataFrame, dependent_columns: list[str]) -> pd.DataFrame:
    variables = dependent_columns + [
        column
        for column in [EXPECTATION_ADJUSTED_AIS, RAW_AIS] + BASELINE_CONTROLS
        if column in df.columns
    ]
    corr = df[variables].apply(pd.to_numeric, errors="coerce").corr()
    corr.insert(0, "variable", corr.index)
    return corr.reset_index(drop=True)


def table_27_baseline_regressions(df: pd.DataFrame, dependent_columns: list[str], cov_type: str) -> pd.DataFrame:
    regressors = [column for column in [EXPECTATION_ADJUSTED_AIS] + BASELINE_CONTROLS if column in df.columns]
    return _multi_dependent_regressions(df, dependent_columns, regressors, "headline_contemporaneous", cov_type)


def table_28_lagged_regressions(df: pd.DataFrame, dependent_columns: list[str], cov_type: str) -> pd.DataFrame:
    regressors = [column for column in [LAGGED_EXPECTATION_ADJUSTED_AIS] + BASELINE_CONTROLS if column in df.columns]
    return _multi_dependent_regressions(df, dependent_columns, regressors, "robustness_lagged_sentiment", cov_type)


def table_29_raw_vs_residual_comparison(df: pd.DataFrame, dependent_columns: list[str], cov_type: str) -> pd.DataFrame:
    rows = []
    specifications = {
        "robustness_raw_ais": RAW_AIS,
        "headline_expectation_adjusted_ais": EXPECTATION_ADJUSTED_AIS,
    }
    for specification_name, sentiment_variable in specifications.items():
        regressors = [column for column in [sentiment_variable] + BASELINE_CONTROLS if column in df.columns]
        table = _multi_dependent_regressions(df, dependent_columns, regressors, specification_name, cov_type)
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _multi_dependent_regressions(
    df: pd.DataFrame,
    dependent_columns: list[str],
    regressors: list[str],
    specification_prefix: str,
    cov_type: str,
) -> pd.DataFrame:
    rows = []
    for dependent in dependent_columns:
        table = regression_table(
            df,
            dependent=dependent,
            specifications={f"{specification_prefix}_{dependent}": regressors},
            cov_type=cov_type,
        )
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


if __name__ == "__main__":
    main()
