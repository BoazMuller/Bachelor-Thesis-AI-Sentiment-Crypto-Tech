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

from thesis.paths import MODELS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT as THESIS_ROOT  # noqa: E402
from thesis.table_output import write_registered_table  # noqa: E402
from thesis.tables.common import add_sentiment_measures, read_daily_time_series  # noqa: E402
from thesis.tables.spillover_regressions import (  # noqa: E402
    infer_dependent_columns,
    merge_sentiment_if_needed,
    table_25_dataset_summary,
    table_26_correlation_matrix,
    table_27_baseline_regressions,
    table_28_lagged_regressions,
    table_29_raw_vs_residual_comparison,
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

if __name__ == "__main__":
    main()
