"""
Build thesis Tables 19-24 for expectation-adjusted AIS construction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from thesis.paths import PROCESSED_DATA_DIR, PROJECT_ROOT as THESIS_ROOT, TABLES_DIR  # noqa: E402
from thesis.table_output import write_dataframe  # noqa: E402
from thesis.tables.common import read_daily_time_series  # noqa: E402
from thesis.tables.expectation_adjusted_sentiment import (  # noqa: E402
    expectation_adjusted_dataset,
    expectation_adjusted_descriptives,
    expectation_adjusted_pre_estimation_diagnostics,
    table_21_correlation_matrix_multicollinearity,
    table_22_orthogonalization_regression_results,
    table_23_residual_ais_validation,
)


DEFAULT_INPUT_CSV = PROCESSED_DATA_DIR / "combined_time_series.csv"
DEFAULT_OUTPUT_DIR = TABLES_DIR / "expectation_adjusted_sentiment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate expectation-adjusted sentiment tables.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--cov-type", default="HC3", help="statsmodels covariance type for orthogonalization regressions.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_daily_time_series(args.input_csv)

    coefficients = table_22_orthogonalization_regression_results(df, cov_type=args.cov_type)
    outputs = [
        write_dataframe(expectation_adjusted_dataset(df), args.output_dir / "expectation_adjusted_sentiment_dataset.csv"),
        write_dataframe(expectation_adjusted_descriptives(df), args.output_dir / "expectation_adjusted_sentiment_descriptives.csv"),
        write_dataframe(expectation_adjusted_pre_estimation_diagnostics(df), args.output_dir / "expectation_adjusted_sentiment_pre_estimation_diagnostics.csv"),
        write_dataframe(table_21_correlation_matrix_multicollinearity(df), args.output_dir / "expectation_adjusted_sentiment_multicollinearity_diagnostics.csv"),
        write_dataframe(coefficients, args.output_dir / "expectation_adjusted_sentiment_coefficients.csv"),
        write_dataframe(_regression_results_summary(coefficients), args.output_dir / "expectation_adjusted_sentiment_results.csv"),
        write_dataframe(table_23_residual_ais_validation(df), args.output_dir / "expectation_adjusted_sentiment_post_estimation_diagnostics.csv"),
    ]

    for path in outputs:
        print(path.relative_to(THESIS_ROOT))


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
