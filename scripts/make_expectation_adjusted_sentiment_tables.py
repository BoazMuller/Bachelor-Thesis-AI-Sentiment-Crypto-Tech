"""
Build thesis Tables 19-24 for expectation-adjusted AIS construction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from thesis.paths import PROCESSED_DATA_DIR, PROJECT_ROOT as THESIS_ROOT  # noqa: E402
from thesis.table_output import write_registered_table  # noqa: E402
from thesis.thesis_tables import (  # noqa: E402
    read_daily_time_series,
    table_19_prediction_market_control_definitions,
    table_20_residual_regression_sample_alignment,
    table_21_correlation_matrix_multicollinearity,
    table_22_orthogonalization_regression_results,
    table_23_residual_ais_validation,
    table_24_raw_ais_versus_residual_ais,
)


DEFAULT_INPUT_CSV = PROCESSED_DATA_DIR / "combined_time_series.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate expectation-adjusted sentiment tables.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--cov-type", default="HC3", help="statsmodels covariance type for orthogonalization regressions.")
    parser.add_argument("--formats", nargs="+", default=["csv", "tex"], choices=["csv", "tex", "md"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_daily_time_series(args.input_csv)

    outputs = []
    outputs.extend(write_registered_table(19, table_19_prediction_market_control_definitions(), formats=args.formats))
    outputs.extend(write_registered_table(20, table_20_residual_regression_sample_alignment(df), formats=args.formats))
    outputs.extend(write_registered_table(21, table_21_correlation_matrix_multicollinearity(df), formats=args.formats))
    outputs.extend(write_registered_table(22, table_22_orthogonalization_regression_results(df, cov_type=args.cov_type), formats=args.formats))
    outputs.extend(write_registered_table(23, table_23_residual_ais_validation(df), formats=args.formats))
    outputs.extend(write_registered_table(24, table_24_raw_ais_versus_residual_ais(df), formats=args.formats))

    for path in outputs:
        print(path.relative_to(THESIS_ROOT))


if __name__ == "__main__":
    main()
