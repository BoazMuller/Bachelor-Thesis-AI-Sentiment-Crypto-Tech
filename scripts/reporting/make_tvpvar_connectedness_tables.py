"""
Build thesis Tables 7-13 and the connectedness regression dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from thesis.paths import MODELS_DIR, PROJECT_ROOT as THESIS_ROOT  # noqa: E402
from thesis.table_output import write_dataframe, write_registered_table  # noqa: E402
from thesis.tables.armax_egarchx import (  # noqa: E402
    table_07_egarch_volatility_extraction_summary,
    table_08_conditional_volatility_descriptives,
)
from thesis.tables.tvpvar_connectedness import (  # noqa: E402
    build_connectedness_regression_dataset,
    read_volatility_panel,
    table_09_tvpvar_system_definition,
    table_10_tvpvar_lag_selection,
    table_11_average_connectedness,
    table_12_average_pairwise_connectedness_matrix,
    table_13_robustness_connectedness,
    tvpvar_lag_selection,
)


DEFAULT_MODELS_DIR = MODELS_DIR / "tvpvar_connectedness"
DEFAULT_VOLATILITY_CSV = DEFAULT_MODELS_DIR / "conditional_volatility_panel.csv"
DEFAULT_REGRESSION_DATASET = DEFAULT_MODELS_DIR / "connectedness_regression_dataset.csv"
DEFAULT_LAG_SELECTION_CSV = DEFAULT_MODELS_DIR / "tvpvar_lag_selection.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EGARCH volatility and TVP-VAR connectedness tables.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--volatility-csv", type=Path, default=DEFAULT_VOLATILITY_CSV)
    parser.add_argument("--maxlags", type=int, default=10)
    parser.add_argument("--lag-selection-only", action="store_true")
    parser.add_argument("--formats", nargs="+", default=["csv", "tex"], choices=["csv", "tex", "md"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    volatility = read_volatility_panel(args.volatility_csv)
    lag_selection = tvpvar_lag_selection(volatility, maxlags=args.maxlags)
    lag_selection_path = write_dataframe(lag_selection, args.models_dir / "tvpvar_lag_selection.csv")

    outputs = [lag_selection_path]
    outputs.extend(write_registered_table(9, table_09_tvpvar_system_definition(), formats=args.formats))
    outputs.extend(write_registered_table(10, table_10_tvpvar_lag_selection(lag_selection), formats=args.formats))

    if not args.lag_selection_only:
        outputs.extend(
            write_registered_table(
                7,
                table_07_egarch_volatility_extraction_summary(
                    args.models_dir / "egarch_volatility_coefficients.csv",
                    args.models_dir / "egarch_volatility_diagnostics.csv",
                ),
                formats=args.formats,
            )
        )
        outputs.extend(
            write_registered_table(
                8,
                table_08_conditional_volatility_descriptives(args.volatility_csv),
                formats=args.formats,
            )
        )
        outputs.extend(write_registered_table(11, table_11_average_connectedness(args.models_dir, horizon=10), formats=args.formats))
        outputs.extend(
            write_registered_table(
                12,
                table_12_average_pairwise_connectedness_matrix(args.models_dir, horizon=10),
                formats=args.formats,
            )
        )
        outputs.extend(write_registered_table(13, table_13_robustness_connectedness(args.models_dir), formats=args.formats))
        outputs.append(
            write_dataframe(
                build_connectedness_regression_dataset(args.models_dir, horizon=10),
                args.models_dir / "connectedness_regression_dataset.csv",
            )
        )

    for path in outputs:
        print(path.relative_to(THESIS_ROOT))


if __name__ == "__main__":
    main()
