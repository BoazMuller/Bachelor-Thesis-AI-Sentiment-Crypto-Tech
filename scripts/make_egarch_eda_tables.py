"""
Build thesis Tables 2-4 and ARMAX-EGARCHX EDA figures.
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
    add_sentiment_measures,
    make_egarch_eda_figures,
    read_daily_time_series,
    table_02_return_sentiment_descriptives,
    table_03_pre_estimation_diagnostics,
    table_04_arma_lag_order_selection,
)


DEFAULT_INPUT_CSV = PROCESSED_DATA_DIR / "combined_time_series.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ARMAX-EGARCHX EDA tables and figures.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--max-p", type=int, default=3)
    parser.add_argument("--max-q", type=int, default=3)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--formats", nargs="+", default=["csv", "tex"], choices=["csv", "tex", "md"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = add_sentiment_measures(read_daily_time_series(args.input_csv))

    outputs = []
    outputs.extend(write_registered_table(2, table_02_return_sentiment_descriptives(df), formats=args.formats))
    outputs.extend(write_registered_table(3, table_03_pre_estimation_diagnostics(df), formats=args.formats))
    outputs.extend(write_registered_table(4, table_04_arma_lag_order_selection(df, max_p=args.max_p, max_q=args.max_q), formats=args.formats))

    if not args.skip_figures:
        outputs.extend(make_egarch_eda_figures(df))

    for path in outputs:
        print(path.relative_to(THESIS_ROOT))


if __name__ == "__main__":
    main()
