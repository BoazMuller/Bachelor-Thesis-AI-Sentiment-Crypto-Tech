"""
Build thesis Table 1 from the processed time-series dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from thesis.paths import PROCESSED_DATA_DIR, PROJECT_ROOT as THESIS_ROOT  # noqa: E402
from thesis.table_output import write_registered_table  # noqa: E402
from thesis.thesis_tables import read_daily_time_series, table_01_data_inventory  # noqa: E402


DEFAULT_INPUT_CSV = PROCESSED_DATA_DIR / "combined_time_series.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis data-inventory table.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--formats", nargs="+", default=["csv", "tex"], choices=["csv", "tex", "md"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_daily_time_series(args.input_csv)
    outputs = write_registered_table(1, table_01_data_inventory(df), formats=args.formats)
    for path in outputs:
        print(path.relative_to(THESIS_ROOT))


if __name__ == "__main__":
    main()
