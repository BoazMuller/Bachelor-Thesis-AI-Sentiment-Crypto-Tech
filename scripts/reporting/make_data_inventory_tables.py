"""
Build thesis Table 1 from the processed time-series dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from thesis.paths import PROCESSED_DATA_DIR, PROJECT_ROOT as THESIS_ROOT, TABLES_DIR  # noqa: E402
from thesis.table_output import write_dataframe  # noqa: E402
from thesis.tables.common import add_sentiment_measures, read_daily_time_series  # noqa: E402
from thesis.tables.data_inventory import table_01_data_inventory  # noqa: E402


DEFAULT_INPUT_CSV = PROCESSED_DATA_DIR / "combined_time_series.csv"
DEFAULT_OUTPUT_CSV = TABLES_DIR / "01_data_inventory" / "table_01_data_sources_definitions_transformations_availability.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis data-inventory table.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = add_sentiment_measures(read_daily_time_series(args.input_csv))
    output = write_dataframe(table_01_data_inventory(df), args.output_csv)
    print(output.relative_to(THESIS_ROOT))


if __name__ == "__main__":
    main()
