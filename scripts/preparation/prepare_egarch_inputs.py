"""
Prepare clean Python-to-R inputs for ARMAX-EGARCHX estimation.
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
from thesis.tables.model_inputs import prepare_egarch_full_input, prepare_egarch_restricted_input  # noqa: E402


DEFAULT_INPUT_CSV = PROCESSED_DATA_DIR / "combined_time_series.csv"
DEFAULT_OUTPUT_CSV = TABLES_DIR / "armax_egarchx" / "armax_egarchx_dataset.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ARMAX-EGARCHX input CSV for R rugarch.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_daily_time_series(args.input_csv)
    output_dir = args.output_csv.parent
    
    full_df = prepare_egarch_full_input(df)
    restricted_df = prepare_egarch_restricted_input(df)
    
    out_full = write_dataframe(full_df, output_dir / "armax_egarchx_full_dataset.csv")
    out_restricted = write_dataframe(restricted_df, output_dir / "armax_egarchx_restricted_dataset.csv")
    # For safety/legacy compatibility:
    out_legacy = write_dataframe(restricted_df, args.output_csv)
    
    print(out_full.relative_to(THESIS_ROOT))
    print(out_restricted.relative_to(THESIS_ROOT))
    print(out_legacy.relative_to(THESIS_ROOT))


if __name__ == "__main__":
    main()
