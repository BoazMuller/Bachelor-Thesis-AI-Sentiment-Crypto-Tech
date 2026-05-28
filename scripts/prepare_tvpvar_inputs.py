"""
Prepare clean return-panel inputs for first-stage volatility extraction before TVP-VAR.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from thesis.paths import MODELS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT as THESIS_ROOT  # noqa: E402
from thesis.table_output import write_dataframe  # noqa: E402
from thesis.thesis_tables import prepare_tvpvar_return_input, read_daily_time_series  # noqa: E402


DEFAULT_INPUT_CSV = PROCESSED_DATA_DIR / "combined_time_series.csv"
DEFAULT_OUTPUT_CSV = MODELS_DIR / "tvpvar_connectedness" / "first_stage_return_panel.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare return panel for first-stage EGARCH volatility extraction.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_daily_time_series(args.input_csv)
    output = write_dataframe(prepare_tvpvar_return_input(df), args.output_csv)
    print(output.relative_to(THESIS_ROOT))


if __name__ == "__main__":
    main()
