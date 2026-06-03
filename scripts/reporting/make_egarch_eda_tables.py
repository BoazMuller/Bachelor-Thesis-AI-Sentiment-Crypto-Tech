"""
Build thesis Tables 2-4 and ARMAX-EGARCHX EDA figures.
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
from thesis.tables.egarch_eda import (  # noqa: E402
    make_egarch_eda_figures,
    table_02a_full_sample_descriptives,
    table_02b_restricted_sample_descriptives,
    table_03_combined_pre_estimation_diagnostics,
)
from thesis.tables.model_inputs import prepare_egarch_full_input, prepare_egarch_restricted_input  # noqa: E402


DEFAULT_INPUT_CSV = PROCESSED_DATA_DIR / "combined_time_series.csv"
DEFAULT_OUTPUT_DIR = TABLES_DIR / "armax_egarchx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ARMAX-EGARCHX EDA tables and figures.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = read_daily_time_series(args.input_csv)
    df_full = prepare_egarch_full_input(source)
    df_restricted = prepare_egarch_restricted_input(source)

    outputs = [
        write_dataframe(df_full, args.output_dir / "armax_egarchx_full_dataset.csv"),
        write_dataframe(df_restricted, args.output_dir / "armax_egarchx_restricted_dataset.csv"),
        write_dataframe(df_restricted, args.output_dir / "armax_egarchx_dataset.csv"),
        write_dataframe(table_02a_full_sample_descriptives(df_full), args.output_dir / "armax_egarchx_full_descriptives.csv"),
        write_dataframe(table_02b_restricted_sample_descriptives(df_restricted), args.output_dir / "armax_egarchx_restricted_descriptives.csv"),
        write_dataframe(table_02b_restricted_sample_descriptives(df_restricted), args.output_dir / "armax_egarchx_descriptives.csv"),
        write_dataframe(table_03_combined_pre_estimation_diagnostics(df_full, df_restricted), args.output_dir / "armax_egarchx_pre_estimation_diagnostics.csv"),
    ]

    if not args.skip_figures:
        outputs.extend(make_egarch_eda_figures(source))

    for path in outputs:
        print(path.relative_to(THESIS_ROOT))


if __name__ == "__main__":
    main()
