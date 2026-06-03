"""
Build thesis Tables 5-6 from ARMAX-EGARCHX model outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from thesis.paths import MODELS_DIR, PROJECT_ROOT as THESIS_ROOT, TABLES_DIR  # noqa: E402
from thesis.table_output import write_dataframe  # noqa: E402
from thesis.tables.armax_egarchx import (  # noqa: E402
    armax_egarchx_results,
    table_05_armax_egarchx_estimation_results,
    table_06_post_estimation_diagnostics,
)
from thesis.tables.common import read_daily_time_series  # noqa: E402
from thesis.tables.egarch_eda import (  # noqa: E402
    table_02a_full_sample_descriptives,
    table_02b_restricted_sample_descriptives,
    table_03_combined_pre_estimation_diagnostics,
)


DEFAULT_OUTPUT_DIR = TABLES_DIR / "armax_egarchx"
DEFAULT_FALLBACK_MODELS_DIR = MODELS_DIR / "armax_egarchx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ARMAX-EGARCHX estimation and diagnostics tables.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fallback-models-dir", type=Path, default=DEFAULT_FALLBACK_MODELS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    full_dataset_csv = args.output_dir / "armax_egarchx_full_dataset.csv"
    if not full_dataset_csv.exists():
        full_dataset_csv = _existing_path(args.output_dir / "armax_egarchx_dataset.csv", args.fallback_models_dir / "egarch_input_returns_sentiment.csv")
        
    restricted_dataset_csv = args.output_dir / "armax_egarchx_restricted_dataset.csv"
    if not restricted_dataset_csv.exists():
        restricted_dataset_csv = _existing_path(args.output_dir / "armax_egarchx_dataset.csv", args.fallback_models_dir / "egarch_input_returns_sentiment.csv")
        
    coefficients_csv = _existing_path(args.output_dir / "armax_egarchx_coefficients.csv", args.fallback_models_dir / "armax_egarchx_coefficients.csv")
    diagnostics_csv = args.output_dir / "armax_egarchx_post_estimation_diagnostics.csv"
    if not diagnostics_csv.exists():
        diagnostics_csv = _existing_path(args.output_dir / "armax_egarchx_diagnostics.csv", args.fallback_models_dir / "armax_egarchx_diagnostics.csv")
    lag_selection_csv = _existing_path(args.output_dir / "armax_egarchx_lag_selection.csv", args.fallback_models_dir / "armax_egarchx_lag_selection.csv")
    volatility_csv = _existing_path(args.output_dir / "armax_egarchx_conditional_volatility.csv", args.fallback_models_dir / "armax_egarchx_conditional_volatility.csv")
    
    df_full = read_daily_time_series(full_dataset_csv)
    df_restricted = read_daily_time_series(restricted_dataset_csv)

    outputs = [
        write_dataframe(df_full, args.output_dir / "armax_egarchx_full_dataset.csv"),
        write_dataframe(df_restricted, args.output_dir / "armax_egarchx_restricted_dataset.csv"),
        write_dataframe(df_restricted, args.output_dir / "armax_egarchx_dataset.csv"),
        write_dataframe(table_02a_full_sample_descriptives(df_full), args.output_dir / "armax_egarchx_full_descriptives.csv"),
        write_dataframe(table_02b_restricted_sample_descriptives(df_restricted), args.output_dir / "armax_egarchx_restricted_descriptives.csv"),
        write_dataframe(table_02b_restricted_sample_descriptives(df_restricted), args.output_dir / "armax_egarchx_descriptives.csv"),
        write_dataframe(table_03_combined_pre_estimation_diagnostics(df_full, df_restricted), args.output_dir / "armax_egarchx_pre_estimation_diagnostics.csv"),
        write_dataframe(table_05_armax_egarchx_estimation_results(coefficients_csv), args.output_dir / "armax_egarchx_coefficients.csv"),
        write_dataframe(armax_egarchx_results(diagnostics_csv), args.output_dir / "armax_egarchx_results.csv"),
        write_dataframe(table_06_post_estimation_diagnostics(diagnostics_csv), args.output_dir / "armax_egarchx_post_estimation_diagnostics.csv"),
    ]
    if lag_selection_csv.exists():
        outputs.append(write_dataframe(read_daily_or_plain_csv(lag_selection_csv), args.output_dir / "armax_egarchx_lag_selection.csv"))
    if volatility_csv.exists():
        outputs.append(write_dataframe(read_daily_or_plain_csv(volatility_csv), args.output_dir / "armax_egarchx_conditional_volatility.csv"))

    for path in outputs:
        print(path.relative_to(THESIS_ROOT))


def read_daily_or_plain_csv(path: Path):
    import pandas as pd

    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _existing_path(primary: Path, fallback: Path) -> Path:
    return primary if primary.exists() else fallback


if __name__ == "__main__":
    main()
