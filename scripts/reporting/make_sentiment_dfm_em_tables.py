"""
Build thesis Tables 14-18 for source-level sentiment coverage and DFM-EM AIS validation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from thesis.paths import FIGURES_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT as THESIS_ROOT, TABLES_DIR  # noqa: E402
from thesis.table_output import write_dataframe  # noqa: E402
from thesis.tables.common import read_daily_time_series  # noqa: E402
from thesis.tables.sentiment_dfm_em import (  # noqa: E402
    dfm_em_dataset,
    dfm_em_pre_estimation_diagnostics,
    make_dfm_em_factor_comparison_plots,
    table_15_daily_sentiment_descriptives_by_source,
    table_17_dynamic_factor_results,
    table_18_ais_construction_validation,
)


DEFAULT_TIME_SERIES_CSV = PROCESSED_DATA_DIR / "combined_time_series.csv"
DEFAULT_SENTIMENT_RECORDS_CSV = PROCESSED_DATA_DIR / "combined_dedup_roberta_sentiment.csv"
DEFAULT_OUTPUT_DIR = TABLES_DIR / "dfm_em"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis sentiment DFM-EM tables.")
    parser.add_argument("--time-series-csv", type=Path, default=DEFAULT_TIME_SERIES_CSV)
    parser.add_argument("--sentiment-records-csv", type=Path, default=DEFAULT_SENTIMENT_RECORDS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_daily_time_series(args.time_series_csv)
    results = table_17_dynamic_factor_results(df)
    outputs = [
        write_dataframe(dfm_em_dataset(df), args.output_dir / "dfm_em_dataset.csv"),
        write_dataframe(table_15_daily_sentiment_descriptives_by_source(df), args.output_dir / "dfm_em_descriptives.csv"),
        write_dataframe(dfm_em_pre_estimation_diagnostics(df), args.output_dir / "dfm_em_pre_estimation_diagnostics.csv"),
        write_dataframe(results, args.output_dir / "dfm_em_results.csv"),
        write_dataframe(results[results["metric"].eq("factor_loading")].reset_index(drop=True), args.output_dir / "dfm_em_coefficients.csv"),
        write_dataframe(table_18_ais_construction_validation(df), args.output_dir / "dfm_em_post_estimation_diagnostics.csv"),
    ]
    if not args.skip_figures:
        outputs.extend(make_dfm_em_factor_comparison_plots(df, FIGURES_DIR / "dfm_em"))

    for path in outputs:
        print(path.relative_to(THESIS_ROOT))


if __name__ == "__main__":
    main()
