"""
Build thesis Tables 14-18 for source-level sentiment coverage and AIS PCA validation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from thesis.paths import PROCESSED_DATA_DIR, PROJECT_ROOT as THESIS_ROOT  # noqa: E402
from thesis.table_output import write_registered_table  # noqa: E402
from thesis.tables.common import read_daily_time_series  # noqa: E402
from thesis.tables.sentiment_pca import (  # noqa: E402
    table_14_text_data_coverage_by_source,
    table_15_daily_sentiment_descriptives_by_source,
    table_16_sentiment_source_correlation_matrix,
    table_17_pca_results,
    table_18_ais_construction_validation,
)


DEFAULT_TIME_SERIES_CSV = PROCESSED_DATA_DIR / "combined_time_series.csv"
DEFAULT_SENTIMENT_RECORDS_CSV = PROCESSED_DATA_DIR / "combined_dedup_roberta_sentiment.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis sentiment PCA tables.")
    parser.add_argument("--time-series-csv", type=Path, default=DEFAULT_TIME_SERIES_CSV)
    parser.add_argument("--sentiment-records-csv", type=Path, default=DEFAULT_SENTIMENT_RECORDS_CSV)
    parser.add_argument("--formats", nargs="+", default=["csv", "tex"], choices=["csv", "tex", "md"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_daily_time_series(args.time_series_csv)
    sentiment_records = pd.read_csv(args.sentiment_records_csv)

    outputs = []
    outputs.extend(write_registered_table(14, table_14_text_data_coverage_by_source(sentiment_records), formats=args.formats))
    outputs.extend(write_registered_table(15, table_15_daily_sentiment_descriptives_by_source(df), formats=args.formats))
    outputs.extend(write_registered_table(16, table_16_sentiment_source_correlation_matrix(df), formats=args.formats))
    outputs.extend(write_registered_table(17, table_17_pca_results(df), formats=args.formats))
    outputs.extend(write_registered_table(18, table_18_ais_construction_validation(df), formats=args.formats))

    for path in outputs:
        print(path.relative_to(THESIS_ROOT))


if __name__ == "__main__":
    main()
