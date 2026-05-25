"""
Collect daily market data from Yahoo Finance.

The default date range is inclusive: 2024-04-01 through 2026-03-31.

Usage:
    python scripts/collect_yahoo_finance.py
    python scripts/collect_yahoo_finance.py --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from thesis.paths import RAW_DATA_DIR  # noqa: E402


START_DATE = "2024-04-01"
END_DATE = "2026-03-31"
DEFAULT_OUTPUT_CSV = (
    RAW_DATA_DIR / "finance" / "yahoo_finance_daily_2024_04_01_to_2026_03_31.csv"
)

SERIES_SPECS: dict[str, dict[str, str]] = {
    "ndx_adj_close": {
        "ticker": "^NDX",
        "field": "Adj Close",
        "description": "Nasdaq-100 adjusted close",
    },
    "bitcoin_adj_close": {
        "ticker": "BTC-USD",
        "field": "Adj Close",
        "description": "Bitcoin adjusted close in USD",
    },
    "nvda_adj_close": {
        "ticker": "NVDA",
        "field": "Adj Close",
        "description": "NVIDIA adjusted close",
    },
    "googl_adj_close": {
        "ticker": "GOOGL",
        "field": "Adj Close",
        "description": "Alphabet Class A adjusted close",
    },
    "msft_adj_close": {
        "ticker": "MSFT",
        "field": "Adj Close",
        "description": "Microsoft adjusted close",
    },
    "sp500_adj_close": {
        "ticker": "^GSPC",
        "field": "Adj Close",
        "description": "S&P 500 adjusted close",
    },
    "dxy_close": {
        "ticker": "DX-Y.NYB",
        "field": "Close",
        "description": "U.S. Dollar Index close",
    },
    "vix_close": {
        "ticker": "^VIX",
        "field": "Close",
        "description": "CBOE Volatility Index close",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect selected daily financial series from Yahoo Finance.",
    )
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path for the downloaded daily price CSV.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Path for metadata JSON. Defaults to OUTPUT_CSV with .metadata.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def validate_date_range(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start > end:
        raise ValueError("--start-date must be on or before --end-date.")

    return start, end


def metadata_path_for(output_csv: Path, metadata_json: Path | None) -> Path:
    if metadata_json is not None:
        return metadata_json

    return output_csv.with_name(f"{output_csv.stem}.metadata.json")


def validate_outputs(output_csv: Path, metadata_json: Path, force: bool) -> None:
    existing_paths = [path for path in (output_csv, metadata_json) if path.exists()]

    if existing_paths and not force:
        existing = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(f"Output already exists: {existing}. Pass --force to overwrite.")


def import_yfinance() -> Any:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: yfinance. Install it with `pip install yfinance` "
            "or update your environment from requirements.txt/environment.yml."
        ) from exc

    return yf


def download_yahoo_data(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    yf = import_yfinance()
    tickers = sorted({spec["ticker"] for spec in SERIES_SPECS.values()})
    yahoo_end = end + pd.Timedelta(days=1)

    downloaded = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=yahoo_end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        actions=False,
        group_by="column",
        progress=False,
        threads=True,
    )

    if downloaded.empty:
        raise ValueError("Yahoo Finance returned no rows for the requested date range.")

    return downloaded


def normalize_price_series(series: pd.Series) -> pd.Series:
    normalized = series.copy()
    normalized.index = pd.to_datetime(normalized.index).tz_localize(None).normalize()
    normalized = normalized.sort_index()

    if normalized.index.has_duplicates:
        normalized = normalized.groupby(level=0).last()

    return normalized


def select_price_series(downloaded: pd.DataFrame, ticker: str, field: str) -> pd.Series:
    if isinstance(downloaded.columns, pd.MultiIndex):
        if (field, ticker) in downloaded.columns:
            return normalize_price_series(downloaded[(field, ticker)])

        if (ticker, field) in downloaded.columns:
            return normalize_price_series(downloaded[(ticker, field)])

        available = [tuple(map(str, column)) for column in downloaded.columns]
        raise KeyError(f"Missing Yahoo column for {ticker} {field}. Available: {available}")

    if field not in downloaded.columns:
        raise KeyError(f"Missing Yahoo column: {field}. Available: {list(downloaded.columns)}")

    return normalize_price_series(downloaded[field])


def build_daily_prices(downloaded: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    date_index = pd.date_range(start=start, end=end, freq="D", name="date")
    prices = pd.DataFrame(index=date_index)

    for output_column, spec in SERIES_SPECS.items():
        prices[output_column] = select_price_series(
            downloaded=downloaded,
            ticker=spec["ticker"],
            field=spec["field"],
        ).reindex(date_index)

        if prices[output_column].notna().sum() == 0:
            raise ValueError(f"Downloaded series is entirely missing: {output_column}")

    return prices.reset_index()


def write_outputs(prices: pd.DataFrame, output_csv: Path, metadata_json: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)

    temp_csv = output_csv.with_name(f".{output_csv.stem}.tmp{output_csv.suffix}")
    temp_metadata = metadata_json.with_name(f".{metadata_json.stem}.tmp{metadata_json.suffix}")

    prices.to_csv(temp_csv, index=False)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Yahoo Finance via yfinance",
        "date_range": {
            "start_date": prices["date"].min().strftime("%Y-%m-%d"),
            "end_date": prices["date"].max().strftime("%Y-%m-%d"),
            "inclusive": True,
        },
        "rows": int(len(prices)),
        "series": SERIES_SPECS,
        "missing_values_by_column": {
            column: int(prices[column].isna().sum())
            for column in prices.columns
            if column != "date"
        },
        "note": (
            "The output uses a daily calendar and does not forward-fill missing values. "
            "Missing values are expected for equities and indices on non-trading days."
        ),
    }

    temp_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    os.replace(temp_csv, output_csv)
    os.replace(temp_metadata, metadata_json)


def main() -> None:
    args = parse_args()
    start, end = validate_date_range(args.start_date, args.end_date)
    output_csv = args.output_csv.resolve()
    metadata_json = metadata_path_for(output_csv, args.metadata_json).resolve()

    validate_outputs(output_csv, metadata_json, args.force)

    print(f"Downloading Yahoo Finance data from {start.date()} through {end.date()}...")
    downloaded = download_yahoo_data(start=start, end=end)
    prices = build_daily_prices(downloaded=downloaded, start=start, end=end)
    write_outputs(prices=prices, output_csv=output_csv, metadata_json=metadata_json)

    print(f"Wrote {len(prices):,} daily rows to {output_csv}")
    print(f"Wrote metadata to {metadata_json}")
    print("Missing values by column:")
    for column, missing_count in prices.drop(columns="date").isna().sum().items():
        print(f"  {column}: {missing_count:,}")


if __name__ == "__main__":
    main()
