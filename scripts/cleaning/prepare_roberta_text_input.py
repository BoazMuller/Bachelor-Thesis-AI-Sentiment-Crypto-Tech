"""
Prepare Reddit and GDELT text fields for RoBERTa sentiment classification.

Usage:
    python scripts/cleaning/prepare_roberta_text_input.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from thesis.paths import INTERIM_DATA_DIR, RAW_DATA_DIR  # noqa: E402


DEFAULT_REDDIT_CSV = INTERIM_DATA_DIR / "reddit_merged.csv"
DEFAULT_GDELT_CSV = (
    RAW_DATA_DIR / "gdelt" / "gdelt_ai_headlines_100_per_day_2024_04_01_to_2026_04_01.csv"
)
DEFAULT_OUTPUT_CSV = INTERIM_DATA_DIR / "combined_dedup_df.csv"

URL_PATTERN = re.compile(
    r"""(?ix)
    \b(
        https?://\S+ |
        www\.\S+
    )
    """
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare deduplicated Reddit and GDELT text for RoBERTa sentiment.",
    )
    parser.add_argument("--reddit-csv", type=Path, default=DEFAULT_REDDIT_CSV)
    parser.add_argument("--gdelt-csv", type=Path, default=DEFAULT_GDELT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def clean_text_for_sentiment(text: object, replace_urls: bool = True) -> str:
    if pd.isna(text):
        return ""

    cleaned = str(text)
    cleaned = URL_PATTERN.sub("<URL>" if replace_urls else "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def prepare_text_input(reddit_csv: Path, gdelt_csv: Path) -> pd.DataFrame:
    reddit_df = pd.read_csv(reddit_csv)
    gdelt_df = pd.read_csv(gdelt_csv)

    gdelt_clean = (
        gdelt_df.rename(columns={"gdelt_date": "date", "headline": "text"})[["date", "text"]]
        .reset_index(drop=True)
    )
    gdelt_clean["id"] = "gdelt_" + gdelt_clean.index.astype(str)
    gdelt_clean["source"] = "gdelt"

    reddit_clean = (
        reddit_df.rename(columns={"created_day": "date", "full_text": "text"})[["date", "text"]]
        .reset_index(drop=True)
    )
    reddit_clean["id"] = "reddit_" + reddit_clean.index.astype(str)
    reddit_clean["source"] = "reddit"

    combined = pd.concat([gdelt_clean, reddit_clean], ignore_index=True)
    deduped = combined[["date", "text", "source"]].drop_duplicates().copy()
    deduped["text_clean"] = deduped["text"].apply(clean_text_for_sentiment)
    return deduped[["date", "text_clean", "source"]]


def main() -> None:
    args = parse_args()
    output = prepare_text_input(args.reddit_csv, args.gdelt_csv)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(output):,} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
