"""
collect_gdelt_ai_headlines.py

Purpose:
    Collect English-language GDELT news headlines related to artificial intelligence
    from Google BigQuery, dated 2024-04-01 through 2026-04-01, with a maximum of
    100 pseudo-randomly selected headlines per day.

What this script does:
    1. Queries the public GDELT 2.1 Global Knowledge Graph table in BigQuery.
    2. Filters records to the date range 2024-04-01 through 2026-04-01.
    3. Excludes translated GDELT records by filtering out GKGRECORDID values
       containing "-T".
    4. Extracts article titles from the GDELT Extras field using PAGE_TITLE.
    5. Keeps only rows where the headline/title itself contains one of the AI keywords.
    6. Caps the result at MAX_ARTICLES_PER_DAY headlines per day.
    7. Saves the final headlines, sources, dates, URLs, and metadata to
       `data/raw/gdelt/` by default.
    8. Prints daily article counts so you can check coverage.

Important:
    This version does NOT visit every article URL.
    That makes it much faster than scraping titles from websites.

Cost note:
    BigQuery has a monthly free tier, but large queries can exceed it.
    This script performs a dry run first and stops if the estimated query size is above
    MAX_BYTES_BILLED.

Requirements:
    pip install -r requirements.txt

Authentication:
    gcloud auth application-default login
    gcloud config set project YOUR_PROJECT_ID

Usage:
    python scripts/collect_gdelt_ai_headlines.py --project-id YOUR_PROJECT_ID
"""

import argparse
import html
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from google.cloud import bigquery


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"

DEFAULT_PROJECT_ID = (
    os.environ.get("GOOGLE_CLOUD_PROJECT")
    or os.environ.get("GCLOUD_PROJECT")
)

START_DATE = "2024-04-01"
END_DATE = "2026-04-01"

OUTPUT_CSV = (
    DEFAULT_OUTPUT_DIR
    / "gdelt_ai_headlines_100_per_day_2024_04_01_to_2026_04_01.csv"
)

# This is the key setting:
# maximum number of headlines to keep per day.
MAX_ARTICLES_PER_DAY = 100

# Safety cap.
MAX_BYTES_BILLED = 700 * 1024**3

KEYWORDS = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "large language model",
    "large language models",
    "llm",
    "llms",
    "chatgpt",
    "openai",
    "claude",
    "anthropic",
    "gemini",
    "bard",
    "copilot",
    "generative ai",
    "genai",
    "ai model",
    "ai models",
    "ai chatbot",
    "ai chatbots",
    "ai-generated",
    "agi",
]


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def build_keyword_regex_for_bigquery(keywords: list[str]) -> str:
    """
    Build a stricter regex for matching keywords in headlines.

    Short or ambiguous terms get word boundaries so they do not match inside
    other words. For example:
        bard should not match Bardon
        llm should not match a random longer word
    """
    patterns = []

    for keyword in keywords:
        keyword_lower = keyword.lower()
        escaped = re.escape(keyword_lower)

        if keyword_lower in {
            "llm",
            "llms",
            "agi",
            "bard",
            "claude",
            "gemini",
            "copilot",
            "openai",
            "chatgpt",
            "anthropic",
            "genai",
        }:
            patterns.append(rf"\b{escaped}\b")
        else:
            patterns.append(escaped)

    return "(" + "|".join(patterns) + ")"


def clean_headline(value: Optional[str]) -> Optional[str]:
    """
    Clean a headline/title string.
    """
    if not value or not isinstance(value, str):
        return None

    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value)
    value = value.strip()

    if not value:
        return None

    return value


def bytes_to_gib(num_bytes: int) -> float:
    """
    Convert bytes to GiB.
    """
    return num_bytes / 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect AI-related GDELT headlines from BigQuery.",
    )
    parser.add_argument(
        "--project-id",
        default=DEFAULT_PROJECT_ID,
        help=(
            "Google Cloud project ID used for BigQuery billing. Defaults to "
            "GOOGLE_CLOUD_PROJECT, GCLOUD_PROJECT, or the Google Cloud SDK default."
        ),
    )
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTPUT_CSV,
        help="Path for the collected headline CSV.",
    )
    parser.add_argument(
        "--max-articles-per-day",
        type=int,
        default=MAX_ARTICLES_PER_DAY,
    )
    parser.add_argument(
        "--max-bytes-billed",
        type=int,
        default=MAX_BYTES_BILLED,
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print("Step 1/5: Building keyword pattern...")
    keyword_regex = build_keyword_regex_for_bigquery(KEYWORDS)

    print("Step 2/5: Connecting to BigQuery...")
    client = bigquery.Client(project=args.project_id)

    query = """
    WITH extracted AS (
      SELECT
        DATE,
        SourceCommonName,
        DocumentIdentifier AS url,
        GKGRECORDID,

        -- Extract article title from GDELT Extras.
        REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>([^<]*)</PAGE_TITLE>') AS gdelt_page_title,

        V2Themes,
        V2Persons,
        V2Organizations,
        V2Locations,
        AllNames,
        V2Tone
      FROM `gdelt-bq.gdeltv2.gkg_partitioned`
      WHERE
        DATE(_PARTITIONTIME) BETWEEN DATE(@start_date) AND DATE(@end_date)

        -- Exclude GDELT translated records.
        AND NOT REGEXP_CONTAINS(GKGRECORDID, r"-T")

        -- Keep web article URLs only.
        AND STARTS_WITH(DocumentIdentifier, "http")
    ),

    filtered AS (
      SELECT
        *,
        DATE(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING))) AS article_day
      FROM extracted
      WHERE
        gdelt_page_title IS NOT NULL

        -- Important: only keep headlines containing an AI keyword.
        AND REGEXP_CONTAINS(LOWER(gdelt_page_title), @keyword_regex)

        -- Remove obvious bad titles.
        AND LENGTH(gdelt_page_title) >= 15
        AND LOWER(gdelt_page_title) NOT IN (
          "domain parked",
          "access denied",
          "page not found",
          "404 not found"
        )
    ),

    ranked AS (
      SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY article_day
            ORDER BY FARM_FINGERPRINT(url)
        ) AS row_num_for_day
      FROM filtered
    )

    SELECT
      DATE,
      SourceCommonName,
      gdelt_page_title,
      url,
      GKGRECORDID,
      V2Themes,
      V2Persons,
      V2Organizations,
      V2Locations,
      AllNames,
      V2Tone
    FROM ranked
    WHERE row_num_for_day <= @max_articles_per_day
    ORDER BY DATE ASC
    """

    query_parameters = [
        bigquery.ScalarQueryParameter("start_date", "STRING", args.start_date),
        bigquery.ScalarQueryParameter("end_date", "STRING", args.end_date),
        bigquery.ScalarQueryParameter("keyword_regex", "STRING", keyword_regex),
        bigquery.ScalarQueryParameter(
            "max_articles_per_day",
            "INT64",
            args.max_articles_per_day,
        ),
    ]

    print("Step 3/5: Estimating BigQuery query size...")

    dry_run_config = bigquery.QueryJobConfig(
        query_parameters=query_parameters,
        dry_run=True,
        use_query_cache=False,
    )

    dry_run_job = client.query(query, job_config=dry_run_config)
    estimated_bytes = dry_run_job.total_bytes_processed

    print()
    print("BigQuery dry-run estimate")
    print("-------------------------")
    print(f"Estimated bytes processed: {estimated_bytes:,}")
    print(f"Estimated GiB processed:   {bytes_to_gib(estimated_bytes):,.2f} GiB")
    print(f"Safety cap GiB:            {bytes_to_gib(args.max_bytes_billed):,.2f} GiB")
    print()

    if estimated_bytes > args.max_bytes_billed:
        raise RuntimeError(
            "The estimated query size is larger than MAX_BYTES_BILLED. "
            "Use a shorter date range or increase MAX_BYTES_BILLED only after "
            "you are comfortable with the possible BigQuery usage."
        )

    print("Step 4/5: Running BigQuery query...")

    run_config = bigquery.QueryJobConfig(
        query_parameters=query_parameters,
        maximum_bytes_billed=args.max_bytes_billed,
        use_query_cache=True,
    )

    query_job = client.query(query, job_config=run_config)
    df = query_job.to_dataframe()

    print(f"Rows returned before Python cleaning: {len(df):,}")

    if df.empty:
        print("No rows found. Try expanding the date range or checking the keywords.")
        return

    print("Step 5/5: Cleaning and saving results...")

    # Clean headline.
    df["headline"] = df["gdelt_page_title"].apply(clean_headline)

    # Convert GDELT date to readable date.
    df["gdelt_date_raw"] = df["DATE"]
    df["gdelt_date"] = pd.to_datetime(
        df["DATE"].astype(str).str[:8],
        format="%Y%m%d",
        errors="coerce",
    )

    # Remove rows where headline cleaning failed.
    df = df[df["headline"].notna()].copy()

    # Remove duplicates.
    df = df.drop_duplicates(subset=["headline", "url"])

    # Sort by date.
    df = df.sort_values(["gdelt_date", "SourceCommonName", "headline"])

    # Clean output columns.
    output_columns = [
        "gdelt_date",
        "SourceCommonName",
        "headline",
        "url",
        "GKGRECORDID",
        "V2Themes",
        "V2Persons",
        "V2Organizations",
        "V2Locations",
        "AllNames",
        "V2Tone",
    ]

    df = df[output_columns]

    # Save final CSV.
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print()
    print(f"Saved: {args.output_csv}")
    print()

    # Daily coverage diagnostics.
    daily_counts = df.groupby("gdelt_date").size().reset_index(name="news_count")

    print("Daily coverage summary:")
    print(daily_counts.describe())
    print()

    print("Lowest-count days:")
    print(daily_counts.sort_values("news_count").head(20))
    print()

    print("Highest-count days:")
    print(daily_counts.sort_values("news_count", ascending=False).head(20))
    print()

    print("Preview:")
    print(df[["gdelt_date", "SourceCommonName", "headline", "url"]].head(20))


if __name__ == "__main__":
    main()
