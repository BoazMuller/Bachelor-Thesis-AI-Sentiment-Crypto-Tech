"""
Merge filtered Reddit submission CSV files into one analysis-ready dataset.

By default this script reads every CSV file in ``data/interim`` and writes a
single merged Reddit dataset to ``data/processed/reddit/reddit_dataset.csv``.

The merger is intentionally conservative:
    - CSV files are read with pandas so quoted multiline Reddit text is parsed
      correctly.
    - Inputs are sorted so repeated runs produce the same row order.
    - Column names are stripped of whitespace/BOM markers.
    - Accidental exported index columns, such as ``Unnamed: 0``, are removed.
    - Source filenames and paths are preserved.
    - Different schemas are merged with the union of all columns.

Usage:
    python3 scripts/cleaning/merge_reddit_submission_csvs.py
    python3 scripts/cleaning/merge_reddit_submission_csvs.py --input-folder data/interim
    python3 scripts/cleaning/merge_reddit_submission_csvs.py --output-file data/processed/reddit/my_file.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FOLDER = PROJECT_ROOT / "data" / "interim"
DEFAULT_OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "interim" / "reddit_merged.csv"
)

PROVENANCE_COLUMNS = ("source_file", "source_path")
ACCIDENTAL_INDEX_PREFIX = "Unnamed:"
CANONICAL_COLUMN_ORDER = [
    "subreddit",
    "id",
    "created_utc",
    "created_date",
    "created_day",
    "author",
    "score",
    "num_comments",
    "title",
    "selftext",
    "full_text",
    "url",
    "permalink",
    "matched_keywords",
    "source_file",
    "source_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Reddit submission CSV files into one dataset.",
    )
    parser.add_argument(
        "--input-folder",
        type=Path,
        default=DEFAULT_INPUT_FOLDER,
        help="Folder containing Reddit CSV files.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=(
            "Merged CSV output path. If a directory is supplied, "
            f"{DEFAULT_OUTPUT_FILE.name!r} is created inside it."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Filename pattern to merge from the input folder.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for CSV files recursively below the input folder.",
    )
    return parser.parse_args()


def resolve_output_file(output_file: Path) -> Path:
    output_file = Path(output_file)

    if output_file.suffix.lower() != ".csv":
        output_file = output_file / DEFAULT_OUTPUT_FILE.name

    return output_file


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def find_csv_files(
    input_folder: Path,
    pattern: str,
    recursive: bool,
    output_file: Path,
) -> list[Path]:
    input_folder = Path(input_folder)

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_folder}")

    iterator: Iterable[Path]
    if recursive:
        iterator = input_folder.rglob(pattern)
    else:
        iterator = input_folder.glob(pattern)

    output_file = output_file.resolve()
    csv_files = sorted(
        path.resolve()
        for path in iterator
        if path.is_file() and path.resolve() != output_file
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files matching {pattern!r} found in: {input_folder}"
        )

    return csv_files


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(column).replace("\ufeff", "").strip()
        for column in df.columns
    ]

    accidental_index_columns = [
        column
        for column in df.columns
        if column.startswith(ACCIDENTAL_INDEX_PREFIX)
    ]
    if accidental_index_columns:
        df = df.drop(columns=accidental_index_columns)

    return df


def protect_existing_provenance_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        column: f"input_{column}"
        for column in PROVENANCE_COLUMNS
        if column in df.columns
    }

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def read_reddit_csv(file_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig", low_memory=False)
    except pd.errors.EmptyDataError:
        print(f"Skipping empty CSV: {display_path(file_path)}")
        return pd.DataFrame()
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not parse CSV file {file_path}: {exc}") from exc
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin-1", low_memory=False)

    df = clean_columns(df)
    df = protect_existing_provenance_columns(df)

    df["source_file"] = file_path.name
    df["source_path"] = display_path(file_path)

    for column in ("id", "subreddit", "author"):
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()

    return df


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    canonical_columns = [
        column
        for column in CANONICAL_COLUMN_ORDER
        if column in df.columns
    ]
    remaining_columns = [
        column
        for column in df.columns
        if column not in canonical_columns
    ]

    return df[canonical_columns + remaining_columns]


def merge_csv_files(
    input_folder: Path,
    output_file: Path,
    pattern: str = "*.csv",
    recursive: bool = False,
) -> pd.DataFrame:
    output_file = resolve_output_file(output_file)
    csv_files = find_csv_files(input_folder, pattern, recursive, output_file)

    print(f"Found {len(csv_files)} CSV file(s):")
    for file_path in csv_files:
        print(f"  - {display_path(file_path)}")

    dataframes = []
    for file_path in csv_files:
        df = read_reddit_csv(file_path)
        if df.empty:
            continue

        print(f"Read {len(df):,} row(s): {file_path.name}")
        dataframes.append(df)

    if not dataframes:
        raise ValueError("No non-empty CSV files were available to merge.")

    merged_df = pd.concat(dataframes, ignore_index=True, sort=False)
    merged_df = order_columns(merged_df)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\nMerge summary:")
    print(f"  Output rows: {len(merged_df):,}")
    print(f"  Output columns: {len(merged_df.columns):,}")
    print(f"  Saved to: {display_path(output_file)}")

    return merged_df


def main() -> None:
    args = parse_args()
    merge_csv_files(
        input_folder=args.input_folder,
        output_file=args.output_file,
        pattern=args.pattern,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
