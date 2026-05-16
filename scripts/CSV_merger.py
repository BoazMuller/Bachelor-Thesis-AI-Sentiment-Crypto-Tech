"""
CSV Merger for Filtered Reddit Submission Data

This script merges multiple filtered Reddit submission CSV files into one
combined CSV file for downstream analysis.

The input CSV files are expected to be outputs from `filter_pushshift_ai.py`.
By default, this script reads filtered CSV files from
`data/interim/reddit/ai_submissions/` and writes the merged analysis-ready
dataset to `data/processed/reddit/`.

The script:
    - Finds all `.csv` files in the configured input folder.
    - Reads each CSV file into a pandas DataFrame.
    - Adds a `source_file` column to preserve the origin of each row.
    - Concatenates all files into one combined dataset.
    - Saves the merged output as a single CSV file.

Note:
    Large merged datasets are not included in this GitHub repository.
    This script is provided so the merged dataset can be reproduced locally.

Requirements:
    pip install -r requirements.txt

Usage:
    python scripts/CSV_merger.py
    python scripts/CSV_merger.py --input-folder data/interim/reddit/ai_submissions
"""

import argparse
import glob
import os
from pathlib import Path

import pandas as pd


# ============================================================
# Folder settings
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FOLDER = PROJECT_ROOT / "data" / "interim"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed"


# ============================================================
# Merge CSV files
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge filtered Reddit submission CSV files.",
    )
    parser.add_argument(
        "--input-folder",
        type=Path,
        default=INPUT_FOLDER,
        help="Folder containing filtered Reddit CSV files.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=OUTPUT_FILE,
        help="Merged CSV output path.",
    )
    return parser.parse_args()


def merge_csv_files(input_folder: Path, output_file: Path):
    input_folder = Path(input_folder)
    output_file = Path(output_file)

    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_folder}")

    print(f"Found {len(csv_files)} CSV file(s):")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")

    dataframes = []

    for file in csv_files:
        print(f"Reading: {os.path.basename(file)}")

        df = pd.read_csv(
            file,
            encoding="utf-8",
            low_memory=False
        )

        # Keep track of which CSV each row came from
        df["source_file"] = os.path.basename(file)

        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)

    print(f"\nTotal merged rows: {len(merged_df):,}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    merged_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Merged file saved to:")
    print(output_file)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    merge_csv_files(args.input_folder, args.output_file)
