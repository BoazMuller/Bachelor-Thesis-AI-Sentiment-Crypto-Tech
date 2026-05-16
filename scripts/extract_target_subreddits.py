"""
Extract target subreddits from Pushshift Reddit submission dumps.

This script streams monthly Pushshift Reddit submission dump files in `.zst`
format, filters records belonging to selected target subreddits, and writes
the matching records to compressed `.zst` output files. By default, raw
Pushshift submission dumps are read from `data/raw/reddit/submissions/`, and
filtered monthly files are written to `data/interim/reddit/target_subreddits/`.

The script is adapted from:
https://github.com/Watchful1/PushshiftDumps/blob/master/scripts/filter_file.py

The original repository, Watchful1/PushshiftDumps, provides example Python
scripts for processing Reddit dump files created by Pushshift and is licensed
under the MIT License.

Modifications in this version:
    - Filters only selected subreddit names.
    - Uses fast byte-level matching before optional timestamp filtering.
    - Supports optional filtering by `created_utc`.
    - Writes one compressed output file per input monthly dump.
    - Adds rotating file and console logging for reproducibility.

Research use:
    This script was used to extract Reddit submissions from selected
    AI- and technology-related subreddits for downstream research analysis.

Input:
    Pushshift Reddit submission dump files in `.zst` format.

Output:
    Filtered Reddit submission dump files in `.zst` format.

Dependencies:
    pip install -r requirements.txt

Usage:
    python scripts/extract_target_subreddits.py
    python scripts/extract_target_subreddits.py --input-folder /path/to/RS_dumps

Author modification:
    Adapted and modified for this research project.
"""

import argparse
import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

import zstandard


# ============================================================
# File settings
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FOLDER = PROJECT_ROOT / "data" / "raw"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "interim"
LOG_FOLDER = PROJECT_ROOT / "logs"


# ============================================================
# Subreddit filter settings
# ============================================================

TARGET_SUBREDDITS = {
    "technology",
    "artificial",
    "ArtificialInteligence",
}

TARGET_SUBREDDIT_BYTE_PATTERNS = [
    b'"subreddit":"technology"',
    b'"subreddit":"artificial"',
    b'"subreddit":"ArtificialInteligence"',
]


# ============================================================
# Optional date filter
# ============================================================

# Since your files are already monthly, you can leave this off.
USE_DATE_FILTER = False

FROM_DATE = datetime.strptime("2024-04-01", "%Y-%m-%d")
TO_DATE = datetime.strptime("2026-04-01", "%Y-%m-%d")


# ============================================================
# Logging setup
# ============================================================

log = logging.getLogger("extract_subreddits")
log.setLevel(logging.INFO)

log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
log.addHandler(console_handler)

LOG_FOLDER.mkdir(parents=True, exist_ok=True)

file_handler = logging.handlers.RotatingFileHandler(
    LOG_FOLDER / "extract_target_subreddits.log",
    maxBytes=1024 * 1024 * 16,
    backupCount=5,
    encoding="utf-8",
)
file_handler.setFormatter(log_formatter)
log.addHandler(file_handler)


# ============================================================
# Helper functions
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract selected subreddit submissions from Pushshift dumps.",
    )
    parser.add_argument(
        "--input-folder",
        type=Path,
        default=INPUT_FOLDER,
        help="Folder containing raw monthly RS_*.zst Pushshift submission dumps.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=OUTPUT_FOLDER,
        help="Folder for filtered .zst files.",
    )
    return parser.parse_args()


def normalize_filename(path: str) -> str:
    name = os.path.basename(path)

    if name.endswith(".zst"):
        name = name[:-4]

    return name


def is_monthly_rs_zst_file(path: str) -> bool:
    if os.path.isdir(path):
        return False

    name = os.path.basename(path).lower()

    if name.endswith(".part"):
        return False

    return name.startswith("rs_") and name.endswith(".zst")


def raw_line_might_be_target_subreddit(raw_line: bytes) -> bool:
    return any(pattern in raw_line for pattern in TARGET_SUBREDDIT_BYTE_PATTERNS)


def raw_line_date_is_in_range(raw_line: bytes) -> bool:
    """
    Optional safety filter by created_utc.

    This decodes/parses only the line substring around created_utc manually enough
    to avoid importing json for every line. You can leave USE_DATE_FILTER = False
    for monthly files.
    """
    try:
        marker = b'"created_utc":'
        idx = raw_line.find(marker)

        if idx == -1:
            return True

        start = idx + len(marker)
        end = start

        while end < len(raw_line) and raw_line[end:end + 1] in b"0123456789.":
            end += 1

        created_utc = float(raw_line[start:end])
        created = datetime.utcfromtimestamp(created_utc)

        return FROM_DATE <= created < TO_DATE

    except Exception:
        return True


def read_lines_zst_bytes(file_path: str):
    """
    Streams raw byte lines from a .zst compressed NDJSON file.
    Does not decode or parse JSON unless the line passes the fast subreddit check.
    """
    with open(file_path, "rb") as file_handle:
        dctx = zstandard.ZstdDecompressor(max_window_size=2**31)

        with dctx.stream_reader(file_handle) as reader:
            buffer = b""
            chunk_size = 1024 * 1024 * 64  # 64 MB

            while True:
                chunk = reader.read(chunk_size)

                if not chunk:
                    break

                buffer += chunk
                lines = buffer.split(b"\n")

                for line in lines[:-1]:
                    if line:
                        yield line

                buffer = lines[-1]

            if buffer:
                yield buffer


def write_line_zst(writer, raw_line: bytes):
    writer.write(raw_line)
    writer.write(b"\n")


def process_file(input_path: str, output_folder: str):
    file_name = normalize_filename(input_path)

    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(
        output_folder,
        f"{file_name}_target_subreddits.zst",
    )

    log.info("=" * 80)
    log.info(f"Input file:  {input_path}")
    log.info(f"Output file: {output_path}")

    total_lines = 0
    subreddit_matches = 0
    date_matches = 0

    compressor = zstandard.ZstdCompressor(level=3)

    with open(output_path, "wb") as output_handle:
        with compressor.stream_writer(output_handle) as writer:
            for raw_line in read_lines_zst_bytes(input_path):
                total_lines += 1

                if total_lines % 1_000_000 == 0:
                    log.info(
                        f"{file_name}: processed={total_lines:,}, "
                        f"subreddit_matches={subreddit_matches:,}, "
                        f"written={date_matches:,}"
                    )

                if not raw_line_might_be_target_subreddit(raw_line):
                    continue

                subreddit_matches += 1

                if USE_DATE_FILTER:
                    if not raw_line_date_is_in_range(raw_line):
                        continue

                date_matches += 1
                write_line_zst(writer, raw_line)

    log.info(
        f"Complete: {file_name} | "
        f"processed={total_lines:,}, "
        f"subreddit_matches={subreddit_matches:,}, "
        f"written={date_matches:,}"
    )


def find_input_files(input_folder: str) -> list[str]:
    files = []

    for name in os.listdir(input_folder):
        path = os.path.join(input_folder, name)

        if is_monthly_rs_zst_file(path):
            files.append(path)

    return sorted(files)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    log.info("Starting target subreddit extraction")
    log.info(f"Input folder: {args.input_folder}")
    log.info(f"Output folder: {args.output_folder}")
    log.info(f"Target subreddits: {sorted(TARGET_SUBREDDITS)}")
    log.info(f"Use date filter: {USE_DATE_FILTER}")

    if USE_DATE_FILTER:
        log.info(
            f"Date range: {FROM_DATE.strftime('%Y-%m-%d')} "
            f"to {TO_DATE.strftime('%Y-%m-%d')}"
        )

    if not args.input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {args.input_folder}")

    input_files = find_input_files(args.input_folder)

    if not input_files:
        log.warning("No matching RS_*.zst files found.")
        return

    log.info(f"Found {len(input_files)} file(s) to process:")

    for path in input_files:
        log.info(f"  - {path}")

    for path in input_files:
        process_file(path, args.output_folder)

    log.info("All done.")


if __name__ == "__main__":
    main()
