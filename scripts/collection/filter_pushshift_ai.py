"""
Filter Pushshift Reddit submissions for AI-related content and export results to CSV.

This script processes compressed Pushshift Reddit submission dump files in `.zst`
format. It reads each file line by line, parses each Reddit submission as JSON,
filters submissions by date, subreddit, and AI-related keyword matches, and writes
the matching records to CSV files for downstream research analysis. By default,
it reads the extracted subreddit `.zst` files from
`data/interim/reddit/target_subreddits/` and writes filtered CSV files to
`data/interim/reddit/ai_submissions/`.

The script is adapted from:
https://github.com/Watchful1/PushshiftDumps/blob/master/scripts/filter_file.py

The original repository, Watchful1/PushshiftDumps, provides example Python
scripts for processing Reddit dump files created by Pushshift and is licensed
under the MIT License.

Modifications in this version:
    - Parses compressed Pushshift `.zst` submission files.
    - Filters submissions from selected target subreddits.
    - Searches submission titles and selftexts for AI-related keywords.
    - Restricts results to a specified date range.
    - Exports selected metadata and full text to CSV.
    - Adds logging, error handling, and progress reporting.

Research use:
    This script was used to identify Reddit submissions related to artificial
    intelligence, machine learning, large language models, and generative AI
    within selected technology-related subreddits.

Input:
    Pushshift Reddit submission dump files in `.zst` format.

Output:
    One CSV file per input `.zst` file, containing only submissions that match
    the configured date, subreddit, and keyword filters.

Dependencies:
    pip install -r requirements.txt

Usage:
    python scripts/collection/filter_pushshift_ai.py
    python scripts/collection/filter_pushshift_ai.py --input-folder data/interim/reddit/target_subreddits

Author modification:
    Adapted and modified for this research project.
"""

import argparse
import csv
import json
import logging
import logging.handlers
import os
import traceback
from datetime import datetime
from pathlib import Path

import zstandard


# ============================================================
# File settings
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FOLDER = PROJECT_ROOT / "data" / "interim" 
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "interim" 
LOG_FOLDER = PROJECT_ROOT / "logs"


# ============================================================
# Filter settings
# ============================================================

FROM_DATE = datetime.strptime("2024-04-01", "%Y-%m-%d")
TO_DATE = datetime.strptime("2030-12-31", "%Y-%m-%d")
SEARCH_FIELDS = ["title", "selftext"]
TARGET_SUBREDDITS = {
    "technology",
    "artificial",
    "ArtificialInteligence",
}

# AI-related keywords.
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

# Output CSV columns.
CSV_COLUMNS = [
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
]

# ============================================================
# LOGGING SETUP
# ============================================================

log = logging.getLogger("pushshift_filter")
log.setLevel(logging.INFO)

log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
log.addHandler(console_handler)

LOG_FOLDER.mkdir(parents=True, exist_ok=True)

file_handler = logging.handlers.RotatingFileHandler(
    LOG_FOLDER / "filter_pushshift_ai.log",
    maxBytes=1024 * 1024 * 16,
    backupCount=5,
    encoding="utf-8",
)
file_handler.setFormatter(log_formatter)
log.addHandler(file_handler)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter extracted Pushshift submissions for AI-related content.",
    )
    parser.add_argument(
        "--input-folder",
        type=Path,
        default=INPUT_FOLDER,
        help="Folder containing extracted subreddit .zst files.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=OUTPUT_FOLDER,
        help="Folder for filtered CSV files.",
    )
    return parser.parse_args()


def connect_title_selftext(obj: dict) -> str:
    title = str(obj.get("title", "") or "").strip()
    selftext = str(obj.get("selftext", "") or "").strip()

    bad_values = {"[removed]", "[deleted]", "removed", "deleted", "nan", "none", "null"}

    title_clean = "" if title.lower() in bad_values else title
    selftext_clean = "" if selftext.lower() in bad_values else selftext

    combined = f"{title_clean}. {selftext_clean}".strip()

    # Remove extra whitespace
    combined = " ".join(combined.split())

    return combined


def normalize_filename(path: str) -> str:
    """
    Returns a clean base name for output files.
    """
    name = os.path.basename(path)

    if name.endswith(".zst"):
        name = name[:-4]

    return name


def is_probably_zst_file(path: str) -> bool:
    """
    Windows may hide the .zst extension, so this script accepts:
      *.zst
      files without extension

    It rejects:
      *.part
      directories
    """
    if os.path.isdir(path):
        return False

    lower = path.lower()

    if lower.endswith(".part"):
        return False

    if lower.endswith(".zst"):
        return True

    # Accept extensionless files because Windows may be hiding .zst,
    # or the file may have been saved without showing the extension.
    _, ext = os.path.splitext(path)
    if ext == "":
        return True

    return False


def read_lines_zst(file_path: str):
    """
    Streams lines from a .zst compressed newline-delimited JSON file.
    Does not load the whole file into memory.
    """
    with open(file_path, "rb") as file_handle:
        dctx = zstandard.ZstdDecompressor(max_window_size=2**31)

        with dctx.stream_reader(file_handle) as reader:
            buffer = b""
            chunk_size = 1024 * 1024 * 64  # 64 MB chunks

            while True:
                chunk = reader.read(chunk_size)

                if not chunk:
                    break

                buffer += chunk
                lines = buffer.split(b"\n")

                for line in lines[:-1]:
                    if line:
                        yield line.decode("utf-8", errors="replace")

                buffer = lines[-1]

            if buffer:
                yield buffer.decode("utf-8", errors="replace")


def find_matching_keywords(obj: dict, fields: list[str], keywords: list[str]) -> list[str]:
    """
    Searches selected fields for any of the keywords.
    Returns the list of matched keywords.
    """
    combined_text_parts = []

    for field in fields:
        value = obj.get(field)
        if value is not None:
            combined_text_parts.append(str(value).lower())

    combined_text = "\n".join(combined_text_parts)

    matched = []
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in combined_text:
            matched.append(keyword)

    return matched


def get_created_date(obj: dict):
    """
    Converts created_utc to a datetime object.
    Returns None if the value is missing or invalid.
    """
    created_utc = obj.get("created_utc")

    if created_utc is None:
        return None

    try:
        return datetime.utcfromtimestamp(int(float(created_utc)))
    except Exception:
        return None


def reddit_permalink(obj: dict) -> str:
    """
    Returns a full Reddit permalink if available.
    """
    permalink = obj.get("permalink")

    if permalink:
        if permalink.startswith("http"):
            return permalink
        return f"https://www.reddit.com{permalink}"

    subreddit = obj.get("subreddit", "")
    post_id = obj.get("id", "")

    if subreddit and post_id:
        return f"https://www.reddit.com/r/{subreddit}/comments/{post_id}/"

    return ""


def write_csv_header(writer):
    writer.writerow(CSV_COLUMNS)


def write_submission_row(writer, obj: dict, created: datetime, matched_keywords: list[str], full_text: str):
    writer.writerow([
        obj.get("subreddit", ""),
        obj.get("id", ""),
        obj.get("created_utc", ""),
        created.strftime("%Y-%m-%d %H:%M:%S"),
        created.strftime("%Y-%m-%d"),
        obj.get("author", ""),
        obj.get("score", ""),
        obj.get("num_comments", ""),
        obj.get("title", ""),
        obj.get("selftext", ""),
        full_text,
        obj.get("url", ""),
        reddit_permalink(obj),
        "; ".join(matched_keywords),
    ])


def process_file(input_path: str, output_folder: str):
    """
    Processes one Pushshift .zst file and writes a filtered CSV.
    """
    file_name = normalize_filename(input_path)

    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, f"{file_name}_ai_from_2024_04.csv")

    log.info("=" * 80)
    log.info(f"Input file:  {input_path}")
    log.info(f"Output file: {output_path}")

    total_lines = 0
    matched_lines = 0
    bad_lines = 0

    file_size = os.path.getsize(input_path)

    with open(output_path, "w", encoding="utf-8", newline="") as output_handle:
        writer = csv.writer(output_handle)
        write_csv_header(writer)

        for line in read_lines_zst(input_path):
            total_lines += 1

            if total_lines % 100000 == 0:
                log.info(
                    f"{file_name}: processed={total_lines:,}, "
                    f"matched={matched_lines:,}, bad={bad_lines:,}"
                )

            try:
                obj = json.loads(line)

                created = get_created_date(obj)
                if created is None:
                    bad_lines += 1
                    continue

                if created < FROM_DATE:
                    continue

                if created > TO_DATE:
                    continue
                if obj.get("subreddit", "") not in TARGET_SUBREDDITS:
                    continue
                full_text = connect_title_selftext(obj)

                if len(full_text) < 5:
                    continue

                matched_keywords = find_matching_keywords(
                    obj=obj,
                    fields=SEARCH_FIELDS,
                    keywords=KEYWORDS,
                )

                if not matched_keywords:
                    continue

                matched_lines += 1
                write_submission_row(writer, obj, created, matched_keywords, full_text)

            except json.JSONDecodeError:
                bad_lines += 1
            except Exception:
                bad_lines += 1
                log.warning(f"Unexpected error on line {total_lines:,}")
                log.warning(traceback.format_exc())

    log.info(
        f"Complete: {file_name} | "
        f"processed={total_lines:,}, matched={matched_lines:,}, bad={bad_lines:,}, "
        f"size={file_size / (1024 * 1024):.2f} MB"
    )


def find_input_files(input_folder: str) -> list[str]:
    """
    Finds files to process.
    Ignores .part files.
    """
    files = []

    for name in os.listdir(input_folder):
        path = os.path.join(input_folder, name)

        if not is_probably_zst_file(path):
            continue
        files.append(path)

    return sorted(files)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    log.info("Starting Pushshift AI filter")
    log.info(f"Input folder: {args.input_folder}")
    log.info(f"Output folder: {args.output_folder}")
    log.info(f"Date range: {FROM_DATE.strftime('%Y-%m-%d')} to {TO_DATE.strftime('%Y-%m-%d')}")
    log.info(f"Search fields: {SEARCH_FIELDS}")
    log.info(f"Keywords: {KEYWORDS}")

    if not args.input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {args.input_folder}")

    input_files = find_input_files(args.input_folder)

    if not input_files:
        log.warning("No matching input files found.")
        log.warning("Check whether your files are in the input folder and are not still .part files.")
        return

    log.info(f"Found {len(input_files)} file(s) to process:")

    for path in input_files:
        log.info(f"  - {path}")

    for path in input_files:
        try:
            process_file(path, args.output_folder)
        except Exception:
            log.error(f"Failed processing file: {path}")
            log.error(traceback.format_exc())

    log.info("All done.")


if __name__ == "__main__":
    main()
