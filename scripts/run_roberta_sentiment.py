"""
Run chunked RoBERTa sentiment classification on the combined text dataset.

The default model is CardiffNLP's Twitter RoBERTa sentiment model, which returns
negative, neutral, and positive probabilities for each text.

Usage:
    python scripts/run_roberta_sentiment.py
    python scripts/run_roberta_sentiment.py --chunksize 500 --model-batch-size 16
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from thesis.paths import INTERIM_DATA_DIR, PROCESSED_DATA_DIR  # noqa: E402
from thesis.sentiment import (  # noqa: E402
    RobertaSentimentClassifier,
    append_roberta_sentiment,
)


DEFAULT_INPUT_CSV = INTERIM_DATA_DIR / "combined_dedup_df.csv"
DEFAULT_OUTPUT_CSV = PROCESSED_DATA_DIR / "combined_dedup_roberta_sentiment.csv"
DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append RoBERTa sentiment probabilities to a CSV in chunks.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input CSV containing the text column to classify.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV with appended sentiment probability columns.",
    )
    parser.add_argument(
        "--text-column",
        default="text_clean",
        help="Name of the text column used for classification.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="Hugging Face model name or local model path.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=2_000,
        help="Number of CSV rows to read and write per chunk.",
    )
    parser.add_argument(
        "--model-batch-size",
        type=int,
        default=32,
        help="Number of texts to classify per model forward pass.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum tokenizer length. RoBERTa commonly uses up to 512 tokens.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, mps, etc.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the Hugging Face model only from the local cache.",
    )
    parser.add_argument(
        "--no-twitter-preprocessing",
        action="store_true",
        help="Disable Twitter-style URL and mention normalization before inference.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output CSV if it already exists.",
    )
    parser.add_argument(
        "--overwrite-input",
        action="store_true",
        help="Allow output-csv to be the same path as input-csv.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    input_csv = args.input_csv.resolve()
    output_csv = args.output_csv.resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive.")

    if args.model_batch_size <= 0:
        raise ValueError("--model-batch-size must be positive.")

    if input_csv == output_csv and not args.overwrite_input:
        raise ValueError(
            "Refusing to write over the input CSV without --overwrite-input. "
            "Use a different --output-csv or pass --overwrite-input."
        )

    if output_csv.exists() and not args.force and input_csv != output_csv:
        raise FileExistsError(
            f"Output CSV already exists: {output_csv}. Pass --force to overwrite it."
        )


def classify_csv(args: argparse.Namespace) -> None:
    input_csv = args.input_csv.resolve()
    output_csv = args.output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_csv = output_csv.with_name(f".{output_csv.stem}.tmp{output_csv.suffix}")

    if temp_csv.exists():
        temp_csv.unlink()

    classifier = RobertaSentimentClassifier(
        model_name=args.model_name,
        device=args.device,
        max_length=args.max_length,
        local_files_only=args.local_files_only,
    )

    print(f"Input: {input_csv}", flush=True)
    print(f"Output: {output_csv}", flush=True)
    print(f"Model: {args.model_name}", flush=True)
    print(f"Device: {classifier.device}", flush=True)
    print(f"CSV chunksize: {args.chunksize}", flush=True)
    print(f"Model batch size: {args.model_batch_size}", flush=True)
    print(f"Max token length: {args.max_length}", flush=True)

    total_rows = 0
    header = True

    for chunk_number, chunk in enumerate(
        pd.read_csv(input_csv, chunksize=args.chunksize),
        start=1,
    ):
        enriched_chunk = append_roberta_sentiment(
            chunk=chunk,
            classifier=classifier,
            text_column=args.text_column,
            model_batch_size=args.model_batch_size,
            apply_twitter_preprocessing=not args.no_twitter_preprocessing,
        )
        enriched_chunk.to_csv(temp_csv, mode="a", index=False, header=header)

        header = False
        total_rows += len(enriched_chunk)
        print(
            f"Finished chunk {chunk_number:,}; total rows written: {total_rows:,}",
            flush=True,
        )

    os.replace(temp_csv, output_csv)
    print(f"Done. Wrote {total_rows:,} rows to {output_csv}", flush=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    classify_csv(args)


if __name__ == "__main__":
    main()
