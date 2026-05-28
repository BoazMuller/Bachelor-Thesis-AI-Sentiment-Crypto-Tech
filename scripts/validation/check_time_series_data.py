"""
Create data inventory, descriptive diagnostics, and model-readiness checks.

The script reads the processed trading-day time-series dataset and writes CSV
tables under results/tables/data_validation/. It does not estimate thesis
models; it checks whether the data is suitable for those models.

Usage:
    python scripts/validation/check_time_series_data.py
    python scripts/validation/check_time_series_data.py --input-csv data/processed/combined_time_series.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "code" / "src"
sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/thesis_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/thesis_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

from thesis.paths import PROCESSED_DATA_DIR, TABLES_DIR  # noqa: E402
from thesis.data.inventory import variable_inventory  # noqa: E402
from thesis.modeling.diagnostics import (  # noqa: E402
    arch_lm_tests,
    autocorrelation_tests,
    correlation_matrix,
    date_integrity_checks,
    default_model_check_columns,
    default_var_groups,
    descriptive_statistics,
    missing_spans,
    missing_value_summary,
    outlier_summary,
    read_time_series,
    stationarity_tests,
    var_stability_checks,
)


DEFAULT_INPUT_CSV = PROCESSED_DATA_DIR / "combined_time_series.csv"
DEFAULT_OUTPUT_DIR = TABLES_DIR / "data_validation"
DEFAULT_INVENTORY_DIR = TABLES_DIR / "01_data_inventory"
TABLE_01_BASENAME = "table_01_data_sources_definitions_transformations_availability"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate data inventory and model-readiness checks.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Processed time-series CSV to validate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where validation tables will be written.",
    )
    parser.add_argument(
        "--var-maxlags",
        type=int,
        default=10,
        help="Maximum lag order considered for VAR stability checks.",
    )
    return parser.parse_args()


def write_table(df, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    df.to_csv(path, index=False)
    return path


def write_matrix(df, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    df.to_csv(path)
    return path


def write_markdown(df, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    headers = list(df.columns)
    rows = [
        "| " + " | ".join(_markdown_cell(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(_markdown_cell(row[column]) for column in headers) + " |")
    path.write_text("\n".join(rows) + "\n")
    return path


def write_latex(df, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(
        df.to_latex(
            index=False,
            escape=True,
            longtable=True,
            caption="Data sources, definitions, transformations, and availability",
            label="tab:data_inventory",
        )
    )
    return path


def _markdown_cell(value) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> None:
    args = parse_args()
    df = read_time_series(args.input_csv)
    check_columns = default_model_check_columns(df)
    return_columns = [column for column in check_columns if column.endswith("_log_return")]
    inventory = variable_inventory(df, project_root=PROJECT_ROOT)

    outputs = [
        write_table(inventory, DEFAULT_INVENTORY_DIR, f"{TABLE_01_BASENAME}.csv"),
        write_markdown(inventory, DEFAULT_INVENTORY_DIR, f"{TABLE_01_BASENAME}.md"),
        write_latex(inventory, DEFAULT_INVENTORY_DIR, f"{TABLE_01_BASENAME}.tex"),
        write_table(date_integrity_checks(df), args.output_dir, "date_integrity_checks.csv"),
        write_table(missing_value_summary(df), args.output_dir, "missing_values.csv"),
        write_table(missing_spans(df), args.output_dir, "missing_spans.csv"),
        write_table(descriptive_statistics(df), args.output_dir, "descriptive_statistics.csv"),
        write_table(outlier_summary(df), args.output_dir, "outliers.csv"),
        write_matrix(correlation_matrix(df, method="pearson"), args.output_dir, "correlation_pearson.csv"),
        write_matrix(correlation_matrix(df, method="spearman"), args.output_dir, "correlation_spearman.csv"),
        write_table(stationarity_tests(df, check_columns), args.output_dir, "stationarity_tests.csv"),
        write_table(autocorrelation_tests(df, check_columns), args.output_dir, "autocorrelation_ljungbox.csv"),
        write_table(arch_lm_tests(df, return_columns), args.output_dir, "arch_lm_tests.csv"),
        write_table(
            var_stability_checks(
                df=df,
                variable_groups=default_var_groups(),
                maxlags=args.var_maxlags,
            ),
            args.output_dir,
            "var_stability_checks.csv",
        ),
    ]

    print(f"Validated {args.input_csv}")
    print(f"Rows: {len(df):,}; columns: {len(df.columns):,}")
    print(f"Wrote {len(outputs):,} validation and inventory tables")
    for path in outputs:
        print(f"- {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
