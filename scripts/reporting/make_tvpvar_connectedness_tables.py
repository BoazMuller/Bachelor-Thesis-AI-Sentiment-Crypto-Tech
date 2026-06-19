"""
Build thesis Tables 7-13 and the connectedness regression dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

import pandas as pd

from thesis.paths import MODELS_DIR, PROJECT_ROOT as THESIS_ROOT, TABLES_DIR  # noqa: E402
from thesis.table_output import write_dataframe  # noqa: E402
from thesis.tables.tvpvar_connectedness import (  # noqa: E402
    build_connectedness_regression_dataset,
    read_volatility_panel,
    table_09_tvpvar_system_definition,
    table_10_tvpvar_lag_selection,
    table_11_average_connectedness,
    table_12_average_pairwise_connectedness_matrix,
    table_13_robustness_connectedness,
    tvpvar_descriptives,
    tvpvar_lag_selection,
    tvpvar_pre_estimation_diagnostics,
)


DEFAULT_OUTPUT_DIR = TABLES_DIR / "tvpvar_connectedness"
DEFAULT_MODELS_DIR = MODELS_DIR / "tvpvar_connectedness"
DEFAULT_VOLATILITY_CSV = DEFAULT_OUTPUT_DIR / "tvpvar_connectedness_dataset.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EGARCH volatility and TVP-VAR connectedness tables.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fallback-models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--volatility-csv", type=Path, default=DEFAULT_VOLATILITY_CSV)
    parser.add_argument("--maxlags", type=int, default=10)
    parser.add_argument("--lag-selection-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    volatility_csv = _existing_path(args.volatility_csv, args.fallback_models_dir / "conditional_volatility_panel.csv")
    volatility = read_volatility_panel(volatility_csv)
    lag_selection = tvpvar_lag_selection(volatility, maxlags=args.maxlags)
    source_dir = _connectedness_source_dir(args.source_dir, args.fallback_models_dir)

    outputs = [
        write_dataframe(
            table_09_tvpvar_system_definition(),
            args.output_dir / "tvpvar_connectedness_system_definition.csv",
        ),
        write_dataframe(volatility, args.output_dir / "tvpvar_connectedness_dataset.csv"),
        write_dataframe(tvpvar_descriptives(volatility), args.output_dir / "tvpvar_connectedness_descriptives.csv"),
        write_dataframe(tvpvar_pre_estimation_diagnostics(volatility), args.output_dir / "tvpvar_connectedness_pre_estimation_diagnostics.csv"),
        write_dataframe(table_10_tvpvar_lag_selection(lag_selection), args.output_dir / "tvpvar_connectedness_lag_selection.csv"),
    ]

    if not args.lag_selection_only:
        coefficients_csv = _existing_path(args.output_dir / "egarch_volatility_coefficients.csv", args.fallback_models_dir / "egarch_volatility_coefficients.csv")
        diagnostics_csv = _existing_path(args.output_dir / "egarch_volatility_post_estimation_diagnostics.csv", args.fallback_models_dir / "egarch_volatility_diagnostics.csv")
        if coefficients_csv.exists():
            outputs.append(write_dataframe(pd.read_csv(coefficients_csv), args.output_dir / "tvpvar_connectedness_coefficients.csv"))
        if diagnostics_csv.exists():
            outputs.append(write_dataframe(pd.read_csv(diagnostics_csv), args.output_dir / "egarch_volatility_post_estimation_diagnostics.csv"))
        outputs.append(write_dataframe(table_11_average_connectedness(source_dir, horizon=10), args.output_dir / "tvpvar_connectedness_results.csv"))
        outputs.append(write_dataframe(table_12_average_pairwise_connectedness_matrix(source_dir, horizon=10), args.output_dir / "tvpvar_connectedness_pairwise_results.csv"))
        outputs.append(
            write_dataframe(
                table_12_average_pairwise_connectedness_matrix(source_dir, horizon=100),
                args.output_dir / "tvpvar_connectedness_pairwise_robustness_results.csv",
            )
        )
        outputs.append(write_dataframe(table_13_robustness_connectedness(source_dir), args.output_dir / "tvpvar_connectedness_robustness_results.csv"))
        outputs.extend(_write_connectedness_components(source_dir, args.output_dir))
        outputs.extend(_write_tvpvar_diagnostics(source_dir, args.output_dir))
        outputs.append(
            write_dataframe(
                build_connectedness_regression_dataset(source_dir, horizon=10),
                args.output_dir / "connectedness_regression_dataset.csv",
            )
        )

    for path in outputs:
        print(path.relative_to(THESIS_ROOT))


def _existing_path(primary: Path, fallback: Path) -> Path:
    return primary if primary.exists() else fallback


def _connectedness_source_dir(primary: Path, fallback: Path) -> Path:
    if any(primary.glob("*_h*/*.csv")):
        return primary
    return fallback


def _write_connectedness_components(source_dir: Path, output_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    for system_dir in source_dir.glob("*_h*"):
        if not system_dir.is_dir():
            continue
        for component in ["connectedness_table", "tci", "to", "from", "net", "npdc"]:
            for source_csv in system_dir.glob(f"{component}_h*.csv"):
                target_name = f"tvpvar_connectedness_{system_dir.name}_{source_csv.stem}.csv"
                outputs.append(write_dataframe(pd.read_csv(source_csv), output_dir / target_name))
    return outputs


def _write_tvpvar_diagnostics(source_dir: Path, output_dir: Path) -> list[Path]:
    diagnostic_frames: list[pd.DataFrame] = []
    completeness_frames: list[pd.DataFrame] = []
    outputs: list[Path] = []
    for system_dir in source_dir.glob("*_h*"):
        if not system_dir.is_dir():
            continue
        system, horizon = _parse_system_dir(system_dir.name)
        for source_csv in system_dir.glob("tvpvar_post_estimation_diagnostics_h*.csv"):
            frame = pd.read_csv(source_csv)
            frame.insert(0, "horizon", horizon)
            frame.insert(0, "system", system)
            diagnostic_frames.append(frame)
        for source_csv in system_dir.glob("tvpvar_output_completeness_h*.csv"):
            frame = pd.read_csv(source_csv)
            frame.insert(0, "horizon", horizon)
            frame.insert(0, "system", system)
            completeness_frames.append(frame)
        for source_csv in system_dir.glob("standardized_innovations_h*.csv"):
            target_name = f"tvpvar_connectedness_{system_dir.name}_{source_csv.stem}.csv"
            outputs.append(write_dataframe(pd.read_csv(source_csv), output_dir / target_name))

    if diagnostic_frames:
        outputs.append(
            write_dataframe(
                pd.concat(diagnostic_frames, ignore_index=True),
                output_dir / "tvpvar_connectedness_post_estimation_diagnostics.csv",
            )
        )
    if completeness_frames:
        outputs.append(
            write_dataframe(
                pd.concat(completeness_frames, ignore_index=True),
                output_dir / "tvpvar_connectedness_output_completeness.csv",
            )
        )
    return outputs


def _parse_system_dir(name: str) -> tuple[str, int]:
    system, horizon = name.rsplit("_h", maxsplit=1)
    return system, int(horizon)


if __name__ == "__main__":
    main()
