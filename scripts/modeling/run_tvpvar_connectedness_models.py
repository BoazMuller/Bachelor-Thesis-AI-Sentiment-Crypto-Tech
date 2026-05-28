"""
Run TVP-VAR connectedness models with BIC-selected lag orders.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from thesis.paths import MODELS_DIR  # noqa: E402
from thesis.tables.tvpvar_connectedness import TVPVAR_SYSTEMS, selected_lags  # noqa: E402


DEFAULT_MODELS_DIR = MODELS_DIR / "tvpvar_connectedness"
DEFAULT_VOLATILITY_CSV = DEFAULT_MODELS_DIR / "conditional_volatility_panel.csv"
DEFAULT_LAG_SELECTION_CSV = DEFAULT_MODELS_DIR / "tvpvar_lag_selection.csv"
DEFAULT_R_SCRIPT = PROJECT_ROOT / "r" / "modeling" / "run_tvpvar_connectedness.R"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TVP-VAR connectedness models using BIC-selected lags.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--volatility-csv", type=Path, default=DEFAULT_VOLATILITY_CSV)
    parser.add_argument("--lag-selection-csv", type=Path, default=DEFAULT_LAG_SELECTION_CSV)
    parser.add_argument("--r-script", type=Path, default=DEFAULT_R_SCRIPT)
    parser.add_argument("--horizons", nargs="+", type=int, default=[10, 100])
    parser.add_argument("--systems", nargs="+", default=list(TVPVAR_SYSTEMS))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lag_selection = pd.read_csv(args.lag_selection_csv)
    lags = selected_lags(lag_selection)

    for system_name in args.systems:
        if system_name not in TVPVAR_SYSTEMS:
            raise ValueError(f"Unknown system {system_name!r}. Expected one of {list(TVPVAR_SYSTEMS)}")
        if system_name not in lags:
            raise ValueError(f"No BIC-selected lag found for system {system_name!r} in {args.lag_selection_csv}")

        columns = ",".join(TVPVAR_SYSTEMS[system_name].columns)
        for horizon in args.horizons:
            output_dir = args.models_dir / f"{system_name}_h{horizon}"
            command = [
                "Rscript",
                str(args.r_script),
                str(args.volatility_csv),
                str(output_dir),
                str(lags[system_name]),
                str(horizon),
                columns,
            ]
            print(" ".join(command))
            if not args.dry_run:
                subprocess.run(command, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
