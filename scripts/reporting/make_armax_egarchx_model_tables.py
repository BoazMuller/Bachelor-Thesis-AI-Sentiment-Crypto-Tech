"""
Build thesis Tables 5-6 from ARMAX-EGARCHX model outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from thesis.paths import MODELS_DIR, PROJECT_ROOT as THESIS_ROOT  # noqa: E402
from thesis.table_output import write_registered_table  # noqa: E402
from thesis.tables.armax_egarchx import (  # noqa: E402
    table_05_armax_egarchx_estimation_results,
    table_06_post_estimation_diagnostics,
)


DEFAULT_MODELS_DIR = MODELS_DIR / "armax_egarchx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ARMAX-EGARCHX estimation and diagnostics tables.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--formats", nargs="+", default=["csv", "tex"], choices=["csv", "tex", "md"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coefficients_csv = args.models_dir / "armax_egarchx_coefficients.csv"
    diagnostics_csv = args.models_dir / "armax_egarchx_diagnostics.csv"

    outputs = []
    outputs.extend(
        write_registered_table(
            5,
            table_05_armax_egarchx_estimation_results(coefficients_csv),
            formats=args.formats,
        )
    )
    outputs.extend(
        write_registered_table(
            6,
            table_06_post_estimation_diagnostics(diagnostics_csv),
            formats=args.formats,
        )
    )

    for path in outputs:
        print(path.relative_to(THESIS_ROOT))


if __name__ == "__main__":
    main()
