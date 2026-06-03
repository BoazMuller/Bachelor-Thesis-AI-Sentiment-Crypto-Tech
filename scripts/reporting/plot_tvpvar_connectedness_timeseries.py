"""
Plot TVP-VAR total and net connectedness time series.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

tmp_cache = PROJECT_ROOT / "results" / "tmp_mpl"
tmp_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(tmp_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(tmp_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis.paths import FIGURES_DIR, TABLES_DIR


DATE_COLUMN = "date"

VARIABLE_LABELS = {
    "bitcoin_conditional_volatility": "BTC",
    "ndx_conditional_volatility": "NDX",
    "msft_conditional_volatility": "MSFT",
    "nvda_conditional_volatility": "NVDA",
    "googl_conditional_volatility": "GOOGL",
}

REQUESTED_NET_SERIES = {
    "benchmark": (
        "bitcoin_conditional_volatility",
        "ndx_conditional_volatility",
    ),
    "ai_equity": (
        "bitcoin_conditional_volatility",
        "msft_conditional_volatility",
        "nvda_conditional_volatility",
        "googl_conditional_volatility",
    ),
}


@dataclass(frozen=True)
class SeriesSpec:
    system: str
    component: str
    column: str
    title: str
    y_label: str
    output_stem: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TVP-VAR connectedness time-series figures.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=TABLES_DIR / "tvpvar_connectedness",
        help="Directory containing TVP-VAR connectedness CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_DIR / "tvpvar_connectedness",
        help="Directory for connectedness time-series figures.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Forecast horizon to plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for spec in _series_specs(args.horizon):
        data = _load_component(args.input_dir, spec.system, spec.component, args.horizon)
        if data.empty or spec.column not in data.columns:
            print(f"Skipping {spec.output_stem}; missing {spec.component} column {spec.column!r}.")
            continue

        output_path = args.output_dir / f"{spec.output_stem}_h{args.horizon}.png"
        _plot_series(data, spec, output_path)
        outputs.append(output_path)

    for output in outputs:
        print(output.relative_to(PROJECT_ROOT))


def _series_specs(horizon: int) -> list[SeriesSpec]:
    specs: list[SeriesSpec] = []
    for system in ("benchmark", "ai_equity"):
        specs.append(
            SeriesSpec(
                system=system,
                component="tci",
                column="TCI",
                title=f"Total Connectedness, H = {horizon}",
                y_label="Total connectedness index",
                output_stem=f"timeseries_{system}_total_connectedness",
            )
        )
        for column in REQUESTED_NET_SERIES[system]:
            label = VARIABLE_LABELS.get(column, column)
            specs.append(
                SeriesSpec(
                    system=system,
                    component="net",
                    column=column,
                    title=f"NET {label}, H = {horizon}",
                    y_label="Net directional connectedness",
                    output_stem=f"timeseries_{system}_net_{label.lower()}",
                )
            )
    return specs


def _load_component(input_dir: Path, system: str, component: str, horizon: int) -> pd.DataFrame:
    path = input_dir / f"{system}_h{horizon}" / f"{component}_h{horizon}.csv"
    if not path.exists():
        path = input_dir / f"tvpvar_connectedness_{system}_h{horizon}_{component}_h{horizon}.csv"
    if not path.exists():
        return pd.DataFrame()

    data = pd.read_csv(path)
    if DATE_COLUMN not in data.columns:
        raise ValueError(f"Expected a '{DATE_COLUMN}' column in {path}")
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data.sort_values(DATE_COLUMN).reset_index(drop=True)


def _plot_series(data: pd.DataFrame, spec: SeriesSpec, output_path: Path) -> None:
    dates = mdates.date2num(data[DATE_COLUMN].to_numpy(dtype="datetime64[ns]"))
    values = pd.to_numeric(data[spec.column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(dates) & np.isfinite(values)
    if not valid.any():
        raise ValueError(f"No finite values available for {spec.title}")

    dates = dates[valid]
    values = values[valid]

    fig, ax = plt.subplots(figsize=(12, 4.2), constrained_layout=True)
    ax.fill_between(dates, values, 0.0, color="#0000ff", linewidth=0)
    ax.plot(dates, values, color="#ff0000", linewidth=0.8)

    ax.set_title(spec.title, fontsize=19, fontweight="bold", pad=12)
    ax.set_ylabel(spec.y_label, fontsize=10)
    ax.set_xlabel("")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.axhline(0, color="#666666", linewidth=0.7)
    ax.set_xlim(float(dates.min()), float(dates.max()))
    ax.xaxis_date()
    _format_date_axis(ax, dates)
    _format_y_axis(ax, values, is_total=spec.component == "tci")

    for spine in ax.spines.values():
        spine.set_color("#5f5f5f")
        spine.set_linewidth(0.9)

    ax.tick_params(axis="both", labelsize=9, colors="black")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _format_date_axis(ax: plt.Axes, dates: np.ndarray) -> None:
    start = mdates.num2date(float(dates.min()))
    end = mdates.num2date(float(dates.max()))
    months = (end.year - start.year) * 12 + end.month - start.month + 1
    interval = max(1, math.ceil(months / 8))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="x", length=7, width=0.9)


def _format_y_axis(ax: plt.Axes, values: np.ndarray, *, is_total: bool) -> None:
    if is_total:
        upper = max(100.0, float(np.nanmax(values)) * 1.05)
        ax.set_ylim(0.0, upper)
        return

    max_abs = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 1.0)
    limit = max_abs * 1.08
    ax.set_ylim(-limit, limit)


if __name__ == "__main__":
    main()
