from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from thesis.tables.common import DATE_COLUMN


@dataclass(frozen=True)
class TvpvarSystem:
    name: str
    description: str
    columns: tuple[str, ...]


TVPVAR_SYSTEMS: dict[str, TvpvarSystem] = {
    "benchmark": TvpvarSystem(
        name="benchmark",
        description="Bitcoin and NASDAQ-100 conditional volatilities",
        columns=("bitcoin_conditional_volatility", "ndx_conditional_volatility"),
    ),
    "ai_equity": TvpvarSystem(
        name="ai_equity",
        description="Bitcoin and selected AI-exposed equity conditional volatilities",
        columns=(
            "bitcoin_conditional_volatility",
            "nvda_conditional_volatility",
            "googl_conditional_volatility",
            "msft_conditional_volatility",
        ),
    ),
}


def read_volatility_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Conditional volatility panel not found: {path}")
    df = pd.read_csv(path)
    if DATE_COLUMN not in df.columns:
        raise ValueError(f"Expected a '{DATE_COLUMN}' column in {path}")
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    return df.sort_values(DATE_COLUMN).reset_index(drop=True)


def table_09_tvpvar_system_definition() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "system": system.name,
                "description": system.description,
                "variables": ", ".join(system.columns),
                "variable_count": len(system.columns),
                "input": "EGARCH(1,1) conditional volatility",
            }
            for system in TVPVAR_SYSTEMS.values()
        ]
    )


def tvpvar_lag_selection(
    volatility_panel: pd.DataFrame,
    *,
    maxlags: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for system in TVPVAR_SYSTEMS.values():
        missing = [column for column in system.columns if column not in volatility_panel.columns]
        if missing:
            rows.append(
                {
                    "system": system.name,
                    "lag": pd.NA,
                    "bic": np.nan,
                    "selected_by_bic": False,
                    "nobs": 0,
                    "variables": ", ".join(system.columns),
                    "note": f"Missing columns: {', '.join(missing)}",
                }
            )
            continue

        panel = volatility_panel[list(system.columns)].apply(pd.to_numeric, errors="coerce").dropna()
        candidate_rows: list[dict[str, object]] = []
        effective_maxlags = min(maxlags, max(1, len(panel) // 8))
        for lag in range(1, effective_maxlags + 1):
            try:
                result = VAR(panel).fit(lag)
                candidate_rows.append(
                    {
                        "system": system.name,
                        "lag": lag,
                        "bic": float(result.bic),
                        "selected_by_bic": False,
                        "nobs": int(result.nobs),
                        "variables": ", ".join(system.columns),
                        "note": "",
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive reporting
                candidate_rows.append(
                    {
                        "system": system.name,
                        "lag": lag,
                        "bic": np.nan,
                        "selected_by_bic": False,
                        "nobs": int(len(panel)),
                        "variables": ", ".join(system.columns),
                        "note": f"{type(exc).__name__}: {exc}",
                    }
                )

        valid = [row for row in candidate_rows if pd.notna(row["bic"])]
        if valid:
            selected = min(valid, key=lambda row: row["bic"])
            for row in candidate_rows:
                row["selected_by_bic"] = row["lag"] == selected["lag"]
        rows.extend(candidate_rows)
    return pd.DataFrame(rows)


def selected_lags(lag_selection: pd.DataFrame) -> dict[str, int]:
    selected = lag_selection[lag_selection["selected_by_bic"].astype(bool)]
    return {
        str(row["system"]): int(row["lag"])
        for _, row in selected.iterrows()
        if pd.notna(row["lag"])
    }


def table_10_tvpvar_lag_selection(lag_selection: pd.DataFrame) -> pd.DataFrame:
    return lag_selection.sort_values(["system", "lag"]).reset_index(drop=True)


def table_11_average_connectedness(models_dir: Path, *, horizon: int = 10) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for system_name in TVPVAR_SYSTEMS:
        output_dir = models_dir / f"{system_name}_h{horizon}"
        for component in ["tci", "to", "from", "net"]:
            path = output_dir / f"{component}_h{horizon}.csv"
            if not path.exists():
                rows.append(_missing_connectedness_row(system_name, horizon, component, path))
                continue
            values = _read_time_component(path)
            numeric_columns = [column for column in values.columns if column != DATE_COLUMN]
            for column in numeric_columns:
                series = pd.to_numeric(values[column], errors="coerce")
                rows.append(
                    {
                        "system": system_name,
                        "horizon": horizon,
                        "component": component,
                        "variable": column,
                        "mean": float(series.mean()),
                        "standard_deviation": float(series.std(ddof=1)),
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "nobs": int(series.notna().sum()),
                        "note": "",
                    }
                )
    return pd.DataFrame(rows)


def table_12_average_pairwise_connectedness_matrix(models_dir: Path, *, horizon: int = 10) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for system_name in TVPVAR_SYSTEMS:
        path = models_dir / f"{system_name}_h{horizon}" / f"connectedness_table_h{horizon}.csv"
        if not path.exists():
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "system": system_name,
                            "horizon": horizon,
                            "row_variable": "",
                            "column_variable": "",
                            "value": np.nan,
                            "note": f"Missing connectedness table: {path}",
                        }
                    ]
                )
            )
            continue
        rows.append(_melt_connectedness_table(path, system_name, horizon))
    return pd.concat(rows, ignore_index=True)


def table_13_robustness_connectedness(models_dir: Path) -> pd.DataFrame:
    benchmark = table_11_average_connectedness(models_dir, horizon=10)
    robustness = table_11_average_connectedness(models_dir, horizon=100)
    combined = benchmark.merge(
        robustness,
        on=["system", "component", "variable"],
        suffixes=("_h10", "_h100"),
        how="outer",
    )
    combined["difference_h100_minus_h10"] = combined["mean_h100"] - combined["mean_h10"]
    keep_columns = [
        "system",
        "component",
        "variable",
        "mean_h10",
        "mean_h100",
        "difference_h100_minus_h10",
        "nobs_h10",
        "nobs_h100",
        "note_h10",
        "note_h100",
    ]
    return combined[[column for column in keep_columns if column in combined.columns]]


def build_connectedness_regression_dataset(models_dir: Path, *, horizon: int = 10) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for system_name in TVPVAR_SYSTEMS:
        output_dir = models_dir / f"{system_name}_h{horizon}"
        system_frames: list[pd.DataFrame] = []
        for component in ["tci", "to", "from", "net"]:
            path = output_dir / f"{component}_h{horizon}.csv"
            if not path.exists():
                continue
            values = _read_time_component(path)
            rename = {
                column: f"{system_name}_{component}_{_clean_column_name(column)}"
                for column in values.columns
                if column != DATE_COLUMN
            }
            system_frames.append(values.rename(columns=rename))
        if not system_frames:
            continue
        system_data = _merge_on_date(system_frames)
        merged = system_data if merged is None else merged.merge(system_data, on=DATE_COLUMN, how="outer")

    if merged is None:
        return pd.DataFrame(columns=[DATE_COLUMN])
    return merged.sort_values(DATE_COLUMN).reset_index(drop=True)


def _missing_connectedness_row(system_name: str, horizon: int, component: str, path: Path) -> dict[str, object]:
    return {
        "system": system_name,
        "horizon": horizon,
        "component": component,
        "variable": "",
        "mean": np.nan,
        "standard_deviation": np.nan,
        "min": np.nan,
        "max": np.nan,
        "nobs": 0,
        "note": f"Missing connectedness output: {path}",
    }


def _read_time_component(path: Path) -> pd.DataFrame:
    values = pd.read_csv(path)
    if DATE_COLUMN in values.columns:
        values[DATE_COLUMN] = pd.to_datetime(values[DATE_COLUMN])
    else:
        values.insert(0, DATE_COLUMN, pd.NaT)
    return values


def _melt_connectedness_table(path: Path, system_name: str, horizon: int) -> pd.DataFrame:
    table = pd.read_csv(path)
    first_column = table.columns[0]
    if first_column.startswith("Unnamed") or first_column.lower() not in {"variable", "row_variable"}:
        table = table.rename(columns={first_column: "row_variable"})
    elif first_column.lower() == "variable":
        table = table.rename(columns={first_column: "row_variable"})

    value_columns = [column for column in table.columns if column != "row_variable"]
    melted = table.melt(
        id_vars="row_variable",
        value_vars=value_columns,
        var_name="column_variable",
        value_name="value",
    )
    melted.insert(0, "horizon", horizon)
    melted.insert(0, "system", system_name)
    melted["note"] = ""
    return melted


def _merge_on_date(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=DATE_COLUMN, how="outer")
    return merged


def _clean_column_name(value: object) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower())
    return cleaned.strip("_")
