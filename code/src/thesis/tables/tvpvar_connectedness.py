from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from thesis.modeling.diagnostics import autocorrelation_tests, stationarity_tests
from thesis.tables.common import DATE_COLUMN, EXPECTATION_ADJUSTED_AIS, descriptive_stats_table


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
    "benchmark_eais": TvpvarSystem(
        name="benchmark_eais",
        description="Bitcoin and NASDAQ-100 conditional volatilities with EAIS",
        columns=(
            "bitcoin_conditional_volatility",
            "ndx_conditional_volatility",
            EXPECTATION_ADJUSTED_AIS,
        ),
    ),
    "ai_equity_eais": TvpvarSystem(
        name="ai_equity_eais",
        description="Bitcoin and selected AI-exposed equity conditional volatilities with EAIS",
        columns=(
            "bitcoin_conditional_volatility",
            "nvda_conditional_volatility",
            "googl_conditional_volatility",
            "msft_conditional_volatility",
            EXPECTATION_ADJUSTED_AIS,
        ),
    ),
}

ORIGINAL_TVPVAR_SYSTEMS = ("benchmark", "ai_equity")
AUGMENTED_TVPVAR_SYSTEMS = ("benchmark_eais", "ai_equity_eais")


def read_volatility_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Conditional volatility panel not found: {path}")
    df = pd.read_csv(path)
    if DATE_COLUMN not in df.columns:
        raise ValueError(f"Expected a '{DATE_COLUMN}' column in {path}")
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    return df.sort_values(DATE_COLUMN).reset_index(drop=True)


def tvpvar_descriptives(volatility_panel: pd.DataFrame) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for system in TVPVAR_SYSTEMS.values():
        if not set(system.columns).issubset(volatility_panel.columns):
            continue
        panel = volatility_panel[[DATE_COLUMN, *system.columns]].dropna()
        table = descriptive_stats_table(panel, system.columns)
        table.insert(0, "system", system.name)
        table["system_nobs"] = len(panel)
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def tvpvar_pre_estimation_diagnostics(volatility_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for system in TVPVAR_SYSTEMS.values():
        missing_columns = [column for column in system.columns if column not in volatility_panel.columns]
        if missing_columns:
            rows.append(
                _diagnostic_row(
                    system.name,
                    "sample_coverage",
                    test="availability",
                    interpretation=f"Missing columns: {', '.join(missing_columns)}",
                )
            )
            continue

        system_data = volatility_panel[[DATE_COLUMN, *system.columns]]
        panel = system_data.dropna().reset_index(drop=True)
        rows.append(
            _diagnostic_row(
                system.name,
                "sample_coverage",
                test="complete_case_sample",
                statistic=float(len(panel)),
                nobs=len(panel),
                interpretation=(
                    f"{panel[DATE_COLUMN].min().date()} to {panel[DATE_COLUMN].max().date()}"
                    if not panel.empty
                    else "No complete observations"
                ),
            )
        )
        for column in system.columns:
            rows.append(
                _diagnostic_row(
                    system.name,
                    "missingness",
                    variable=column,
                    test="missing_count",
                    statistic=float(system_data[column].isna().sum()),
                    nobs=len(system_data),
                    interpretation=f"{system_data[column].isna().mean():.4%} missing",
                )
            )

        stationarity = stationarity_tests(panel, system.columns)
        for _, row in stationarity.iterrows():
            interpretation = str(row.get("interpretation", ""))
            if row["variable"] == EXPECTATION_ADJUSTED_AIS and row["test"] == "KPSS":
                interpretation = f"{interpretation}; mixed EAIS stationarity evidence is retained as a limitation"
            rows.append(
                _diagnostic_row(
                    system.name,
                    "stationarity",
                    variable=row["variable"],
                    test=row["test"],
                    lag=row.get("lags", np.nan),
                    statistic=row.get("statistic", np.nan),
                    p_value=row.get("p_value", np.nan),
                    nobs=row.get("nobs", np.nan),
                    interpretation=interpretation,
                )
            )
        autocorr = autocorrelation_tests(panel, system.columns)
        for _, row in autocorr.iterrows():
            rows.append(
                _diagnostic_row(
                    system.name,
                    "autocorrelation",
                    variable=row["variable"],
                    test="Ljung-Box",
                    lag=row["lag"],
                    statistic=row["lb_stat"],
                    p_value=row["p_value"],
                    nobs=row["nobs"],
                    interpretation=row["interpretation"],
                )
            )

        correlations = panel[list(system.columns)].corr()
        for row_variable in system.columns:
            for column_variable in system.columns:
                rows.append(
                    _diagnostic_row(
                        system.name,
                        "correlation",
                        variable=row_variable,
                        test=column_variable,
                        statistic=correlations.loc[row_variable, column_variable],
                        nobs=len(panel),
                        interpretation="Pearson correlation",
                    )
                )

        _append_var_stability(rows, system, panel)
    return pd.DataFrame(rows)


def table_09_tvpvar_system_definition() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "system": system.name,
                "description": system.description,
                "variables": ", ".join(system.columns),
                "variable_count": len(system.columns),
                "input": "EGARCH(1,1) conditional volatility, plus EAIS where configured",
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

        panel = (
            volatility_panel[list(system.columns)]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
            .reset_index(drop=True)
        )
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
                mean = float(series.mean())
                rows.append(
                    {
                        "system": system_name,
                        "horizon": horizon,
                        "component": component,
                        "variable": column,
                        "mean": mean,
                        "standard_deviation": float(series.std(ddof=1)),
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "nobs": int(series.notna().sum()),
                        "role": (
                            "net_transmitter"
                            if component == "net" and mean >= 0
                            else "net_receiver"
                            if component == "net"
                            else ""
                        ),
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
        "role_h10",
        "role_h100",
        "note_h10",
        "note_h100",
    ]
    return combined[[column for column in keep_columns if column in combined.columns]]


def build_connectedness_regression_dataset(models_dir: Path, *, horizon: int = 10) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for system_name in ORIGINAL_TVPVAR_SYSTEMS:
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
        "role": "",
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


def _diagnostic_row(
    system: str,
    diagnostic: str,
    *,
    variable: str = "",
    test: str = "",
    lag: object = np.nan,
    statistic: object = np.nan,
    p_value: object = np.nan,
    nobs: object = np.nan,
    interpretation: str = "",
) -> dict[str, object]:
    return {
        "system": system,
        "diagnostic": diagnostic,
        "variable": variable,
        "test": test,
        "lag": lag,
        "statistic": statistic,
        "p_value": p_value,
        "nobs": nobs,
        "interpretation": interpretation,
    }


def _append_var_stability(
    rows: list[dict[str, object]],
    system: TvpvarSystem,
    panel: pd.DataFrame,
) -> None:
    lag = _selected_bic_lag(panel[list(system.columns)])
    if lag is None:
        rows.append(
            _diagnostic_row(
                system.name,
                "constant_var_stability",
                test="characteristic_roots",
                nobs=len(panel),
                interpretation="Could not estimate a positive-lag constant VAR",
            )
        )
        return

    result = VAR(panel[list(system.columns)]).fit(lag)
    roots = np.abs(result.roots)
    rows.append(
        _diagnostic_row(
            system.name,
            "constant_var_stability",
            test="characteristic_roots",
            lag=lag,
            statistic=float(roots.min()) if len(roots) else np.nan,
            nobs=result.nobs,
            interpretation=(
                f"stable={result.is_stable(verbose=False)}; "
                f"min_abs_root={roots.min():.6f}; max_abs_root={roots.max():.6f}"
            ),
        )
    )


def _selected_bic_lag(panel: pd.DataFrame, maxlags: int = 10) -> int | None:
    effective_maxlags = min(maxlags, max(1, len(panel) // 8))
    candidates: list[tuple[float, int]] = []
    for lag in range(1, effective_maxlags + 1):
        try:
            result = VAR(panel).fit(lag)
        except Exception:
            continue
        if np.isfinite(result.bic):
            candidates.append((float(result.bic), lag))
    return min(candidates)[1] if candidates else None
