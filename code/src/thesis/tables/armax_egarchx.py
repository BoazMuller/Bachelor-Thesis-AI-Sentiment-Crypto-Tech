from __future__ import annotations

from pathlib import Path

import pandas as pd

from thesis.tables.common import descriptive_stats_table


def read_model_output(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required model output not found: {path}")
    return pd.read_csv(path)


def table_05_armax_egarchx_estimation_results(coefficients_csv: Path) -> pd.DataFrame:
    coefficients = read_model_output(coefficients_csv)
    keep_columns = [
        "asset",
        "specification",
        "selected_p",
        "selected_q",
        "standard_error_type",
        "term",
        "estimate",
        "standard_error",
        "t_value",
        "p_value",
    ]
    return coefficients[[column for column in keep_columns if column in coefficients.columns]]


def table_06_post_estimation_diagnostics(diagnostics_csv: Path) -> pd.DataFrame:
    diagnostics = read_model_output(diagnostics_csv)
    keep_columns = [
        "asset",
        "specification",
        "selected_p",
        "selected_q",
        "diagnostic",
        "lag",
        "statistic",
        "p_value",
        "nobs",
        "note",
    ]
    return diagnostics[[column for column in keep_columns if column in diagnostics.columns]]


def table_07_egarch_volatility_extraction_summary(
    coefficients_csv: Path,
    diagnostics_csv: Path,
) -> pd.DataFrame:
    coefficients = read_model_output(coefficients_csv)
    diagnostics = read_model_output(diagnostics_csv)
    convergence = diagnostics[diagnostics["diagnostic"].eq("convergence_code")].copy()
    convergence = convergence.rename(columns={"statistic": "convergence_code"})

    rows: list[dict[str, object]] = []
    for asset, asset_coefficients in coefficients.groupby("asset"):
        asset_convergence = convergence[convergence["asset"].eq(asset)]
        rows.append(
            {
                "asset": asset,
                "model": "EGARCH(1,1), constant mean, Student-t innovations",
                "arma_terms_in_mean": "none",
                "sentiment_included": False,
                "n_parameters": int(asset_coefficients["term"].nunique()),
                "nobs": int(asset_coefficients["nobs"].dropna().iloc[0]) if "nobs" in asset_coefficients and asset_coefficients["nobs"].notna().any() else pd.NA,
                "convergence_code": float(asset_convergence["convergence_code"].iloc[0]) if not asset_convergence.empty else pd.NA,
                "note": "Used as first-stage conditional volatility input for TVP-VAR",
            }
        )
    return pd.DataFrame(rows)


def table_08_conditional_volatility_descriptives(volatility_panel_csv: Path) -> pd.DataFrame:
    volatility = read_model_output(volatility_panel_csv)
    columns = [column for column in volatility.columns if column != "date"]
    return descriptive_stats_table(volatility, columns)
