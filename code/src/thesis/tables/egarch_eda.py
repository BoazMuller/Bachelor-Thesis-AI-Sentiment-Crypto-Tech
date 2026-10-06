from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "thesis_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from thesis.modeling.diagnostics import arch_lm_tests, autocorrelation_tests, stationarity_tests, one_lag_granger_test
from thesis.paths import FIGURES_DIR
from thesis.tables.common import (
    DATE_COLUMN,
    EXPECTATION_ADJUSTED_AIS,
    RAW_AIS,
    RETURN_COLUMNS,
    add_sentiment_measures,
    descriptive_stats_table,
)

def table_02a_full_sample_descriptives(df_full: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in RETURN_COLUMNS + [RAW_AIS]
        if column in df_full.columns
    ]
    return descriptive_stats_table(df_full, columns)

def table_02b_restricted_sample_descriptives(df_restricted: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in RETURN_COLUMNS + [EXPECTATION_ADJUSTED_AIS]
        if column in df_restricted.columns
    ]
    return descriptive_stats_table(df_restricted, columns)

def table_02_return_sentiment_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    return table_02b_restricted_sample_descriptives(df)

def table_03_pre_estimation_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    sentiment_columns = [
        column
        for column in [RAW_AIS, EXPECTATION_ADJUSTED_AIS]
        if column in df.columns
    ]
    diagnostic_columns = [column for column in RETURN_COLUMNS if column in df.columns] + sentiment_columns
    rows: list[dict[str, object]] = []

    stationarity = stationarity_tests(df, diagnostic_columns)
    for _, row in stationarity.iterrows():
        rows.append(
            {
                "diagnostic": "stationarity",
                "variable": row["variable"],
                "test": row["test"],
                "lag": row.get("lags", np.nan),
                "statistic": row.get("statistic", np.nan),
                "p_value": row.get("p_value", np.nan),
                "nobs": row.get("nobs", np.nan),
                "interpretation": row.get("interpretation", ""),
            }
        )

    autocorr = autocorrelation_tests(df, [column for column in RETURN_COLUMNS if column in df.columns] + sentiment_columns)
    for _, row in autocorr.iterrows():
        var_name = row["variable"]
        diag_name = "return_autocorrelation" if "return" in str(var_name) else "sentiment_autocorrelation"
        rows.append(
            {
                "diagnostic": diag_name,
                "variable": var_name,
                "test": "Ljung-Box",
                "lag": row["lag"],
                "statistic": row["lb_stat"],
                "p_value": row["p_value"],
                "nobs": row["nobs"],
                "interpretation": row["interpretation"],
            }
        )

    arch = arch_lm_tests(df, [column for column in RETURN_COLUMNS if column in df.columns])
    for _, row in arch.iterrows():
        rows.append(
            {
                "diagnostic": "conditional_heteroskedasticity",
                "variable": row["variable"],
                "test": "ARCH-LM",
                "lag": row["lag"],
                "statistic": row["lm_stat"],
                "p_value": row["lm_p_value"],
                "nobs": row["nobs"],
                "interpretation": row["interpretation"],
            }
        )

    # 4. Granger Causality Tests (1-lag check)
    for sentiment_col in sentiment_columns:
        for return_col in [column for column in RETURN_COLUMNS if column in df.columns]:
            data = df[[return_col, sentiment_col]].dropna()
            if len(data) > 10:
                try:
                    test = one_lag_granger_test(df, return_col, sentiment_col)
                    f_stat, p_value = test["statistic"], test["p_value"]
                    interpretation = (
                        "Rejects no causality at 5%"
                        if p_value < 0.05
                        else "Does not reject no causality at 5%"
                    )
                    rows.append(
                        {
                            "diagnostic": "granger_causality",
                            "variable": sentiment_col,
                            "test": f"Granger causes {return_col}",
                            "lag": 1,
                            "statistic": float(f_stat),
                            "p_value": float(p_value),
                            "nobs": int(len(data)),
                            "interpretation": interpretation,
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "diagnostic": "granger_causality",
                            "variable": sentiment_col,
                            "test": f"Granger causes {return_col}",
                            "lag": 1,
                            "statistic": np.nan,
                            "p_value": np.nan,
                            "nobs": int(len(data)),
                            "interpretation": f"Error: {type(exc).__name__}: {exc}",
                        }
                    )

    return pd.DataFrame(rows)

def table_03_combined_pre_estimation_diagnostics(df_full: pd.DataFrame, df_restricted: pd.DataFrame) -> pd.DataFrame:
    diag_full = table_03_pre_estimation_diagnostics(df_full)
    diag_restricted = table_03_pre_estimation_diagnostics(df_restricted)
    diag_restricted = diag_restricted[
        diag_restricted["variable"].isin([EXPECTATION_ADJUSTED_AIS])
    ]
    return pd.concat([diag_full, diag_restricted], ignore_index=True)

def make_egarch_eda_figures(df: pd.DataFrame) -> list[Path]:
    enriched = add_sentiment_measures(df)
    output_dir = FIGURES_DIR / "armax_egarchx"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for column in [column for column in RETURN_COLUMNS if column in enriched.columns]:
        axes[0].plot(enriched[DATE_COLUMN], enriched[column], label=column)
    axes[0].set_title("Daily log returns")
    axes[0].legend(loc="best")
    axes[1].plot(enriched[DATE_COLUMN], enriched[EXPECTATION_ADJUSTED_AIS], color="black")
    axes[1].set_title("Expectation-adjusted AIS")
    axes[2].plot(enriched[DATE_COLUMN], enriched[RAW_AIS], color="tab:gray")
    axes[2].set_title("Raw AIS")
    fig.tight_layout()
    paths.append(_save_figure(fig, output_dir / "returns_and_expectation_adjusted_ais.png"))

    fig, ax = plt.subplots(figsize=(10, 5))
    for column in [column for column in RETURN_COLUMNS if column in enriched.columns]:
        rolling_vol = pd.to_numeric(enriched[column], errors="coerce").rolling(30).std() * np.sqrt(252)
        ax.plot(enriched[DATE_COLUMN], rolling_vol, label=column)
    ax.set_title("Thirty-day rolling annualized volatility")
    ax.legend(loc="best")
    fig.tight_layout()
    paths.append(_save_figure(fig, output_dir / "rolling_volatility.png"))

    for column in [column for column in RETURN_COLUMNS if column in enriched.columns]:
        series = pd.to_numeric(enriched[column], errors="coerce").dropna()
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        plot_acf(series, ax=axes[0, 0], lags=30, title=f"ACF {column}")
        plot_pacf(series, ax=axes[0, 1], lags=30, title=f"PACF {column}")
        plot_acf(series.pow(2), ax=axes[1, 0], lags=30, title=f"ACF squared {column}")
        plot_pacf(series.pow(2), ax=axes[1, 1], lags=30, title=f"PACF squared {column}")
        fig.tight_layout()
        paths.append(_save_figure(fig, output_dir / f"acf_pacf_{column}.png"))

    lead_lag = lead_lag_correlations(enriched)
    fig, ax = plt.subplots(figsize=(10, 5))
    for target, group in lead_lag.groupby("target"):
        ax.plot(group["sentiment_lag"], group["correlation"], marker="o", label=target)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Lead-lag correlations with expectation-adjusted AIS")
    ax.set_xlabel("Sentiment lag; positive means sentiment is lagged")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    paths.append(_save_figure(fig, output_dir / "lead_lag_cross_correlations.png"))
    lead_lag.to_csv(output_dir / "lead_lag_cross_correlations.csv", index=False)
    paths.append(output_dir / "lead_lag_cross_correlations.csv")

    return paths

def lead_lag_correlations(df: pd.DataFrame, max_lag: int = 10) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sentiment = pd.to_numeric(df[EXPECTATION_ADJUSTED_AIS], errors="coerce")
    target_series: dict[str, pd.Series] = {}
    for column in [column for column in RETURN_COLUMNS if column in df.columns]:
        returns = pd.to_numeric(df[column], errors="coerce")
        target_series[column] = returns
        target_series[f"abs_{column}"] = returns.abs()
        target_series[f"squared_{column}"] = returns.pow(2)
    for target, series in target_series.items():
        for lag in range(-max_lag, max_lag + 1):
            rows.append(
                {
                    "target": target,
                    "sentiment_lag": lag,
                    "correlation": float(series.corr(sentiment.shift(lag))),
                    "nobs": int(pd.concat([series, sentiment.shift(lag)], axis=1).dropna().shape[0]),
                }
            )
    return pd.DataFrame(rows)

def _save_figure(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path
