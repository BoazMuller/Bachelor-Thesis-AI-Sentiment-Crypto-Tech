from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, kurtosis, skew
from statsmodels.stats.diagnostic import acorr_ljungbox

from thesis.tables.common import (
    DATE_COLUMN,
    DFM_IDIOSYNCRATIC_AR1,
    GDELT_SENTIMENT,
    RAW_AIS,
    REDDIT_SENTIMENT,
    SOURCE_SENTIMENT_COLUMNS,
    construct_ais,
    descriptive_stats_table,
)

SOURCE_LABELS = {
    GDELT_SENTIMENT: "gdelt_news",
    REDDIT_SENTIMENT: "reddit_social_media",
}

LJUNG_BOX_LAGS = (1, 5, 10, 20, 30)
AUTOCORRELATION_LAGS = (1, 5)
TABLE_15_COLUMNS = [
    "source",
    "variable",
    "diagnostic",
    "statistic",
    "lag",
    "value",
    "p_value",
    "nobs",
    "missing",
    "interpretation",
]

def table_14_text_data_coverage_by_source(sentiment_records: pd.DataFrame) -> pd.DataFrame:
    records = sentiment_records.copy()
    records[DATE_COLUMN] = pd.to_datetime(records[DATE_COLUMN])
    rows: list[dict[str, object]] = []
    for source, source_df in records.groupby("source"):
        daily_counts = source_df.groupby(DATE_COLUMN).size()
        full_index = pd.date_range(daily_counts.index.min(), daily_counts.index.max(), freq="D")
        aligned = daily_counts.reindex(full_index, fill_value=0)
        rows.append(
            {
                "source": source,
                "raw_text_count": int(len(source_df)),
                "start_date": full_index.min().date(),
                "end_date": full_index.max().date(),
                "days_covered": int((aligned > 0).sum()),
                "average_texts_per_day": float(aligned.mean()),
                "median_texts_per_day": float(aligned.median()),
                "zero_text_days": int((aligned == 0).sum()),
                "missing_days": int((aligned == 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("source")

def table_15_daily_sentiment_descriptives_by_source(df: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in SOURCE_SENTIMENT_COLUMNS if column in df.columns]
    rows: list[dict[str, object]] = []
    for column in columns:
        rows.extend(_raw_sentiment_diagnostic_rows(df, column))
    return pd.DataFrame(rows, columns=TABLE_15_COLUMNS)

def _raw_sentiment_diagnostic_rows(df: pd.DataFrame, column: str) -> list[dict[str, object]]:
    full_series = pd.to_numeric(df[column], errors="coerce")
    series = full_series.dropna()
    source = SOURCE_LABELS.get(column, column)
    missing = int(full_series.isna().sum())
    nobs = int(series.shape[0])
    rows: list[dict[str, object]] = []

    descriptive_values = _raw_sentiment_descriptive_values(series, full_series)
    for statistic, value in descriptive_values.items():
        rows.append(
            _table_15_row(
                source=source,
                variable=column,
                diagnostic="descriptive",
                statistic=statistic,
                value=value,
                nobs=nobs,
                missing=missing,
                interpretation="",
            )
        )

    if nobs >= 3 and series.nunique() >= 2:
        jb = jarque_bera(series)
        rows.append(
            _table_15_row(
                source=source,
                variable=column,
                diagnostic="normality",
                statistic="jarque_bera",
                value=float(jb.statistic),
                p_value=float(jb.pvalue),
                nobs=nobs,
                missing=missing,
                interpretation="Rejects normality at 5%" if jb.pvalue < 0.05 else "Does not reject normality at 5%",
            )
        )

    for lag in AUTOCORRELATION_LAGS:
        rows.append(
            _table_15_row(
                source=source,
                variable=column,
                diagnostic="autocorrelation",
                statistic="acf",
                lag=lag,
                value=float(series.autocorr(lag)) if nobs > lag else np.nan,
                nobs=nobs,
                missing=missing,
                interpretation="",
            )
        )

    rows.extend(
        _ljung_box_rows(
            series,
            source=source,
            variable=column,
            diagnostic="ljung_box_levels",
            missing=missing,
            interpretation_when_significant="Serial correlation at 5%",
            interpretation_when_not_significant="No serial correlation at 5%",
        )
    )
    squared_demeaned = (series - series.mean()).pow(2)
    rows.extend(
        _ljung_box_rows(
            squared_demeaned,
            source=source,
            variable=column,
            diagnostic="ljung_box_squared_demeaned",
            missing=missing,
            interpretation_when_significant="Serial dependence in squared sentiment at 5%",
            interpretation_when_not_significant="No serial dependence in squared sentiment at 5%",
        )
    )
    return rows

def _raw_sentiment_descriptive_values(
    series: pd.Series,
    full_series: pd.Series,
) -> dict[str, float]:
    if series.empty:
        return {
            "N": 0.0,
            "missing": float(full_series.isna().sum()),
            "missing_pct": float(full_series.isna().mean()),
        }

    return {
        "N": float(series.shape[0]),
        "missing": float(full_series.isna().sum()),
        "missing_pct": float(full_series.isna().mean()),
        "mean": float(series.mean()),
        "standard_deviation": float(series.std(ddof=1)),
        "min": float(series.min()),
        "p01": float(series.quantile(0.01)),
        "p05": float(series.quantile(0.05)),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
        "skewness": float(skew(series, bias=False)) if len(series) > 2 else np.nan,
        "kurtosis": float(kurtosis(series, fisher=False, bias=False)) if len(series) > 3 else np.nan,
    }

def _ljung_box_rows(
    series: pd.Series,
    *,
    source: str,
    variable: str,
    diagnostic: str,
    missing: int,
    interpretation_when_significant: str,
    interpretation_when_not_significant: str,
) -> list[dict[str, object]]:
    nobs = int(series.dropna().shape[0])
    valid_lags = [lag for lag in LJUNG_BOX_LAGS if lag < nobs - 1]
    if not valid_lags:
        return []

    result = acorr_ljungbox(series.dropna(), lags=valid_lags, return_df=True)
    rows: list[dict[str, object]] = []
    for lag, row in result.iterrows():
        p_value = float(row["lb_pvalue"])
        interpretation = (
            interpretation_when_significant
            if p_value < 0.05
            else interpretation_when_not_significant
        )
        interpretation = f"{interpretation}; missing observations dropped before Ljung-Box test"
        rows.append(
            _table_15_row(
                source=source,
                variable=variable,
                diagnostic=diagnostic,
                statistic="Ljung-Box",
                lag=int(lag),
                value=float(row["lb_stat"]),
                p_value=p_value,
                nobs=nobs,
                missing=missing,
                interpretation=interpretation,
            )
        )
    return rows

def _table_15_row(
    *,
    source: str,
    variable: str,
    diagnostic: str,
    statistic: str,
    value: float,
    nobs: int,
    missing: int,
    interpretation: str,
    p_value: float | str = "",
    lag: int | str = "",
) -> dict[str, object]:
    return {
        "source": source,
        "variable": variable,
        "diagnostic": diagnostic,
        "statistic": statistic,
        "lag": lag,
        "value": value,
        "p_value": p_value,
        "nobs": nobs,
        "missing": missing,
        "interpretation": interpretation,
    }

def table_16_sentiment_source_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in SOURCE_SENTIMENT_COLUMNS if column in df.columns]
    corr = df[columns].corr()
    corr.insert(0, "variable", corr.index)
    return corr.reset_index(drop=True)

def table_17_dynamic_factor_results(df: pd.DataFrame) -> pd.DataFrame:
    dfm_result = construct_ais(df)
    rows: list[dict[str, object]] = []

    for _, row in dfm_result.lag_selection.iterrows():
        selected = int(row["factor_order"]) == dfm_result.selected_factor_order
        for metric in ["bic", "aic", "llf", "converged", "iterations", "idiosyncratic_ar1", "fit_success"]:
            value = row[metric]
            note = "Selected by minimum BIC with r = 1" if metric == "bic" and selected else ""
            if metric == "bic" and not bool(row["converged"]):
                value = "N/A"
                note = "BIC not reported because EM did not converge"
            rows.append(
                {
                    "section": "lag_selection",
                    "metric": metric,
                    "factor_order": int(row["factor_order"]),
                    "n_factors": int(row["n_factors"]),
                    "source": "",
                    "value": value,
                    "selected": selected,
                    "note": note,
                }
            )
        if row.get("error", ""):
            rows.append(
                {
                    "section": "lag_selection",
                    "metric": "error",
                    "factor_order": int(row["factor_order"]),
                    "n_factors": int(row["n_factors"]),
                    "source": "",
                    "value": row["error"],
                    "selected": selected,
                    "note": "Estimation error for this candidate factor order",
                }
            )

    for source, loading in dfm_result.loadings.items():
        rows.append(
            {
                "section": "measurement_equation",
                "metric": "factor_loading",
                "factor_order": dfm_result.selected_factor_order,
                "n_factors": 1,
                "source": source,
                "value": float(loading),
                "selected": True,
                "note": "Loading from selected one-factor dynamic factor model",
            }
        )

    return pd.DataFrame(rows)

def table_18_ais_construction_validation(df: pd.DataFrame) -> pd.DataFrame:
    dfm_result = construct_ais(df)
    enriched = df.copy()
    enriched[RAW_AIS] = dfm_result.scores
    rows = []
    for source in SOURCE_SENTIMENT_COLUMNS:
        rows.append(
            {
                "validation_item": f"Correlation with {source}",
                "value": float(enriched[[RAW_AIS, source]].corr().iloc[0, 1]),
                "note": "Filtered DFM factor sign is normalized so higher AIS means more positive average source sentiment",
            }
        )
    rows.extend(
        [
            {
                "validation_item": "selected_factor_order",
                "value": dfm_result.selected_factor_order,
                "note": "Selected by BIC over factor-order candidates 1-5 with r = 1",
            },
            {
                "validation_item": "selected_model_bic",
                "value": dfm_result.bic,
                "note": "BIC for selected dynamic factor model",
            },
            {
                "validation_item": "selected_model_log_likelihood",
                "value": dfm_result.llf,
                "note": "Log likelihood for selected dynamic factor model",
            },
            {
                "validation_item": "selected_model_converged",
                "value": int(dfm_result.converged),
                "note": "EM convergence indicator",
            },
            {
                "validation_item": "selected_model_iterations",
                "value": dfm_result.iterations,
                "note": "EM iterations used by selected model",
            },
            {
                "validation_item": "idiosyncratic_ar1",
                "value": int(DFM_IDIOSYNCRATIC_AR1),
                "note": "Measurement disturbances are white noise; BIC selects lags in the latent factor process",
            },
        ]
    )
    stats = descriptive_stats_table(enriched, [RAW_AIS])
    for _, row in stats.iterrows():
        rows.append(
            {
                "validation_item": f"AIS {row['statistic']}",
                "value": row["value"],
                "note": "",
            }
        )
    return pd.DataFrame(rows)
