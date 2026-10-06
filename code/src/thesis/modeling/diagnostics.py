from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests

RETURN_SUFFIX = "_log_return"

def one_lag_granger_test(data: pd.DataFrame, dependent: str, predictor: str) -> dict[str, float]:
    """Bivariate one-lag SSR F test, matching the thesis return diagnostic.

    Complete cases must form a contiguous trading-day sample; missing internal
    observations are rejected rather than silently turned into adjacent lags.
    """
    pair = data[[dependent, predictor]].apply(pd.to_numeric, errors="raise")
    valid = pair.notna().all(axis=1).to_numpy()
    positions = np.flatnonzero(valid)
    if len(positions) < 11 or np.any(np.diff(positions) != 1):
        raise ValueError("Granger test needs at least 11 contiguous complete observations")
    pair = pair.iloc[positions]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = grangercausalitytests(pair, maxlag=[1], verbose=False)
    f_stat, p_value, df_denom, df_num = result[1][0]["ssr_ftest"]
    return {"statistic": float(f_stat), "p_value": float(p_value),
            "nobs": len(pair) - 1, "input_nobs": len(pair),
            "df_denom": float(df_denom), "df_num": float(df_num)}

def read_time_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"Expected a 'date' column in {path}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

def date_integrity_checks(df: pd.DataFrame) -> pd.DataFrame:
    dates = df["date"]
    gaps = dates.diff().dt.days.dropna()
    checks = {
        "rows": len(df),
        "start_date": dates.min().date(),
        "end_date": dates.max().date(),
        "duplicate_dates": int(dates.duplicated().sum()),
        "is_monotonic_increasing": bool(dates.is_monotonic_increasing),
        "weekend_rows": int(dates.dt.weekday.ge(5).sum()),
        "calendar_gaps_gt_1_day": int(gaps.gt(1).sum()),
        "max_calendar_gap_days": int(gaps.max()) if len(gaps) else 0,
        "trading_day_frequency_note": "Weekend/holiday gaps are expected after equity-calendar alignment",
    }
    return pd.DataFrame(
        [{"check": key, "value": value} for key, value in checks.items()]
    )

def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "variable": df.columns,
            "missing": df.isna().sum().to_numpy(),
            "missing_pct": df.isna().mean().to_numpy(),
            "non_missing": df.notna().sum().to_numpy(),
        }
    )
    return summary.sort_values(["missing", "variable"], ascending=[False, True])

def missing_spans(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in df.columns:
        if column == "date":
            continue
        is_missing = df[column].isna()
        if not is_missing.any():
            continue

        group = is_missing.ne(is_missing.shift()).cumsum()
        for _, span in df.loc[is_missing].groupby(group[is_missing]):
            rows.append(
                {
                    "variable": column,
                    "start_date": span["date"].iloc[0].date(),
                    "end_date": span["date"].iloc[-1].date(),
                    "length": int(len(span)),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["variable", "start_date", "end_date", "length"])
    return pd.DataFrame(rows).sort_values(["variable", "start_date"])

def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    from thesis.tables.common import descriptive_stats_table
    numeric = df.select_dtypes(include=[np.number])
    # The descriptive_stats_table returns rows like:
    # [{"variable": col, "statistic": "mean", "value": X}, ...]
    # But for a flat wide table like descriptive_statistics used to be, we should pivot it
    stats_long = descriptive_stats_table(df, numeric.columns)
    
    # Pivot to wide format: rows=variable, columns=statistic
    stats_wide = stats_long.pivot(index="variable", columns="statistic", values="value").reset_index()
    
    # Missing count is currently duplicated across all statistics in the long format, 
    # let's extract it and add it as a standalone column
    missing_counts = stats_long.drop_duplicates(subset=["variable"])[["variable", "missing"]]
    
    # Merge back and clean up
    result = stats_wide.merge(missing_counts, on="variable", how="left")
    
    # Define preferred column order
    preferred_order = [
        "variable", "N", "missing", "mean", "standard_deviation", 
        "min", "max", "skewness", "kurtosis", "jarque_bera", "jarque_bera_p_value"
    ]
    return result[[col for col in preferred_order if col in result.columns]]

def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    numeric = df.select_dtypes(include=[np.number])

    for column in numeric.columns:
        series = numeric[column].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_iqr = q1 - 3.0 * iqr
        upper_iqr = q3 + 3.0 * iqr
        iqr_mask = series.lt(lower_iqr) | series.gt(upper_iqr)

        median = series.median()
        mad = float(np.median(np.abs(series - median)))
        if mad == 0:
            robust_z = pd.Series(np.nan, index=series.index)
            mad_count = 0
            max_abs_robust_z = np.nan
        else:
            robust_z = 0.6745 * (series - median) / mad
            mad_count = int(robust_z.abs().gt(6).sum())
            max_abs_robust_z = float(robust_z.abs().max())

        rows.append(
            {
                "variable": column,
                "iqr_outliers_3x": int(iqr_mask.sum()),
                "iqr_outlier_pct": float(iqr_mask.mean()),
                "lower_iqr_3x": float(lower_iqr),
                "upper_iqr_3x": float(upper_iqr),
                "mad_outliers_abs_z_gt_6": mad_count,
                "max_abs_robust_z": max_abs_robust_z,
                "min_date": df.loc[series.idxmin(), "date"].date(),
                "min_value": float(series.min()),
                "max_date": df.loc[series.idxmax(), "date"].date(),
                "max_value": float(series.max()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["iqr_outliers_3x", "mad_outliers_abs_z_gt_6", "variable"],
        ascending=[False, False, True],
    )

def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    return numeric.corr(method=method)

def default_model_check_columns(df: pd.DataFrame) -> list[str]:
    priority = [
        "bitcoin_adj_close_log_return",
        "ndx_adj_close_log_return",
        "nvda_adj_close_log_return",
        "googl_adj_close_log_return",
        "msft_adj_close_log_return",
        "sp500_adj_close_log_return",
        "dxy_close_log_return",
        "vix_close",
        "epu",
        "gpr",
        "sentiment_gdelt_sentiment_compound",
        "sentiment_reddit_sentiment_compound",
        "kalshi_before_2030",
        "metaculus_recency_weighted_days_until_median",
    ]
    return [column for column in priority if column in df.columns]

def stationarity_tests(
    df: pd.DataFrame,
    columns: Iterable[str],
    maxlag: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(series) < 20 or series.nunique() < 3:
            rows.append(
                {
                    "variable": column,
                    "test": "skipped",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "lags": np.nan,
                    "nobs": int(len(series)),
                    "interpretation": "Too few non-missing or unique observations",
                }
            )
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                adf_stat, adf_p, adf_lags, adf_nobs, *_ = adfuller(
                    series,
                    maxlag=maxlag,
                    autolag="BIC" if maxlag is None else None,
                )
                rows.append(
                    {
                        "variable": column,
                        "test": "ADF",
                        "statistic": float(adf_stat),
                        "p_value": float(adf_p),
                        "lags": int(adf_lags),
                        "nobs": int(adf_nobs),
                        "interpretation": "Rejects unit root at 5%" if adf_p < 0.05 else "Does not reject unit root at 5%",
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive diagnostics
                rows.append(_error_row(column, "ADF", exc, len(series)))

            try:
                kpss_stat, kpss_p, kpss_lags, _ = kpss(series, regression="c", nlags="auto")
                rows.append(
                    {
                        "variable": column,
                        "test": "KPSS",
                        "statistic": float(kpss_stat),
                        "p_value": float(kpss_p),
                        "lags": int(kpss_lags),
                        "nobs": int(len(series)),
                        "interpretation": "Rejects level stationarity at 5%" if kpss_p < 0.05 else "Does not reject level stationarity at 5%",
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive diagnostics
                rows.append(_error_row(column, "KPSS", exc, len(series)))

    return pd.DataFrame(rows)

def autocorrelation_tests(
    df: pd.DataFrame,
    columns: Iterable[str],
    lags: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        valid_lags = [lag for lag in lags if lag < len(series) - 1]
        if not valid_lags:
            continue
        try:
            result = acorr_ljungbox(series, lags=valid_lags, return_df=True)
        except Exception as exc:  # pragma: no cover - defensive diagnostics
            rows.append(_error_row(column, "Ljung-Box", exc, len(series)))
            continue
        for lag, row in result.iterrows():
            rows.append(
                {
                    "variable": column,
                    "lag": int(lag),
                    "lb_stat": float(row["lb_stat"]),
                    "p_value": float(row["lb_pvalue"]),
                    "nobs": int(len(series)),
                    "interpretation": "Serial correlation at 5%" if row["lb_pvalue"] < 0.05 else "No serial correlation at 5%",
                }
            )
    return pd.DataFrame(rows)

def arch_lm_tests(
    df: pd.DataFrame,
    columns: Iterable[str],
    lags: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        series = series - series.mean()
        for lag in lags:
            if len(series) <= lag + 5:
                continue
            try:
                lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(series, nlags=lag)
                rows.append(
                    {
                        "variable": column,
                        "lag": lag,
                        "lm_stat": float(lm_stat),
                        "lm_p_value": float(lm_pvalue),
                        "f_stat": float(f_stat),
                        "f_p_value": float(f_pvalue),
                        "nobs": int(len(series)),
                        "interpretation": "ARCH effects at 5%" if lm_pvalue < 0.05 else "No ARCH effects at 5%",
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive diagnostics
                rows.append(_error_row(column, f"ARCH-LM lag {lag}", exc, len(series)))
    return pd.DataFrame(rows)

def var_stability_checks(
    df: pd.DataFrame,
    variable_groups: dict[str, list[str]],
    maxlags: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_name, columns in variable_groups.items():
        available = [column for column in columns if column in df.columns]
        panel = df[available].dropna()
        if len(available) < 2 or len(panel) < max(30, len(available) * 10):
            rows.append(
                {
                    "group": group_name,
                    "variables": ", ".join(available),
                    "selected_lag": np.nan,
                    "nobs": int(len(panel)),
                    "is_stable": pd.NA,
                    "min_abs_root": np.nan,
                    "max_abs_root": np.nan,
                    "note": "Skipped: insufficient variables or observations",
                }
            )
            continue

        effective_maxlags = min(maxlags, max(1, len(panel) // 8))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = VAR(panel)
                result = model.fit(maxlags=effective_maxlags, ic="bic")
            if result.k_ar == 0:
                rows.append(
                    {
                        "group": group_name,
                        "variables": ", ".join(available),
                        "selected_lag": int(result.k_ar),
                        "nobs": int(result.nobs),
                        "is_stable": pd.NA,
                        "min_abs_root": np.nan,
                        "max_abs_root": np.nan,
                        "note": "BIC selected VAR(0); no characteristic roots to evaluate",
                    }
                )
                continue

            roots = np.abs(result.roots)
            rows.append(
                {
                    "group": group_name,
                    "variables": ", ".join(available),
                    "selected_lag": int(result.k_ar),
                    "nobs": int(result.nobs),
                    "is_stable": bool(result.is_stable(verbose=False)),
                    "min_abs_root": float(np.min(roots)) if len(roots) else np.nan,
                    "max_abs_root": float(np.max(roots)) if len(roots) else np.nan,
                    "note": "Stable VAR requires characteristic roots outside the unit circle",
                }
            )
        except Exception as exc:  # pragma: no cover - defensive diagnostics
            rows.append(
                {
                    "group": group_name,
                    "variables": ", ".join(available),
                    "selected_lag": np.nan,
                    "nobs": int(len(panel)),
                    "is_stable": pd.NA,
                    "min_abs_root": np.nan,
                    "max_abs_root": np.nan,
                    "note": f"Error: {type(exc).__name__}: {exc}",
                }
            )
    return pd.DataFrame(rows)

def default_var_groups() -> dict[str, list[str]]:
    return {
        "benchmark_returns": [
            "bitcoin_adj_close_log_return",
            "ndx_adj_close_log_return",
        ],
        "ai_equity_and_bitcoin_returns": [
            "bitcoin_adj_close_log_return",
            "nvda_adj_close_log_return",
            "googl_adj_close_log_return",
            "msft_adj_close_log_return",
        ],
        "market_controls": [
            "sp500_adj_close_log_return",
            "dxy_close_log_return",
            "vix_close",
            "epu",
            "gpr",
        ],
    }

def _error_row(
    variable: str,
    test: str,
    exc: Exception,
    nobs: int,
) -> dict[str, object]:
    return {
        "variable": variable,
        "test": test,
        "statistic": np.nan,
        "p_value": np.nan,
        "lags": np.nan,
        "nobs": int(nobs),
        "interpretation": f"Error: {type(exc).__name__}: {exc}",
    }
