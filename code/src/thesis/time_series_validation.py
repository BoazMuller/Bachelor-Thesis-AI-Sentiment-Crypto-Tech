from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss


RETURN_SUFFIX = "_log_return"


@dataclass(frozen=True)
class VariableMetadata:
    symbol: str
    source: str
    raw_frequency: str
    transformed_frequency: str
    transformation: str
    expected_sign_or_role: str
    units: str
    treatment_of_missing_values: str
    notes: str = ""


VARIABLE_METADATA: dict[str, VariableMetadata] = {
    "date": VariableMetadata(
        symbol="t",
        source="Merged trading-day calendar from Yahoo Finance",
        raw_frequency="Daily calendar",
        transformed_frequency="Trading day",
        transformation="Parsed as date; non-trading days removed from final dataset",
        expected_sign_or_role="Time index for all aligned empirical series",
        units="Date",
        treatment_of_missing_values="Dates without S&P 500 observations are excluded from the final trading-day dataset",
        notes="Defines the common sample after equity-market calendar alignment",
    ),
    "ndx_adj_close": VariableMetadata(
        symbol="P_NDX,t",
        source="Yahoo Finance (^NDX)",
        raw_frequency="Daily calendar download",
        transformed_frequency="Trading day",
        transformation="Adjusted close level",
        expected_sign_or_role="NASDAQ-100 market level used to derive returns",
        units="Index points",
        treatment_of_missing_values="Non-trading days are excluded through the common trading-day calendar",
    ),
    "bitcoin_adj_close": VariableMetadata(
        symbol="P_BTC,t",
        source="Yahoo Finance (BTC-USD)",
        raw_frequency="Daily calendar",
        transformed_frequency="Trading day",
        transformation="Adjusted close level; weekend observations excluded",
        expected_sign_or_role="Bitcoin market level used to derive returns",
        units="USD",
        treatment_of_missing_values="Weekend observations are excluded to align with the equity-market calendar",
        notes="Bitcoin trades continuously, but the thesis sample uses equity trading days",
    ),
    "nvda_adj_close": VariableMetadata(
        symbol="P_NVDA,t",
        source="Yahoo Finance (NVDA)",
        raw_frequency="Daily calendar download",
        transformed_frequency="Trading day",
        transformation="Adjusted close level",
        expected_sign_or_role="AI-exposed equity price level for TVP-VAR systems",
        units="USD",
        treatment_of_missing_values="Non-trading days are excluded through the common trading-day calendar",
    ),
    "googl_adj_close": VariableMetadata(
        symbol="P_GOOGL,t",
        source="Yahoo Finance (GOOGL)",
        raw_frequency="Daily calendar download",
        transformed_frequency="Trading day",
        transformation="Adjusted close level",
        expected_sign_or_role="AI-exposed equity price level for TVP-VAR systems",
        units="USD",
        treatment_of_missing_values="Non-trading days are excluded through the common trading-day calendar",
    ),
    "msft_adj_close": VariableMetadata(
        symbol="P_MSFT,t",
        source="Yahoo Finance (MSFT)",
        raw_frequency="Daily calendar download",
        transformed_frequency="Trading day",
        transformation="Adjusted close level",
        expected_sign_or_role="AI-exposed equity price level for TVP-VAR systems",
        units="USD",
        treatment_of_missing_values="Non-trading days are excluded through the common trading-day calendar",
    ),
    "sp500_adj_close": VariableMetadata(
        symbol="P_SP500,t",
        source="Yahoo Finance (^GSPC)",
        raw_frequency="Daily calendar download",
        transformed_frequency="Trading day",
        transformation="Adjusted close level",
        expected_sign_or_role="Broad market level used to derive market-return control",
        units="Index points",
        treatment_of_missing_values="Non-trading days are excluded through the common trading-day calendar",
    ),
    "dxy_close": VariableMetadata(
        symbol="DXY_t",
        source="Yahoo Finance (DX-Y.NYB)",
        raw_frequency="Daily calendar download",
        transformed_frequency="Trading day",
        transformation="Close level",
        expected_sign_or_role="US dollar control; stronger dollar may affect crypto and equity conditions",
        units="Index points",
        treatment_of_missing_values="Non-trading days are excluded through the common trading-day calendar",
    ),
    "dxy_close_log_return": VariableMetadata(
        symbol="r_DXY,t",
        source="Yahoo Finance (DX-Y.NYB)",
        raw_frequency="Derived from daily DXY close",
        transformed_frequency="Trading day",
        transformation="log(DXY_t) - log(DXY_t-1)",
        expected_sign_or_role="US dollar return control for robustness and spillover regressions",
        units="Log return",
        treatment_of_missing_values=(
            "First observation is missing because returns require a previous "
            "trading-day value; non-trading days are excluded through the common calendar"
        ),
    ),
    "vix_close": VariableMetadata(
        symbol="VIX_t",
        source="Yahoo Finance (^VIX)",
        raw_frequency="Daily calendar download",
        transformed_frequency="Trading day",
        transformation="Close level",
        expected_sign_or_role="Market uncertainty/risk-aversion control; expected to relate positively to volatility and connectedness",
        units="Index points",
        treatment_of_missing_values="Non-trading days are excluded through the common trading-day calendar",
    ),
    "epu": VariableMetadata(
        symbol="EPU_t",
        source="Economic Policy Uncertainty daily policy index",
        raw_frequency="Daily calendar",
        transformed_frequency="Trading day",
        transformation="Daily policy index level",
        expected_sign_or_role="Policy uncertainty control; expected to capture macro uncertainty",
        units="Index value",
        treatment_of_missing_values="Restricted to the analysis window and matched to trading days",
    ),
    "gpr": VariableMetadata(
        symbol="GPR_t",
        source="Geopolitical Risk daily index",
        raw_frequency="Daily calendar",
        transformed_frequency="Trading day",
        transformation="GPRD level",
        expected_sign_or_role="Geopolitical risk control; expected to capture geopolitical stress",
        units="Index value",
        treatment_of_missing_values="Restricted to the analysis window and matched to trading days",
    ),
    "sentiment_gdelt_sentiment_compound": VariableMetadata(
        symbol="S_News,t",
        source="GDELT AI headlines with RoBERTa sentiment",
        raw_frequency="Text-level news headlines",
        transformed_frequency="Trading day",
        transformation="Mean positive probability minus negative probability",
        expected_sign_or_role="News component of raw AI sentiment",
        units="Sentiment score in [-1, 1]",
        treatment_of_missing_values="Text-level sentiment is aggregated by source-date; non-trading days are assigned to the next trading day",
        notes="GDELT/news has a known upstream coverage gap in June-July 2025",
    ),
    "sentiment_reddit_sentiment_compound": VariableMetadata(
        symbol="S_Reddit,t",
        source="Reddit AI submissions with RoBERTa sentiment",
        raw_frequency="Text-level Reddit submissions",
        transformed_frequency="Trading day",
        transformation="Mean positive probability minus negative probability",
        expected_sign_or_role="Social-media component of raw AI sentiment",
        units="Sentiment score in [-1, 1]",
        treatment_of_missing_values="Text-level sentiment is aggregated by source-date; non-trading days are assigned to the next trading day",
    ),
    "kalshi_before_2030": VariableMetadata(
        symbol="Kalshi_t",
        source="Kalshi AGI-related market price",
        raw_frequency="Intermittent daily market observations",
        transformed_frequency="Trading day",
        transformation="Before 2030 contract price; forward-filled after first observation",
        expected_sign_or_role="Prediction-market AI expectation control; higher values imply higher probability of AGI before 2030",
        units="Market price / implied probability scale",
        treatment_of_missing_values="Forward-filled after first observed value; pre-first-observation dates remain missing",
    ),
    "metaculus_recency_weighted_days_until_median": VariableMetadata(
        symbol="MetaculusDays_t",
        source="Metaculus AGI date forecast history",
        raw_frequency="Intermittent forecast updates",
        transformed_frequency="Trading day",
        transformation="Median forecast date minus forecast timestamp in days; forward-filled",
        expected_sign_or_role="Forecast expectation control; larger raw values imply later expected AGI arrival",
        units="Days",
        treatment_of_missing_values="Forward-filled from most recent available recency-weighted forecast",
    ),
    "metaculus_inverted_days_until_median": VariableMetadata(
        symbol="MetaculusSooner_t",
        source="Derived from Metaculus AGI date forecast history",
        raw_frequency="Intermittent forecast updates",
        transformed_frequency="Trading day",
        transformation="Inverted Metaculus days-until-median so larger values imply earlier expected AI arrival",
        expected_sign_or_role="Forecast expectation control aligned with Kalshi direction",
        units="Negative days",
        treatment_of_missing_values="Inherits Metaculus alignment and forward-fill rule",
    ),
    "raw_ais": VariableMetadata(
        symbol="AIS_t",
        source="PCA of standardized GDELT/news and Reddit/social-media sentiment",
        raw_frequency="Daily source-level sentiment",
        transformed_frequency="Trading day",
        transformation="First principal component of standardized source sentiment; sign normalized positive",
        expected_sign_or_role="AI sentiment index; robustness check for expectation-adjusted sentiment",
        units="PCA score",
        treatment_of_missing_values="Complete-case PCA over dates with both source sentiment measures",
        notes="Higher values mean more positive AI sentiment",
    ),
    "expectation_adjusted_ais": VariableMetadata(
        symbol="EAIS_t",
        source="Residual from AIS orthogonalization regression",
        raw_frequency="Trading day",
        transformed_frequency="Trading day",
        transformation="Residual from AIS on Kalshi, inverted Metaculus, S&P 500 return, and VIX",
        expected_sign_or_role="Headline sentiment variable for spillover regressions",
        units="OLS residual",
        treatment_of_missing_values="Complete-case sample for AIS and baseline controls",
        notes="Benchmark sentiment variable throughout the thesis",
    ),
    "robust_expectation_adjusted_ais": VariableMetadata(
        symbol="EAIS_robust,t",
        source="Residual from expanded AIS orthogonalization regression",
        raw_frequency="Trading day",
        transformed_frequency="Trading day",
        transformation="Residual from AIS on Kalshi, inverted Metaculus, S&P 500 return, VIX, EPU, DXY return, and GPR",
        expected_sign_or_role="Robustness version of expectation-adjusted AI sentiment",
        units="OLS residual",
        treatment_of_missing_values="Complete-case sample for AIS and expanded controls",
        notes="Robustness check, not the headline sentiment variable",
    ),
    "lagged_expectation_adjusted_ais": VariableMetadata(
        symbol="EAIS_t-1",
        source="Lag of expectation-adjusted AIS",
        raw_frequency="Trading day",
        transformed_frequency="Trading day",
        transformation="One-trading-day lag of expectation-adjusted AIS",
        expected_sign_or_role="Lagged-sentiment robustness check for spillover regressions",
        units="OLS residual",
        treatment_of_missing_values="First residual observation is missing after lagging",
    ),
    "egarch_conditional_volatility": VariableMetadata(
        symbol="sigma_i,t",
        source="ARMAX-EGARCHX fitted model",
        raw_frequency="Trading day returns",
        transformed_frequency="Trading day",
        transformation="Conditional standard deviation estimated from EGARCH(1,1)",
        expected_sign_or_role="Volatility input for TVP-VAR connectedness systems",
        units="Return standard deviation",
        treatment_of_missing_values="Not yet generated; unavailable until EGARCH models are estimated",
        notes="Will be produced separately for Bitcoin, NASDAQ-100, NVIDIA, Alphabet, and Microsoft",
    ),
    "tvpvar_connectedness_measures": VariableMetadata(
        symbol="TO_i,t; FROM_i,t; NET_i,t; NPDC_ij,t; TCI_t",
        source="R ConnectednessApproach TVP-VAR",
        raw_frequency="Trading day EGARCH conditional volatilities",
        transformed_frequency="Trading day",
        transformation="Generalized FEVD-based connectedness measures from TVP-VAR",
        expected_sign_or_role="Dependent variables in connectedness-on-sentiment regressions",
        units="Share / percentage of forecast error variance",
        treatment_of_missing_values="Not yet generated; unavailable until TVP-VAR is estimated",
        notes="Benchmark horizon H=10; robustness horizon H=100",
    ),
}


INVENTORY_COLUMNS = [
    "variable",
    "symbol",
    "source",
    "raw_frequency",
    "transformed_frequency",
    "transformation",
    "expected_sign_or_role",
    "sample_start",
    "sample_end",
    "raw_observations",
    "final_usable_observations",
    "missing_days",
    "date_range",
    "units",
    "treatment_of_missing_values",
    "notes",
]


def metadata_for_column(column: str) -> VariableMetadata:
    if column in VARIABLE_METADATA:
        return VARIABLE_METADATA[column]
    if column.endswith(RETURN_SUFFIX):
        base = column[: -len(RETURN_SUFFIX)]
        base_meta = VARIABLE_METADATA.get(base)
        source = base_meta.source if base_meta else "Derived from price level"
        if base_meta and base_meta.symbol.startswith("P_"):
            symbol = base_meta.symbol.replace("P_", "r_", 1)
        else:
            symbol = f"r_{base},t"
        return VariableMetadata(
            symbol=symbol,
            source=source,
            raw_frequency="Derived from daily price level",
            transformed_frequency="Trading day",
            transformation="log(value_t) - log(value_t-1)",
            expected_sign_or_role=f"Return series derived from {base}",
            units="Log return",
            treatment_of_missing_values=(
                "First observation is missing because returns require a previous "
                "trading-day value; non-trading days are excluded through the common calendar"
            ),
        )
    return VariableMetadata(
        symbol="not documented",
        source="Derived or undocumented source",
        raw_frequency="Unknown",
        transformed_frequency="Unknown",
        transformation="Not documented in validation metadata",
        expected_sign_or_role="Unclassified variable",
        units="Unknown",
        treatment_of_missing_values="Not documented",
    )


def read_time_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"Expected a 'date' column in {path}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def variable_inventory(df: pd.DataFrame, project_root: Path | None = None) -> pd.DataFrame:
    raw_counts = raw_observation_counts(project_root, df) if project_root else {}
    rows: list[dict[str, object]] = []

    inventory_variables = list(df.columns)
    for planned_variable in [
        "egarch_conditional_volatility",
        "tvpvar_connectedness_measures",
    ]:
        if planned_variable not in inventory_variables:
            inventory_variables.append(planned_variable)

    for column in inventory_variables:
        meta = metadata_for_column(column)
        if column in df.columns:
            series = df[column]
            if column == "date":
                valid_dates = df["date"]
            else:
                valid_dates = df.loc[series.notna(), "date"]
            sample_start = _format_date(valid_dates.min()) if len(valid_dates) else "not available"
            sample_end = _format_date(valid_dates.max()) if len(valid_dates) else "not available"
            date_range = f"{sample_start} to {sample_end}" if len(valid_dates) else "not available"
            final_usable_observations: object = int(series.notna().sum())
            missing_days: object = int(series.isna().sum())
        else:
            sample_start = "not yet generated"
            sample_end = "not yet generated"
            date_range = "not yet generated"
            final_usable_observations = "not yet generated"
            missing_days = "not yet generated"

        row: dict[str, object] = {
            "variable": column,
            "symbol": meta.symbol,
            "source": meta.source,
            "raw_frequency": meta.raw_frequency,
            "transformed_frequency": meta.transformed_frequency,
            "transformation": meta.transformation,
            "expected_sign_or_role": meta.expected_sign_or_role,
            "sample_start": sample_start,
            "sample_end": sample_end,
            "raw_observations": raw_counts.get(column, "not available"),
            "final_usable_observations": final_usable_observations,
            "missing_days": missing_days,
            "date_range": date_range,
            "units": meta.units,
            "treatment_of_missing_values": meta.treatment_of_missing_values,
            "notes": meta.notes,
        }
        rows.append(row)

    return pd.DataFrame(rows, columns=INVENTORY_COLUMNS)


def raw_observation_counts(
    project_root: Path | None,
    df: pd.DataFrame,
) -> dict[str, object]:
    if project_root is None or "date" not in df:
        return {}

    counts: dict[str, object] = {}
    start = df["date"].min()
    end = df["date"].max()

    _add_yahoo_counts(project_root, counts, start, end)
    _add_epu_count(project_root, counts, start, end)
    _add_gpr_count(project_root, counts, start, end)
    _add_sentiment_counts(project_root, counts, start, end)
    _add_kalshi_count(project_root, counts, start, end)
    _add_metaculus_count(project_root, counts, start, end)

    counts["date"] = counts.get("sp500_adj_close", "not available")
    for planned_variable in [
        "egarch_conditional_volatility",
        "tvpvar_connectedness_measures",
    ]:
        counts.setdefault(planned_variable, "not yet generated")

    return counts


def _add_yahoo_counts(
    project_root: Path,
    counts: dict[str, object],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    path = project_root / "data" / "raw" / "finance" / "yahoo_finance_daily_2024_04_01_to_2026_03_31.csv"
    if not path.exists():
        return

    raw = pd.read_csv(path)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[raw["date"].between(start, end)]
    for column in [
        "ndx_adj_close",
        "bitcoin_adj_close",
        "nvda_adj_close",
        "googl_adj_close",
        "msft_adj_close",
        "sp500_adj_close",
        "dxy_close",
        "vix_close",
    ]:
        if column in raw:
            count = int(raw[column].notna().sum())
            counts[column] = count
            counts[f"{column}_log_return"] = count


def _add_epu_count(
    project_root: Path,
    counts: dict[str, object],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    path = project_root / "data" / "raw" / "finance" / "All_Daily_Policy_Data.csv"
    if not path.exists():
        return

    raw = pd.read_csv(path)
    raw["date"] = pd.to_datetime(
        {"year": raw["year"], "month": raw["month"], "day": raw["day"]}
    )
    counts["epu"] = int(raw["date"].between(start, end).sum())


def _add_gpr_count(
    project_root: Path,
    counts: dict[str, object],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    path = project_root / "data" / "raw" / "finance" / "data_gpr_daily_recent.csv"
    if not path.exists():
        return

    raw = pd.read_csv(path)
    raw["date"] = pd.to_datetime(raw["date"], format="%m/%d/%y", errors="coerce")
    counts["gpr"] = int(raw["date"].between(start, end).sum())


def _add_sentiment_counts(
    project_root: Path,
    counts: dict[str, object],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    path = project_root / "data" / "processed" / "combined_dedup_roberta_sentiment.csv"
    if not path.exists():
        return

    raw = pd.read_csv(path, usecols=["date", "source"])
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    raw = raw[raw["date"].between(start, end)]
    source_counts = raw["source"].value_counts()
    counts["sentiment_gdelt_sentiment_compound"] = int(source_counts.get("gdelt", 0))
    counts["sentiment_reddit_sentiment_compound"] = int(source_counts.get("reddit", 0))


def _add_kalshi_count(
    project_root: Path,
    counts: dict[str, object],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    path = project_root / "data" / "raw" / "kalshi" / "Kalshi Prices.csv"
    if not path.exists():
        return

    raw = pd.read_csv(path)
    raw["date"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(None).dt.normalize()
    raw = raw[raw["date"].between(start, end)]
    if "Before 2030" in raw:
        counts["kalshi_before_2030"] = int(raw["Before 2030"].notna().sum())


def _add_metaculus_count(
    project_root: Path,
    counts: dict[str, object],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    path = project_root / "data" / "raw" / "metaculus" / "Metaculus_forecast_data.csv"
    if not path.exists():
        return

    raw = pd.read_csv(path, usecols=["Forecaster Username", "Start Time"])
    raw = raw[raw["Forecaster Username"].eq("recency_weighted")].copy()
    raw["date"] = pd.to_datetime(raw["Start Time"], utc=True).dt.tz_convert(None).dt.normalize()
    counts["metaculus_recency_weighted_days_until_median"] = int(
        raw["date"].between(start, end).sum()
    )


def _format_date(value: object) -> str:
    if pd.isna(value):
        return "not available"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


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
    numeric = df.select_dtypes(include=[np.number])
    desc = numeric.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    desc.insert(0, "variable", desc.index)
    return desc.reset_index(drop=True)


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
