from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

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
    "log_epu": VariableMetadata(
        symbol="log(EPU_t)",
        source="Economic Policy Uncertainty daily policy index",
        raw_frequency="Daily calendar",
        transformed_frequency="Trading day",
        transformation="Natural logarithm of the daily policy index",
        expected_sign_or_role="Policy uncertainty control (log-transformed to mitigate outliers)",
        units="Log points",
        treatment_of_missing_values="Inherits EPU missing value treatment",
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
    "log_gpr": VariableMetadata(
        symbol="log(GPR_t)",
        source="Geopolitical Risk daily index",
        raw_frequency="Daily calendar",
        transformed_frequency="Trading day",
        transformation="Natural logarithm of the GPRD level",
        expected_sign_or_role="Geopolitical risk control (log-transformed to mitigate outliers)",
        units="Log points",
        treatment_of_missing_values="Inherits GPR missing value treatment",
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
    "kalshi_before_2030_diff": VariableMetadata(
        symbol="ΔKalshi_t",
        source="Kalshi AGI-related market price",
        raw_frequency="Intermittent daily market observations",
        transformed_frequency="Trading day",
        transformation="First difference of Before 2030 contract price",
        expected_sign_or_role="Prediction-market AI expectation control (differenced); positive changes imply increasing probability of AGI before 2030",
        units="Percentage points",
        treatment_of_missing_values="First observation is missing after differencing",
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
    "metaculus_inverted_days_until_median_diff": VariableMetadata(
        symbol="ΔMetaculusSooner_t",
        source="Derived from Metaculus AGI date forecast history",
        raw_frequency="Intermittent forecast updates",
        transformed_frequency="Trading day",
        transformation="First difference of inverted Metaculus days-until-median",
        expected_sign_or_role="Forecast expectation control aligned with Kalshi direction (differenced)",
        units="Days",
        treatment_of_missing_values="First observation is missing after differencing",
    ),
    "raw_ais": VariableMetadata(
        symbol="AIS_t",
        source="One-factor dynamic factor model of standardized GDELT/news and Reddit/social-media sentiment",
        raw_frequency="Daily source-level sentiment",
        transformed_frequency="Trading day",
        transformation="One-sided filtered latent DFM factor estimated by EM; factor-order lag length selected by BIC among converged candidates; sign normalized positive",
        expected_sign_or_role="AI sentiment index; robustness check for expectation-adjusted sentiment",
        units="Standardized latent factor score",
        treatment_of_missing_values="State-space DFM filtering over available standardized source sentiment measures",
        notes="Higher values mean more positive AI sentiment",
    ),
    "expectation_adjusted_ais": VariableMetadata(
        symbol="EAIS_t",
        source="Residual from AIS orthogonalization regression",
        raw_frequency="Trading day",
        transformed_frequency="Trading day",
        transformation="Residual from AIS on Kalshi diff, inverted Metaculus diff, S&P 500 return, VIX, log(EPU), DXY return, and log(GPR)",
        expected_sign_or_role="Headline sentiment variable for spillover regressions",
        units="OLS residual",
        treatment_of_missing_values="Complete-case sample for AIS and baseline controls",
        notes="Benchmark sentiment variable throughout the thesis",
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
    "used_in_models",
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
    if column.endswith("_conditional_volatility"):
        asset = column.removesuffix("_conditional_volatility")
        return VariableMetadata(
            symbol=f"sigma_{asset},t",
            source="Plain EGARCH(1,1) fitted model",
            raw_frequency="Trading day returns",
            transformed_frequency="Trading day",
            transformation="Conditional standard deviation estimated from EGARCH(1,1)",
            expected_sign_or_role="Volatility input for TVP-VAR connectedness systems",
            units="Return standard deviation",
            treatment_of_missing_values="Complete-case EGARCH volatility sample",
        )
    if _is_connectedness_measure(column):
        return VariableMetadata(
            symbol="C_t",
            source="R ConnectednessApproach TVP-VAR",
            raw_frequency="Trading day EGARCH conditional volatilities",
            transformed_frequency="Trading day",
            transformation="Generalized FEVD-based connectedness measure from TVP-VAR",
            expected_sign_or_role="Dependent variable in connectedness-on-sentiment regressions",
            units="Share / percentage of forecast error variance",
            treatment_of_missing_values="Complete-case connectedness output sample",
        )
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
    if project_root is None:
        from thesis.paths import PROJECT_ROOT
        project_root = PROJECT_ROOT

    if project_root:
        vol_path = project_root / "results" / "tables" / "tvpvar_connectedness" / "tvpvar_connectedness_dataset.csv"
        if vol_path.exists():
            vol_df = pd.read_csv(vol_path)
            vol_df["date"] = pd.to_datetime(vol_df["date"])
            cols_to_use = [col for col in vol_df.columns if col not in df.columns or col == "date"]
            if len(cols_to_use) > 1:
                df = pd.merge(df, vol_df[cols_to_use], on="date", how="left")

        conn_path = project_root / "results" / "tables" / "tvpvar_connectedness" / "connectedness_regression_dataset.csv"
        if conn_path.exists():
            conn_df = pd.read_csv(conn_path)
            conn_df["date"] = pd.to_datetime(conn_df["date"])
            if "benchmark_tci_tci" in conn_df.columns and "tvpvar_connectedness_measures" not in df.columns:
                df = pd.merge(
                    df,
                    conn_df[["date", "benchmark_tci_tci"]].rename(columns={"benchmark_tci_tci": "tvpvar_connectedness_measures"}),
                    on="date",
                    how="left"
                )

    raw_counts = raw_observation_counts(project_root, df) if project_root else {}
    rows: list[dict[str, object]] = []

    inventory_variables = list(df.columns)
    for planned_variable in [
        "bitcoin_conditional_volatility",
        "ndx_conditional_volatility",
        "nvda_conditional_volatility",
        "googl_conditional_volatility",
        "msft_conditional_volatility",
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
            "used_in_models": models_using_variable(column),
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
        "bitcoin_conditional_volatility",
        "ndx_conditional_volatility",
        "nvda_conditional_volatility",
        "googl_conditional_volatility",
        "msft_conditional_volatility",
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

def models_using_variable(column: str) -> str:
    models: list[str] = []
    if column == "date":
        models = [
            "armax_egarchx",
            "tvpvar_connectedness",
            "dfm_em",
            "expectation_adjusted_sentiment",
            "spillover_regressions",
        ]
    elif column in {
        "sentiment_gdelt_sentiment_compound",
        "sentiment_reddit_sentiment_compound",
    }:
        models = ["dfm_em", "expectation_adjusted_sentiment"]
    elif column in {"raw_ais", "expectation_adjusted_ais"}:
        models = ["armax_egarchx", "expectation_adjusted_sentiment", "spillover_regressions"]
        if column == "raw_ais":
            models.insert(0, "dfm_em")
    elif column == "lagged_expectation_adjusted_ais":
        models = ["armax_egarchx", "spillover_regressions"]
    elif column.endswith("_conditional_volatility") or column == "egarch_conditional_volatility":
        models = ["tvpvar_connectedness", "spillover_regressions"]
    elif column == "tvpvar_connectedness_measures" or _is_connectedness_measure(column):
        models = ["tvpvar_connectedness", "spillover_regressions"]
    elif column.endswith("_log_return"):
        models = []
        if column in {"bitcoin_adj_close_log_return", "ndx_adj_close_log_return"}:
            models = ["armax_egarchx", "tvpvar_connectedness"]
        elif column in {
            "nvda_adj_close_log_return",
            "googl_adj_close_log_return",
            "msft_adj_close_log_return",
        }:
            models = ["tvpvar_connectedness"]
        if column in {
            "sp500_adj_close_log_return",
            "dxy_close_log_return",
        }:
            models.append("expectation_adjusted_sentiment")
            models.append("spillover_regressions")
    elif column in {"kalshi_before_2030", "kalshi_before_2030_diff", "metaculus_recency_weighted_days_until_median", "metaculus_inverted_days_until_median", "metaculus_inverted_days_until_median_diff", "vix_close", "epu", "log_epu", "gpr", "log_gpr"}:
        models = ["expectation_adjusted_sentiment", "spillover_regressions"]
    elif column.endswith("_adj_close") or column == "dxy_close":
        models = ["data_preparation"]

    return ", ".join(dict.fromkeys(models))

def _is_connectedness_measure(column: str) -> bool:
    lower = column.lower()
    return any(token in lower for token in ("connectedness", "_tci_", "_to_", "_from_", "_net_", "_npdc_"))
