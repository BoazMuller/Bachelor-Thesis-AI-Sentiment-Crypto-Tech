from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

START_DATE = pd.Timestamp("2024-04-01")
END_DATE = pd.Timestamp("2026-03-31")

FILL_METHOD = "ffill"

RAW_FINANCE_DIR = PROJECT_ROOT / "data" / "raw" / "finance"
RAW_KALSHI_DIR = PROJECT_ROOT / "data" / "raw" / "kalshi"
RAW_METACULUS_DIR = PROJECT_ROOT / "data" / "raw" / "metaculus"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

YAHOO_FINANCE_CSV = RAW_FINANCE_DIR / "yahoo_finance_daily_2024_04_01_to_2026_03_31.csv"
EPU_CSV = RAW_FINANCE_DIR / "All_Daily_Policy_Data.csv"
GPR_CSV = RAW_FINANCE_DIR / "data_gpr_daily_recent.csv"
SENTIMENT_CSV = PROCESSED_DATA_DIR / "combined_dedup_roberta_sentiment.csv"
KALSHI_CSV = RAW_KALSHI_DIR / "Kalshi Prices.csv"
METACULUS_FORECAST_CSV = RAW_METACULUS_DIR / "Metaculus_forecast_data.csv"
OUTPUT_CSV = PROCESSED_DATA_DIR / "combined_time_series.csv"

PRICE_COLUMNS = [
    "ndx_adj_close",
    "bitcoin_adj_close",
    "nvda_adj_close",
    "googl_adj_close",
    "msft_adj_close",
    "sp500_adj_close",
    "dxy_close",
]

SENTIMENT_VALUE_COLUMNS = ["roberta_sentiment_compound"]


def clean_column_name(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def date_range() -> pd.DatetimeIndex:
    return pd.date_range(START_DATE, END_DATE, freq="D")


def historical_date_range(df: pd.DataFrame) -> pd.DatetimeIndex:
    start = min(df["date"].min(), START_DATE)
    return pd.date_range(start, END_DATE, freq="D")


def filter_date_range(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["date"].between(START_DATE, END_DATE)].copy()


def fill_daily(df: pd.DataFrame, fill_method: str = FILL_METHOD) -> pd.DataFrame:
    if fill_method != "ffill":
        raise ValueError("fill_method must be 'ffill'.")

    filled = (
        df.sort_values("date")
        .drop_duplicates(subset="date", keep="last")
        .set_index("date")
        .reindex(historical_date_range(df))
    )
    filled.index.name = "date"

    return filter_date_range(getattr(filled, fill_method)().reset_index())


def load_yahoo_finance() -> tuple[pd.DataFrame, pd.Series]:
    finance = pd.read_csv(YAHOO_FINANCE_CSV)
    finance["date"] = pd.to_datetime(finance["date"])
    finance = filter_date_range(finance).sort_values("date")

    trading_dates = finance.loc[finance["sp500_adj_close"].notna(), "date"].copy()
    trading_finance = finance[finance["date"].isin(trading_dates)].copy()

    for column in PRICE_COLUMNS:
        trading_finance[f"{column}_log_return"] = np.log(trading_finance[column]).diff()

    return trading_finance, trading_dates


def load_epu() -> pd.DataFrame:
    epu = pd.read_csv(EPU_CSV)
    epu["date"] = pd.to_datetime(
        {
            "year": epu["year"],
            "month": epu["month"],
            "day": epu["day"],
        }
    )

    return (
        filter_date_range(epu)
        .rename(columns={"daily_policy_index": "epu"})
        [["date", "epu"]]
    )


def load_gpr() -> pd.DataFrame:
    gpr = pd.read_csv(GPR_CSV)
    gpr["date"] = pd.to_datetime(gpr["date"], format="%m/%d/%y")

    return (
        filter_date_range(gpr)
        .rename(columns={"GPRD": "gpr"})
        [["date", "gpr"]]
    )


def assign_to_next_trading_day(
    df: pd.DataFrame,
    trading_dates: pd.Series,
) -> pd.DataFrame:
    trading_calendar = pd.DataFrame({"trading_date": pd.to_datetime(trading_dates)})

    return pd.merge_asof(
        df.sort_values("date"),
        trading_calendar.sort_values("trading_date"),
        left_on="date",
        right_on="trading_date",
        direction="forward",
    )


def load_sentiment(trading_dates: pd.Series) -> pd.DataFrame:
    sentiment = pd.read_csv(SENTIMENT_CSV)
    sentiment["date"] = pd.to_datetime(sentiment["date"]).dt.normalize()
    sentiment = sentiment[sentiment["date"].between(START_DATE, END_DATE)].copy()

    matched = assign_to_next_trading_day(sentiment, trading_dates)
    matched = matched.dropna(subset=["trading_date"])

    grouped = (
        matched.groupby(["trading_date", "source"], as_index=False)
        .agg({column: "mean" for column in SENTIMENT_VALUE_COLUMNS})
    )

    wide = grouped.pivot(index="trading_date", columns="source")
    wide.columns = [
        f"sentiment_{clean_column_name(source)}_{clean_column_name(metric).replace('roberta_', '')}"
        for metric, source in wide.columns
    ]

    return wide.reset_index().rename(columns={"trading_date": "date"})


def load_kalshi(trading_dates: pd.Series, fill_method: str = FILL_METHOD) -> pd.DataFrame:
    kalshi = pd.read_csv(KALSHI_CSV)
    kalshi["date"] = (
        pd.to_datetime(kalshi["timestamp"], utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )

    value_columns = ["Before 2030"]
    kalshi = kalshi[["date", *value_columns]].rename(
        columns={column: f"kalshi_{clean_column_name(column)}" for column in value_columns}
    )

    filled = fill_daily(kalshi[kalshi["date"].le(END_DATE)].copy(), fill_method=fill_method)
    return filled[filled["date"].isin(trading_dates)].reset_index(drop=True)


def parse_metaculus_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%m/%d/%y", errors="coerce")


def load_metaculus(
    trading_dates: pd.Series,
    fill_method: str = FILL_METHOD,
) -> pd.DataFrame:
    forecasts = pd.read_csv(METACULUS_FORECAST_CSV)
    forecasts = forecasts[forecasts["Forecaster Username"].eq("recency_weighted")].copy()
    forecasts["start_time"] = (
        pd.to_datetime(forecasts["Start Time"], utc=True)
        .dt.tz_convert(None)
    )
    forecasts["date"] = forecasts["start_time"].dt.normalize()
    forecasts["median_date"] = parse_metaculus_date(forecasts["Median"])
    forecasts["days_until_median"] = (
        forecasts["median_date"] - forecasts["start_time"]
    ).dt.total_seconds() / 86_400

    forecasts = forecasts.sort_values("start_time")
    daily = (
        forecasts.groupby(["date", "Forecaster Username"], as_index=False)
        .tail(1)
        .rename(columns={"Forecaster Username": "forecast_source"})
    )

    value_columns = ["days_until_median"]
    daily = daily[["date", "forecast_source", *value_columns]]

    wide = daily.pivot(index="date", columns="forecast_source", values=value_columns)
    wide.columns = [
        f"metaculus_{clean_column_name(source)}_{clean_column_name(metric)}"
        for metric, source in wide.columns
    ]
    wide = wide.reset_index()

    filled = fill_daily(wide[wide["date"].le(END_DATE)].copy(), fill_method=fill_method)
    return filled[filled["date"].isin(trading_dates)].reset_index(drop=True)


def build_time_series(fill_method: str = FILL_METHOD) -> pd.DataFrame:
    finance, trading_dates = load_yahoo_finance()

    controls = (
        finance.merge(load_epu(), how="left", on="date")
        .merge(load_gpr(), how="left", on="date")
        .merge(load_sentiment(trading_dates), how="left", on="date")
        .merge(load_kalshi(trading_dates, fill_method=fill_method), how="left", on="date")
        .merge(load_metaculus(trading_dates, fill_method=fill_method), how="left", on="date")
    )

    return controls.sort_values("date").reset_index(drop=True)


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    time_series = build_time_series(fill_method=FILL_METHOD)
    time_series.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote {len(time_series):,} rows and {len(time_series.columns):,} columns to {OUTPUT_CSV}")
    print(time_series.isna().sum().sort_values(ascending=False).head(15))


if __name__ == "__main__":
    main()
