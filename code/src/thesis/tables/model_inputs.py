from __future__ import annotations

import pandas as pd

from thesis.tables.common import (
    DATE_COLUMN,
    EXPECTATION_ADJUSTED_AIS,
    LAGGED_EXPECTATION_ADJUSTED_AIS,
    RAW_AIS,
    RETURN_COLUMNS,
    ROBUST_EXPECTATION_ADJUSTED_AIS,
    add_sentiment_measures,
)

def prepare_egarch_input(df: pd.DataFrame) -> pd.DataFrame:
    enriched = add_sentiment_measures(df)
    columns = [DATE_COLUMN] + [column for column in RETURN_COLUMNS if column in enriched.columns] + [
        RAW_AIS,
        EXPECTATION_ADJUSTED_AIS,
        ROBUST_EXPECTATION_ADJUSTED_AIS,
        LAGGED_EXPECTATION_ADJUSTED_AIS,
    ]
    return enriched[columns].dropna().reset_index(drop=True)

def prepare_tvpvar_return_input(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        DATE_COLUMN,
        "bitcoin_adj_close_log_return",
        "ndx_adj_close_log_return",
        "nvda_adj_close_log_return",
        "googl_adj_close_log_return",
        "msft_adj_close_log_return",
    ]
    return df[[column for column in columns if column in df.columns]].dropna().reset_index(drop=True)
