from __future__ import annotations

import pandas as pd

from thesis.tables.common import (
    DATE_COLUMN,
    EXPECTATION_ADJUSTED_AIS,
    LAGGED_EXPECTATION_ADJUSTED_AIS,
    RAW_AIS,
    RETURN_COLUMNS,
    add_sentiment_measures,
    expectation_adjusted_sample,
)

def prepare_egarch_restricted_input(df: pd.DataFrame) -> pd.DataFrame:
    enriched = expectation_adjusted_sample(
        df,
        include_lagged=True,
        extra_columns=RETURN_COLUMNS,
    )
    columns = [DATE_COLUMN] + [column for column in RETURN_COLUMNS if column in enriched.columns] + [
        EXPECTATION_ADJUSTED_AIS,
        LAGGED_EXPECTATION_ADJUSTED_AIS,
    ]
    return enriched[columns].reset_index(drop=True)

def prepare_egarch_full_input(df: pd.DataFrame) -> pd.DataFrame:
    enriched = add_sentiment_measures(df)
    columns = [DATE_COLUMN] + [column for column in RETURN_COLUMNS if column in enriched.columns] + [
        RAW_AIS,
    ]
    return enriched[columns].dropna(subset=columns).reset_index(drop=True)

def prepare_egarch_input(df: pd.DataFrame) -> pd.DataFrame:
    return prepare_egarch_restricted_input(df)

def prepare_tvpvar_return_input(df: pd.DataFrame) -> pd.DataFrame:
    enriched = add_sentiment_measures(df)
    columns = [
        DATE_COLUMN,
        "bitcoin_adj_close_log_return",
        "ndx_adj_close_log_return",
        "nvda_adj_close_log_return",
        "googl_adj_close_log_return",
        "msft_adj_close_log_return",
        EXPECTATION_ADJUSTED_AIS,
    ]
    return enriched[[column for column in columns if column in enriched.columns]].dropna(
        subset=[
            column
            for column in columns
            if column in enriched.columns and column != EXPECTATION_ADJUSTED_AIS
        ]
    ).reset_index(drop=True)
