from __future__ import annotations

import numpy as np
import pandas as pd

from thesis.tables.common import (
    BASELINE_CONTROLS,
    DATE_COLUMN,
    EXPECTATION_ADJUSTED_AIS,
    LAGGED_EXPECTATION_ADJUSTED_AIS,
    LOG_EPU,
    LOG_GPR,
    RAW_AIS,
    descriptive_stats_table,
    expectation_adjusted_sample,
    regression_table,
    raw_ais_sample,
)

CONNECTEDNESS_REGRESSION_CONTROLS = [
    "dxy_close_log_return",
    "vix_close",
    LOG_GPR,
    LOG_EPU,
]

CONNECTEDNESS_DEPENDENT_PATTERNS = ("_tci_", "_to_", "_from_")

def merge_sentiment_if_needed(dataset: pd.DataFrame, time_series: pd.DataFrame) -> pd.DataFrame:
    required = [RAW_AIS, EXPECTATION_ADJUSTED_AIS, LAGGED_EXPECTATION_ADJUSTED_AIS]
    if all(column in dataset.columns for column in required):
        return dataset
    merge_columns = [DATE_COLUMN] + [
        column
        for column in [RAW_AIS, EXPECTATION_ADJUSTED_AIS, LAGGED_EXPECTATION_ADJUSTED_AIS] + CONNECTEDNESS_REGRESSION_CONTROLS
        if column in time_series.columns
    ]
    return dataset.merge(time_series[merge_columns], on=DATE_COLUMN, how="left")

def infer_dependent_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        DATE_COLUMN,
        RAW_AIS,
        EXPECTATION_ADJUSTED_AIS,
        LAGGED_EXPECTATION_ADJUSTED_AIS,
        *BASELINE_CONTROLS,
    }
    columns: list[str] = []
    for column in df.select_dtypes(include=[np.number]).columns:
        lower = column.lower()
        if column in excluded or column.endswith("_log_return"):
            continue
        if any(pattern in lower for pattern in CONNECTEDNESS_DEPENDENT_PATTERNS):
            columns.append(column)
    return columns

def table_25_dataset_summary(df: pd.DataFrame, dependent_columns: list[str]) -> pd.DataFrame:
    stats = descriptive_stats_table(df, dependent_columns)
    if DATE_COLUMN in df:
        sample_period = f"{df[DATE_COLUMN].min().date()} to {df[DATE_COLUMN].max().date()}"
        stats["sample_period"] = sample_period
    return stats

def table_26_correlation_matrix(df: pd.DataFrame, dependent_columns: list[str]) -> pd.DataFrame:
    variables = dependent_columns + [
        column
        for column in [EXPECTATION_ADJUSTED_AIS, RAW_AIS] + CONNECTEDNESS_REGRESSION_CONTROLS
        if column in df.columns
    ]
    corr = df[variables].apply(pd.to_numeric, errors="coerce").corr()
    corr.insert(0, "variable", corr.index)
    return corr.reset_index(drop=True)

def table_27_baseline_regressions(df: pd.DataFrame, dependent_columns: list[str], cov_type: str) -> pd.DataFrame:
    regressors = [column for column in [EXPECTATION_ADJUSTED_AIS] + CONNECTEDNESS_REGRESSION_CONTROLS if column in df.columns]
    sample = expectation_adjusted_sample(df, extra_columns=dependent_columns + regressors)
    return _multi_dependent_regressions(sample, dependent_columns, regressors, "headline_contemporaneous", cov_type)

def table_28_lagged_regressions(df: pd.DataFrame, dependent_columns: list[str], cov_type: str) -> pd.DataFrame:
    regressors = [
        column
        for column in [EXPECTATION_ADJUSTED_AIS, LAGGED_EXPECTATION_ADJUSTED_AIS] + CONNECTEDNESS_REGRESSION_CONTROLS
        if column in df.columns
    ]
    sample = expectation_adjusted_sample(df, include_lagged=True, extra_columns=dependent_columns + regressors)
    return _multi_dependent_regressions(sample, dependent_columns, regressors, "robustness_lagged_sentiment", cov_type)

def table_29_raw_ais_regressions(df: pd.DataFrame, dependent_columns: list[str], cov_type: str) -> pd.DataFrame:
    regressors = [column for column in [RAW_AIS] + CONNECTEDNESS_REGRESSION_CONTROLS if column in df.columns]
    sample = raw_ais_sample(df, extra_columns=dependent_columns + regressors)
    return _multi_dependent_regressions(sample, dependent_columns, regressors, "robustness_raw_ais", cov_type)

def _multi_dependent_regressions(
    df: pd.DataFrame,
    dependent_columns: list[str],
    regressors: list[str],
    specification_prefix: str,
    cov_type: str,
) -> pd.DataFrame:
    rows = []
    for dependent in dependent_columns:
        table = regression_table(
            df,
            dependent=dependent,
            specifications={f"{specification_prefix}_{dependent}": regressors},
            cov_type=cov_type,
        )
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
