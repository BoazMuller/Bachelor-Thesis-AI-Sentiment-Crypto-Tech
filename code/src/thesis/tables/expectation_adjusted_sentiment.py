from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from thesis.tables.common import (
    BASELINE_CONTROLS,
    EXPECTATION_ADJUSTED_AIS,
    INVERTED_METACULUS_DIFF,
    KALSHI_DIFF,
    LOG_EPU,
    LOG_GPR,
    RAW_AIS,
    add_sentiment_measures,
    descriptive_stats_table,
    expectation_adjusted_sample,
    regression_table,
)
from thesis.modeling.diagnostics import stationarity_tests

def expectation_adjusted_dataset(df: pd.DataFrame) -> pd.DataFrame:
    sample = expectation_adjusted_sample(df, extra_columns=[RAW_AIS])
    columns = [
        "date",
        RAW_AIS,
        EXPECTATION_ADJUSTED_AIS,
        *[column for column in BASELINE_CONTROLS if column in sample.columns],
    ]
    return sample[[column for column in columns if column in sample.columns]]

def expectation_adjusted_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    dataset = expectation_adjusted_dataset(df)
    columns = [
        column
        for column in [RAW_AIS, EXPECTATION_ADJUSTED_AIS] + BASELINE_CONTROLS
        if column in dataset.columns
    ]
    return descriptive_stats_table(dataset, columns)

def expectation_adjusted_pre_estimation_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    dataset = expectation_adjusted_dataset(df)
    columns = [
        column
        for column in [RAW_AIS, *BASELINE_CONTROLS]
        if column in dataset.columns
    ]
    rows = []
    if columns:
        stationarity = stationarity_tests(dataset, columns)
        for _, row in stationarity.iterrows():
            rows.append(
                {
                    "diagnostic": "stationarity",
                    "variable": row["variable"],
                    "test": row["test"],
                    "lag": row.get("lags", pd.NA),
                    "statistic": row.get("statistic", pd.NA),
                    "p_value": row.get("p_value", pd.NA),
                    "nobs": row.get("nobs", pd.NA),
                    "interpretation": row.get("interpretation", ""),
                }
            )
    return pd.DataFrame(rows)

def table_19_prediction_market_control_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variable": KALSHI_DIFF,
                "source": "Kalshi AGI-related prediction market",
                "role": "Prediction-market expectation control (differenced)",
                "transformation": "First difference of Before 2030 contract price, aligned to trading days",
                "units": "Percentage points",
                "timing": "Contemporaneous daily control",
            },
            {
                "variable": INVERTED_METACULUS_DIFF,
                "source": "Metaculus AGI forecast history",
                "role": "Forecast expectation control (differenced)",
                "transformation": "First difference of negative days until median forecast date; positive values indicate nearer expected AGI timing",
                "units": "Days",
                "timing": "Contemporaneous daily control",
            },
            {
                "variable": "sp500_adj_close_log_return",
                "source": "Yahoo Finance (^GSPC)",
                "role": "Broad equity-market control",
                "transformation": "Daily log return",
                "units": "Log points",
                "timing": "Contemporaneous daily control",
            },
            {
                "variable": "vix_close",
                "source": "Yahoo Finance (^VIX)",
                "role": "Market uncertainty control",
                "transformation": "Close level",
                "units": "Index points",
                "timing": "Contemporaneous daily control",
            },
            {
                "variable": LOG_EPU,
                "source": "Economic Policy Uncertainty daily policy index",
                "role": "Robustness macro-financial control",
                "transformation": "Natural logarithm of daily index level",
                "units": "Log points",
                "timing": "Contemporaneous daily control",
            },
            {
                "variable": "dxy_close_log_return",
                "source": "Yahoo Finance (DX-Y.NYB)",
                "role": "Robustness dollar control",
                "transformation": "Daily log return",
                "units": "Log points",
                "timing": "Contemporaneous daily control",
            },
            {
                "variable": LOG_GPR,
                "source": "Geopolitical Risk daily index",
                "role": "Robustness geopolitical-risk control",
                "transformation": "Natural logarithm of daily index level",
                "units": "Log points",
                "timing": "Contemporaneous daily control",
            },
        ]
    )

def table_20_residual_regression_sample_alignment(df: pd.DataFrame) -> pd.DataFrame:
    enriched = add_sentiment_measures(df)
    groups = {
        "ais_only": [RAW_AIS],
        "orthogonalization": [RAW_AIS] + [column for column in BASELINE_CONTROLS if column in enriched.columns],
    }
    rows: list[dict[str, object]] = []
    for sample_name, columns in groups.items():
        sample = enriched[columns]
        rows.append(
            {
                "sample": sample_name,
                "variables": ", ".join(columns),
                "n_before_dropna": int(len(sample)),
                "n_after_dropna": int(sample.dropna().shape[0]),
                "missing_observations_by_variable": "; ".join(
                    f"{column}: {int(sample[column].isna().sum())}" for column in columns
                ),
            }
        )
    return pd.DataFrame(rows)

def table_21_correlation_matrix_multicollinearity(df: pd.DataFrame) -> pd.DataFrame:
    enriched = expectation_adjusted_dataset(df)
    variables = [RAW_AIS] + [column for column in BASELINE_CONTROLS if column in enriched.columns]
    complete = enriched[variables].apply(pd.to_numeric, errors="coerce").dropna()
    rows: list[dict[str, object]] = []
    corr = complete.corr()
    for left in corr.index:
        for right in corr.columns:
            rows.append(
                {
                    "diagnostic": "correlation",
                    "variable": left,
                    "comparison_variable": right,
                    "value": float(corr.loc[left, right]),
                    "sample": "baseline_controls_complete_case",
                }
            )

    if len(complete) > len(variables) + 1:
        x = sm.add_constant(complete[variables], has_constant="add")
        for idx, column in enumerate(x.columns):
            if column == "const":
                continue
            rows.append(
                {
                    "diagnostic": "vif",
                    "variable": column,
                    "comparison_variable": "",
                    "value": float(variance_inflation_factor(x.to_numpy(), idx)),
                    "sample": "baseline_controls_complete_case",
                }
            )
    return pd.DataFrame(rows)

def table_22_orthogonalization_regression_results(
    df: pd.DataFrame,
    *,
    cov_type: str = "HC3",
) -> pd.DataFrame:
    enriched = expectation_adjusted_sample(df, extra_columns=[RAW_AIS])
    return regression_table(
        enriched,
        dependent=RAW_AIS,
        specifications={
            "baseline": [column for column in BASELINE_CONTROLS if column in enriched.columns],
            "prediction_markets_only": [
                column for column in [KALSHI_DIFF, INVERTED_METACULUS_DIFF] if column in enriched.columns
            ],
        },
        cov_type=cov_type,
    )

def table_23_residual_ais_validation(df: pd.DataFrame) -> pd.DataFrame:
    enriched = expectation_adjusted_dataset(df)
    rows: list[dict[str, object]] = []
    for variable in [EXPECTATION_ADJUSTED_AIS]:
        stats = descriptive_stats_table(enriched, [variable])
        for _, row in stats.iterrows():
            rows.append(
                {
                    "measure": variable,
                    "validation_item": row["statistic"],
                    "value": row["value"],
                }
            )
        rows.append(
            {
                "measure": variable,
                "validation_item": f"correlation_with_{RAW_AIS}",
                "value": float(enriched[[variable, RAW_AIS]].corr().iloc[0, 1]),
            }
        )
        for control in [column for column in BASELINE_CONTROLS if column in enriched.columns]:
            rows.append(
                {
                    "measure": variable,
                    "validation_item": f"correlation_with_{control}",
                    "value": float(enriched[[variable, control]].corr().iloc[0, 1]),
                }
            )
    return pd.DataFrame(rows)

def table_24_raw_ais_versus_residual_ais(df: pd.DataFrame) -> pd.DataFrame:
    enriched = expectation_adjusted_dataset(df)
    columns = [RAW_AIS, EXPECTATION_ADJUSTED_AIS]
    stats = descriptive_stats_table(enriched, columns)
    corr = enriched[columns].corr()
    rows = stats.to_dict("records")
    for left in columns:
        for right in columns:
            rows.append(
                {
                    "variable": left,
                    "statistic": f"correlation_with_{right}",
                    "value": float(corr.loc[left, right]),
                    "missing": int(enriched[left].isna().sum()),
                }
            )
    return pd.DataFrame(rows)
