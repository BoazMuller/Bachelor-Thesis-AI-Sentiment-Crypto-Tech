from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from thesis.tables.common import (
    BASELINE_CONTROLS,
    EXPECTATION_ADJUSTED_AIS,
    INVERTED_METACULUS,
    RAW_AIS,
    ROBUST_CONTROLS,
    ROBUST_EXPECTATION_ADJUSTED_AIS,
    add_sentiment_measures,
    descriptive_stats_table,
    regression_table,
)

def table_19_prediction_market_control_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variable": "kalshi_before_2030",
                "source": "Kalshi AGI-related prediction market",
                "role": "Prediction-market expectation control",
                "transformation": "Before 2030 contract price, aligned to trading days",
                "units": "Contract-implied price/probability points",
                "timing": "Contemporaneous daily control",
            },
            {
                "variable": INVERTED_METACULUS,
                "source": "Metaculus AGI forecast history",
                "role": "Forecast expectation control",
                "transformation": "Negative days until median forecast date; larger values indicate nearer expected AGI timing",
                "units": "Negative days",
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
                "variable": "epu",
                "source": "Economic Policy Uncertainty daily policy index",
                "role": "Robustness macro-financial control",
                "transformation": "Daily index level",
                "units": "Index points",
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
                "variable": "gpr",
                "source": "Geopolitical Risk daily index",
                "role": "Robustness geopolitical-risk control",
                "transformation": "Daily index level",
                "units": "Index points",
                "timing": "Contemporaneous daily control",
            },
        ]
    )

def table_20_residual_regression_sample_alignment(df: pd.DataFrame) -> pd.DataFrame:
    enriched = add_sentiment_measures(df)
    groups = {
        "ais_only": [RAW_AIS],
        "baseline_orthogonalization": [RAW_AIS] + [column for column in BASELINE_CONTROLS if column in enriched.columns],
        "robust_orthogonalization": [RAW_AIS] + [column for column in ROBUST_CONTROLS if column in enriched.columns],
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
    enriched = add_sentiment_measures(df)
    variables = [RAW_AIS] + [column for column in ROBUST_CONTROLS if column in enriched.columns]
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
                    "sample": "robust_controls_complete_case",
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
                    "sample": "robust_controls_complete_case",
                }
            )
    return pd.DataFrame(rows)

def table_22_orthogonalization_regression_results(
    df: pd.DataFrame,
    *,
    cov_type: str = "HC3",
) -> pd.DataFrame:
    enriched = add_sentiment_measures(df)
    return regression_table(
        enriched,
        dependent=RAW_AIS,
        specifications={
            "baseline": [column for column in BASELINE_CONTROLS if column in enriched.columns],
            "robust": [column for column in ROBUST_CONTROLS if column in enriched.columns],
        },
        cov_type=cov_type,
    )

def table_23_residual_ais_validation(df: pd.DataFrame) -> pd.DataFrame:
    enriched = add_sentiment_measures(df)
    rows: list[dict[str, object]] = []
    for variable in [EXPECTATION_ADJUSTED_AIS, ROBUST_EXPECTATION_ADJUSTED_AIS]:
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
        controls = BASELINE_CONTROLS if variable == EXPECTATION_ADJUSTED_AIS else ROBUST_CONTROLS
        for control in [column for column in controls if column in enriched.columns]:
            rows.append(
                {
                    "measure": variable,
                    "validation_item": f"correlation_with_{control}",
                    "value": float(enriched[[variable, control]].corr().iloc[0, 1]),
                }
            )
    return pd.DataFrame(rows)

def table_24_raw_ais_versus_residual_ais(df: pd.DataFrame) -> pd.DataFrame:
    enriched = add_sentiment_measures(df)
    columns = [RAW_AIS, EXPECTATION_ADJUSTED_AIS, ROBUST_EXPECTATION_ADJUSTED_AIS]
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
