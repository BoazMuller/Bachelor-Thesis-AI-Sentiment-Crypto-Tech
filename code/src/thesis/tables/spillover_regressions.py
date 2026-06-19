from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
LAGGED_RAW_AIS = "lagged_raw_ais"
EAIS_LAG_COLUMNS = [
    EXPECTATION_ADJUSTED_AIS,
    *[f"{EXPECTATION_ADJUSTED_AIS}_lag_{lag}" for lag in range(1, 6)],
]

def merge_sentiment_if_needed(dataset: pd.DataFrame, time_series: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        RAW_AIS,
        EXPECTATION_ADJUSTED_AIS,
        LAGGED_EXPECTATION_ADJUSTED_AIS,
        *CONNECTEDNESS_REGRESSION_CONTROLS,
    ]
    missing = [column for column in candidates if column not in dataset.columns and column in time_series.columns]
    if not missing:
        return dataset
    return dataset.merge(time_series[[DATE_COLUMN, *missing]], on=DATE_COLUMN, how="left")

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
    enriched = add_lagged_raw_ais(df)
    variables = dependent_columns + [
        column
        for column in [EXPECTATION_ADJUSTED_AIS, RAW_AIS, LAGGED_RAW_AIS] + CONNECTEDNESS_REGRESSION_CONTROLS
        if column in enriched.columns
    ]
    corr = enriched[variables].apply(pd.to_numeric, errors="coerce").corr()
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
    lagged = add_lagged_raw_ais(df)
    regressors = [
        column
        for column in [RAW_AIS, LAGGED_RAW_AIS] + CONNECTEDNESS_REGRESSION_CONTROLS
        if column in lagged.columns
    ]
    sample = raw_ais_sample(lagged, extra_columns=dependent_columns + regressors)
    return _multi_dependent_regressions(
        sample,
        dependent_columns,
        regressors,
        "robustness_raw_ais_contemporaneous_and_lagged",
        cov_type,
    )


def add_lagged_raw_ais(df: pd.DataFrame) -> pd.DataFrame:
    lagged = df.sort_values(DATE_COLUMN).reset_index(drop=True).copy()
    lagged[LAGGED_RAW_AIS] = pd.to_numeric(lagged[RAW_AIS], errors="coerce").shift(1)
    return lagged


def add_five_eais_lags(df: pd.DataFrame) -> pd.DataFrame:
    lagged = df.sort_values(DATE_COLUMN).reset_index(drop=True).copy()
    eais = pd.to_numeric(lagged[EXPECTATION_ADJUSTED_AIS], errors="coerce")
    for lag in range(1, 6):
        lagged[f"{EXPECTATION_ADJUSTED_AIS}_lag_{lag}"] = eais.shift(lag)
    return lagged


def table_30_five_lag_hac_regressions(
    df: pd.DataFrame,
    dependent_columns: list[str],
    *,
    maxlags: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    lagged = add_five_eais_lags(df)
    regressors = [
        column
        for column in EAIS_LAG_COLUMNS + CONNECTEDNESS_REGRESSION_CONTROLS
        if column in lagged.columns
    ]
    for dependent in dependent_columns:
        fitted = _fit_ols(lagged, dependent, regressors, cov_type="HAC", maxlags=maxlags)
        if fitted is None:
            rows.append(_skipped_regression_row("robustness_eais_t_to_t_minus_5", dependent, lagged))
            continue
        for term in fitted.params.index:
            rows.append(
                {
                    "specification": "robustness_eais_t_to_t_minus_5",
                    "dependent_variable": dependent,
                    "term": term,
                    "coefficient": float(fitted.params[term]),
                    "standard_error": float(fitted.bse[term]),
                    "p_value": float(fitted.pvalues[term]),
                    "r_squared": float(fitted.rsquared),
                    "adjusted_r_squared": float(fitted.rsquared_adj),
                    "nobs": int(fitted.nobs),
                    "covariance_type": f"HAC(maxlags={maxlags})",
                    "note": "",
                }
            )
    return pd.DataFrame(rows)


def table_31_five_lag_hac_joint_tests(
    df: pd.DataFrame,
    dependent_columns: list[str],
    *,
    maxlags: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    lagged = add_five_eais_lags(df)
    regressors = [
        column
        for column in EAIS_LAG_COLUMNS + CONNECTEDNESS_REGRESSION_CONTROLS
        if column in lagged.columns
    ]
    eais_terms = [column for column in EAIS_LAG_COLUMNS if column in regressors]
    for dependent in dependent_columns:
        fitted = _fit_ols(lagged, dependent, regressors, cov_type="HAC", maxlags=maxlags)
        if fitted is None:
            continue
        rows.extend(_eais_effect_tests(fitted, dependent, eais_terms, maxlags))
    return pd.DataFrame(rows)


def spillover_regression_diagnostics(
    df: pd.DataFrame,
    dependent_columns: list[str],
    *,
    maxlags: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    lagged = add_five_eais_lags(df)
    eais_complete = lagged.dropna(subset=EAIS_LAG_COLUMNS)
    rows.append(
        {
            "diagnostic": "five_lag_eais_sample",
            "dependent_variable": "",
            "variable": "",
            "lag": maxlags,
            "statistic": float(len(eais_complete)),
            "p_value": np.nan,
            "nobs": len(eais_complete),
            "note": "EAIS observations after constructing terms t through t-5, before dependent/control filtering",
        }
    )

    correlations = eais_complete[EAIS_LAG_COLUMNS].corr()
    for row_variable in EAIS_LAG_COLUMNS:
        for column_variable in EAIS_LAG_COLUMNS:
            rows.append(
                {
                    "diagnostic": "eais_lag_correlation",
                    "dependent_variable": "",
                    "variable": f"{row_variable}|{column_variable}",
                    "lag": np.nan,
                    "statistic": float(correlations.loc[row_variable, column_variable]),
                    "p_value": np.nan,
                    "nobs": len(eais_complete),
                    "note": "Pearson correlation",
                }
            )

    regressors = [
        column
        for column in EAIS_LAG_COLUMNS + CONNECTEDNESS_REGRESSION_CONTROLS
        if column in lagged.columns
    ]
    for dependent in dependent_columns:
        fitted = _fit_ols(lagged, dependent, regressors, cov_type="HAC", maxlags=maxlags)
        if fitted is None:
            continue
        model_data = lagged[[dependent, *regressors]].apply(pd.to_numeric, errors="coerce").dropna()
        design = sm.add_constant(model_data[regressors], has_constant="add")
        for index, variable in enumerate(design.columns):
            if variable == "const":
                continue
            rows.append(
                {
                    "diagnostic": "variance_inflation_factor",
                    "dependent_variable": dependent,
                    "variable": variable,
                    "lag": np.nan,
                    "statistic": float(variance_inflation_factor(design.to_numpy(), index)),
                    "p_value": np.nan,
                    "nobs": len(model_data),
                    "note": "",
                }
            )
        for lag_value, result in acorr_ljungbox(
            fitted.resid,
            lags=[lag for lag in (5, 10, 20) if lag < fitted.nobs - 1],
            return_df=True,
        ).iterrows():
            rows.append(
                {
                    "diagnostic": "ljung_box_regression_residual",
                    "dependent_variable": dependent,
                    "variable": "",
                    "lag": int(lag_value),
                    "statistic": float(result["lb_stat"]),
                    "p_value": float(result["lb_pvalue"]),
                    "nobs": int(fitted.nobs),
                    "note": "",
                }
            )
        for test in _eais_effect_tests(fitted, dependent, EAIS_LAG_COLUMNS, maxlags):
            rows.append(
                {
                    "diagnostic": f"hac_{test['test']}",
                    "dependent_variable": dependent,
                    "variable": " + ".join(EAIS_LAG_COLUMNS),
                    "lag": maxlags,
                    "statistic": test["test_statistic"],
                    "p_value": test["p_value"],
                    "nobs": test["nobs"],
                    "note": f"estimate={test['estimate']}; standard_error={test['standard_error']}",
                }
            )
    return pd.DataFrame(rows)


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


def _fit_ols(
    df: pd.DataFrame,
    dependent: str,
    regressors: list[str],
    *,
    cov_type: str,
    maxlags: int,
):
    model_data = df[[dependent, *regressors]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(model_data) <= len(regressors) + 2:
        return None
    x = sm.add_constant(model_data[regressors], has_constant="add")
    if cov_type == "HAC":
        return sm.OLS(model_data[dependent], x).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return sm.OLS(model_data[dependent], x).fit(cov_type=cov_type)


def _eais_effect_tests(fitted, dependent: str, eais_terms: list[str], maxlags: int) -> list[dict[str, object]]:
    terms = [term for term in eais_terms if term in fitted.params.index]
    indices = [fitted.params.index.get_loc(term) for term in terms]
    joint_restriction = np.zeros((len(indices), len(fitted.params)))
    for row, index in enumerate(indices):
        joint_restriction[row, index] = 1.0
    joint = fitted.wald_test(joint_restriction, scalar=True)

    cumulative_restriction = np.zeros((1, len(fitted.params)))
    cumulative_restriction[0, indices] = 1.0
    cumulative = fitted.wald_test(cumulative_restriction, scalar=True)
    cumulative_estimate = float(fitted.params[terms].sum())
    covariance = fitted.cov_params().loc[terms, terms].to_numpy()
    cumulative_se = float(np.sqrt(np.ones(len(terms)) @ covariance @ np.ones(len(terms))))
    common = {
        "specification": "robustness_eais_t_to_t_minus_5",
        "dependent_variable": dependent,
        "terms": ", ".join(terms),
        "nobs": int(fitted.nobs),
        "covariance_type": f"HAC(maxlags={maxlags})",
    }
    return [
        {
            **common,
            "test": "cumulative_effect",
            "estimate": cumulative_estimate,
            "standard_error": cumulative_se,
            "test_statistic": float(np.asarray(cumulative.statistic).squeeze()),
            "p_value": float(np.asarray(cumulative.pvalue).squeeze()),
            "degrees_of_freedom": 1,
        },
        {
            **common,
            "test": "joint_zero",
            "estimate": np.nan,
            "standard_error": np.nan,
            "test_statistic": float(np.asarray(joint.statistic).squeeze()),
            "p_value": float(np.asarray(joint.pvalue).squeeze()),
            "degrees_of_freedom": len(terms),
        },
    ]


def _skipped_regression_row(specification: str, dependent: str, df: pd.DataFrame) -> dict[str, object]:
    return {
        "specification": specification,
        "dependent_variable": dependent,
        "term": "",
        "coefficient": np.nan,
        "standard_error": np.nan,
        "p_value": np.nan,
        "r_squared": np.nan,
        "adjusted_r_squared": np.nan,
        "nobs": len(df),
        "covariance_type": "HAC(maxlags=5)",
        "note": "Skipped: insufficient complete observations",
    }
