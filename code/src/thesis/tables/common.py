from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import jarque_bera, kurtosis, skew
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

DATE_COLUMN = "date"
GDELT_SENTIMENT = "sentiment_gdelt_sentiment_compound"
REDDIT_SENTIMENT = "sentiment_reddit_sentiment_compound"
RAW_AIS = "raw_ais"
EXPECTATION_ADJUSTED_AIS = "expectation_adjusted_ais"
ROBUST_EXPECTATION_ADJUSTED_AIS = "robust_expectation_adjusted_ais"
INVERTED_METACULUS = "metaculus_inverted_days_until_median"
LAGGED_EXPECTATION_ADJUSTED_AIS = "lagged_expectation_adjusted_ais"

RETURN_COLUMNS = [
    "bitcoin_adj_close_log_return",
    "ndx_adj_close_log_return",
]

BASELINE_CONTROLS = [
    "kalshi_before_2030",
    INVERTED_METACULUS,
    "sp500_adj_close_log_return",
    "vix_close",
]

ROBUST_CONTROLS = BASELINE_CONTROLS + [
    "epu",
    "dxy_close_log_return",
    "gpr",
]

SOURCE_SENTIMENT_COLUMNS = [GDELT_SENTIMENT, REDDIT_SENTIMENT]
DFM_FACTOR_COUNT = 1
DFM_FACTOR_ORDER_CANDIDATES = tuple(range(1, 6))
DFM_EM_MAXITER = 1000
DFM_EM_TOLERANCE = 1e-6
DFM_IDIOSYNCRATIC_AR1 = False
_AIS_CONSTRUCTION_CACHE: dict[str, "DynamicFactorConstruction"] = {}


@dataclass(frozen=True)
class DynamicFactorConstruction:
    scores: pd.Series
    loadings: pd.Series
    lag_selection: pd.DataFrame
    selected_factor_order: int
    bic: float
    llf: float
    converged: bool
    iterations: int
    standardized_sources: pd.DataFrame


PCAConstruction = DynamicFactorConstruction

def read_daily_time_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if DATE_COLUMN not in df.columns:
        raise ValueError(f"Expected a '{DATE_COLUMN}' column in {path}")
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    return df.sort_values(DATE_COLUMN).reset_index(drop=True)

def add_sentiment_measures(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    if "metaculus_recency_weighted_days_until_median" in enriched.columns:
        enriched[INVERTED_METACULUS] = -pd.to_numeric(
            enriched["metaculus_recency_weighted_days_until_median"],
            errors="coerce",
        )

    dfm_result = construct_ais(enriched)
    enriched[RAW_AIS] = dfm_result.scores
    enriched[EXPECTATION_ADJUSTED_AIS] = residualize_series(
        enriched,
        dependent=RAW_AIS,
        controls=BASELINE_CONTROLS,
    )
    enriched[ROBUST_EXPECTATION_ADJUSTED_AIS] = residualize_series(
        enriched,
        dependent=RAW_AIS,
        controls=ROBUST_CONTROLS,
    )
    enriched[LAGGED_EXPECTATION_ADJUSTED_AIS] = enriched[EXPECTATION_ADJUSTED_AIS].shift(1)
    return enriched

def construct_ais(df: pd.DataFrame) -> DynamicFactorConstruction:
    available = [column for column in SOURCE_SENTIMENT_COLUMNS if column in df.columns]
    if len(available) < 2:
        raise ValueError("AIS construction requires both GDELT/news and Reddit/social-media sentiment columns")

    source_data = df[available].apply(pd.to_numeric, errors="coerce")
    cache_key = _ais_cache_key(source_data)
    if cache_key in _AIS_CONSTRUCTION_CACHE:
        return _AIS_CONSTRUCTION_CACHE[cache_key]

    complete = source_data.dropna()
    if len(complete) < 3:
        raise ValueError("AIS construction requires at least three complete sentiment observations")

    means = complete.mean()
    stds = complete.std(ddof=1).replace(0, np.nan)
    standardized = (source_data - means) / stds

    lag_selection = _dfm_lag_selection(standardized[available])
    converged_models = lag_selection[
        lag_selection["fit_success"].eq(True) & lag_selection["converged"].eq(True)
    ]
    if converged_models.empty:
        errors = "; ".join(lag_selection["error"].dropna().astype(str))
        raise RuntimeError(f"AIS DFM estimation did not converge for any candidate factor order: {errors}")

    selected_row = converged_models.sort_values(["bic", "factor_order"]).iloc[0]
    selected_factor_order = int(selected_row["factor_order"])
    result = selected_row["result"]

    raw_scores = result.factors.filtered.iloc[:, 0].reindex(df.index)
    loadings = _dfm_loading_params(result, available)

    average_standardized_sentiment = standardized[available].mean(axis=1)
    if raw_scores.corr(average_standardized_sentiment) < 0:
        raw_scores = -raw_scores
        loadings = -loadings

    score_mean = raw_scores.mean()
    score_std = raw_scores.std(ddof=1)
    scores = ((raw_scores - score_mean) / score_std).rename(RAW_AIS)
    if not np.isfinite(score_std) or score_std == 0:
        scores = raw_scores.rename(RAW_AIS)
    else:
        loadings = loadings * score_std

    construction = DynamicFactorConstruction(
        scores=scores,
        loadings=loadings.rename("factor_loading"),
        lag_selection=lag_selection.drop(columns=["result"]).reset_index(drop=True),
        selected_factor_order=selected_factor_order,
        bic=float(result.bic),
        llf=float(result.llf),
        converged=_mle_converged(result),
        iterations=_mle_iterations(result),
        standardized_sources=standardized,
    )
    _AIS_CONSTRUCTION_CACHE[cache_key] = construction
    return construction

def _ais_cache_key(source_data: pd.DataFrame) -> str:
    values_hash = pd.util.hash_pandas_object(source_data, index=True).to_numpy().tobytes()
    columns = "|".join(source_data.columns).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(columns)
    digest.update(str(source_data.shape).encode("utf-8"))
    digest.update(values_hash)
    return digest.hexdigest()

def _dfm_lag_selection(standardized_sources: pd.DataFrame) -> pd.DataFrame:
    nobs = int(standardized_sources.dropna(how="all").shape[0])
    candidate_orders = [order for order in DFM_FACTOR_ORDER_CANDIDATES if order < max(nobs - 1, 1)]
    if not candidate_orders:
        raise ValueError("AIS DFM lag selection requires enough sentiment observations for at least one factor lag")

    rows: list[dict[str, object]] = []
    for factor_order in candidate_orders:
        result = None
        try:
            model = DynamicFactorMQ(
                standardized_sources,
                factors=DFM_FACTOR_COUNT,
                factor_orders=factor_order,
                idiosyncratic_ar1=DFM_IDIOSYNCRATIC_AR1,
                standardize=False,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
                result = model.fit(
                    method="em",
                    maxiter=DFM_EM_MAXITER,
                    tolerance=DFM_EM_TOLERANCE,
                    disp=False,
                    cov_type="none",
                )
        except Exception as exc:
            rows.append(
                {
                    "factor_order": factor_order,
                    "n_factors": DFM_FACTOR_COUNT,
                    "bic": np.nan,
                    "aic": np.nan,
                    "llf": np.nan,
                    "nobs": nobs,
                    "idiosyncratic_ar1": DFM_IDIOSYNCRATIC_AR1,
                    "converged": False,
                    "iterations": 0,
                    "fit_success": False,
                    "error": str(exc),
                    "result": result,
                }
            )
            continue
        converged = _mle_converged(result)
        rows.append(
            {
                "factor_order": factor_order,
                "n_factors": DFM_FACTOR_COUNT,
                "bic": float(result.bic) if converged else np.nan,
                "aic": float(result.aic),
                "llf": float(result.llf),
                "nobs": int(result.nobs),
                "idiosyncratic_ar1": DFM_IDIOSYNCRATIC_AR1,
                "converged": converged,
                "iterations": _mle_iterations(result),
                "fit_success": True,
                "error": "",
                "result": result,
            }
        )
    return pd.DataFrame(rows)

def _dfm_loading_params(result: object, variables: Sequence[str]) -> pd.Series:
    params = result.params
    loadings = {}
    for variable in variables:
        parameter_name = f"loading.0->{variable}"
        loadings[variable] = float(params[parameter_name]) if parameter_name in params.index else np.nan
    return pd.Series(loadings)

def _mle_iterations(result: object) -> int:
    iterations = result.mle_retvals.get("iter", result.mle_retvals.get("iterations", np.nan))
    return int(iterations) if pd.notna(iterations) else 0

def _mle_converged(result: object) -> bool:
    converged = result.mle_retvals.get("converged")
    if converged is not None:
        return bool(converged)
    iterations = _mle_iterations(result)
    return iterations > 0 and iterations < DFM_EM_MAXITER

def residualize_series(
    df: pd.DataFrame,
    *,
    dependent: str,
    controls: Sequence[str],
    cov_type: str | None = None,
) -> pd.Series:
    available_controls = [column for column in controls if column in df.columns]
    model_data = df[[dependent] + available_controls].apply(pd.to_numeric, errors="coerce").dropna()
    residuals = pd.Series(np.nan, index=df.index, name=f"{dependent}_residual")
    if len(model_data) <= len(available_controls) + 2:
        return residuals

    x = sm.add_constant(model_data[available_controls], has_constant="add")
    model = sm.OLS(model_data[dependent], x).fit(cov_type=cov_type) if cov_type else sm.OLS(model_data[dependent], x).fit()
    residuals.loc[model_data.index] = model.resid
    return residuals

def descriptive_stats_table(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce")
        non_missing = series.dropna()
        if non_missing.empty:
            continue
        jb = jarque_bera(non_missing)
        values = {
            "N": int(non_missing.shape[0]),
            "mean": float(non_missing.mean()),
            "standard_deviation": float(non_missing.std(ddof=1)),
            "min": float(non_missing.min()),
            "max": float(non_missing.max()),
            "skewness": float(skew(non_missing, bias=False)) if len(non_missing) > 2 else np.nan,
            "kurtosis": float(kurtosis(non_missing, fisher=False, bias=False)) if len(non_missing) > 3 else np.nan,
            "jarque_bera": float(jb.statistic),
            "jarque_bera_p_value": float(jb.pvalue),
            "missing": int(series.isna().sum()),
        }
        for statistic, value in values.items():
            rows.append(
                {
                    "variable": column,
                    "statistic": statistic,
                    "value": value,
                    "missing": int(series.isna().sum()),
                }
            )
    return pd.DataFrame(rows)

def regression_table(
    df: pd.DataFrame,
    *,
    dependent: str,
    specifications: dict[str, Sequence[str]],
    cov_type: str = "HC3",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec_name, regressors in specifications.items():
        model_data = df[[dependent] + list(regressors)].apply(pd.to_numeric, errors="coerce").dropna()
        if len(model_data) <= len(regressors) + 2:
            rows.append(
                {
                    "specification": spec_name,
                    "dependent_variable": dependent,
                    "term": "",
                    "coefficient": np.nan,
                    "standard_error": np.nan,
                    "p_value": np.nan,
                    "r_squared": np.nan,
                    "adjusted_r_squared": np.nan,
                    "nobs": int(len(model_data)),
                    "note": "Skipped: insufficient complete observations",
                }
            )
            continue
        x = sm.add_constant(model_data[list(regressors)], has_constant="add")
        model = sm.OLS(model_data[dependent], x).fit(cov_type=cov_type)
        for term in model.params.index:
            rows.append(
                {
                    "specification": spec_name,
                    "dependent_variable": dependent,
                    "term": term,
                    "coefficient": float(model.params[term]),
                    "standard_error": float(model.bse[term]),
                    "p_value": float(model.pvalues[term]),
                    "r_squared": float(model.rsquared),
                    "adjusted_r_squared": float(model.rsquared_adj),
                    "nobs": int(model.nobs),
                    "covariance_type": cov_type,
                    "note": "",
                }
            )
    return pd.DataFrame(rows)
