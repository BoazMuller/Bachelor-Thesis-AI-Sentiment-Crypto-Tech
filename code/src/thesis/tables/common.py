from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import jarque_bera, kurtosis, skew
from sklearn.decomposition import PCA

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


@dataclass(frozen=True)
class PCAConstruction:
    scores: pd.Series
    loadings: pd.Series
    explained_variance: pd.Series
    explained_variance_ratio: pd.Series
    standardized_sources: pd.DataFrame

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

    pca_result = construct_ais(enriched)
    enriched[RAW_AIS] = pca_result.scores
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

def construct_ais(df: pd.DataFrame) -> PCAConstruction:
    available = [column for column in SOURCE_SENTIMENT_COLUMNS if column in df.columns]
    if len(available) < 2:
        raise ValueError("AIS construction requires both GDELT/news and Reddit/social-media sentiment columns")

    source_data = df[available].apply(pd.to_numeric, errors="coerce")
    complete = source_data.dropna()
    if len(complete) < 3:
        raise ValueError("AIS construction requires at least three complete sentiment observations")

    means = complete.mean()
    stds = complete.std(ddof=1).replace(0, np.nan)
    standardized = (source_data - means) / stds
    fit_data = standardized.loc[complete.index]

    pca = PCA(n_components=len(available), random_state=42)
    transformed = pca.fit_transform(fit_data)
    pc1 = pd.Series(transformed[:, 0], index=fit_data.index, name=RAW_AIS)
    loadings = pd.Series(pca.components_[0], index=available, name="pc1_loading")

    average_standardized_sentiment = fit_data.mean(axis=1)
    if pc1.corr(average_standardized_sentiment) < 0:
        pc1 = -pc1
        loadings = -loadings

    scores = pd.Series(np.nan, index=df.index, name=RAW_AIS)
    scores.loc[pc1.index] = pc1

    return PCAConstruction(
        scores=scores,
        loadings=loadings,
        explained_variance=pd.Series(pca.explained_variance_, name="eigenvalue"),
        explained_variance_ratio=pd.Series(pca.explained_variance_ratio_, name="variance_explained"),
        standardized_sources=standardized,
    )

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
