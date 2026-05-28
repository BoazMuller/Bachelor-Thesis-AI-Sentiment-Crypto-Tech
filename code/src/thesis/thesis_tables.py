from __future__ import annotations

import warnings
from dataclasses import dataclass
import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "thesis_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import jarque_bera, kurtosis, skew
from sklearn.decomposition import PCA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.arima.model import ARIMA

from thesis.paths import FIGURES_DIR
from thesis.time_series_validation import (
    arch_lm_tests,
    autocorrelation_tests,
    metadata_for_column,
    stationarity_tests,
    variable_inventory,
)


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


def table_01_data_inventory(df: pd.DataFrame) -> pd.DataFrame:
    inventory = variable_inventory(df)
    if "role" not in inventory.columns and "expected_sign_or_role" in inventory.columns:
        inventory["role"] = inventory["expected_sign_or_role"]
    if "frequency" not in inventory.columns and "transformed_frequency" in inventory.columns:
        inventory["frequency"] = inventory["transformed_frequency"]
    if "observations" not in inventory.columns and "final_usable_observations" in inventory.columns:
        inventory["observations"] = inventory["final_usable_observations"]
    if "missing" not in inventory.columns and "missing_days" in inventory.columns:
        inventory["missing"] = inventory["missing_days"]
    if "missing_pct" not in inventory.columns:
        numeric_missing = pd.to_numeric(inventory["missing"], errors="coerce")
        inventory["missing_pct"] = numeric_missing / len(df)
    for column in ["units", "notes"]:
        if column not in inventory.columns:
            inventory[column] = inventory["variable"].map(
                _units_for_variable if column == "units" else _notes_for_variable
            )
    return inventory[
        [
            "variable",
            "source",
            "role",
            "date_range",
            "frequency",
            "units",
            "transformation",
            "observations",
            "missing",
            "missing_pct",
            "notes",
        ]
    ]


def table_02_return_sentiment_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in RETURN_COLUMNS + [RAW_AIS, EXPECTATION_ADJUSTED_AIS, ROBUST_EXPECTATION_ADJUSTED_AIS]
        if column in df.columns
    ]
    return descriptive_stats_table(df, columns)


def table_03_pre_estimation_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    sentiment_columns = [
        column
        for column in [RAW_AIS, EXPECTATION_ADJUSTED_AIS, ROBUST_EXPECTATION_ADJUSTED_AIS]
        if column in df.columns
    ]
    diagnostic_columns = [column for column in RETURN_COLUMNS if column in df.columns] + sentiment_columns
    rows: list[dict[str, object]] = []

    stationarity = stationarity_tests(df, diagnostic_columns)
    for _, row in stationarity.iterrows():
        rows.append(
            {
                "diagnostic": "stationarity",
                "variable": row["variable"],
                "test": row["test"],
                "lag": row.get("lags", np.nan),
                "statistic": row.get("statistic", np.nan),
                "p_value": row.get("p_value", np.nan),
                "nobs": row.get("nobs", np.nan),
                "interpretation": row.get("interpretation", ""),
            }
        )

    autocorr = autocorrelation_tests(df, [column for column in RETURN_COLUMNS if column in df.columns])
    for _, row in autocorr.iterrows():
        rows.append(
            {
                "diagnostic": "return_autocorrelation",
                "variable": row["variable"],
                "test": "Ljung-Box",
                "lag": row["lag"],
                "statistic": row["lb_stat"],
                "p_value": row["p_value"],
                "nobs": row["nobs"],
                "interpretation": row["interpretation"],
            }
        )

    arch = arch_lm_tests(df, [column for column in RETURN_COLUMNS if column in df.columns])
    for _, row in arch.iterrows():
        rows.append(
            {
                "diagnostic": "conditional_heteroskedasticity",
                "variable": row["variable"],
                "test": "ARCH-LM",
                "lag": row["lag"],
                "statistic": row["lm_stat"],
                "p_value": row["lm_p_value"],
                "nobs": row["nobs"],
                "interpretation": row["interpretation"],
            }
        )

    return pd.DataFrame(rows)


def table_04_arma_lag_order_selection(
    df: pd.DataFrame,
    *,
    max_p: int = 3,
    max_q: int = 3,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in [column for column in RETURN_COLUMNS if column in df.columns]:
        series = pd.to_numeric(df[column], errors="coerce").dropna() * 100
        candidate_rows: list[dict[str, object]] = []
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = ARIMA(
                            series,
                            order=(p, 0, q),
                            trend="c",
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit()
                    candidate_rows.append(
                        {
                            "asset_return": column,
                            "p": p,
                            "q": q,
                            "bic": float(result.bic),
                            "aic": float(result.aic),
                            "nobs": int(result.nobs),
                            "converged": bool(result.mle_retvals.get("converged", False)),
                            "note": "",
                        }
                    )
                except Exception as exc:
                    candidate_rows.append(
                        {
                            "asset_return": column,
                            "p": p,
                            "q": q,
                            "bic": np.nan,
                            "aic": np.nan,
                            "nobs": int(len(series)),
                            "converged": False,
                            "note": f"{type(exc).__name__}: {exc}",
                        }
                    )
        valid = [row for row in candidate_rows if pd.notna(row["bic"])]
        best = min(valid, key=lambda row: row["bic"]) if valid else None
        for row in candidate_rows:
            row["selected_by_bic"] = bool(best and row["p"] == best["p"] and row["q"] == best["q"])
            rows.append(row)
    return pd.DataFrame(rows)


def table_14_text_data_coverage_by_source(sentiment_records: pd.DataFrame) -> pd.DataFrame:
    records = sentiment_records.copy()
    records[DATE_COLUMN] = pd.to_datetime(records[DATE_COLUMN])
    rows: list[dict[str, object]] = []
    for source, source_df in records.groupby("source"):
        daily_counts = source_df.groupby(DATE_COLUMN).size()
        full_index = pd.date_range(daily_counts.index.min(), daily_counts.index.max(), freq="D")
        aligned = daily_counts.reindex(full_index, fill_value=0)
        rows.append(
            {
                "source": source,
                "raw_text_count": int(len(source_df)),
                "start_date": full_index.min().date(),
                "end_date": full_index.max().date(),
                "days_covered": int((aligned > 0).sum()),
                "average_texts_per_day": float(aligned.mean()),
                "median_texts_per_day": float(aligned.median()),
                "zero_text_days": int((aligned == 0).sum()),
                "missing_days": int((aligned == 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("source")


def table_15_daily_sentiment_descriptives_by_source(df: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in SOURCE_SENTIMENT_COLUMNS if column in df.columns]
    table = descriptive_stats_table(df, columns)
    table["source"] = table["variable"].map(
        {
            GDELT_SENTIMENT: "gdelt_news",
            REDDIT_SENTIMENT: "reddit_social_media",
        }
    )
    return table[["source"] + [column for column in table.columns if column != "source"]]


def table_16_sentiment_source_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in SOURCE_SENTIMENT_COLUMNS if column in df.columns]
    corr = df[columns].corr()
    corr.insert(0, "variable", corr.index)
    return corr.reset_index(drop=True)


def table_17_pca_results(df: pd.DataFrame) -> pd.DataFrame:
    pca_result = construct_ais(df)
    rows: list[dict[str, object]] = []
    cumulative = pca_result.explained_variance_ratio.cumsum()
    for component_idx, (eigenvalue, variance_share) in enumerate(
        zip(pca_result.explained_variance, pca_result.explained_variance_ratio),
        start=1,
    ):
        row = {
            "component": f"PC{component_idx}",
            "eigenvalue": float(eigenvalue),
            "variance_explained": float(variance_share),
            "cumulative_variance_explained": float(cumulative.iloc[component_idx - 1]),
        }
        if component_idx == 1:
            for source, loading in pca_result.loadings.items():
                row[f"{source}_loading"] = float(loading)
        rows.append(row)
    return pd.DataFrame(rows)


def table_18_ais_construction_validation(df: pd.DataFrame) -> pd.DataFrame:
    enriched = add_sentiment_measures(df)
    rows = []
    for source in SOURCE_SENTIMENT_COLUMNS:
        rows.append(
            {
                "validation_item": f"Correlation with {source}",
                "value": float(enriched[[RAW_AIS, source]].corr().iloc[0, 1]),
                "note": "PC1 sign is normalized so higher AIS means more positive average source sentiment",
            }
        )
    stats = descriptive_stats_table(enriched, [RAW_AIS])
    for _, row in stats.iterrows():
        rows.append(
            {
                "validation_item": f"AIS {row['statistic']}",
                "value": row["value"],
                "note": "",
            }
        )
    return pd.DataFrame(rows)


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


def make_egarch_eda_figures(df: pd.DataFrame) -> list[Path]:
    enriched = add_sentiment_measures(df)
    output_dir = FIGURES_DIR / "armax_egarchx"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for column in [column for column in RETURN_COLUMNS if column in enriched.columns]:
        axes[0].plot(enriched[DATE_COLUMN], enriched[column], label=column)
    axes[0].set_title("Daily log returns")
    axes[0].legend(loc="best")
    axes[1].plot(enriched[DATE_COLUMN], enriched[EXPECTATION_ADJUSTED_AIS], color="black")
    axes[1].set_title("Expectation-adjusted AIS")
    axes[2].plot(enriched[DATE_COLUMN], enriched[RAW_AIS], color="tab:gray")
    axes[2].set_title("Raw AIS")
    fig.tight_layout()
    paths.append(_save_figure(fig, output_dir / "returns_and_expectation_adjusted_ais.png"))

    fig, ax = plt.subplots(figsize=(10, 5))
    for column in [column for column in RETURN_COLUMNS if column in enriched.columns]:
        rolling_vol = pd.to_numeric(enriched[column], errors="coerce").rolling(30).std() * np.sqrt(252)
        ax.plot(enriched[DATE_COLUMN], rolling_vol, label=column)
    ax.set_title("Thirty-day rolling annualized volatility")
    ax.legend(loc="best")
    fig.tight_layout()
    paths.append(_save_figure(fig, output_dir / "rolling_volatility.png"))

    for column in [column for column in RETURN_COLUMNS if column in enriched.columns]:
        series = pd.to_numeric(enriched[column], errors="coerce").dropna()
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        plot_acf(series, ax=axes[0, 0], lags=30, title=f"ACF {column}")
        plot_pacf(series, ax=axes[0, 1], lags=30, title=f"PACF {column}")
        plot_acf(series.pow(2), ax=axes[1, 0], lags=30, title=f"ACF squared {column}")
        plot_pacf(series.pow(2), ax=axes[1, 1], lags=30, title=f"PACF squared {column}")
        fig.tight_layout()
        paths.append(_save_figure(fig, output_dir / f"acf_pacf_{column}.png"))

    lead_lag = lead_lag_correlations(enriched)
    fig, ax = plt.subplots(figsize=(10, 5))
    for target, group in lead_lag.groupby("target"):
        ax.plot(group["sentiment_lag"], group["correlation"], marker="o", label=target)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Lead-lag correlations with expectation-adjusted AIS")
    ax.set_xlabel("Sentiment lag; positive means sentiment is lagged")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    paths.append(_save_figure(fig, output_dir / "lead_lag_cross_correlations.png"))
    lead_lag.to_csv(output_dir / "lead_lag_cross_correlations.csv", index=False)
    paths.append(output_dir / "lead_lag_cross_correlations.csv")

    return paths


def lead_lag_correlations(df: pd.DataFrame, max_lag: int = 10) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sentiment = pd.to_numeric(df[EXPECTATION_ADJUSTED_AIS], errors="coerce")
    target_series: dict[str, pd.Series] = {}
    for column in [column for column in RETURN_COLUMNS if column in df.columns]:
        returns = pd.to_numeric(df[column], errors="coerce")
        target_series[column] = returns
        target_series[f"abs_{column}"] = returns.abs()
        target_series[f"squared_{column}"] = returns.pow(2)
    for target, series in target_series.items():
        for lag in range(-max_lag, max_lag + 1):
            rows.append(
                {
                    "target": target,
                    "sentiment_lag": lag,
                    "correlation": float(series.corr(sentiment.shift(lag))),
                    "nobs": int(pd.concat([series, sentiment.shift(lag)], axis=1).dropna().shape[0]),
                }
            )
    return pd.DataFrame(rows)


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


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _units_for_variable(column: str) -> str:
    meta = metadata_for_column(column)
    if hasattr(meta, "units"):
        return meta.units
    if column.endswith("_log_return"):
        return "Log points"
    if "sentiment" in column or column.endswith("_ais"):
        return "Sentiment index units"
    if "vix" in column or column in {"epu", "gpr"}:
        return "Index points"
    if "kalshi" in column:
        return "Price/probability points"
    if "metaculus" in column:
        return "Days"
    if column == DATE_COLUMN:
        return "Date"
    return "Price or source units"


def _notes_for_variable(column: str) -> str:
    if column == RAW_AIS:
        return "PC1 of standardized GDELT/news and Reddit/social-media sentiment, sign-normalized positive"
    if column == EXPECTATION_ADJUSTED_AIS:
        return "Residual from baseline expectation orthogonalization"
    if column == ROBUST_EXPECTATION_ADJUSTED_AIS:
        return "Residual from robustness orthogonalization including macro-risk controls"
    meta = metadata_for_column(column)
    if hasattr(meta, "notes") and meta.notes:
        return meta.notes
    if hasattr(meta, "expected_sign_or_role") and meta.expected_sign_or_role != "Unclassified variable":
        return meta.expected_sign_or_role
    return ""
