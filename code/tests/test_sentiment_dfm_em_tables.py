from types import SimpleNamespace

import numpy as np
import pandas as pd

from thesis.tables import common as common_tables
from thesis.tables import sentiment_dfm_em as sentiment_dfm_em_tables
from thesis.tables.common import GDELT_SENTIMENT, REDDIT_SENTIMENT
from thesis.tables.sentiment_dfm_em import (
    table_15_daily_sentiment_descriptives_by_source,
    table_17_dynamic_factor_results,
    table_18_ais_construction_validation,
)


def test_table_15_includes_both_raw_sentiment_sources():
    df = _sentiment_panel()

    table = table_15_daily_sentiment_descriptives_by_source(df)

    assert set(table["variable"]) == {GDELT_SENTIMENT, REDDIT_SENTIMENT}
    assert set(table["source"]) == {"gdelt_news", "reddit_social_media"}


def test_table_15_includes_quantiles_and_ljung_box_rows():
    df = _sentiment_panel()

    table = table_15_daily_sentiment_descriptives_by_source(df)

    gdelt = table[table["variable"].eq(GDELT_SENTIMENT)]
    descriptive_stats = set(gdelt.loc[gdelt["diagnostic"].eq("descriptive"), "statistic"])
    assert {"p01", "p05", "p25", "median", "p75", "p95", "p99"}.issubset(descriptive_stats)

    level_ljung_box = gdelt[gdelt["diagnostic"].eq("ljung_box_levels")]
    squared_ljung_box = gdelt[gdelt["diagnostic"].eq("ljung_box_squared_demeaned")]
    assert set(level_ljung_box["lag"]) == {1, 5, 10, 20, 30}
    assert set(squared_ljung_box["lag"]) == {1, 5, 10, 20, 30}
    assert level_ljung_box["p_value"].notna().all()
    assert squared_ljung_box["p_value"].notna().all()
    assert level_ljung_box["interpretation"].str.contains("missing observations dropped").all()
    assert squared_ljung_box["interpretation"].str.contains("missing observations dropped").all()


def test_table_15_handles_missing_gdelt_observations():
    df = _sentiment_panel()

    table = table_15_daily_sentiment_descriptives_by_source(df)

    gdelt_n = table[
        table["variable"].eq(GDELT_SENTIMENT)
        & table["diagnostic"].eq("descriptive")
        & table["statistic"].eq("N")
    ].iloc[0]
    gdelt_missing = table[
        table["variable"].eq(GDELT_SENTIMENT)
        & table["diagnostic"].eq("descriptive")
        & table["statistic"].eq("missing")
    ].iloc[0]

    assert gdelt_n["value"] == 43
    assert gdelt_missing["value"] == 2
    assert gdelt_missing["missing"] == 2


def test_dynamic_factor_ais_uses_bic_selected_single_factor(monkeypatch):
    _use_fast_dfm_settings(monkeypatch)
    df = _sentiment_panel()

    result = common_tables.construct_ais(df)

    assert result.selected_factor_order in {3, 4}
    assert result.lag_selection["n_factors"].eq(1).all()
    assert np.isclose(result.scores.mean(), 0.0)
    assert np.isclose(result.scores.std(ddof=1), 1.0)

    table_17 = table_17_dynamic_factor_results(df)
    selected_bic = table_17[table_17["metric"].eq("bic") & table_17["selected"].eq(True)]
    loadings = table_17[table_17["metric"].eq("factor_loading")]
    assert len(selected_bic) == 1
    assert set(loadings["source"]) == {GDELT_SENTIMENT, REDDIT_SENTIMENT}

    table_18 = table_18_ais_construction_validation(df)
    assert "selected_factor_order" in set(table_18["validation_item"])
    assert "AIS N" in set(table_18["validation_item"])


def test_dynamic_factor_ais_uses_filtered_factor_not_smoothed(monkeypatch):
    common_tables._AIS_CONSTRUCTION_CACHE.clear()
    df = pd.DataFrame(
        {
            GDELT_SENTIMENT: [1.0, 2.0, 3.0, 4.0, 5.0],
            REDDIT_SENTIMENT: [1.5, 2.5, 3.5, 4.5, 5.5],
        }
    )
    filtered = pd.DataFrame({0: [1.0, 2.0, 3.0, 4.0, 5.0]})
    smoothed = pd.DataFrame({0: [10.0, 10.0, 10.0, 10.0, 10.0]})
    fake_result = SimpleNamespace(
        factors=SimpleNamespace(filtered=filtered, smoothed=smoothed),
        params=pd.Series(
            {
                f"loading.0->{GDELT_SENTIMENT}": 1.0,
                f"loading.0->{REDDIT_SENTIMENT}": 1.0,
            }
        ),
        bic=1.0,
        llf=-1.0,
        mle_retvals={"converged": True, "iter": 3},
    )
    fake_lag_selection = pd.DataFrame(
        [
            {
                "factor_order": 1,
                "n_factors": 1,
                "bic": 1.0,
                "aic": 1.0,
                "llf": -1.0,
                "nobs": len(df),
                "idiosyncratic_ar1": False,
                "converged": True,
                "iterations": 3,
                "fit_success": True,
                "error": "",
                "result": fake_result,
            }
        ]
    )
    monkeypatch.setattr(common_tables, "_dfm_lag_selection", lambda _: fake_lag_selection)

    result = common_tables.construct_ais(df)

    expected = (filtered.iloc[:, 0] - filtered.iloc[:, 0].mean()) / filtered.iloc[:, 0].std(ddof=1)
    assert np.allclose(result.scores.to_numpy(), expected.to_numpy())


def test_nonconverged_dynamic_factor_bic_is_not_reported(monkeypatch):
    class FakeDynamicFactorMQ:
        def __init__(self, *args, factor_orders, **kwargs):
            self.factor_order = factor_orders

        def fit(self, **kwargs):
            converged = self.factor_order == 2
            return SimpleNamespace(
                bic=10.0 if self.factor_order == 1 else 20.0,
                aic=5.0,
                llf=-2.0,
                nobs=5,
                mle_retvals={"converged": converged, "iter": 1},
            )

    monkeypatch.setattr(common_tables, "DynamicFactorMQ", FakeDynamicFactorMQ)
    monkeypatch.setattr(common_tables, "DFM_FACTOR_ORDER_CANDIDATES", (1, 2))
    lag_selection = common_tables._dfm_lag_selection(
        pd.DataFrame(
            {
                GDELT_SENTIMENT: [1.0, 2.0, 3.0, 4.0, 5.0],
                REDDIT_SENTIMENT: [1.5, 2.5, 3.5, 4.5, 5.5],
            }
        )
    )

    nonconverged = lag_selection[lag_selection["factor_order"].eq(1)].iloc[0]
    converged = lag_selection[lag_selection["factor_order"].eq(2)].iloc[0]
    assert pd.isna(nonconverged["bic"])
    assert converged["bic"] == 20.0


def test_table_17_reports_nonconverged_bic_as_na(monkeypatch):
    fake_result = SimpleNamespace(
        lag_selection=pd.DataFrame(
            [
                {
                    "factor_order": 1,
                    "n_factors": 1,
                    "bic": np.nan,
                    "aic": 1.0,
                    "llf": -1.0,
                    "converged": False,
                    "iterations": 50,
                    "idiosyncratic_ar1": False,
                    "fit_success": True,
                    "error": "",
                },
                {
                    "factor_order": 2,
                    "n_factors": 1,
                    "bic": 2.0,
                    "aic": 1.0,
                    "llf": -1.0,
                    "converged": True,
                    "iterations": 25,
                    "idiosyncratic_ar1": False,
                    "fit_success": True,
                    "error": "",
                },
            ]
        ),
        selected_factor_order=2,
        loadings=pd.Series({GDELT_SENTIMENT: 0.5, REDDIT_SENTIMENT: 0.5}),
    )
    monkeypatch.setattr(sentiment_dfm_em_tables, "construct_ais", lambda _: fake_result)

    table = sentiment_dfm_em_tables.table_17_dynamic_factor_results(pd.DataFrame())

    nonconverged_bic = table[
        table["metric"].eq("bic") & table["factor_order"].eq(1)
    ].iloc[0]
    assert nonconverged_bic["value"] == "N/A"
    assert "did not converge" in nonconverged_bic["note"]


def _use_fast_dfm_settings(monkeypatch):
    common_tables._AIS_CONSTRUCTION_CACHE.clear()
    monkeypatch.setattr(common_tables, "DFM_FACTOR_ORDER_CANDIDATES", (3, 4))
    monkeypatch.setattr(common_tables, "DFM_EM_MAXITER", 50)


def _sentiment_panel() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=45, freq="B")
    trend = np.linspace(-1.0, 1.0, len(dates))
    seasonal = np.sin(np.linspace(0.0, 4.0 * np.pi, len(dates)))
    gdelt = 0.08 * trend + 0.04 * seasonal
    reddit = 0.10 * trend - 0.03 * seasonal
    gdelt[[10, 11]] = np.nan
    return pd.DataFrame(
        {
            "date": dates,
            GDELT_SENTIMENT: gdelt,
            REDDIT_SENTIMENT: reddit,
        }
    )
