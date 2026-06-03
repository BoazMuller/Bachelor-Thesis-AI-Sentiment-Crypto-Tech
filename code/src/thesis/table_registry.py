from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from thesis.paths import FIGURES_DIR, TABLES_DIR


@dataclass(frozen=True)
class TableSpec:
    number: int
    title: str
    folder: str
    stem: str

    @property
    def output_dir(self) -> Path:
        return TABLES_DIR / self.folder

    @property
    def csv_filename(self) -> str:
        return f"{self.stem}.csv"

    @property
    def tex_filename(self) -> str:
        return f"{self.stem}.tex"

    @property
    def md_filename(self) -> str:
        return f"{self.stem}.md"

    @property
    def latex_label(self) -> str:
        return f"tab:{self.stem}"


TABLE_REGISTRY: dict[int, TableSpec] = {
    1: TableSpec(1, "Data sources, definitions, transformations, and availability", "01_data_inventory", "table_01_data_sources_definitions_transformations_availability"),
    2: TableSpec(2, "Return and sentiment descriptive statistics", "02_armax_egarchx", "table_02_return_sentiment_descriptives"),
    3: TableSpec(3, "Pre-estimation time-series diagnostics", "02_armax_egarchx", "table_03_pre_estimation_time_series_diagnostics"),
    4: TableSpec(4, "ARMAX-EGARCHX BIC lag-order selection", "02_armax_egarchx", "armax_egarchx_lag_selection"),
    5: TableSpec(5, "ARMAX-EGARCHX estimation results", "02_armax_egarchx", "table_05_armax_egarchx_estimation_results"),
    6: TableSpec(6, "Post-estimation diagnostics", "02_armax_egarchx", "table_06_post_estimation_diagnostics"),
    7: TableSpec(7, "First-stage EGARCH volatility extraction summary", "03_tvpvar_connectedness", "table_07_egarch_volatility_extraction_summary"),
    8: TableSpec(8, "Conditional volatility descriptive statistics", "03_tvpvar_connectedness", "table_08_conditional_volatility_descriptives"),
    9: TableSpec(9, "TVP-VAR system definition table", "03_tvpvar_connectedness", "table_09_tvpvar_system_definition"),
    10: TableSpec(10, "TVP-VAR lag selection", "03_tvpvar_connectedness", "table_10_tvpvar_lag_selection"),
    11: TableSpec(11, "Average connectedness table", "03_tvpvar_connectedness", "table_11_average_connectedness"),
    12: TableSpec(12, "Average pairwise connectedness matrix", "03_tvpvar_connectedness", "table_12_average_pairwise_connectedness_matrix"),
    13: TableSpec(13, "Robustness connectedness table", "03_tvpvar_connectedness", "table_13_robustness_connectedness"),
    14: TableSpec(14, "Text data coverage by source", "04_sentiment_dfm_em", "table_14_text_data_coverage_by_source"),
    15: TableSpec(15, "Daily sentiment descriptive statistics by source", "04_sentiment_dfm_em", "table_15_daily_sentiment_descriptives_by_source"),
    16: TableSpec(16, "Sentiment source correlation matrix", "04_sentiment_dfm_em", "table_16_sentiment_source_correlation_matrix"),
    17: TableSpec(17, "DFM-EM AIS results", "04_sentiment_dfm_em", "table_17_dynamic_factor_results"),
    18: TableSpec(18, "AIS construction validation", "04_sentiment_dfm_em", "table_18_ais_construction_validation"),
    19: TableSpec(19, "Prediction-market and control-variable definitions", "05_expectation_adjusted_sentiment", "table_19_prediction_market_control_definitions"),
    20: TableSpec(20, "Residual-regression sample alignment", "05_expectation_adjusted_sentiment", "table_20_residual_regression_sample_alignment"),
    21: TableSpec(21, "Correlation matrix and multicollinearity diagnostics", "05_expectation_adjusted_sentiment", "table_21_correlation_matrix_multicollinearity_diagnostics"),
    22: TableSpec(22, "Orthogonalization regression results", "05_expectation_adjusted_sentiment", "table_22_orthogonalization_regression_results"),
    23: TableSpec(23, "Residual AIS validation table", "05_expectation_adjusted_sentiment", "table_23_residual_ais_validation"),
    24: TableSpec(24, "Raw AIS versus residual AIS comparison", "05_expectation_adjusted_sentiment", "table_24_raw_ais_versus_residual_ais_comparison"),
    25: TableSpec(25, "Connectedness-regression dataset summary", "06_spillover_regressions", "table_25_connectedness_regression_dataset_summary"),
    26: TableSpec(26, "Connectedness-regression correlation matrix", "06_spillover_regressions", "table_26_connectedness_regression_correlation_matrix"),
    27: TableSpec(27, "Headline contemporaneous spillover regression results", "06_spillover_regressions", "table_27_baseline_spillover_regression_results"),
    28: TableSpec(28, "Lagged sentiment robustness results", "06_spillover_regressions", "table_28_lagged_sentiment_robustness_results"),
    29: TableSpec(29, "Raw AIS robustness regression comparison", "06_spillover_regressions", "table_29_raw_ais_versus_residual_ais_regression_comparison"),
    30: TableSpec(30, "RoBERTa daily text counts", "RoBERTa", "roberta_daily_counts"),
    31: TableSpec(31, "RoBERTa daily count descriptive statistics", "RoBERTa", "roberta_count_descriptives"),
    32: TableSpec(32, "RoBERTa 100-sample manual verification table", "RoBERTa", "roberta_validation_sample_100"),
    33: TableSpec(33, "RoBERTa extreme sentiment examples", "RoBERTa", "roberta_extreme_sentiment"),
    34: TableSpec(34, "RoBERTa sentiment label distribution", "RoBERTa", "roberta_label_distribution"),
    35: TableSpec(35, "RoBERTa daily text counts (shifted to trading days)", "RoBERTa", "roberta_daily_counts_shifted"),
    36: TableSpec(36, "RoBERTa daily count descriptive statistics (shifted to trading days)", "RoBERTa", "roberta_count_descriptives_shifted"),
}


def get_table_spec(number: int) -> TableSpec:
    try:
        return TABLE_REGISTRY[number]
    except KeyError as exc:
        raise KeyError(f"No table registered for table number {number}") from exc


def ensure_output_structure() -> None:
    TABLE_REGISTRY[1].output_dir.mkdir(parents=True, exist_ok=True)
    for folder in [
        "armax_egarchx",
        "tvpvar_connectedness",
        "dfm_em",
        "expectation_adjusted_sentiment",
        "spillover_regressions",
        "RoBERTa",
    ]:
        (FIGURES_DIR / folder).mkdir(parents=True, exist_ok=True)
