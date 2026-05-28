from __future__ import annotations

import pandas as pd

from thesis.tables.common import (
    DATE_COLUMN,
    GDELT_SENTIMENT,
    RAW_AIS,
    REDDIT_SENTIMENT,
    SOURCE_SENTIMENT_COLUMNS,
    add_sentiment_measures,
    construct_ais,
    descriptive_stats_table,
)

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
