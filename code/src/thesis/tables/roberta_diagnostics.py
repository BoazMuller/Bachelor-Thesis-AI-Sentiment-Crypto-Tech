from __future__ import annotations

import numpy as np
import pandas as pd


def table_roberta_daily_counts(sentiment_records: pd.DataFrame) -> pd.DataFrame:
    """
    Generate daily text counts for GDELT (news) and Reddit (social media) across the sample period.
    Reindexes to a full daily frequency, filling missing days with 0 counts.
    """
    df = sentiment_records.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Group by date and source to count articles/posts
    counts = df.groupby(["date", "source"]).size().unstack(fill_value=0)

    # Ensure both 'gdelt' and 'reddit' columns exist in the output index
    for col in ["gdelt", "reddit"]:
        if col not in counts.columns:
            counts[col] = 0

    # Reindex to a complete daily date range to identify gaps/zero-text days
    if not counts.empty:
        full_index = pd.date_range(counts.index.min(), counts.index.max(), freq="D")
        counts = counts.reindex(full_index, fill_value=0)

    counts.index.name = "date"
    counts = counts.reset_index()
    counts = counts.rename(columns={"gdelt": "gdelt_count", "reddit": "reddit_count"})
    counts["total_count"] = counts["gdelt_count"] + counts["reddit_count"]
    
    # Format date as YYYY-MM-DD string
    counts["date"] = counts["date"].dt.strftime("%Y-%m-%d")
    return counts


def table_roberta_count_descriptives(sentiment_records: pd.DataFrame) -> pd.DataFrame:
    """
    Generate descriptive statistics for the daily post/headline counts.
    """
    daily_counts = table_roberta_daily_counts(sentiment_records)

    # Convert the date column back to datetime for start/end analysis
    dates = pd.to_datetime(daily_counts["date"])

    if not daily_counts.empty:
        start_date = dates.min().strftime("%Y-%m-%d")
        end_date = dates.max().strftime("%Y-%m-%d")
        total_days = len(daily_counts)
    else:
        start_date = "N/A"
        end_date = "N/A"
        total_days = 0

    stats = {
        "start_date": lambda s: start_date,
        "end_date": lambda s: end_date,
        "days_in_sample": lambda s: total_days,
        "total_texts": lambda s: int(s.sum()),
        "mean_per_day": lambda s: float(s.mean()) if len(s) > 0 else np.nan,
        "std_per_day": lambda s: float(s.std()) if len(s) > 1 else np.nan,
        "min_per_day": lambda s: int(s.min()) if len(s) > 0 else np.nan,
        "p25_per_day": lambda s: float(s.quantile(0.25)) if len(s) > 0 else np.nan,
        "median_per_day": lambda s: float(s.median()) if len(s) > 0 else np.nan,
        "p75_per_day": lambda s: float(s.quantile(0.75)) if len(s) > 0 else np.nan,
        "max_per_day": lambda s: int(s.max()) if len(s) > 0 else np.nan,
        "zero_text_days": lambda s: int((s == 0).sum()) if len(s) > 0 else np.nan,
    }

    sources = ["gdelt_count", "reddit_count", "total_count"]
    source_labels = {"gdelt_count": "gdelt", "reddit_count": "reddit", "total_count": "total"}

    rows = []
    for metric_name, func in stats.items():
        row = {"metric": metric_name}
        for col in sources:
            label = source_labels[col]
            series = daily_counts[col]
            row[label] = func(series)
        rows.append(row)

    return pd.DataFrame(rows)


def table_roberta_validation_sample(
    sentiment_records: pd.DataFrame,
    n: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Randomly select a sample of classifications for manual verification.
    Uses source-balanced sampling to include GDELT and Reddit texts.
    """
    df = sentiment_records.copy()
    if df.empty:
        sample_df = pd.DataFrame(columns=df.columns)
    elif "source" in df.columns:
        sources = [source for source in ["gdelt", "reddit"] if source in set(df["source"].dropna())]
        if not sources:
            sources = sorted(df["source"].dropna().unique())
        sample_df = _source_balanced_sample(df, sources=sources, n=n, seed=seed)
    else:
        sample_size = min(len(df), n)
        sample_df = df.sample(n=sample_size, random_state=seed) if sample_size else pd.DataFrame(columns=df.columns)

    columns_to_keep = [
        "date",
        "source",
        "text_clean",
        "roberta_sentiment_label",
        "roberta_sentiment_score",
        "roberta_sentiment_compound",
        "roberta_prob_negative",
        "roberta_prob_neutral",
        "roberta_prob_positive",
    ]
    columns_to_keep = [col for col in columns_to_keep if col in sample_df.columns]

    sample_df = sample_df[columns_to_keep].copy()
    if "roberta_sentiment_label" in sample_df.columns:
        sample_df = sample_df.rename(columns={"roberta_sentiment_label": "model_label"})
    elif {"roberta_prob_negative", "roberta_prob_neutral", "roberta_prob_positive"}.issubset(sample_df.columns):
        sample_df["model_label"] = _model_label_from_probabilities(sample_df)
    else:
        sample_df["model_label"] = ""

    sample_df["manual_label"] = ""
    sample_df["manual_notes"] = ""
    sample_df.insert(0, "validation_id", range(1, len(sample_df) + 1))

    if "date" in sample_df.columns:
        sample_df["date"] = pd.to_datetime(sample_df["date"])
        sort_columns = ["source", "date"] if "source" in sample_df.columns else ["date"]
        sample_df = sample_df.sort_values(sort_columns).reset_index(drop=True)
        sample_df["date"] = sample_df["date"].dt.strftime("%Y-%m-%d")
    else:
        sample_df = sample_df.reset_index(drop=True)
    sample_df["validation_id"] = range(1, len(sample_df) + 1)

    return sample_df


def _source_balanced_sample(
    df: pd.DataFrame,
    *,
    sources: list[str],
    n: int,
    seed: int,
) -> pd.DataFrame:
    if n <= 0 or not sources:
        return pd.DataFrame(columns=df.columns)

    base_n = n // len(sources)
    remainder = n % len(sources)
    samples = []

    for offset, source in enumerate(sources):
        source_df = df[df["source"] == source]
        requested = base_n + (1 if offset < remainder else 0)
        sample_size = min(len(source_df), requested)
        if sample_size:
            samples.append(source_df.sample(n=sample_size, random_state=seed + offset))

    if not samples:
        return pd.DataFrame(columns=df.columns)

    sample_df = pd.concat(samples)
    if len(sample_df) < n:
        remaining = df.drop(sample_df.index)
        topup_size = min(len(remaining), n - len(sample_df))
        if topup_size:
            sample_df = pd.concat([sample_df, remaining.sample(n=topup_size, random_state=seed + len(sources))])

    return sample_df


def _model_label_from_probabilities(df: pd.DataFrame) -> pd.Series:
    label_columns = {
        "negative": "roberta_prob_negative",
        "neutral": "roberta_prob_neutral",
        "positive": "roberta_prob_positive",
    }
    probabilities = df[list(label_columns.values())].apply(pd.to_numeric, errors="coerce")
    inverse_labels = {column: label for label, column in label_columns.items()}
    return probabilities.idxmax(axis=1).map(inverse_labels).fillna("")


def table_roberta_extreme_sentiment(
    sentiment_records: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Identify the top N most positive and top N most negative texts for each source.
    """
    df = sentiment_records.copy()
    if "roberta_sentiment_compound" not in df.columns:
        return pd.DataFrame()

    df["roberta_sentiment_compound"] = pd.to_numeric(df["roberta_sentiment_compound"], errors="coerce")
    df = df.dropna(subset=["roberta_sentiment_compound"])

    rows = []
    for source in sorted(df["source"].unique()):
        src_df = df[df["source"] == source]

        pos = src_df.nlargest(top_n, "roberta_sentiment_compound")
        for _, row in pos.iterrows():
            rows.append({
                "source": source,
                "extreme_group": "top_positive",
                "date": row.get("date"),
                "model_label": row.get("roberta_sentiment_label", ""),
                "compound_score": row["roberta_sentiment_compound"],
                "roberta_prob_negative": row.get("roberta_prob_negative", np.nan),
                "roberta_prob_neutral": row.get("roberta_prob_neutral", np.nan),
                "roberta_prob_positive": row.get("roberta_prob_positive", np.nan),
                "text_clean": row.get("text_clean", ""),
            })

        neg = src_df.nsmallest(top_n, "roberta_sentiment_compound")
        for _, row in neg.iterrows():
            rows.append({
                "source": source,
                "extreme_group": "top_negative",
                "date": row.get("date"),
                "model_label": row.get("roberta_sentiment_label", ""),
                "compound_score": row["roberta_sentiment_compound"],
                "roberta_prob_negative": row.get("roberta_prob_negative", np.nan),
                "roberta_prob_neutral": row.get("roberta_prob_neutral", np.nan),
                "roberta_prob_positive": row.get("roberta_prob_positive", np.nan),
                "text_clean": row.get("text_clean", ""),
            })

    return pd.DataFrame(rows)


def table_roberta_label_distribution(sentiment_records: pd.DataFrame) -> pd.DataFrame:
    """
    Break down the count and percentage of sentiment labels for each source.
    """
    df = sentiment_records.copy()
    if "roberta_sentiment_label" not in df.columns:
        return pd.DataFrame()

    df["roberta_sentiment_label"] = df["roberta_sentiment_label"].astype(str).str.lower()

    # Group by source and label
    counts = df.groupby(["source", "roberta_sentiment_label"]).size().unstack(fill_value=0)

    # Calculate percentage
    totals = counts.sum(axis=1)
    percentages = counts.div(totals, axis=0) * 100

    rows = []
    for source in sorted(counts.index):
        for label in ["negative", "neutral", "positive"]:
            count_val = counts.loc[source, label] if label in counts.columns else 0
            pct_val = percentages.loc[source, label] if label in percentages.columns else 0.0
            rows.append({
                "source": source,
                "label": label,
                "count": int(count_val),
                "percentage": float(pct_val),
            })

    return pd.DataFrame(rows)


def table_roberta_daily_counts_shifted(
    sentiment_records: pd.DataFrame,
    trading_dates: pd.Series,
) -> pd.DataFrame:
    """
    Generate daily text counts for GDELT (news) and Reddit (social media)
    after shifting non-trading day observations to the next available trading day.
    """
    df = sentiment_records.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Align dates to next trading day
    trading_calendar = pd.DataFrame({"trading_date": pd.to_datetime(trading_dates)}).sort_values("trading_date")
    matched = pd.merge_asof(
        df.sort_values("date"),
        trading_calendar,
        left_on="date",
        right_on="trading_date",
        direction="forward",
    )

    # Drop records that couldn't be matched to any future trading day (if any)
    matched = matched.dropna(subset=["trading_date"])

    # Count by trading_date and source
    counts = matched.groupby(["trading_date", "source"]).size().unstack(fill_value=0)

    # Ensure both 'gdelt' and 'reddit' columns exist in the output index
    for col in ["gdelt", "reddit"]:
        if col not in counts.columns:
            counts[col] = 0

    # Reindex to the exact trading dates series to ensure we represent the full calendar
    trading_dates_dt = pd.to_datetime(trading_dates)
    counts = counts.reindex(trading_dates_dt, fill_value=0)

    counts.index.name = "date"
    counts = counts.reset_index()
    counts = counts.rename(columns={"gdelt": "gdelt_count", "reddit": "reddit_count"})
    counts["total_count"] = counts["gdelt_count"] + counts["reddit_count"]

    counts["date"] = counts["date"].dt.strftime("%Y-%m-%d")
    return counts


def table_roberta_count_descriptives_shifted(
    sentiment_records: pd.DataFrame,
    trading_dates: pd.Series,
) -> pd.DataFrame:
    """
    Generate descriptive statistics for the daily post/headline counts after
    assigning non-trading day observations to the next trading day.
    """
    daily_counts = table_roberta_daily_counts_shifted(sentiment_records, trading_dates)

    # Convert the date column back to datetime for start/end analysis
    dates = pd.to_datetime(daily_counts["date"])

    if not daily_counts.empty:
        start_date = dates.min().strftime("%Y-%m-%d")
        end_date = dates.max().strftime("%Y-%m-%d")
        total_days = len(daily_counts)
    else:
        start_date = "N/A"
        end_date = "N/A"
        total_days = 0

    stats = {
        "start_date": lambda s: start_date,
        "end_date": lambda s: end_date,
        "days_in_sample": lambda s: total_days,
        "total_texts": lambda s: int(s.sum()),
        "mean_per_day": lambda s: float(s.mean()) if len(s) > 0 else np.nan,
        "std_per_day": lambda s: float(s.std()) if len(s) > 1 else np.nan,
        "min_per_day": lambda s: int(s.min()) if len(s) > 0 else np.nan,
        "p25_per_day": lambda s: float(s.quantile(0.25)) if len(s) > 0 else np.nan,
        "median_per_day": lambda s: float(s.median()) if len(s) > 0 else np.nan,
        "p75_per_day": lambda s: float(s.quantile(0.75)) if len(s) > 0 else np.nan,
        "max_per_day": lambda s: int(s.max()) if len(s) > 0 else np.nan,
        "zero_text_days": lambda s: int((s == 0).sum()) if len(s) > 0 else np.nan,
    }

    sources = ["gdelt_count", "reddit_count", "total_count"]
    source_labels = {"gdelt_count": "gdelt", "reddit_count": "reddit", "total_count": "total"}

    rows = []
    for metric_name, func in stats.items():
        row = {"metric": metric_name}
        for col in sources:
            label = source_labels[col]
            series = daily_counts[col]
            row[label] = func(series)
        rows.append(row)

    return pd.DataFrame(rows)
