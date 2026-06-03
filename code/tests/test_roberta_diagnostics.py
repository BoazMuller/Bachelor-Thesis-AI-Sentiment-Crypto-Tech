from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis.tables.roberta_diagnostics import (
    table_roberta_count_descriptives,
    table_roberta_count_descriptives_shifted,
    table_roberta_daily_counts,
    table_roberta_daily_counts_shifted,
    table_roberta_extreme_sentiment,
    table_roberta_label_distribution,
    table_roberta_validation_sample,
)


@pytest.fixture
def dummy_sentiment_records() -> pd.DataFrame:
    # Create 10 dummy records spread across 3 days
    dates = ["2026-05-01", "2026-05-01", "2026-05-01", "2026-05-02", "2026-05-02", "2026-05-04", "2026-05-04", "2026-05-04", "2026-05-04", "2026-05-04"]
    sources = ["gdelt", "reddit", "gdelt", "gdelt", "gdelt", "reddit", "reddit", "reddit", "gdelt", "reddit"]
    labels = ["positive", "neutral", "negative", "neutral", "positive", "negative", "neutral", "positive", "neutral", "negative"]
    scores = [0.8, 0.5, 0.9, 0.4, 0.7, 0.6, 0.5, 0.9, 0.4, 0.8]
    compounds = [0.6, 0.0, -0.8, 0.1, 0.5, -0.4, 0.0, 0.8, 0.1, -0.7]
    texts = [f"Text {i}" for i in range(10)]

    return pd.DataFrame({
        "date": dates,
        "source": sources,
        "text_clean": texts,
        "roberta_sentiment_label": labels,
        "roberta_sentiment_score": scores,
        "roberta_sentiment_compound": compounds,
        "roberta_prob_negative": [0.1] * 10,
        "roberta_prob_neutral": [0.2] * 10,
        "roberta_prob_positive": [0.7] * 10,
    })


def test_table_roberta_daily_counts(dummy_sentiment_records):
    counts = table_roberta_daily_counts(dummy_sentiment_records)

    # Date range is 2026-05-01 to 2026-05-04 (4 days including 2026-05-03 which has 0 texts)
    assert len(counts) == 4
    assert counts.columns.tolist() == ["date", "gdelt_count", "reddit_count", "total_count"]
    
    # Check zero-text day (2026-05-03)
    row_empty = counts[counts["date"] == "2026-05-03"].iloc[0]
    assert row_empty["gdelt_count"] == 0
    assert row_empty["reddit_count"] == 0
    assert row_empty["total_count"] == 0

    # Check 2026-05-01
    row_first = counts[counts["date"] == "2026-05-01"].iloc[0]
    assert row_first["gdelt_count"] == 2
    assert row_first["reddit_count"] == 1
    assert row_first["total_count"] == 3


def test_table_roberta_count_descriptives(dummy_sentiment_records):
    descriptives = table_roberta_count_descriptives(dummy_sentiment_records)

    assert "metric" in descriptives.columns
    assert "gdelt" in descriptives.columns
    assert "reddit" in descriptives.columns
    assert "total" in descriptives.columns

    # Verify key metrics
    start_date_row = descriptives[descriptives["metric"] == "start_date"].iloc[0]
    assert start_date_row["total"] == "2026-05-01"

    end_date_row = descriptives[descriptives["metric"] == "end_date"].iloc[0]
    assert end_date_row["total"] == "2026-05-04"

    days_row = descriptives[descriptives["metric"] == "days_in_sample"].iloc[0]
    assert days_row["total"] == 4

    total_texts_row = descriptives[descriptives["metric"] == "total_texts"].iloc[0]
    assert total_texts_row["gdelt"] == 5
    assert total_texts_row["reddit"] == 5
    assert total_texts_row["total"] == 10

    zero_days_row = descriptives[descriptives["metric"] == "zero_text_days"].iloc[0]
    assert zero_days_row["gdelt"] == 2  # 2026-05-02 and 2026-05-03 have no reddit/gdelt
    assert zero_days_row["reddit"] == 2  # 2026-05-02 and 2026-05-03
    assert zero_days_row["total"] == 1   # Only 2026-05-03 has zero total count


def test_table_roberta_validation_sample(dummy_sentiment_records):
    sample = table_roberta_validation_sample(dummy_sentiment_records, n=4, seed=42)

    # Should select 4 rows (2 from gdelt, 2 from reddit due to stratified sampling)
    assert len(sample) == 4
    assert set(sample["source"]) == {"gdelt", "reddit"}
    assert (sample["source"] == "gdelt").sum() == 2
    assert (sample["source"] == "reddit").sum() == 2

    # Verify manual labeling columns are added and blank
    assert "manual_label" in sample.columns
    assert "manual_notes" in sample.columns
    assert sample["manual_label"].eq("").all()
    assert sample["manual_notes"].eq("").all()

    # Reproducibility check: same seed should give identical dataframe
    sample2 = table_roberta_validation_sample(dummy_sentiment_records, n=4, seed=42)
    pd.testing.assert_frame_equal(sample, sample2)


def test_table_roberta_extreme_sentiment(dummy_sentiment_records):
    extreme = table_roberta_extreme_sentiment(dummy_sentiment_records, top_n=2)

    # 2 positive and 2 negative per source = 8 rows total
    assert len(extreme) == 8
    assert set(extreme["sentiment_class"]) == {"Extreme Positive", "Extreme Negative"}
    assert set(extreme["source"]) == {"gdelt", "reddit"}

    # GDELT most positive compound score in dummy is 0.6, most negative is -0.8
    gdelt_extreme = extreme[extreme["source"] == "gdelt"]
    assert gdelt_extreme[gdelt_extreme["sentiment_class"] == "Extreme Positive"]["compound_score"].max() == 0.6
    assert gdelt_extreme[gdelt_extreme["sentiment_class"] == "Extreme Negative"]["compound_score"].min() == -0.8


def test_table_roberta_label_distribution(dummy_sentiment_records):
    dist = table_roberta_label_distribution(dummy_sentiment_records)

    assert len(dist) == 6  # 3 labels (negative, neutral, positive) * 2 sources
    assert set(dist["label"]) == {"negative", "neutral", "positive"}

    # Check GDELT label counts
    # Positive: index 0 (0.6 compound), index 4 (0.5 compound) -> 2
    # Neutral: index 3 (0.1 compound), index 8 (0.1 compound) -> 2
    # Negative: index 2 (-0.8 compound) -> 1
    gdelt_positive = dist[(dist["source"] == "gdelt") & (dist["label"] == "positive")].iloc[0]
    assert gdelt_positive["count"] == 2
    assert gdelt_positive["percentage"] == 40.0


def test_table_roberta_daily_counts_shifted(dummy_sentiment_records):
    # Let's define a trading dates calendar.
    # Dummy records dates:
    # 2026-05-01 (Friday) - 3 texts
    # 2026-05-02 (Saturday) - 2 texts
    # 2026-05-04 (Monday) - 5 texts
    # Let's say 2026-05-02 is a Saturday (non-trading day), and 2026-05-01 and 2026-05-04 are trading days.
    # Trading days series: ["2026-05-01", "2026-05-04"]
    trading_dates = pd.Series(["2026-05-01", "2026-05-04"])

    counts = table_roberta_daily_counts_shifted(dummy_sentiment_records, trading_dates)

    # It should only contain the 2 trading days
    assert len(counts) == 2
    assert counts.columns.tolist() == ["date", "gdelt_count", "reddit_count", "total_count"]

    # 2026-05-01 should have only its own texts:
    # 2026-05-01: gdelt (index 0, 2), reddit (index 1) -> gdelt_count=2, reddit_count=1, total=3
    row_first = counts[counts["date"] == "2026-05-01"].iloc[0]
    assert row_first["gdelt_count"] == 2
    assert row_first["reddit_count"] == 1
    assert row_first["total_count"] == 3

    # 2026-05-04 is a Monday. The Saturday (2026-05-02) texts should be shifted forward to 2026-05-04.
    # 2026-05-02 texts: gdelt (index 3, 4) -> 2 gdelt texts, 0 reddit texts.
    # 2026-05-04 texts: reddit (index 5, 6, 7, 9), gdelt (index 8) -> 1 gdelt text, 4 reddit texts.
    # Combined for 2026-05-04:
    # gdelt: 2 (from Saturday) + 1 (from Monday) = 3
    # reddit: 0 (from Saturday) + 4 (from Monday) = 4
    # total: 7
    row_second = counts[counts["date"] == "2026-05-04"].iloc[0]
    assert row_second["gdelt_count"] == 3
    assert row_second["reddit_count"] == 4
    assert row_second["total_count"] == 7


def test_table_roberta_count_descriptives_shifted(dummy_sentiment_records):
    trading_dates = pd.Series(["2026-05-01", "2026-05-04"])
    descriptives = table_roberta_count_descriptives_shifted(dummy_sentiment_records, trading_dates)

    assert "metric" in descriptives.columns
    assert "gdelt" in descriptives.columns
    assert "reddit" in descriptives.columns
    assert "total" in descriptives.columns

    # Verify key metrics
    start_date_row = descriptives[descriptives["metric"] == "start_date"].iloc[0]
    assert start_date_row["total"] == "2026-05-01"

    end_date_row = descriptives[descriptives["metric"] == "end_date"].iloc[0]
    assert end_date_row["total"] == "2026-05-04"

    days_row = descriptives[descriptives["metric"] == "days_in_sample"].iloc[0]
    assert days_row["total"] == 2

    total_texts_row = descriptives[descriptives["metric"] == "total_texts"].iloc[0]
    assert total_texts_row["gdelt"] == 5
    assert total_texts_row["reddit"] == 5
    assert total_texts_row["total"] == 10

    zero_days_row = descriptives[descriptives["metric"] == "zero_text_days"].iloc[0]
    assert zero_days_row["gdelt"] == 0
    assert zero_days_row["reddit"] == 0
    assert zero_days_row["total"] == 0

