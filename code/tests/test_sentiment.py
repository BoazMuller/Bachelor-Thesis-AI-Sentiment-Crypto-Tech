import pandas as pd
import pytest

from thesis.sentiment import (
    append_roberta_sentiment,
    normalize_for_twitter_roberta,
    ordered_sentiment_labels,
)


def test_normalize_for_twitter_roberta_replaces_urls_and_mentions():
    text = "Read https://example.com from @SomeUser"

    assert normalize_for_twitter_roberta(text) == "Read http from @user"


def test_ordered_sentiment_labels_accepts_cardiffnlp_labels():
    id_to_label = {0: "Negative", 1: "Neutral", 2: "Positive"}

    assert ordered_sentiment_labels(id_to_label) == ["negative", "neutral", "positive"]


def test_ordered_sentiment_labels_rejects_non_three_class_model():
    id_to_label = {
        0: "strongly negative",
        1: "negative",
        2: "negative or neutral",
        3: "positive",
        4: "strongly positive",
    }

    with pytest.raises(ValueError, match="three-class"):
        ordered_sentiment_labels(id_to_label)


def test_append_roberta_sentiment_leaves_empty_text_unclassified():
    class FakeClassifier:
        def predict_proba(self, texts, batch_size):
            assert texts == ["AI is useful"]
            assert batch_size == 2
            return pd.DataFrame(
                {
                    "roberta_prob_negative": [0.1],
                    "roberta_prob_neutral": [0.2],
                    "roberta_prob_positive": [0.7],
                    "roberta_sentiment_label": ["positive"],
                    "roberta_sentiment_score": [0.7],
                    "roberta_sentiment_compound": [0.6],
                }
            )

    chunk = pd.DataFrame({"text_clean": ["AI is useful", "  "]})

    result = append_roberta_sentiment(
        chunk=chunk,
        classifier=FakeClassifier(),
        text_column="text_clean",
        model_batch_size=2,
    )

    assert result.loc[0, "roberta_sentiment_label"] == "positive"
    assert pd.isna(result.loc[1, "roberta_sentiment_label"])
