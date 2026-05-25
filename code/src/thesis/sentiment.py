from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


SENTIMENT_LABELS = ("negative", "neutral", "positive")
PROBABILITY_COLUMNS = tuple(f"roberta_prob_{label}" for label in SENTIMENT_LABELS)

URL_PATTERN = re.compile(r"\b(?:https?://\S+|www\.\S+)", flags=re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")


def normalize_for_twitter_roberta(text: Any) -> str:
    if pd.isna(text):
        return ""

    value = str(text)
    value = URL_PATTERN.sub("http", value)
    value = value.replace("<URL>", "http").replace("<url>", "http")
    value = MENTION_PATTERN.sub("@user", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_sentiment_label(label: Any, index: int | None = None) -> str:
    value = str(label).strip().lower()
    value = value.replace("label_", "").replace("LABEL_", "")

    if value in {"0", "negative", "neg"} or index == 0 and value in {"label_0", ""}:
        return "negative"
    if value in {"1", "neutral", "neu"} or index == 1 and value in {"label_1", ""}:
        return "neutral"
    if value in {"2", "positive", "pos"} or index == 2 and value in {"label_2", ""}:
        return "positive"

    if "negative" in value and "neutral" not in value:
        return "negative"
    if "neutral" in value:
        return "neutral"
    if "positive" in value:
        return "positive"

    raise ValueError(f"Unsupported sentiment label: {label!r}")


def ordered_sentiment_labels(id_to_label: dict[int | str, Any]) -> list[str]:
    ordered_items = sorted((int(key), value) for key, value in id_to_label.items())
    labels = [
        normalize_sentiment_label(raw_label, index=index)
        for index, raw_label in ordered_items
    ]

    if set(labels) != set(SENTIMENT_LABELS) or len(labels) != len(SENTIMENT_LABELS):
        raise ValueError(
            "Expected a three-class negative/neutral/positive sentiment model. "
            f"Observed labels: {labels}"
        )

    return labels


class RobertaSentimentClassifier:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_length: int = 512,
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "RoBERTa sentiment classification requires torch and transformers. "
                "Install them with `python -m pip install torch transformers`."
            ) from exc

        self.torch = torch
        self.device = self._resolve_device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model.to(self.device)
        self.model.eval()
        self.labels = ordered_sentiment_labels(self.model.config.id2label)

    def _resolve_device(self, requested_device: str) -> str:
        if requested_device != "auto":
            return requested_device

        if self.torch.cuda.is_available():
            return "cuda"

        if (
            hasattr(self.torch.backends, "mps")
            and self.torch.backends.mps.is_available()
        ):
            return "mps"

        return "cpu"

    def predict_proba(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> pd.DataFrame:
        rows: list[dict[str, float | str]] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with self.torch.inference_mode():
                output = self.model(**encoded)
                probabilities = (
                    self.torch.softmax(output.logits, dim=-1).detach().cpu().numpy()
                )

            for probability_vector in probabilities:
                probability_by_label = {
                    f"roberta_prob_{label}": float(probability_vector[index])
                    for index, label in enumerate(self.labels)
                }
                predicted_index = int(np.argmax(probability_vector))
                predicted_label = self.labels[predicted_index]

                rows.append(
                    {
                        **probability_by_label,
                        "roberta_sentiment_label": predicted_label,
                        "roberta_sentiment_score": float(
                            probability_vector[predicted_index]
                        ),
                        "roberta_sentiment_compound": (
                            probability_by_label["roberta_prob_positive"]
                            - probability_by_label["roberta_prob_negative"]
                        ),
                    }
                )

        return pd.DataFrame(rows)


def append_roberta_sentiment(
    chunk: pd.DataFrame,
    classifier: RobertaSentimentClassifier,
    text_column: str,
    model_batch_size: int,
    apply_twitter_preprocessing: bool = True,
) -> pd.DataFrame:
    if text_column not in chunk.columns:
        raise ValueError(
            f"Column {text_column!r} was not found. Available columns: "
            f"{list(chunk.columns)}"
        )

    enriched = chunk.copy()
    output_columns = [
        *PROBABILITY_COLUMNS,
        "roberta_sentiment_label",
        "roberta_sentiment_score",
        "roberta_sentiment_compound",
    ]

    for column in output_columns:
        if column == "roberta_sentiment_label":
            enriched[column] = pd.Series(pd.NA, index=enriched.index, dtype="object")
        else:
            enriched[column] = np.nan

    text_values = enriched[text_column].fillna("").astype(str)
    non_empty_mask = text_values.str.strip().ne("")
    if not non_empty_mask.any():
        return enriched

    texts = text_values.loc[non_empty_mask].tolist()
    if apply_twitter_preprocessing:
        texts = [normalize_for_twitter_roberta(text) for text in texts]

    predictions = classifier.predict_proba(texts, batch_size=model_batch_size)
    enriched.loc[non_empty_mask, predictions.columns] = predictions.to_numpy()
    return enriched
