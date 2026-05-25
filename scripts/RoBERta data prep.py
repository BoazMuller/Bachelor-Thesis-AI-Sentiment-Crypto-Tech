"""
Prepare text fields for RoBERTa sentiment classification.
"""

import re
import pandas as pd

#Load Data
reddit_df = pd.read_csv('/Users/boazmuller/thesis/data/interim/reddit_merged.csv')
gdelt_df = pd.read_csv('/Users/boazmuller/thesis/data/raw/gdelt/gdelt_ai_headlines_100_per_day_2024_04_01_to_2026_04_01.csv')

#Format Data
gdelt_clean = (
    gdelt_df
    .rename(columns={
        "gdelt_date": "date",
        "headline": "text",
    })
    [["date", "text"]]
    .reset_index(drop=True)
)

gdelt_clean["id"] = "gdelt_" + gdelt_clean.index.astype(str)
gdelt_clean["source"] = "gdelt"

reddit_clean = (
    reddit_df
    .rename(columns={
        "created_day": "date",
        "full_text": "text",
    })
    [["date", "text"]]
    .reset_index(drop=True)
)

reddit_clean["id"] = "reddit_" + reddit_clean.index.astype(str)
reddit_clean["source"] = "reddit"

combined_df = pd.concat(
    [gdelt_clean, reddit_clean],
    ignore_index=True
)

combined_df = combined_df[["id", "date", "text", "source"]]

#Remove duplicates
combined_dedup_df = combined_df[['date', 'text', 'source']].drop_duplicates()


#Clean Data
url_pattern = re.compile(
    r"""(?ix)
    \b(
        https?://\S+ |
        www\.\S+
    )
    """
)
"""Prepare text fields for RoBERTa sentiment classification."""


def clean_text_for_sentiment(text: str, replace_urls: bool = True) -> str:
    if pd.isna(text):
        return ""

    text = str(text)

    if replace_urls:
        text = url_pattern.sub("<URL>", text)
    else:
        text = url_pattern.sub("", text)

    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text

combined_dedup_df["text_clean"] = combined_dedup_df["text"].apply(clean_text_for_sentiment)

combined_dedup_df[['date', 'text_clean', 'source']].to_csv('/Users/boazmuller/thesis/data/interim/combined_dedup_df.csv', index=False)
