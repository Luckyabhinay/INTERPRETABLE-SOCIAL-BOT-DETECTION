"""Tweet summaries, train-only lexical transformation, and overlap flags."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .feature_schema import TEXT_SUMMARY_COLUMNS, TFIDF_CONFIG

_HASHTAG = re.compile(r"(?<!\w)#[\w]+", re.UNICODE)
_MENTION = re.compile(r"(?<![\w@])@[\w]+", re.UNICODE)
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_WORD = re.compile(r"(?u)\b\w\w+\b")


def _string_tweets(value: Any) -> tuple[bool, tuple[str, ...], int, int]:
    if not isinstance(value, list):
        return False, (), 0, 0
    strings = tuple(item for item in value if isinstance(item, str))
    invalid_items = len(value) - len(strings)
    empty_items = sum(not text.strip() for text in strings)
    return True, strings, invalid_items, empty_items


def extract_text_features(
    tweet_lists: Sequence[Any],
) -> tuple[pd.DataFrame, tuple[tuple[str, ...], ...]]:
    """Extract account summaries and unmodified ordered tweet documents."""
    rows: list[dict[str, float]] = []
    documents: list[tuple[str, ...]] = []
    for value in tweet_lists:
        available, strings, invalid_count, empty_count = _string_tweets(value)
        lengths = [len(text) for text in strings]
        word_counts = [len(text.split()) for text in strings]
        unique_count = len(set(strings))
        valid_count = len(strings)
        rows.append(
            {
                "tweet.list_available": float(available),
                "tweet.item_count": float(len(value) if available else 0),
                "tweet.valid_string_count": float(valid_count),
                "tweet.nonempty_count": float(
                    sum(bool(text.strip()) for text in strings)
                ),
                "tweet.total_char_count": float(sum(lengths)),
                "tweet.mean_char_count": float(np.mean(lengths)) if lengths else 0.0,
                "tweet.median_char_count": float(np.median(lengths)) if lengths else 0.0,
                "tweet.total_word_count": float(sum(word_counts)),
                "tweet.mean_word_count": (
                    float(np.mean(word_counts)) if word_counts else 0.0
                ),
                "tweet.median_word_count": (
                    float(np.median(word_counts)) if word_counts else 0.0
                ),
                "tweet.unique_exact_text_count": float(unique_count),
                "tweet.exact_duplicate_fraction": (
                    float((valid_count - unique_count) / valid_count)
                    if valid_count
                    else 0.0
                ),
                "tweet.hashtag_count": float(
                    sum(len(_HASHTAG.findall(text)) for text in strings)
                ),
                "tweet.mention_count": float(
                    sum(len(_MENTION.findall(text)) for text in strings)
                ),
                "tweet.url_count": float(
                    sum(len(_URL.findall(text)) for text in strings)
                ),
                "tweet.empty_or_invalid_item_count": float(
                    empty_count + invalid_count
                ),
            }
        )
        documents.append(strings)

    frame = pd.DataFrame(rows, columns=TEXT_SUMMARY_COLUMNS, dtype=np.float64)
    if frame.columns.tolist() != list(TEXT_SUMMARY_COLUMNS):
        raise AssertionError("Tweet summary feature order does not match the schema")
    return frame, tuple(documents)


def analyze_tweet_document(document: Iterable[str]) -> Iterable[str]:
    """Generate word unigrams/bigrams without crossing tweet boundaries."""
    for tweet in document:
        words = _WORD.findall(tweet.lower())
        yield from words
        yield from (
            f"{left} {right}" for left, right in zip(words, words[1:])
        )


def create_tfidf_vectorizer() -> TfidfVectorizer:
    """Create the fixed train-only text transformer specified by the schema."""
    return TfidfVectorizer(
        analyzer=analyze_tweet_document,
        min_df=TFIDF_CONFIG["min_df"],
        max_features=TFIDF_CONFIG["max_features"],
        sublinear_tf=TFIDF_CONFIG["sublinear_tf"],
        norm=TFIDF_CONFIG["norm"],
        use_idf=TFIDF_CONFIG["use_idf"],
        smooth_idf=TFIDF_CONFIG["smooth_idf"],
        binary=TFIDF_CONFIG["binary"],
    )


def normalize_overlap_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _text_hash(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def overlap_exposure(
    documents: Sequence[Sequence[str]],
    splits: Sequence[str],
) -> pd.DataFrame:
    """Return sidecar exposure flags against train text, retaining every row."""
    if len(documents) != len(splits):
        raise ValueError("Document and split lengths differ")
    train_exact: set[bytes] = set()
    train_normalized: set[bytes] = set()
    for document, split in zip(documents, splits):
        if split != "train":
            continue
        for text in document:
            train_exact.add(_text_hash(text))
            train_normalized.add(_text_hash(normalize_overlap_text(text)))

    rows: list[dict[str, Any]] = []
    for document, split in zip(documents, splits):
        if split == "train":
            exact_exposure: bool | None = None
            normalized_exposure: bool | None = None
        elif split in {"dev", "test"}:
            exact_exposure = any(_text_hash(text) in train_exact for text in document)
            normalized_exposure = any(
                _text_hash(normalize_overlap_text(text)) in train_normalized
                for text in document
            )
        else:
            raise ValueError(f"Unexpected official split: {split!r}")
        rows.append(
            {
                "text_overlap_exact_train": exact_exposure,
                "text_overlap_normalized_train": normalized_exposure,
            }
        )
    return pd.DataFrame(
        rows,
        columns=("text_overlap_exact_train", "text_overlap_normalized_train"),
    )


__all__ = [
    "TEXT_SUMMARY_COLUMNS",
    "analyze_tweet_document",
    "create_tfidf_vectorizer",
    "extract_text_features",
    "normalize_overlap_text",
    "overlap_exposure",
]
