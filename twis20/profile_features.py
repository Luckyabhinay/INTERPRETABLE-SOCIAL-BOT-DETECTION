"""Deterministic profile-only feature extraction."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .feature_schema import (
    PROFILE_BOOLEAN_FIELDS,
    PROFILE_COLUMNS,
    PROFILE_NUMERIC_FIELDS,
)

_HASHTAG = re.compile(r"(?<!\w)#[\w]+", re.UNICODE)
_MENTION = re.compile(r"(?<![\w@])@[\w]+", re.UNICODE)
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_MISSING_STRINGS = {"", "none", "null", "nan"}


def is_missing(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().casefold() in _MISSING_STRINGS
    )


def _finite_nonnegative_number(value: Any) -> float:
    if is_missing(value) or isinstance(value, bool):
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(number) or number < 0:
        return math.nan
    return number


def _boolean_state(value: Any) -> str:
    if is_missing(value):
        return "unknown"
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1"}:
        return "true"
    if normalized in {"false", "0"}:
        return "false"
    return "unknown"


def _created_year(value: Any) -> float:
    if is_missing(value) or not isinstance(value, str):
        return math.nan
    text = str(value).strip()
    for date_format in (
        "%a %b %d %H:%M:%S %z %Y",
        "%a %b %d %H:%M:%S %Y",
    ):
        try:
            return float(datetime.strptime(text, date_format).year)
        except ValueError:
            continue
    return math.nan


def _description_counts(profile: Mapping[str, Any]) -> dict[str, float]:
    description = profile.get("description")
    if is_missing(description) or not isinstance(description, str):
        return {
            "profile.description_present": 0.0,
            "profile.description_char_count": math.nan,
            "profile.description_word_count": math.nan,
            "profile.description_hashtag_count": math.nan,
            "profile.description_mention_count": math.nan,
            "profile.description_url_count": math.nan,
        }
    text = description
    return {
        "profile.description_present": 1.0,
        "profile.description_char_count": float(len(text)),
        "profile.description_word_count": float(len(text.split())),
        "profile.description_hashtag_count": float(len(_HASHTAG.findall(text))),
        "profile.description_mention_count": float(len(_MENTION.findall(text))),
        "profile.description_url_count": float(len(_URL.findall(text))),
    }


def extract_profile_features(
    profiles: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Extract a fixed-order, non-identity profile feature frame."""
    rows: list[dict[str, float]] = []
    for profile in profiles:
        row: dict[str, float] = {}
        for field in PROFILE_NUMERIC_FIELDS:
            name = f"profile.{field}"
            value = _finite_nonnegative_number(profile.get(field))
            row[name] = value
            row[f"{name}__missing"] = float(math.isnan(value))

        for field in PROFILE_BOOLEAN_FIELDS:
            state = _boolean_state(profile.get(field))
            for candidate in ("true", "false", "unknown"):
                row[f"profile.{field}__{candidate}"] = float(state == candidate)

        row["profile.created_at_year"] = _created_year(profile.get("created_at"))
        row["profile.created_at_year__missing"] = float(
            math.isnan(row["profile.created_at_year"])
        )
        row.update(_description_counts(profile))
        for field in (
            "description_char_count",
            "description_word_count",
            "description_hashtag_count",
            "description_mention_count",
            "description_url_count",
        ):
            name = f"profile.{field}"
            row[f"{name}__missing"] = float(math.isnan(row[name]))
        rows.append(row)

    frame = pd.DataFrame(rows, columns=PROFILE_COLUMNS, dtype=np.float64)
    if frame.columns.tolist() != list(PROFILE_COLUMNS):
        raise AssertionError("Profile feature order does not match the schema")
    return frame


__all__ = ["PROFILE_COLUMNS", "extract_profile_features"]
