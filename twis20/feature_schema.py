"""Stable feature names and schema metadata for TwiBot-20."""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.0"
SPLITS = ("train", "dev", "test")
FEATURE_GROUP_ORDER = ("profile", "text_summary", "text_tfidf", "network")

PROFILE_NUMERIC_FIELDS = (
    "followers_count",
    "friends_count",
    "statuses_count",
    "favourites_count",
    "listed_count",
)

PROFILE_BOOLEAN_FIELDS = (
    "verified",
    "geo_enabled",
    "protected",
    "default_profile",
    "default_profile_image",
    "contributors_enabled",
    "has_extended_profile",
    "is_translation_enabled",
    "is_translator",
    "profile_background_tile",
    "profile_use_background_image",
)

PROFILE_DERIVED_FIELDS = (
    "created_at_year",
    "description_present",
    "description_char_count",
    "description_word_count",
    "description_hashtag_count",
    "description_mention_count",
    "description_url_count",
)

PROFILE_COLUMNS = tuple(
    column
    for field in PROFILE_NUMERIC_FIELDS
    for column in (f"profile.{field}", f"profile.{field}__missing")
) + tuple(
    f"profile.{field}__{state}"
    for field in PROFILE_BOOLEAN_FIELDS
    for state in ("true", "false", "unknown")
) + tuple(
    column
    for field in PROFILE_DERIVED_FIELDS
    for column in (
        (f"profile.{field}", f"profile.{field}__missing")
        if field in {
            "created_at_year",
            "description_char_count",
            "description_word_count",
            "description_hashtag_count",
            "description_mention_count",
            "description_url_count",
        }
        else (f"profile.{field}",)
    )
)

TEXT_SUMMARY_FIELDS = (
    "list_available",
    "item_count",
    "valid_string_count",
    "nonempty_count",
    "total_char_count",
    "mean_char_count",
    "median_char_count",
    "total_word_count",
    "mean_word_count",
    "median_word_count",
    "unique_exact_text_count",
    "exact_duplicate_fraction",
    "hashtag_count",
    "mention_count",
    "url_count",
    "empty_or_invalid_item_count",
)
TEXT_SUMMARY_COLUMNS = tuple(f"tweet.{field}" for field in TEXT_SUMMARY_FIELDS)

NETWORK_COLUMNS = (
    "network.friend_in_degree",
    "network.friend_out_degree",
    "network.follow_in_degree",
    "network.follow_out_degree",
    "network.combined_in_degree",
    "network.combined_out_degree",
    "network.combined_unique_neighbor_count",
    "network.isolate",
)

TFIDF_CONFIG: dict[str, Any] = {
    "lowercase": True,
    "token_pattern": r"(?u)\b\w\w+\b",
    "ngram_range": [1, 2],
    "min_df": 2,
    "max_features": 50_000,
    "sublinear_tf": True,
    "norm": "l2",
    "use_idf": True,
    "smooth_idf": True,
    "binary": False,
    "analyzer": "per_tweet_word_unigrams_and_bigrams",
}

MISSING_VALUE_RULES = {
    "strings": "trim; empty, none, null, and nan (case-insensitive) are missing",
    "numeric_counts": "finite numeric values >= 0; all other values are missing",
    "boolean_tokens": {
        "true": ["true", "1"],
        "false": ["false", "0"],
        "unknown": "missing or any unrecognized value",
    },
    "learned_imputation": "training-row median; all-missing training columns use 0",
}


def feature_schema_manifest(
    text_tfidf_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return a JSON-serializable description with exact feature order."""
    groups = {
        "profile": list(PROFILE_COLUMNS),
        "text_summary": list(TEXT_SUMMARY_COLUMNS),
        "text_tfidf": [f"tweet.tfidf.{name}" for name in text_tfidf_names],
        "network": list(NETWORK_COLUMNS),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_group_order": list(FEATURE_GROUP_ORDER),
        "feature_names": groups,
        "combined_feature_names": [
            name for group in FEATURE_GROUP_ORDER for name in groups[group]
        ],
        "tfidf": {**TFIDF_CONFIG, "feature_names": list(text_tfidf_names)},
        "missing_value_rules": MISSING_VALUE_RULES,
        "identifiers_and_target": {
            "account_id": "metadata sidecar only; excluded from feature matrices",
            "split": "metadata sidecar only; excluded from feature matrices",
            "label": "target sidecar only; excluded from feature matrices",
        },
        "network_protocol": {
            "relations": ["friend", "follow"],
            "post_edges": "excluded",
            "support_nodes": "unlabeled topology only; may contribute to degrees",
            "transductive": True,
            "validation_test_topology_visible": True,
            "label_derived_features": False,
        },
        "unavailable_features": [
            "tweet likes",
            "tweet replies",
            "tweet reposts or retweets",
            "tweet impressions",
            "tweet engagement rates",
        ],
    }
