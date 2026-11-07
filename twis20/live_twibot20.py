"""Live-compatible TwiBot-20 profile+text inference."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from scipy import sparse

from .profile_features import extract_profile_features
from .text_features import extract_text_features

ROOT = Path(__file__).resolve().parent
MODEL_PACKAGE = ROOT / "phase4_live_profile_text_run01"
MODEL = MODEL_PACKAGE / "profile_text_model.joblib"
PREPROCESSOR = MODEL_PACKAGE / "preprocessor.joblib"
DEPLOYMENT_MANIFEST = MODEL_PACKAGE / "deployment_manifest.json"
GROUPS = ("profile", "text_summary", "text_tfidf")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _created_at(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
        return dt.strftime("%a %b %d %H:%M:%S +0000 %Y")
    except ValueError:
        return None


def normalize_live_profile(user: dict) -> dict:
    """Map only observed X fields to Phase 3 raw profile field names."""
    metrics = (
        user.get("public_metrics", {})
        if isinstance(user.get("public_metrics"), dict)
        else user
    )
    return {
        "followers_count": metrics.get("followers_count", metrics.get("follower_count")),
        "friends_count": metrics.get("following_count", metrics.get("following")),
        "statuses_count": metrics.get("tweet_count"),
        "favourites_count": metrics.get("favourites_count"),
        "listed_count": metrics.get("listed_count"),
        "verified": user.get("verified"),
        "protected": user.get("protected"),
        "created_at": _created_at(user.get("created_at")),
        "description": user.get("description"),
    }


def load_live_artifacts():
    manifest = json.loads(DEPLOYMENT_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("experiment") != "profile_text"
        or manifest.get("feature_groups") != list(GROUPS)
        or manifest.get("network_features_used") is not False
    ):
        raise RuntimeError("Live model feature groups do not match the implementation")
    names = [
        name
        for group in GROUPS
        for name in manifest.get("feature_names", {}).get(group, [])
    ]
    expected_count = manifest.get("feature_count")
    if expected_count != 50072 or len(names) != expected_count or len(set(names)) != expected_count:
        raise RuntimeError("Live model feature schema is incomplete or duplicated")
    threshold = manifest.get("threshold")
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise RuntimeError("Live model threshold is missing or invalid")
    if _sha256_file(MODEL) != manifest.get("model_sha256"):
        raise RuntimeError("Live profile+text model hash mismatch")
    if _sha256_file(PREPROCESSOR) != manifest.get("preprocessor_sha256"):
        raise RuntimeError("Live preprocessing artifact hash mismatch")

    preprocessor = joblib.load(PREPROCESSOR)
    model = joblib.load(MODEL)
    if model.n_features_in_ != len(names):
        raise RuntimeError("Live model input dimension does not match the feature schema")
    return manifest, preprocessor, model, names


def _feature_schema_audit(preprocessor, expected_feature_names, transformed, matrix):
    expected_names = list(expected_feature_names)
    actual_names = (
        list(preprocessor.feature_names_.get("profile", ()))
        + list(preprocessor.feature_names_.get("text_summary", ()))
        + [
            f"tweet.tfidf.{name}"
            for name in preprocessor.feature_names_.get("text_tfidf", ())
        ]
    )
    expected_count = 50072
    valid_names = [
        name for name in expected_names if isinstance(name, str) and name.strip()
    ]
    duplicate_count = len(valid_names) - len(set(valid_names))
    invalid_count = len(expected_names) - len(valid_names)
    order_validated = (
        len(expected_names) == expected_count
        and len(actual_names) == expected_count
        and actual_names == expected_names
        and duplicate_count == 0
        and invalid_count == 0
    )

    profile_medians = preprocessor.profile_medians_
    summary_medians = preprocessor.summary_medians_
    preprocessing_validated = (
        preprocessor.is_fitted_
        and set(profile_medians) == set(preprocessor.profile_columns)
        and set(summary_medians) == set(preprocessor.summary_columns)
        and len(preprocessor.feature_names_.get("profile", ())) == 56
        and len(preprocessor.feature_names_.get("text_summary", ())) == 16
        and len(preprocessor.feature_names_.get("text_tfidf", ())) == 50000
        and all(value == value and abs(value) != float("inf") for value in profile_medians.values())
        and all(value == value and abs(value) != float("inf") for value in summary_medians.values())
        and transformed["profile"].shape[1] == 56
        and transformed["text_summary"].shape[1] == 16
        and transformed["text_tfidf"].shape[1] == 50000
    )
    constructed_count = int(matrix.shape[1])
    return {
        "expected_feature_count": expected_count,
        "constructed_feature_count": constructed_count,
        "profile_feature_count": int(transformed["profile"].shape[1]),
        "tweet_summary_feature_count": int(transformed["text_summary"].shape[1]),
        "tfidf_feature_count": int(transformed["text_tfidf"].shape[1]),
        "missing_feature_count": max(expected_count - constructed_count, 0),
        "duplicate_or_invalid_feature_count": duplicate_count + invalid_count,
        "feature_order_validated": order_validated,
        "preprocessing_validated": bool(preprocessing_validated),
        "preprocessing_details": {
            "profile": "Packaged train-fitted medians applied to profile features.",
            "tweet_summary": "Packaged train-fitted medians applied to tweet-summary features.",
            "tfidf": "Packaged train-fitted TF-IDF vocabulary applied to tweet text.",
            "network": "Not used by this profile+text model.",
        },
    }


def construct_profile_text(
    user: dict,
    tweets: list[dict],
    preprocessor,
    expected_feature_names=None,
):
    profile = extract_profile_features([normalize_live_profile(user)])
    texts = [tweet.get("text", "") for tweet in tweets if isinstance(tweet, dict)]
    summary, documents = extract_text_features([texts])
    transformed = preprocessor.transform(profile, summary, documents)
    matrix = sparse.hstack(
        [
            sparse.csr_matrix(
                transformed["profile"].to_numpy(dtype="float32")
            ),
            sparse.csr_matrix(
                transformed["text_summary"].to_numpy(dtype="float32")
            ),
            transformed["text_tfidf"],
        ],
        format="csr",
        dtype="float32",
    )
    if expected_feature_names is None:
        expected_feature_names = (
            list(preprocessor.profile_columns)
            + list(preprocessor.summary_columns)
            + [
                f"tweet.tfidf.{name}"
                for name in preprocessor.feature_names_.get("text_tfidf", ())
            ]
        )
    audit = _feature_schema_audit(
        preprocessor, expected_feature_names, transformed, matrix
    )
    available = {"profile": 56, "text_summary": 16, "text_tfidf": 50000}
    return matrix, available, [], audit
