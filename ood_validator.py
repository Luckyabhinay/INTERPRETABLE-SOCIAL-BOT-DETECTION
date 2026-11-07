"""
Feature Validation & Out-Of-Distribution (OOD) Detection Module
===============================================================
Performs rigorous, non-destructive audit of live feature vectors
against the empirical Cresci-15 training distributions:

1. Detects window-incompatible cumulative counts (e.g. 20-tweet sample vs historical lifetime sums).
2. Detects extreme out-of-distribution engagement metrics (avg_replies, avg_favorites).
3. Identifies missing and imputed attributes.
4. Checks verification status against the training reality (Cresci-15 had 0 verified accounts).
5. Determines whether an account can be reliably classified or must be flagged
   as 'Unable to Reliably Classify (Out-of-Distribution / Window Mismatch)'.
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_FILE = os.path.join(BASE_DIR, "cresci15_training_distributions.json")

TRAINING_DISTRIBUTIONS = {}
if os.path.exists(DIST_FILE):
    with open(DIST_FILE, "r", encoding="utf-8") as f:
        TRAINING_DISTRIBUTIONS = json.load(f)

# Features that represent cumulative sums across an entire tweet archive in Cresci-15 training
CUMULATIVE_COUNT_FEATURES = [
    "hashtag_count",
    "mentions_count",
    "url_count",
    "retweet_count",
    "reply_count"
]

def evaluate_feature_compatibility_and_ood(
    feature_dict: Dict[str, float],
    available_features: List[str],
    missing_features: List[str],
    tweets_analyzed_count: int,
    account_status_count: float
) -> Dict[str, Any]:
    """
    Audits the live feature vector against Cresci-15 training distributions.
    Does NOT modify, clamp, or invent any feature values.

    Returns an audit report containing:
      - is_reliable: bool (False if critical features are incompatible or severely OOD)
      - recommendation_label: Optional[str] (e.g. 'Unable to Reliably Classify')
      - window_mismatch_detected: bool
      - ood_features: List[Dict[str, Any]] (features outside p01-p99 training range)
      - verification_note: str (scientific note regarding verified=1 in Cresci-15)
      - audit_warnings: List[str]
    """
    ood_features = []
    audit_warnings = []
    is_reliable = True
    recommendation_label = None

    # Cresci-15 training summed hashtags/mentions over hundreds of tweets (median 989 for humans).
    # When a modern API sample provides <= 50 tweets but the account has a status count > 50,
    # the cumulative counters (e.g. hashtag_count=0, mentions_count=0) are fundamentally window-incompatible
    # with the training distribution where genuine humans had hundreds of cumulative hashtags and mentions.
    window_mismatch_detected = False
    if tweets_analyzed_count > 0 and tweets_analyzed_count <= 50 and account_status_count > 50:
        window_mismatch_detected = True
        audit_warnings.append(
            f"Aggregation Window Mismatch: Model feature counts ('hashtag_count', 'mentions_count', etc.) "
            f"were extracted over only {tweets_analyzed_count} recent posts, whereas Cresci-15 training "
            f"summed cumulative counts across an account's entire tweet history. This structural mismatch "
            f"causes cumulative totals to artificially mirror inactive bot profiles."
        )

    for feat, val in feature_dict.items():
        if feat not in TRAINING_DISTRIBUTIONS:
            continue
        stats = TRAINING_DISTRIBUTIONS[feat]
        p01 = stats["p01"]
        p99 = stats["p99"]
        median = stats["median"]
        feat_max = stats["max"]
        feat_min = stats["min"]

        # Note: account_age_days naturally increases with real time and is monitored as HIGH rather than CRITICAL rejection
        if val > p99:
            severity = "CRITICAL" if (val > feat_max and feat != "account_age_days") else "HIGH"
            ood_features.append({
                "feature": feat,
                "live_value": round(val, 4),
                "training_median": round(median, 4),
                "training_p99": round(p99, 4),
                "training_max": round(feat_max, 4),
                "direction": "ABOVE_RANGE",
                "severity": severity
            })
        elif val < p01 and val < feat_min:
            ood_features.append({
                "feature": feat,
                "live_value": round(val, 4),
                "training_median": round(median, 4),
                "training_p01": round(p01, 4),
                "training_min": round(feat_min, 4),
                "direction": "BELOW_RANGE",
                "severity": "HIGH"
            })

    # In Cresci-15 genuine accounts (E13), avg_replies maxed out at 0.7858 (99th pct = 0.39)
    # and avg_favorites maxed out at 10.01 (99th pct = 1.15).
    # Modern active accounts frequently have avg_replies > 1.0 and avg_favorites > 15.0,
    # which in Cresci-15 only occurred in commercial interaction spambots ('INT' bots).
    avg_rep = feature_dict.get("avg_replies", 0.0)
    avg_fav = feature_dict.get("avg_favorites", 0.0)
    if avg_rep > 1.0 or avg_fav > 25.0:
        audit_warnings.append(
            f"Extreme Engagement Domain Shift: Live per-tweet engagement (avg_replies={avg_rep:.2f}, "
            f"avg_favorites={avg_fav:.2f}) exceeds the maximum range of genuine human accounts in Cresci-15 "
            f"(genuine E13 max replies=0.79, max favorites=10.01). In the 2015 dataset, engagement of this level "
            f"was exclusive to commercial interaction-boosting spambots."
        )

    verified_val = feature_dict.get("verified", 0.0)
    verification_note = ""
    if verified_val > 0.0:
        verification_note = (
            "Verification Context: Account is Verified on X (verified=1.0). "
            "However, the Cresci-15 training dataset contained 0 verified accounts across both genuine "
            "and bot classes. Consequently, the trained tree model has no split threshold for 'verified=1' "
            "and cannot use verification status to establish authenticity."
        )

    # If there is an aggregation window mismatch (sampled tweets vs lifetime status count),
    # or extreme OOD engagement (avg_replies > 1.0, avg_favorites > 25.0), or critical range violations,
    # reject definitive classification to prevent presenting misleading predictions.
    critical_ood = [o for o in ood_features if o.get("severity") == "CRITICAL"]
    if window_mismatch_detected or len(critical_ood) > 0 or avg_rep > 1.0 or avg_fav > 25.0:
        is_reliable = False
        recommendation_label = "Unable to Reliably Classify"
        audit_warnings.append(
            "Classification Guard: Account exhibits feature-window incompatibility or out-of-distribution "
            "metrics compared to the historical Cresci-15 training baseline. A definitive 'Likely Bot' or 'Likely Genuine' "
            "classification is rejected as scientifically unreliable on this model."
        )

    return {
        "is_reliable": is_reliable,
        "recommendation_label": recommendation_label,
        "window_mismatch_detected": window_mismatch_detected,
        "ood_features": ood_features,
        "verification_note": verification_note,
        "audit_warnings": audit_warnings,
        "tweets_analyzed": tweets_analyzed_count
    }
