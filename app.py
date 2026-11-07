import os
import sys
import csv
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
import shap
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from lime.lime_tabular import LimeTabularExplainer

from x_api_client import (
    sanitize_username,
    XApiClient,
    extract_features_from_x_data
)
from xfetch_client import (
    XFetchClient,
    extract_features_from_xfetch_data
)
from ood_validator import evaluate_feature_compatibility_and_ood
from twis20.live_twibot20 import construct_profile_text, load_live_artifacts


def _profile_summary_metrics(user: dict) -> dict:
    """Normalize documented X API and XFetch profile fields for display."""
    public_metrics = user.get("public_metrics")
    metrics = public_metrics if isinstance(public_metrics, dict) else {}

    def first_metric(*keys):
        for source in (metrics, user):
            for key in keys:
                value = source.get(key)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    return value
        return None

    account_age_days = None
    created_at = user.get("created_at")
    if isinstance(created_at, str) and created_at:
        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError:
            pass
        else:
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            account_age_days = round(
                max(
                    0.0,
                    (datetime.now(timezone.utc) - created.astimezone(timezone.utc)).total_seconds()
                    / 86400.0,
                ),
                1,
            )

    return {
        "account_age_days": account_age_days,
        "followers_count": first_metric("followers_count", "follower_count"),
        "friends_count": first_metric("following_count", "following", "friends_count"),
        "status_count": first_metric("tweet_count", "status_count", "statuses_count"),
    }


def _binary_uncalibrated_scores(model, features) -> dict[str, float]:
    """Return validated raw scores for the TwiBot-20 labels (0=genuine, 1=bot)."""
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ValueError("model is missing classes_ metadata")

    class_labels = np.asarray(classes).tolist()
    if (
        not isinstance(class_labels, list)
        or len(class_labels) != 2
        or any(
            isinstance(label, (bool, np.bool_))
            or not isinstance(label, (int, np.integer))
            for label in class_labels
        )
        or set(class_labels) != {0, 1}
    ):
        raise ValueError("model classes_ must contain the binary labels 0 and 1")

    outputs = np.asarray(model.predict_proba(features), dtype=np.float64)
    if outputs.shape != (1, 2):
        raise ValueError("predict_proba must return one score for each binary class")
    if (
        not np.isfinite(outputs).all()
        or np.any(outputs < 0)
        or np.any(outputs > 1)
        or not np.isclose(outputs[0].sum(), 1.0, rtol=0.0, atol=1e-6)
    ):
        raise ValueError("predict_proba outputs must be finite, in range, and sum to 1")

    class_indices = {label: index for index, label in enumerate(class_labels)}
    return {
        "Genuine": float(outputs[0, class_indices[0]]),
        "Bot": float(outputs[0, class_indices[1]]),
    }


if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "tuned_lightgbm_cresci15.joblib")
CRESCI15_TRAIN = os.path.join(BASE_DIR, "cresci15_train_split.csv")
CRESCI15_DATA = CRESCI15_TRAIN if os.path.exists(CRESCI15_TRAIN) else os.path.join(BASE_DIR, "cresci15_full_features.csv")

app = FastAPI(
    title="Interpretable Social Bot Detection & Live X Analyzer API",
    description="Explainable ML API & Web Dashboard for detecting spambots and fake followers with SHAP/LIME, featuring Live X (Twitter) Account Analysis",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
lime_explainer = None
shap_explainer = None
feature_cols = []
default_features = {}

@app.on_event("startup")
def load_artifacts():
    global model, lime_explainer, shap_explainer, feature_cols, default_features
    print("[*] Loading trained champion model...")
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("[✓] Model loaded successfully.")
    else:
        print(f"[!] Warning: Model file not found at {MODEL_PATH}")

    if os.path.exists(CRESCI15_DATA):
        df = pd.read_csv(CRESCI15_DATA)
        ignore_cols = ["user_id", "dataset", "label"]
        raw_cols = [c for c in df.columns if c not in ignore_cols]
        if model is not None and hasattr(model, "feature_name_"):
            feature_cols = list(model.feature_name_)
        else:
            feature_cols = sorted(raw_cols)
        X = df[feature_cols].copy()
        
        default_features = X.median().to_dict()

        lime_explainer = LimeTabularExplainer(
            training_data=np.array(X.values, dtype=float),
            feature_names=feature_cols,
            class_names=["Genuine Human", "Spambot / Fake"],
            mode="classification",
            random_state=42
        )
        print(f"[✓] LIME Explainer initialized with {len(feature_cols)} features.")

    if model is not None:
        try:
            shap_explainer = shap.TreeExplainer(model)
            print("[✓] SHAP TreeExplainer initialized successfully.")
        except Exception as e:
            print(f"[!] Warning: Could not initialize SHAP TreeExplainer: {e}")

class AccountInput(BaseModel):
    followers_count: float = 120
    friends_count: float = 85
    status_count: float = 1500
    favorites_count: float = 350
    listed_count: float = 5
    verified: int = 0
    default_profile: int = 0
    default_profile_image: int = 0
    geo_enabled: int = 1
    profile_background_tile: int = 0
    description_binary: int = 1
    profile_completeness: float = 0.83
    hashtag_count: float = 120
    mentions_count: float = 300
    retweet_count: float = 450
    reply_count: float = 80
    url_count: float = 60
    avg_hashtag: float = 0.25
    avg_retweets: float = 12.5
    avg_replies: float = 0.0
    avg_mentions: float = 0.65
    avg_url: float = 0.15
    avg_favorites: float = 0.5
    avg_user_engagement: float = 13.15
    unique_word_count: float = 420
    unique_word_use: float = 0.65
    punctuation_count: float = 350
    avg_sentence_length: float = 95.0
    punctuation_density: float = 0.07
    avg_polarity: float = 0.05
    avg_subjectivity: float = 0.10
    account_age_days: float = 1200.0
    tweets_per_day: float = 1.25

class XAnalyzeRequest(BaseModel):
    username: str = Field(..., description="X (Twitter) handle or profile URL (e.g. '@elonmusk' or 'https://x.com/username')")

@app.post("/twibot20/analyze")
async def analyze_twibot20(req: XAnalyzeRequest):
    """Score live X data with the benchmark-trained profile+text model."""
    def error_response(
        status_code: int,
        status: str,
        error_code: str,
        message: str,
        username: str | None = None,
        feature_audit: dict | None = None,
        manifest: dict | None = None,
    ):
        audit = feature_audit or {
            "expected_feature_count": 50072,
            "constructed_feature_count": None,
            "profile_feature_count": None,
            "tweet_summary_feature_count": None,
            "tfidf_feature_count": None,
            "missing_feature_count": None,
            "duplicate_or_invalid_feature_count": None,
            "feature_order_validated": False,
            "preprocessing_validated": False,
            "preprocessing_details": {
                "status": "not_run",
                "message": "Feature construction did not complete.",
            },
        }
        return JSONResponse(
            status_code=status_code,
            content={
                "status": status,
                "error_code": error_code,
                "message": message,
                "username": username,
                "prediction": None,
                "probability": None,
                "confidence": None,
                "score_calibrated": False,
                "score_interpretation": "No uncalibrated Bot-class model score was produced; expected range is [0, 1].",
                "threshold": (manifest or {}).get("threshold"),
                "model": (manifest or {}).get("model_name"),
                "model_score_uncalibrated": None,
                "class_scores_uncalibrated": {"Genuine": None, "Bot": None},
                "raw_model_prediction": None,
                "probabilities": {"Human": None, "Bot": None},
                "profile_summary": {},
                "feature_count_required": 50072,
                "feature_count_constructed": audit["constructed_feature_count"],
                "available_feature_groups": {},
                "missing_feature_groups": [],
                "feature_audit": audit,
                "network_features_used": False,
                "available_features": [],
                "missing_features": [],
                "explanation": [],
                "lime_explanations": [],
                "ood_validation": {
                    "audit_warnings": [],
                    "verification_note": None,
                },
            },
        )

    clean_user = sanitize_username(req.username)
    if not clean_user:
        return error_response(
            400,
            "error",
            "invalid_username",
            "Provide a valid X username or URL.",
        )

    xfetch = XFetchClient()
    xclient = XApiClient()
    user = {}
    tweets = []
    if xfetch.has_key():
        payload, _, error, status = xfetch.fetch_profile(clean_user)
        if error or not payload:
            return error_response(
                status,
                "error",
                "profile_fetch_failed",
                error or "Live X fetch failed.",
                clean_user,
            )
        user = payload.get("user", {})
        tweets = payload.get("recent_tweets", [])
    elif xclient.has_token():
        user, error, status = xclient.fetch_user_by_username(clean_user)
        if error or not user:
            return error_response(
                status,
                "error",
                "profile_fetch_failed",
                error or "Live X fetch failed.",
                clean_user,
            )
        tweets, _ = xclient.fetch_recent_tweets(user.get("id", ""), max_results=100)
    else:
        return error_response(
            401,
            "error",
            "missing_api_credentials",
            "No XFETCH_API_KEY or X_BEARER_TOKEN is configured.",
            clean_user,
        )

    manifest = None
    try:
        manifest, preprocessor, locked_model, names = load_live_artifacts()
        profile_text, available, missing, feature_audit = construct_profile_text(
            user, tweets, preprocessor, names
        )
    except Exception as exc:
        return error_response(
            500,
            "error",
            "artifact_validation_failed",
            f"TwiBot-20 artifact validation failed: {exc}",
            clean_user,
            manifest=manifest,
        )

    if (
        missing
        or profile_text.shape[1] != len(names)
        or not feature_audit["feature_order_validated"]
        or not feature_audit["preprocessing_validated"]
    ):
        return error_response(
            422,
            "insufficient_features",
            "feature_validation_failed",
            "The profile+text feature schema or preprocessing validation failed; no prediction was made.",
            clean_user,
            feature_audit,
            manifest,
        )

    try:
        class_scores = _binary_uncalibrated_scores(locked_model, profile_text)
    except Exception as exc:
        return error_response(
            500,
            "error",
            "invalid_model_scores",
            f"TwiBot-20 model scores were unavailable or invalid: {exc}",
            clean_user,
            feature_audit,
            manifest,
        )

    bot_score = class_scores["Bot"]
    threshold = float(manifest["threshold"])
    is_bot = bot_score >= threshold
    summary_metrics = _profile_summary_metrics(user)
    prediction = "Likely Bot" if is_bot else "Likely Genuine"

    return JSONResponse(
        content={
            "status": "success",
            "username": clean_user,
            "prediction": prediction,
            "probability": None,
            "confidence": None,
            "threshold": threshold,
            "model": manifest["model_name"],
            "model_score_uncalibrated": bot_score,
            "class_scores_uncalibrated": class_scores,
            "score_calibrated": False,
            "score_interpretation": (
                "Uncalibrated predict_proba outputs for the verified binary classes "
                "Genuine=0 and Bot=1; they sum to 1 and are not calibrated probabilities."
            ),
            "raw_model_prediction": "Bot" if is_bot else "Genuine",
            "probabilities": {
                "Human": None,
                "Bot": None,
            },
            "profile_summary": {
                "name": user.get("name", clean_user),
                "username": user.get("username", clean_user),
                "description": user.get("description", ""),
                "verified": user.get("verified") is True,
                "profile_image_url": user.get("profile_image_url"),
                "account_age_days": summary_metrics["account_age_days"],
                "tweets_analyzed": len(tweets),
                "followers_count": summary_metrics["followers_count"],
                "friends_count": summary_metrics["friends_count"],
                "status_count": summary_metrics["status_count"],
            },
            "feature_count_required": len(names),
            "feature_count_constructed": int(profile_text.shape[1]),
            "available_feature_groups": available,
            "missing_feature_groups": missing,
            "feature_audit": feature_audit,
            "network_features_used": False,
            # Legacy response properties are retained as empty arrays for old clients.
            # TwiBot-20 feature audit details are exposed only in feature_audit.
            "available_features": [],
            "missing_features": [],
            "explanation": [],
            "lime_explanations": [],
            "ood_validation": {
                "audit_warnings": [manifest["live_distribution_warning"]],
                "verification_note": (
                    "This is a profile+text model; it does not use benchmark "
                    "graph/network features."
                ),
            },
        }
    )


@app.get("/", response_class=HTMLResponse)
async def twibot20_dashboard():
    """Serve the standalone TwiBot-20 research dashboard."""
    dashboard_path = os.path.join(BASE_DIR, "twis20", "dashboard.html")
    with open(dashboard_path, encoding="utf-8") as dashboard_file:
        return HTMLResponse(content=dashboard_file.read())


@app.get("/twibot20/benchmark-explainability")
async def twibot20_benchmark_explainability():
    """Return read-only SHAP summaries for the separate locked benchmark model."""
    output_dir = os.path.join(
        BASE_DIR, "twis20", "phase5_explainability", "outputs_run03"
    )
    phase4_dir = os.path.join(BASE_DIR, "twis20", "phase4_results_run03")
    try:
        with open(
            os.path.join(output_dir, "validation.json"), encoding="utf-8"
        ) as stream:
            validation = json.load(stream)
        with open(
            os.path.join(phase4_dir, "locked_selection_before_test.json"),
            encoding="utf-8",
        ) as stream:
            selection_lock = json.load(stream)
        if not isinstance(validation, dict) or not isinstance(selection_lock, dict):
            raise ValueError("benchmark provenance files must contain JSON objects")
        before = validation.get("input_fingerprints_before")
        after = validation.get("input_fingerprints_after")
        selection_metadata = selection_lock.get("selection_metadata", {})
        selection = selection_lock.get("selection", {})
        shap_shape = validation.get("shap_shape")
        if (
            validation.get("model_inputs") != 50080
            or validation.get("feature_names") != 50080
            or not isinstance(shap_shape, list)
            or len(shap_shape) != 2
            or shap_shape[1] != 50080
            or validation.get("sample_rows") != 200
            or validation.get("shap_all_finite") is not True
            or validation.get("error_shap_all_finite") is not True
            or not np.isfinite(float(validation["max_additivity_residual"]))
            or float(validation["max_additivity_residual"]) >= 1e-4
            or not validation.get("phase4_fingerprint_match")
            or before != after
            or not isinstance(selection_metadata, dict)
            or not isinstance(selection, dict)
            or selection_metadata.get("selection_split") != "dev"
            or selection_metadata.get("test_metrics_used") is not False
            or selection.get("experiment") != "profile_text_network"
            or selection.get("algorithm") != "xgboost"
            or selection.get("setting") != "depth_8"
            or not np.isclose(
                float(validation["locked_threshold"]),
                float(selection["threshold"]),
            )
        ):
            raise ValueError("benchmark artifact provenance validation failed")
        with open(
            os.path.join(output_dir, "global_shap_importance.csv"),
            newline="",
            encoding="utf-8",
        ) as stream:
            global_features = list(csv.DictReader(stream))[:12]
        with open(
            os.path.join(output_dir, "feature_group_importance.csv"),
            newline="",
            encoding="utf-8",
        ) as stream:
            feature_groups = list(csv.DictReader(stream))
        with open(
            os.path.join(output_dir, "individual_explanations.csv"),
            newline="",
            encoding="utf-8",
        ) as stream:
            individual_rows = list(csv.DictReader(stream))
    except (OSError, TypeError, ValueError, KeyError, csv.Error) as exc:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error_code": "benchmark_explainability_unavailable",
                "message": f"Benchmark explanation artifacts failed validation: {exc}",
            },
        )

    examples = []
    global_feature_data = []
    feature_group_data = []
    try:
        for row in individual_rows:
            actual_label = int(row["actual_label"])
            predicted_class = int(row["predicted_class"])
            bot_score = float(row["probability_bot"])
            example_threshold = float(row["threshold"])
            toward_bot = json.loads(row["top_toward_bot"])
            toward_genuine = json.loads(row["top_toward_genuine"])
            if (
                actual_label not in (0, 1)
                or predicted_class not in (0, 1)
                or not np.isfinite(bot_score)
                or not 0 <= bot_score <= 1
                or not np.isclose(example_threshold, float(selection["threshold"]))
                or not isinstance(toward_bot, list)
                or not isinstance(toward_genuine, list)
                or not toward_bot
                or not toward_genuine
            ):
                raise ValueError("individual benchmark explanation failed validation")
            for contribution in toward_bot[:3] + toward_genuine[:3]:
                if (
                    not isinstance(contribution, dict)
                    or not isinstance(contribution.get("feature"), str)
                    or isinstance(contribution.get("shap"), bool)
                    or not isinstance(contribution.get("shap"), (int, float))
                    or not np.isfinite(contribution["shap"])
                ):
                    raise ValueError("individual SHAP contribution failed validation")
            examples.append(
                {
                    "sample_row": int(row["sample_row"]),
                    "actual_class": "Bot" if actual_label == 1 else "Genuine",
                    "predicted_class": "Bot" if predicted_class == 1 else "Genuine",
                    "bot_class_score_uncalibrated": bot_score,
                    "threshold": example_threshold,
                    "toward_bot": toward_bot[:3],
                    "toward_genuine": toward_genuine[:3],
                }
            )
        for row in global_features:
            value = float(row["mean_abs_shap"])
            if not np.isfinite(value) or value < 0:
                raise ValueError("global SHAP importance failed validation")
            global_feature_data.append(
                {
                    "feature": row["feature"],
                    "mean_abs_shap": value,
                    "feature_group": row["feature_group"],
                }
            )
        for row in feature_groups:
            value = float(row["mean_abs_shap"])
            if not np.isfinite(value) or value < 0:
                raise ValueError("SHAP group importance failed validation")
            feature_group_data.append(
                {
                    "feature_group": row["feature_group"],
                    "mean_abs_shap": value,
                }
            )
        if not examples or not global_feature_data or not feature_group_data:
            raise ValueError("benchmark explanation artifacts are empty")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error_code": "benchmark_explainability_unavailable",
                "message": f"Benchmark explanation artifacts failed validation: {exc}",
            },
        )

    return {
        "status": "success",
        "model": (
            f"TwiBot-20 {selection.get('experiment', 'benchmark')} "
            f"{selection.get('algorithm', 'model')} {selection.get('setting', '')}"
        ).strip(),
        "feature_count": validation.get("feature_names"),
        "threshold": selection.get("threshold"),
        "explanation_split": "development",
        "explanation_sample_size": validation.get("sample_rows"),
        "shap_output": validation.get("shap_output"),
        "max_additivity_residual": validation.get("max_additivity_residual"),
        "global_features": global_feature_data,
        "feature_groups": feature_group_data,
        "individual_examples": examples,
        "causality_note": (
            "SHAP describes this model's feature contributions on benchmark "
            "examples; it does not establish causal effects."
        ),
    }


@app.get("/cresci15", response_class=HTMLResponse)
async def web_dashboard():
    """Serves an interactive Web UI for demonstrating predictions, XAI explanations, and Live X Analysis"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Interpretable Social Bot Detection Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
        <style>
            body { background-color: #f4f6f9; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            .header-banner { background: linear-gradient(135deg, #0f172a 0%, #1e3c72 100%); color: white; padding: 25px 0; margin-bottom: 25px; border-radius: 0 0 15px 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
            .card { border-radius: 12px; border: none; box-shadow: 0 4px 12px rgba(0,0,0,0.06); margin-bottom: 25px; }
            .nav-pills .nav-link { font-weight: 600; border-radius: 8px; color: #495057; padding: 10px 20px; }
            .nav-pills .nav-link.active { background-color: #1e3c72; color: #fff; }
            .badge-bot { background-color: #dc3545; font-size: 1.1rem; padding: 8px 15px; }
            .badge-human { background-color: #28a745; font-size: 1.1rem; padding: 8px 15px; }
            .badge-unable { background-color: #6c757d; font-size: 1.1rem; padding: 8px 15px; }
            .profile-avatar { width: 72px; height: 72px; border-radius: 50%; object-fit: cover; border: 3px solid #e2e8f0; }
            .disclaimer-box { background-color: #fff8e1; border-left: 4px solid #ffb300; padding: 12px 16px; font-size: 0.85rem; border-radius: 4px; }
            .feature-pill-avail { background-color: #e6f4ea; color: #137333; font-size: 0.75rem; padding: 3px 8px; border-radius: 12px; margin: 2px; display: inline-block; }
            .feature-pill-miss { background-color: #fce8e6; color: #c5221f; font-size: 0.75rem; padding: 3px 8px; border-radius: 12px; margin: 2px; display: inline-block; }
        </style>
    </head>
    <body>
        <div class="header-banner text-center">
            <div class="container">
                <div class="d-flex justify-content-center align-items-center gap-2 mb-1">
                    <h2 class="fw-bold mb-0">Interpretable AI Social Bot Detection</h2>
                </div>
                <p class="mb-0 text-white-50">Trained Champion LightGBM Model &bull; SHAP &amp; LIME Dual-Tier XAI &bull; Live X (Twitter) API v2 Integration</p>
            </div>
        </div>

        <div class="container">
            <ul class="nav nav-pills justify-content-center mb-4" id="pills-tab" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="tab-live-x" data-bs-toggle="pill" data-bs-target="#content-live-x" type="button" role="tab">
                        <i class="bi bi-twitter-x me-1"></i> Live X (Twitter) Account Analyzer
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="tab-manual" data-bs-toggle="pill" data-bs-target="#content-manual" type="button" role="tab">
                        <i class="bi bi-sliders me-1"></i> Manual Feature Simulator
                    </button>
                </li>
            </ul>

            <div class="tab-content" id="pills-tabContent">
                
                <div class="tab-pane fade show active" id="content-live-x" role="tabpanel">
                    <div class="row">
                        <div class="col-lg-5">
                            <div class="card p-4">
                                <div class="d-flex align-items-center mb-3">
                                    <svg class="me-2" width="28" height="28" viewBox="0 0 24 24" fill="#0f1419"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
                                    <h5 class="fw-bold text-secondary mb-0">Live X Profile Lookup</h5>
                                </div>
                                <p class="small text-muted mb-3">Enter an X handle or profile URL to inspect live account attributes via official X API v2 and evaluate bot probability.</p>

                                <div class="mb-3">
                                    <label class="form-label small fw-semibold">X Username or URL</label>
                                    <div class="input-group">
                                        <span class="input-group-text bg-light text-muted">@</span>
                                        <input type="text" class="form-control" id="xUsernameInput" placeholder="elonmusk or https://x.com/username">
                                    </div>
                                    <div class="form-text small">Accepted: <code>@username</code>, <code>username</code>, <code>https://x.com/user</code>, <code>https://twitter.com/user</code></div>
                                </div>

                                <div class="mb-3">
                                    <span class="small fw-semibold text-muted d-block mb-1">Quick Sample Handles:</span>
                                    <button class="btn btn-sm btn-outline-secondary me-1 mb-1" onclick="setHandle('NASA')">@NASA</button>
                                    <button class="btn btn-sm btn-outline-secondary me-1 mb-1" onclick="setHandle('OpenAI')">@OpenAI</button>
                                    <button class="btn btn-sm btn-outline-secondary me-1 mb-1" onclick="setHandle('ylecun')">@ylecun</button>
                                </div>

                                <button type="button" class="btn btn-primary w-100 py-2 fw-semibold" id="btnAnalyzeX" onclick="analyzeXAccount()">
                                    <i class="bi bi-search me-1"></i> Analyze X Account
                                </button>

                                <div class="disclaimer-box mt-4">
                                    <strong><i class="bi bi-info-circle me-1"></i> Scientific Transparency Note:</strong>
                                    Predictions are probabilistic outputs ("Likely Bot" / "Likely Genuine") inferred by a model trained on benchmark Cresci datasets. Real-world X accounts may reflect contemporary domain shift.
                                </div>
                            </div>
                        </div>

                        <div class="col-lg-7">
                            <div class="card p-4">
                                <h5 class="fw-bold text-secondary mb-3">Live Analysis Results</h5>
                                <div id="xResultContainer" class="text-center py-5 text-muted">
                                    <i class="bi bi-twitter-x fs-1 text-black-50 mb-2 d-block"></i>
                                    <p class="mb-0">Enter an X handle and click <strong>"Analyze X Account"</strong> to execute live API lookup, canonical feature mapping, and dual SHAP/LIME explanation.</p>
                                </div>
                            </div>

                            <div class="card p-4" id="xShapCard" style="display: none;">
                                <div class="d-flex justify-content-between align-items-center mb-2">
                                    <h5 class="fw-bold text-secondary mb-0">SHAP Feature Attributions</h5>
                                    <span class="badge bg-light text-dark border">TreeExplainer</span>
                                </div>
                                <p class="small text-muted mb-3">Top feature impacts on model log-odds towards Bot (Red/+) vs Genuine (Green/-):</p>
                                <div id="xShapList"></div>
                            </div>

                            <div class="card p-4" id="xLimeCard" style="display: none;">
                                <div class="d-flex justify-content-between align-items-center mb-2">
                                    <h5 class="fw-bold text-secondary mb-0">LIME Local Surrogate Rules</h5>
                                    <span class="badge bg-light text-dark border">LimeTabularExplainer</span>
                                </div>
                                <div id="xLimeList"></div>
                            </div>

                            <div class="card p-4" id="xFeatureAuditCard" style="display: none;">
                                <h5 class="fw-bold text-secondary mb-2">Feature Audit &amp; Imputation Transparency</h5>
                                <p class="small text-muted mb-2">Detailed mapping across the canonical 34 features:</p>
                                <div class="mb-2">
                                    <strong class="small text-success d-block mb-1">Directly Extracted via API (<span id="availCount">0</span>):</strong>
                                    <div id="availPills"></div>
                                </div>
                                <div>
                                    <strong class="small text-danger d-block mb-1">Imputed with Baseline Training Medians (<span id="missCount">0</span>):</strong>
                                    <div id="missPills"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="tab-pane fade" id="content-manual" role="tabpanel">
                    <div class="row">
                        <div class="col-lg-6">
                            <div class="card p-4">
                                <div class="d-flex justify-content-between align-items-center mb-3">
                                    <h5 class="fw-bold text-secondary mb-0">Manual Account Attributes</h5>
                                    <div>
                                        <button type="button" class="btn btn-sm btn-outline-success me-1" onclick="loadPreset('human')">Preset: Human</button>
                                        <button type="button" class="btn btn-sm btn-outline-danger" onclick="loadPreset('bot')">Preset: Bot</button>
                                    </div>
                                </div>

                                <form id="botForm">
                                    <div class="row g-2">
                                        <div class="col-md-6">
                                            <label class="form-label small fw-semibold">Followers Count</label>
                                            <input type="number" class="form-control form-control-sm" id="followers_count" value="1200">
                                        </div>
                                        <div class="col-md-6">
                                            <label class="form-label small fw-semibold">Friends Count (Following)</label>
                                            <input type="number" class="form-control form-control-sm" id="friends_count" value="450">
                                        </div>
                                        <div class="col-md-6">
                                            <label class="form-label small fw-semibold">Statuses Count (Tweets)</label>
                                            <input type="number" class="form-control form-control-sm" id="status_count" value="3500">
                                        </div>
                                        <div class="col-md-6">
                                            <label class="form-label small fw-semibold">Favorites Count (Likes given)</label>
                                            <input type="number" class="form-control form-control-sm" id="favorites_count" value="1200">
                                        </div>
                                        <div class="col-md-6">
                                            <label class="form-label small fw-semibold">Listed Count</label>
                                            <input type="number" class="form-control form-control-sm" id="listed_count" value="15">
                                        </div>
                                        <div class="col-md-6">
                                            <label class="form-label small fw-semibold">Avg. Replies per Tweet</label>
                                            <input type="number" step="0.01" class="form-control form-control-sm" id="avg_replies" value="0.25">
                                        </div>
                                        <div class="col-md-6">
                                            <label class="form-label small fw-semibold">Avg. Mentions per Tweet</label>
                                            <input type="number" step="0.01" class="form-control form-control-sm" id="avg_mentions" value="0.65">
                                        </div>
                                        <div class="col-md-6">
                                            <label class="form-label small fw-semibold">Account Age (Days)</label>
                                            <input type="number" class="form-control form-control-sm" id="account_age_days" value="1450">
                                        </div>
                                        <div class="col-md-6">
                                            <label class="form-label small fw-semibold">Has Bio/Description</label>
                                            <select class="form-select form-select-sm" id="description_binary">
                                                <option value="1" selected>Yes (1)</option>
                                                <option value="0">No (0)</option>
                                            </select>
                                        </div>
                                        <div class="col-md-6">
                                            <label class="form-label small fw-semibold">Default Avatar Image</label>
                                            <select class="form-select form-select-sm" id="default_profile_image">
                                                <option value="0" selected>No (Custom Avatar)</option>
                                                <option value="1">Yes (Default Egg/Placeholder)</option>
                                            </select>
                                        </div>
                                    </div>

                                    <button type="button" class="btn btn-primary w-100 mt-4 py-2 fw-semibold" onclick="analyzeAccountManual()">
                                        Run Detection &amp; LIME Explanation
                                    </button>
                                </form>
                            </div>
                        </div>

                        <div class="col-lg-6">
                            <div class="card p-4">
                                <h5 class="fw-bold text-secondary mb-3">Model Prediction &amp; Confidence</h5>
                                <div id="manualResultContainer" class="text-center py-4 text-muted">
                                    <p class="mb-0">Click <strong>"Run Detection &amp; LIME Explanation"</strong> to generate prediction.</p>
                                </div>
                            </div>

                            <div class="card p-4" id="manualLimeCard" style="display: none;">
                                <h5 class="fw-bold text-secondary mb-2">Local LIME Feature Contributions</h5>
                                <p class="small text-muted mb-3">Factors driving the prediction towards Bot (Red) vs. Human (Green):</p>
                                <div id="manualLimeList"></div>
                            </div>
                        </div>
                    </div>
                </div>

            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            function setHandle(h) {
                document.getElementById('xUsernameInput').value = h;
            }

            function loadPreset(type) {
                if (type === 'human') {
                    document.getElementById('followers_count').value = 1450;
                    document.getElementById('friends_count').value = 420;
                    document.getElementById('status_count').value = 4500;
                    document.getElementById('favorites_count').value = 1800;
                    document.getElementById('listed_count').value = 25;
                    document.getElementById('avg_replies').value = 0.0;
                    document.getElementById('avg_mentions').value = 0.75;
                    document.getElementById('account_age_days').value = 1800;
                    document.getElementById('description_binary').value = 1;
                    document.getElementById('default_profile_image').value = 0;
                } else {
                    document.getElementById('followers_count').value = 12;
                    document.getElementById('friends_count').value = 450;
                    document.getElementById('status_count').value = 15;
                    document.getElementById('favorites_count').value = 0;
                    document.getElementById('listed_count').value = 0;
                    document.getElementById('avg_replies').value = 0.0;
                    document.getElementById('avg_mentions').value = 0.05;
                    document.getElementById('account_age_days').value = 45;
                    document.getElementById('description_binary').value = 0;
                    document.getElementById('default_profile_image').value = 1;
                }
            }

            async function analyzeAccountManual() {
                const payload = {
                    followers_count: parseFloat(document.getElementById('followers_count').value),
                    friends_count: parseFloat(document.getElementById('friends_count').value),
                    status_count: parseFloat(document.getElementById('status_count').value),
                    favorites_count: parseFloat(document.getElementById('favorites_count').value),
                    listed_count: parseFloat(document.getElementById('listed_count').value),
                    avg_replies: parseFloat(document.getElementById('avg_replies').value),
                    avg_mentions: parseFloat(document.getElementById('avg_mentions').value),
                    account_age_days: parseFloat(document.getElementById('account_age_days').value),
                    description_binary: parseInt(document.getElementById('description_binary').value),
                    default_profile_image: parseInt(document.getElementById('default_profile_image').value)
                };

                const resDiv = document.getElementById('manualResultContainer');
                resDiv.innerHTML = '<div class="spinner-border text-primary" role="status"></div><p class="mt-2 text-muted">Computing model inference and generating local surrogate explanations...</p>';

                try {
                    const response = await fetch('/predict_explain', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const data = await response.json();

                    const isBot = data.prediction.includes('Spambot') || data.prediction.includes('Bot');
                    const badgeClass = isBot ? 'badge-bot' : 'badge-human';
                    const conf = (data.confidence * 100).toFixed(1);

                    resDiv.innerHTML = `
                        <div class="mb-3">
                            <span class="badge ${badgeClass}">${data.prediction}</span>
                        </div>
                        <h4 class="fw-bold mb-1">${conf}% Confidence</h4>
                        <p class="text-muted small">Assigned by Champion Tuned LightGBM Classifier</p>
                        <div class="progress mt-3" style="height: 12px;">
                            <div class="progress-bar bg-success" role="progressbar" style="width: ${(data.probabilities.Human * 100).toFixed(1)}%"></div>
                            <div class="progress-bar bg-danger" role="progressbar" style="width: ${(data.probabilities.Bot * 100).toFixed(1)}%"></div>
                        </div>
                        <div class="d-flex justify-content-between small text-muted mt-1">
                            <span>Human: ${(data.probabilities.Human * 100).toFixed(1)}%</span>
                            <span>Bot: ${(data.probabilities.Bot * 100).toFixed(1)}%</span>
                        </div>
                    `;

                    const limeCard = document.getElementById('manualLimeCard');
                    const limeList = document.getElementById('manualLimeList');
                    limeCard.style.display = 'block';
                    limeList.innerHTML = '';

                    data.lime_explanations.forEach(item => {
                        const isToBot = item.direction.includes('BOT');
                        const colorClass = isToBot ? 'text-danger fw-bold' : 'text-success fw-bold';
                        const barColor = isToBot ? 'bg-danger' : 'bg-success';
                        const widthPct = Math.min(100, Math.abs(item.weight) * 250);

                        limeList.innerHTML += `
                            <div class="mb-2">
                                <div class="d-flex justify-content-between small">
                                    <span><code>${item.feature_rule}</code></span>
                                    <span class="${colorClass}">${item.direction} (${item.weight > 0 ? '+' : ''}${item.weight.toFixed(4)})</span>
                                </div>
                                <div class="progress" style="height: 6px;">
                                    <div class="progress-bar ${barColor}" style="width: ${widthPct}%"></div>
                                </div>
                            </div>
                        `;
                    });
                } catch (err) {
                    resDiv.innerHTML = '<div class="alert alert-danger">Error communicating with server: ' + err.message + '</div>';
                }
            }

            async function analyzeXAccount() {
                const username = document.getElementById('xUsernameInput').value.trim();
                const resDiv = document.getElementById('xResultContainer');
                const shapCard = document.getElementById('xShapCard');
                const limeCard = document.getElementById('xLimeCard');
                const auditCard = document.getElementById('xFeatureAuditCard');

                if (!username) {
                    resDiv.innerHTML = '<div class="alert alert-warning">Please enter an X username or profile URL.</div>';
                    return;
                }

                shapCard.style.display = 'none';
                limeCard.style.display = 'none';
                auditCard.style.display = 'none';
                resDiv.innerHTML = `
                    <div class="spinner-border text-primary my-3" role="status"></div>
                    <h6 class="fw-semibold">Querying Official X API v2 &amp; Computing Explanations...</h6>
                    <p class="small text-muted">Retrieving profile metrics &bull; Computing canonical features &bull; Calculating SHAP &amp; LIME</p>
                `;

                try {
                    const response = await fetch('/analyze_x', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ username: username })
                    });
                    const data = await response.json();

                    if (!response.ok || data.status === 'error') {
                        resDiv.innerHTML = `
                            <div class="alert alert-danger text-start">
                                <h6 class="fw-bold mb-1"><i class="bi bi-exclamation-triangle-fill me-1"></i> Analysis Failed</h6>
                                <p class="mb-0">${data.message || data.error || 'Failed to analyze X account.'}</p>
                            </div>
                        `;
                        return;
                    }

                    const prof = data.profile_summary || {};
                    const isBot = data.prediction === 'Likely Bot';
                    const isUnable = data.prediction === 'Unable to Analyze';
                    const badgeClass = isUnable ? 'badge-unable' : (isBot ? 'badge-bot' : 'badge-human');
                    const conf = (data.confidence * 100).toFixed(1);
                    const avatarUrl = prof.profile_image_url || 'https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png';

                    const creditsHtml = data.credits ? `
                        <div class="mt-2 text-muted small">
                            <span class="badge bg-light text-secondary border">xfetch Credits: ${data.credits.remaining || 'N/A'} remaining (${data.credits.charged || 0} charged)</span>
                        </div>
                    ` : '';

                    resDiv.innerHTML = `
                        <div class="d-flex align-items-center mb-3 text-start bg-light p-3 rounded">
                            <img src="${avatarUrl}" class="profile-avatar me-3" onerror="this.src='https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png'">
                            <div class="flex-grow-1">
                                <h5 class="fw-bold mb-0">${prof.name || data.username} ${prof.verified ? '<i class="bi bi-patch-check-fill text-primary" title="Verified"></i>' : ''}</h5>
                                <div class="text-muted small">@${prof.username || data.username} &bull; Account Age: ${prof.account_age_days ? Math.round(prof.account_age_days) + ' days' : 'N/A'} &bull; Tweets Analyzed: ${prof.tweets_analyzed || 0}</div>
                                <p class="small text-secondary mb-0 mt-1">${prof.description || '<em>No bio provided</em>'}</p>
                            </div>
                        </div>

                        <div class="row text-center mb-3 g-2">
                            <div class="col-4">
                                <div class="p-2 border rounded">
                                    <div class="fw-bold fs-6">${Number(prof.followers_count || 0).toLocaleString()}</div>
                                    <div class="text-muted small" style="font-size:0.75rem;">Followers</div>
                                </div>
                            </div>
                            <div class="col-4">
                                <div class="p-2 border rounded">
                                    <div class="fw-bold fs-6">${Number(prof.friends_count || 0).toLocaleString()}</div>
                                    <div class="text-muted small" style="font-size:0.75rem;">Following</div>
                                </div>
                            </div>
                            <div class="col-4">
                                <div class="p-2 border rounded">
                                    <div class="fw-bold fs-6">${Number(prof.status_count || 0).toLocaleString()}</div>
                                    <div class="text-muted small" style="font-size:0.75rem;">Statuses</div>
                                </div>
                            </div>
                        </div>

                        <div class="my-3">
                            <span class="badge ${badgeClass}">${data.prediction}</span>
                        </div>
                        <h4 class="fw-bold mb-1">${(data.model_score_uncalibrated * 100).toFixed(1)}% Model Score</h4>
                        <p class="text-muted small mb-1">Raw Uncalibrated Tree Score (${data.raw_model_prediction})</p>
                        ${creditsHtml}

                        ${data.ood_validation && data.ood_validation.audit_warnings.length > 0 ? `
                            <div class="alert alert-warning text-start mt-3 mb-2 p-2 small">
                                <strong><i class="bi bi-exclamation-triangle me-1"></i> Out-Of-Distribution (OOD) Audit Warnings:</strong>
                                <ul class="mb-0 ps-3 mt-1">
                                    ${data.ood_validation.audit_warnings.map(w => `<li>${w}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}

                        ${data.ood_validation && data.ood_validation.verification_note ? `
                            <div class="alert alert-info text-start mt-2 mb-2 p-2 small">
                                <i class="bi bi-patch-check me-1"></i> ${data.ood_validation.verification_note}
                            </div>
                        ` : ''}

                        <div class="progress mt-3" style="height: 12px;">
                            <div class="progress-bar bg-success" role="progressbar" style="width: ${(data.probabilities.Human * 100).toFixed(1)}%"></div>
                            <div class="progress-bar bg-danger" role="progressbar" style="width: ${(data.probabilities.Bot * 100).toFixed(1)}%"></div>
                        </div>
                        <div class="d-flex justify-content-between small text-muted mt-1">
                            <span>Likely Genuine: ${(data.probabilities.Human * 100).toFixed(1)}%</span>
                            <span>Likely Bot: ${(data.probabilities.Bot * 100).toFixed(1)}%</span>
                        </div>
                    `;

                    if (data.explanation && data.explanation.length > 0) {
                        shapCard.style.display = 'block';
                        const shapList = document.getElementById('xShapList');
                        shapList.innerHTML = '';
                        data.explanation.forEach(item => {
                            const isBotPush = item.direction.includes('BOT');
                            const colorClass = isBotPush ? 'text-danger fw-bold' : 'text-success fw-bold';
                            const barColor = isBotPush ? 'bg-danger' : 'bg-success';
                            const widthPct = Math.min(100, Math.abs(item.attribution) * 20);

                            shapList.innerHTML += `
                                <div class="mb-2">
                                    <div class="d-flex justify-content-between small">
                                        <span><strong>${item.feature}</strong> = <code>${item.value}</code></span>
                                        <span class="${colorClass}">${item.direction} (${item.attribution > 0 ? '+' : ''}${item.attribution.toFixed(4)})</span>
                                    </div>
                                    <div class="progress" style="height: 6px;">
                                        <div class="progress-bar ${barColor}" style="width: ${widthPct}%"></div>
                                    </div>
                                </div>
                            `;
                        });
                    }

                    if (data.lime_explanations && data.lime_explanations.length > 0) {
                        limeCard.style.display = 'block';
                        const limeList = document.getElementById('xLimeList');
                        limeList.innerHTML = '';
                        data.lime_explanations.forEach(item => {
                            const isToBot = item.direction.includes('BOT');
                            const colorClass = isToBot ? 'text-danger fw-bold' : 'text-success fw-bold';
                            const barColor = isToBot ? 'bg-danger' : 'bg-success';
                            const widthPct = Math.min(100, Math.abs(item.weight) * 250);

                            limeList.innerHTML += `
                                <div class="mb-2">
                                    <div class="d-flex justify-content-between small">
                                        <span><code>${item.feature_rule}</code></span>
                                        <span class="${colorClass}">${item.direction} (${item.weight > 0 ? '+' : ''}${item.weight.toFixed(4)})</span>
                                    </div>
                                    <div class="progress" style="height: 6px;">
                                        <div class="progress-bar ${barColor}" style="width: ${widthPct}%"></div>
                                    </div>
                                </div>
                            `;
                        });
                    }

                    auditCard.style.display = 'block';
                    document.getElementById('availCount').innerText = data.available_features.length;
                    document.getElementById('missCount').innerText = data.missing_features.length;
                    document.getElementById('availPills').innerHTML = data.available_features.map(f => `<span class="feature-pill-avail">${f}</span>`).join(' ');
                    document.getElementById('missPills').innerHTML = data.missing_features.map(f => `<span class="feature-pill-miss">${f}</span>`).join(' ');

                } catch (err) {
                    resDiv.innerHTML = '<div class="alert alert-danger">Error contacting analysis service: ' + err.message + '</div>';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/predict_explain")
async def predict_and_explain(account: AccountInput):
    """Predicts bot classification and computes real-time LIME feature attribution"""
    if model is None or lime_explainer is None:
        return JSONResponse(status_code=500, content={"error": "Model or explainer artifacts not loaded."})

    input_dict = default_features.copy()
    user_inputs = account.model_dump() if hasattr(account, "model_dump") else account.dict()
    for k, v in user_inputs.items():
        if k in input_dict:
            input_dict[k] = v

    fol = input_dict.get("followers_count", 0)
    fri = input_dict.get("friends_count", 0)
    input_dict["ffratio"] = fol / (fol + fri + 1e-5)
    age = max(1.0, input_dict.get("account_age_days", 365))
    input_dict["tweets_per_day"] = input_dict.get("status_count", 0) / age

    row_vec = np.array([input_dict[col] for col in feature_cols], dtype=float)
    row_df = pd.DataFrame([row_vec], columns=feature_cols)

    # Predict with DataFrame to preserve exact feature name alignment
    probs = model.predict_proba(row_df)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = "Spambot / Fake" if pred_idx == 1 else "Genuine Human"

    def predict_wrapper(arr):
        df_arr = pd.DataFrame(arr, columns=feature_cols)
        return model.predict_proba(df_arr)

    exp = lime_explainer.explain_instance(
        data_row=row_vec,
        predict_fn=predict_wrapper,
        num_features=6
    )

    lime_factors = []
    for rule, weight in exp.as_list():
        direction = "Towards BOT" if weight > 0 else "Towards HUMAN"
        lime_factors.append({
            "feature_rule": rule,
            "weight": round(weight, 4),
            "direction": direction
        })

    return {
        "prediction": pred_label,
        "confidence": float(probs[pred_idx]),
        "probabilities": {
            "Human": round(float(probs[0]), 4),
            "Bot": round(float(probs[1]), 4)
        },
        "lime_explanations": lime_factors
    }


@app.post("/analyze_x")
async def analyze_x(req: XAnalyzeRequest):
    """
    Live X (Twitter) Account Bot Analysis:
    1. Validates and sanitizes username or URL.
    2. Queries live data provider:
       - Uses xfetch.io (/v1/profiles/by-username/{username}) when XFETCH_API_KEY is configured.
       - Falls back to official X API v2 when X_BEARER_TOKEN is configured.
    3. Maps attributes + recent tweets onto canonical 34 features expected by champion LightGBM model.
    4. Computes probabilistic prediction ('Likely Bot' / 'Likely Genuine').
    5. Generates SHAP TreeExplainer attributions and LIME local surrogate rules.
    6. Returns complete transparent audit payload with credits info if available.
    """
    if model is None:
        return JSONResponse(status_code=500, content={"status": "error", "message": "Model artifacts not loaded on server."})

    raw_input = req.username
    clean_user = sanitize_username(raw_input)
    if not clean_user:
        return JSONResponse(status_code=400, content={
            "status": "error",
            "message": "Username cannot be empty. Please provide a valid X username or profile URL."
        })

    xfetch_client = XFetchClient()
    x_client = XApiClient()

    user_data = None
    tweets_data = []
    credits_info = None

    if xfetch_client.has_key():
        # Primary: Use xfetch API (fetches profile + recent tweets in 1 call)
        xfetch_res, _, err_msg, status_code = xfetch_client.fetch_profile(clean_user)
        if err_msg or not xfetch_res:
            return JSONResponse(status_code=status_code, content={
                "status": "error",
                "platform": "X (via xfetch)",
                "username": clean_user,
                "message": err_msg or "Failed to retrieve account details from xfetch API."
            })
        user_data = xfetch_res.get("user", {})
        tweets_data = xfetch_res.get("recent_tweets", [])
        credits_info = xfetch_res.get("meta", {}).get("credits", None)

        extracted_features, available_feats, missing_feats = extract_features_from_xfetch_data(
            payload=xfetch_res,
            default_medians=default_features
        )
    elif x_client.has_token():
        # Secondary: Official X API v2
        user_data, err_msg, status_code = x_client.fetch_user_by_username(clean_user)
        if err_msg or not user_data:
            return JSONResponse(status_code=status_code, content={
                "status": "error",
                "platform": "X",
                "username": clean_user,
                "message": err_msg or "Failed to retrieve account details from X."
            })
        user_id = user_data.get("id", "")
        tweets_data, _ = x_client.fetch_recent_tweets(user_id, max_results=10)

        extracted_features, available_feats, missing_feats = extract_features_from_x_data(
            user_data=user_data,
            tweets_data=tweets_data,
            default_medians=default_features
        )
    else:
        return JSONResponse(status_code=401, content={
            "status": "error",
            "platform": "X",
            "username": clean_user,
            "message": "No API credentials configured. Set XFETCH_API_KEY (recommended) or X_BEARER_TOKEN in .env"
        })

    if "followers_count" not in available_feats or "friends_count" not in available_feats:
        return JSONResponse(status_code=422, content={
            "platform": "X",
            "username": clean_user,
            "prediction": "Unable to Reliably Classify",
            "confidence": 0.0,
            "status": "error",
            "message": "Insufficient profile attributes available to reliably evaluate account."
        })

    tweets_analyzed_count = len(tweets_data)
    account_status_count = float(extracted_features.get("status_count", 0))
    ood_report = evaluate_feature_compatibility_and_ood(
        feature_dict=extracted_features,
        available_features=available_feats,
        missing_features=missing_feats,
        tweets_analyzed_count=tweets_analyzed_count,
        account_status_count=account_status_count
    )

    # Construct input vector in exact model canonical order (original uncalibrated feature values strictly preserved)
    row_vec = np.array([extracted_features.get(col, 0.0) for col in feature_cols], dtype=float)
    row_df = pd.DataFrame([row_vec], columns=feature_cols)

    probs = model.predict_proba(row_df)[0]
    p_human = round(float(probs[0]), 4)
    p_bot = round(float(probs[1]), 4)
    raw_pred_label = "Likely Bot" if p_bot >= 0.50 else "Likely Genuine"
    raw_model_score = p_bot if raw_pred_label == "Likely Bot" else p_human

    # Determine final prediction label using OOD guard
    if not ood_report["is_reliable"]:
        pred_label = ood_report["recommendation_label"] or "Unable to Reliably Classify"
    else:
        pred_label = raw_pred_label

    # Compute SHAP explanation (corresponding to exact feature vector)
    shap_explanations = []
    if shap_explainer is not None:
        try:
            sv = shap_explainer.shap_values(row_df)
            if isinstance(sv, list) and len(sv) == 2:
                bot_shap = sv[1][0]
            elif isinstance(sv, np.ndarray):
                bot_shap = sv[0]
            else:
                bot_shap = np.zeros(len(feature_cols))

            top_indices = np.argsort(np.abs(bot_shap))[::-1][:8]
            for idx in top_indices:
                feat_name = feature_cols[idx]
                val = extracted_features.get(feat_name, 0.0)
                attrib = float(bot_shap[idx])
                direction = "Towards BOT" if attrib > 0 else "Towards GENUINE"
                shap_explanations.append({
                    "feature": feat_name,
                    "value": round(val, 4) if isinstance(val, (int, float)) else val,
                    "attribution": round(attrib, 4),
                    "direction": direction
                })
        except Exception as e:
            print(f"[!] SHAP calculation error: {e}")

    lime_explanations = []
    if lime_explainer is not None:
        try:
            def predict_wrapper(arr):
                df_arr = pd.DataFrame(arr, columns=feature_cols)
                return model.predict_proba(df_arr)

            exp = lime_explainer.explain_instance(
                data_row=row_vec,
                predict_fn=predict_wrapper,
                num_features=6
            )
            for rule, weight in exp.as_list():
                direction = "Towards BOT" if weight > 0 else "Towards GENUINE"
                lime_explanations.append({
                    "feature_rule": rule,
                    "weight": round(weight, 4),
                    "direction": direction
                })
        except Exception as e:
            print(f"[!] LIME calculation error: {e}")

    profile_summary = {
        "id": user_data.get("id"),
        "name": user_data.get("name"),
        "username": user_data.get("username"),
        "description": user_data.get("description", ""),
        "created_at": user_data.get("created_at"),
        "account_age_days": extracted_features.get("account_age_days"),
        "followers_count": extracted_features.get("followers_count"),
        "friends_count": extracted_features.get("friends_count"),
        "status_count": extracted_features.get("status_count"),
        "listed_count": extracted_features.get("listed_count"),
        "verified": bool(user_data.get("verified", False)),
        "profile_image_url": user_data.get("profile_image_url"),
        "tweets_analyzed": len(tweets_data)
    }

    resp_payload = {
        "platform": "X",
        "username": clean_user,
        "prediction": pred_label,
        "raw_model_prediction": raw_pred_label,
        "model_score_uncalibrated": round(float(raw_model_score), 4),
        "is_reliable": ood_report["is_reliable"],
        "probabilities": {
            "Human": p_human,
            "Bot": p_bot
        },
        "ood_validation": ood_report,
        "profile_summary": profile_summary,
        "available_features": available_feats,
        "missing_features": missing_feats,
        "explanation": shap_explanations,
        "lime_explanations": lime_explanations,
        "status": "success",
        "scientific_disclaimer": "Predictions are probabilistic estimates derived from the Cresci-15 benchmark training dataset. Model scores are uncalibrated raw tree probabilities. Contemporary X platform dynamics, verified statuses, and sampled tweet windows reflect notable domain shift."
    }
    if credits_info:
        resp_payload["credits"] = credits_info

    return resp_payload


if __name__ == "__main__":
    import uvicorn
    print("\n[+] Starting Interpretable Social Bot Detection Web Server...")
    print("    Local Dashboard URL: http://127.0.0.1:8000")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
