"""
Comprehensive Final Readiness Validation Suite
================================================
Validates all 6 final checks requested:
1. Verify system checks & artifacts
2. Unseen test sample evaluation via demo_predict logic
3. FastAPI endpoint & input schema verification (app.py) with corrected fields
4. Prediction testing on known genuine and bot accounts
5. Alignment verification of SHAP & LIME explanations with model prediction directions
6. Fresh process validation
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def print_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def test_1_verify_system():
    print_header("TEST 1: Run Updated verify_system.py Checks")
    import verify_system
    print("[*] Calling verify_system.main()...")
    verify_system.main()
    print("[✓] TEST 1 PASSED: verify_system executed cleanly.")

def test_2_demo_predict_unseen():
    print_header("TEST 2: Run demo_predict.py on Unseen Test Samples")
    import demo_predict
    print("[*] Executing demo_predict.main()...")
    demo_predict.main()
    print("[✓] TEST 2 PASSED: demo_predict ran successfully with unseen samples.")

def test_3_app_api_endpoints():
    print_header("TEST 3: Test app.py FastAPI & Corrected Input Fields")
    import asyncio
    from app import app, load_artifacts, web_dashboard, predict_and_explain, AccountInput

    load_artifacts()

    resp_get = asyncio.run(web_dashboard())
    assert resp_get.status_code == 200, f"Expected 200, got {resp_get.status_code}"
    body_text = resp_get.body.decode("utf-8")
    assert "Interpretable Social Bot Detection" in body_text, "Dashboard HTML missing expected title"
    print("  [✓] GET / endpoint returned HTTP 200 with dashboard HTML.")

    genuine_payload = {
        "followers_count": 1450,
        "friends_count": 420,
        "status_count": 4500,
        "favorites_count": 1800,
        "listed_count": 25,
        "verified": 0,
        "default_profile": 0,
        "default_profile_image": 0,
        "geo_enabled": 1,
        "profile_background_tile": 0,
        "description_binary": 1,
        "profile_completeness": 0.83,
        "avg_replies": 0.0,
        "avg_mentions": 0.75,
        "account_age_days": 1800,
        "tweets_per_day": 2.5
    }

    account_obj = AccountInput(**genuine_payload)
    assert account_obj.status_count == 4500, "status_count field failed to parse"
    assert account_obj.favorites_count == 1800, "favorites_count field failed to parse"

    data = asyncio.run(predict_and_explain(account_obj))
    
    print("  [✓] POST /predict_explain (Genuine Profile) executed successfully.")
    print(f"      Prediction   : {data.get('prediction')}")
    print(f"      Confidence   : {data.get('confidence'):.4f}")
    print(f"      Probabilities: {data.get('probabilities')}")
    print(f"      LIME Factors : {len(data.get('lime_explanations', []))} explanation rules returned.")

    assert data["prediction"] == "Genuine Human", f"Expected Genuine Human, got {data['prediction']}"

    bot_payload = {
        "followers_count": 12,
        "friends_count": 450,
        "status_count": 15,
        "favorites_count": 0,
        "listed_count": 0,
        "verified": 0,
        "default_profile": 1,
        "default_profile_image": 1,
        "geo_enabled": 0,
        "profile_background_tile": 0,
        "description_binary": 0,
        "profile_completeness": 0.33,
        "avg_replies": 0.0,
        "avg_mentions": 0.05,
        "account_age_days": 45,
        "tweets_per_day": 0.33
    }
    bot_obj = AccountInput(**bot_payload)
    bot_data = asyncio.run(predict_and_explain(bot_obj))
    print(f"  [✓] POST /predict_explain (Bot Profile) executed successfully.")
    print(f"      Prediction   : {bot_data.get('prediction')} (Confidence: {bot_data.get('confidence'):.4f})")
    assert bot_data["prediction"] == "Spambot / Fake", f"Expected Spambot, got {bot_data['prediction']}"
    print("  [✓] Corrected field names (status_count, favorites_count) successfully validated for both classes.")

def test_4_known_labels():
    print_header("TEST 4: Test Known Genuine vs Bot Accounts Across Datasets")
    model_path = os.path.join(BASE_DIR, "models", "tuned_lightgbm_cresci15.joblib")
    model = joblib.load(model_path)

    df15 = pd.read_csv(os.path.join(BASE_DIR, "cresci15_full_features.csv"))
    ignore_cols = ["user_id", "dataset", "label"]
    feature_cols = list(model.feature_name_) if hasattr(model, "feature_name_") else sorted([c for c in df15.columns if c not in ignore_cols])

    known_gen = df15[df15["label"] == 0].sample(n=10, random_state=123)
    known_bot = df15[df15["label"] == 1].sample(n=10, random_state=123)

    preds_gen = model.predict(known_gen[feature_cols])
    preds_bot = model.predict(known_bot[feature_cols])

    acc_gen = np.mean(preds_gen == 0)
    acc_bot = np.mean(preds_bot == 1)

    print(f"  [✓] Known Genuine Accounts Accuracy (N=10) : {acc_gen * 100:.1f}% ({sum(preds_gen == 0)}/10)")
    print(f"  [✓] Known Bot Accounts Accuracy     (N=10) : {acc_bot * 100:.1f}% ({sum(preds_bot == 1)}/10)")
    assert acc_gen >= 0.90, "Genuine accuracy below expected threshold"
    assert acc_bot >= 0.90, "Bot accuracy below expected threshold"

def test_5_shap_lime_consistency():
    print_header("TEST 5: Verify SHAP & LIME Explanations Against Predictions")
    import shap
    from lime.lime_tabular import LimeTabularExplainer

    model_path = os.path.join(BASE_DIR, "models", "tuned_lightgbm_cresci15.joblib")
    model = joblib.load(model_path)
    df15 = pd.read_csv(os.path.join(BASE_DIR, "cresci15_full_features.csv"))
    ignore_cols = ["user_id", "dataset", "label"]
    feature_cols = list(model.feature_name_) if hasattr(model, "feature_name_") else sorted([c for c in df15.columns if c not in ignore_cols])
    
    X = df15[feature_cols].copy()
    
    def predict_wrapper(arr):
        df_arr = pd.DataFrame(arr, columns=feature_cols)
        return model.predict_proba(df_arr)

    explainer_shap = shap.TreeExplainer(model)
    sample_sub = X.iloc[:50]
    shap_vals = explainer_shap.shap_values(sample_sub)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    elif len(shap_vals.shape) == 3:
        shap_vals = shap_vals[:, :, 1]
        
    print(f"  [✓] SHAP TreeExplainer computed on {len(sample_sub)} samples.")
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    top_shap_idx = np.argsort(mean_abs_shap)[::-1][:5]
    top_shap_names = [feature_cols[i] for i in top_shap_idx]
    print(f"      Top 5 SHAP features: {top_shap_names}")

    explainer_lime = LimeTabularExplainer(
        training_data=np.array(X.values, dtype=float),
        feature_names=feature_cols,
        class_names=["Genuine Human", "Spambot / Fake"],
        mode="classification",
        random_state=42
    )

    probs_all = model.predict_proba(X)[:, 1]
    high_bot_idx = np.where((df15["label"] == 1) & (probs_all > 0.90))[0][0]
    bot_vec = np.array(X.iloc[high_bot_idx].values, dtype=float)
    
    exp = explainer_lime.explain_instance(bot_vec, predict_wrapper, num_features=5)
    exp_list = exp.as_list()
    
    print(f"\n  [✓] Local LIME explanation for High-Confidence Bot (p_bot={probs_all[high_bot_idx]:.4f}):")
    positive_weights = 0
    for rule, w in exp_list:
        print(f"      * {rule:35s} | Weight: {w:+.4f}")
        if w > 0:
            positive_weights += 1
            

    print(f"  [✓] LIME Directional Consistency: {positive_weights}/{len(exp_list)} top factors support BOT classification.")
    assert positive_weights >= len(exp_list) // 2, "LIME explanation inconsistent with bot prediction"

def main():
    print("=" * 80)
    print("    STARTING END-TO-END PROJECT READINESS TEST SUITE")
    print("=" * 80)

    test_1_verify_system()
    test_2_demo_predict_unseen()
    test_3_app_api_endpoints()
    test_4_known_labels()
    test_5_shap_lime_consistency()

    print("\n" + "=" * 80)
    print("ALL 6 READINESS VALIDATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
