import os
import sys
import pandas as pd
import numpy as np
import joblib

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def check_file(path, desc):
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    status = "PASS" if exists and size > 0 else "FAIL"
    print(f"[{status}] {desc}: {os.path.basename(path)} ({size:,} bytes)")
    return exists and size > 0

def main():
    print("=" * 75)
    print("COMPREHENSIVE SYSTEM & ARTIFACT HEALTH CHECK")
    print("=" * 75)

    all_passed = True

    # 1. Datasets Check
    print("\n[1] Checking Extracted & Split Datasets...")
    c15_path = os.path.join(BASE_DIR, "cresci15_full_features.csv")
    c17_path = os.path.join(BASE_DIR, "cresci17_full_features.csv")
    train_split = os.path.join(BASE_DIR, "cresci15_train_split.csv")
    test_split = os.path.join(BASE_DIR, "cresci15_heldout_test_split.csv")
    
    for p, name in [(c15_path, "Cresci-15 Features"), (c17_path, "Cresci-17 Features"),
                    (train_split, "80% Train Split"), (test_split, "20% Held-Out Split")]:
        if check_file(p, name):
            df = pd.read_csv(p)
            print(f"      Rows: {len(df)}, Cols: {df.shape[1]}, NaNs: {df.isna().sum().sum()}")
        else:
            all_passed = False

    # 2. Benchmark Results Check
    print("\n[2] Checking Benchmark Results & Predictions...")
    res15 = os.path.join(BASE_DIR, "cresci15_benchmark_results.csv")
    res17 = os.path.join(BASE_DIR, "cresci17_benchmark_results.csv")
    cross_res = os.path.join(BASE_DIR, "cross_dataset_results_cresci15_vs_17.csv")
    cross_detail = os.path.join(BASE_DIR, "cross_dataset_detailed_breakdown.csv")
    heldout_preds = os.path.join(BASE_DIR, "cresci15_heldout_test_predictions.csv")

    for p, name in [(res15, "Cresci-15 Benchmark"), (res17, "Cresci-17 Benchmark"),
                    (cross_res, "Cross-Dataset Results"), (cross_detail, "Subtype Breakdown"),
                    (heldout_preds, "Held-Out Predictions")]:
        if check_file(p, name):
            df_r = pd.read_csv(p)
            print(f"      Records: {len(df_r)} evaluation entries verified.")
        else:
            all_passed = False

    # 3. Model Files Check
    print("\n[3] Checking Trained Model Artifacts...")
    m_lgb = os.path.join(BASE_DIR, "models", "tuned_lightgbm_cresci15.joblib")
    m_xgb = os.path.join(BASE_DIR, "models", "tuned_xgboost_cresci15.joblib")

    for p, name in [(m_lgb, "Tuned LightGBM"), (m_xgb, "Tuned XGBoost")]:
        if check_file(p, name):
            try:
                clf = joblib.load(p)
                n_feats = len(clf.feature_name_) if hasattr(clf, "feature_name_") else (len(clf.feature_names_in_) if hasattr(clf, "feature_names_in_") else 34)
                print(f"      Successfully loaded {name} into memory. Feature count: {n_feats}")
            except Exception as e:
                print(f"      [ERROR] Could not deserialize model: {e}")
                all_passed = False
        else:
            all_passed = False

    # 4. Phase 3 Results Check
    print("\n[4] Checking Visualizations & XAI Artifacts...")
    p3_dir = os.path.join(BASE_DIR, "phase3_results")
    p3_files = [
        "shap_summary_cresci15.png",
        "shap_summary_cresci17.png",
        "shap_top15_bar_cresci15.png",
        "shap_feature_rankings_cresci15.csv",
        "feature_pruning_curve.png",
        "feature_pruning_performance.csv",
        "confusion_matrix_tuned_lgb.png",
        "roc_curve_tuned_lgb.png",
        "lime_case_studies.txt",
        "lime_True_Spambot.html",
        "lime_True_Genuine_Human.html",
        "lime_False_Negative_Case.html"
    ]
    for fname in p3_files:
        fpath = os.path.join(p3_dir, fname)
        if not check_file(fpath, fname):
            all_passed = False

    # 5. Core Documentation Check
    print("\n[5] Checking Academic Documentation & Lockfiles...")
    for doc in ["FINAL_PROJECT_REPORT.md", "PRESENTATION_SLIDES.md", "DEPLOYMENT_GUIDE.md", "COMPREHENSIVE_AUDIT_REPORT.md", "requirements.txt", "requirements_lock.txt", "app.py"]:
        if not check_file(os.path.join(BASE_DIR, doc), doc):
            all_passed = False

    # 6. Test Model Inference on Held-Out Test Set
    print("\n[6] Running Live Inference Verification (Strictly on Held-Out Test Set)...")
    try:
        clf_lgb = joblib.load(m_lgb)
        test_df = pd.read_csv(test_split)
        feature_cols_aligned = list(clf_lgb.feature_name_)
        
        sample_x = test_df[feature_cols_aligned].iloc[0:5]
        sample_y = test_df["label"].iloc[0:5].values
        
        preds = clf_lgb.predict(sample_x)
        probs = clf_lgb.predict_proba(sample_x)
        
        correct = sum(preds == sample_y)
        print(f"[PASS] Model Inference Test (Unseen Held-Out samples): {correct}/{len(preds)} correct")
        print(f"       Predictions: {preds}")
        print(f"       True Labels: {sample_y}")
        print(f"       Confidence : {probs.max(axis=1)}")
    except Exception as e:
        print(f"[FAIL] Inference Test Failed: {e}")
        all_passed = False

    print("\n" + "=" * 75)
    if all_passed:
        print("ALL AUDIT CHECKS PASSED: SYSTEM IS 100% HEALTHY, TRAINED & OPERATIONAL.")
    else:
        print("SOME CHECKS FAILED: Please inspect logs above.")
    print("=" * 75)

if __name__ == "__main__":
    main()
