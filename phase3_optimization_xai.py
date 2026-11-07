import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, brier_score_loss
)

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import shap
from lime.lime_tabular import LimeTabularExplainer

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(BASE_DIR, "cresci15_train_split.csv")
TEST_FILE = os.path.join(BASE_DIR, "cresci15_heldout_test_split.csv")
CRESCI17_PATH = os.path.join(BASE_DIR, "cresci17_full_features.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "phase3_results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PREDICTIONS_CSV = os.path.join(BASE_DIR, "cresci15_heldout_test_predictions.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data(filepath, ignore_extra=None):
    df = pd.read_csv(filepath)
    drop_cols = ["user_id", "dataset", "label"]
    if ignore_extra:
        drop_cols.extend(ignore_extra)
    features = sorted([c for c in df.columns if c not in drop_cols])
    X = df[features].copy()
    y = df["label"].values
    return df, X, y, features

def main():
    print("=" * 80)
    print("PHASE 3: LEAK-FREE TUNING, SHAP PRUNING & DUAL-TIER XAI ON 80/20 SPLIT")
    print("=" * 80)

    df_train, X_train, y_train, feats_train = load_data(TRAIN_FILE)
    df_test, X_test, y_test, feats_test = load_data(TEST_FILE)
    df17, X17, y17, feats17 = load_data(CRESCI17_PATH, ignore_extra=["subtype"])

    common_feats = sorted(list(set(feats_train).intersection(set(feats17))))
    print(f"[+] Canonical Features: {len(common_feats)}")
    print(f"    80% Train Split : {len(df_train)} accounts")
    print(f"    20% Held-Out Set: {len(df_test)} accounts")

    X_train_com = X_train[common_feats].copy()
    X_test_com = X_test[common_feats].copy()
    X17_com = X17[common_feats].copy()

    # Preprocessing strictly fit on training set
    imp = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imp.fit_transform(X_train_com), columns=common_feats)
    X_test_imp = pd.DataFrame(imp.transform(X_test_com), columns=common_feats)
    X17_imp = pd.DataFrame(imp.transform(X17_com), columns=common_feats)

    print("\n" + "=" * 60)
    print("STEP 1: HYPERPARAMETER TUNING VIA 5-FOLD CV ON 80% TRAIN SET")
    print("=" * 60)

    lgb_model_path = os.path.join(MODELS_DIR, "tuned_lightgbm_cresci15.joblib")
    xgb_model_path = os.path.join(MODELS_DIR, "tuned_xgboost_cresci15.joblib")

    lgb_param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.03, 0.08],
        'num_leaves': [20, 31],
        'max_depth': [5, 8],
        'subsample': [0.8, 1.0]
    }

    xgb_param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.03, 0.08],
        'max_depth': [4, 6],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("[*] Tuning LightGBM on 80% Train Split...")
    lgb_base = LGBMClassifier(class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
    grid_lgb = GridSearchCV(lgb_base, lgb_param_grid, cv=skf, scoring='f1', n_jobs=-1)
    grid_lgb.fit(X_train_imp, y_train)
    best_lgb = grid_lgb.best_estimator_
    print(f"    Best LightGBM Parameters: {grid_lgb.best_params_}")
    print(f"    Best LightGBM 5-Fold F1:  {grid_lgb.best_score_:.4f}")

    print("[*] Tuning XGBoost on 80% Train Split...")
    xgb_base = XGBClassifier(scale_pos_weight=1.0, random_state=42, n_jobs=-1, eval_metric="logloss")
    grid_xgb = GridSearchCV(xgb_base, xgb_param_grid, cv=skf, scoring='f1', n_jobs=-1)
    grid_xgb.fit(X_train_imp, y_train)
    best_xgb = grid_xgb.best_estimator_
    print(f"    Best XGBoost Parameters: {grid_xgb.best_params_}")
    print(f"    Best XGBoost 5-Fold F1:  {grid_xgb.best_score_:.4f}")

    joblib.dump(best_lgb, lgb_model_path)
    joblib.dump(best_xgb, xgb_model_path)
    print(f"[✓] Tuned models serialized to: {MODELS_DIR}")

    print("\n" + "=" * 60)
    print("STEP 2: GLOBAL EXPLAINABILITY VIA SHAP")
    print("=" * 60)

    explainer_15 = shap.TreeExplainer(best_lgb)
    shap_values_15 = explainer_15.shap_values(X_train_imp)
    
    if isinstance(shap_values_15, list):
        shap_vals_15 = shap_values_15[1]
    elif len(shap_values_15.shape) == 3:
        shap_vals_15 = shap_values_15[:, :, 1]
    else:
        shap_vals_15 = shap_values_15

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals_15, X_train_imp, max_display=20, show=False)
    plt.title("SHAP Global Feature Importance Beeswarm (Cresci-15 Train)", fontsize=14, pad=15)
    plt.tight_layout()
    beeswarm_path_15 = os.path.join(RESULTS_DIR, "shap_summary_cresci15.png")
    plt.savefig(beeswarm_path_15, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved SHAP Beeswarm Plot: {beeswarm_path_15}")

    mean_abs_shap = np.abs(shap_vals_15).mean(axis=0)
    shap_ranking_df = pd.DataFrame({
        "Feature": common_feats,
        "Mean_Absolute_SHAP": mean_abs_shap
    }).sort_values(by="Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)
    
    ranking_csv = os.path.join(RESULTS_DIR, "shap_feature_rankings_cresci15.csv")
    shap_ranking_df.to_csv(ranking_csv, index=False)
    print(f"[✓] Saved SHAP Feature Rankings: {ranking_csv}")
    print("\nTop 10 Most Influential Features:")
    print(shap_ranking_df.head(10).to_string(index=False))

    plt.figure(figsize=(10, 6))
    sns.barplot(data=shap_ranking_df.head(15), x="Mean_Absolute_SHAP", y="Feature", hue="Feature", palette="viridis", legend=False)
    plt.title("Top 15 Most Predictive Features by Mean |SHAP| (Cresci-15 Train)", fontsize=13)
    plt.xlabel("Mean |SHAP Value| (Impact on Model Output)")
    plt.tight_layout()
    bar_path = os.path.join(RESULTS_DIR, "shap_top15_bar_cresci15.png")
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()

    best_lgb_17 = LGBMClassifier(**best_lgb.get_params())
    best_lgb_17.fit(X17_imp, y17)
    explainer_17 = shap.TreeExplainer(best_lgb_17)
    shap_values_17 = explainer_17.shap_values(X17_imp)
    shap_vals_17 = shap_values_17[1] if isinstance(shap_values_17, list) else (shap_values_17[:, :, 1] if len(shap_values_17.shape) == 3 else shap_values_17)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals_17, X17_imp, max_display=20, show=False)
    plt.title("SHAP Global Feature Importance Beeswarm (Cresci-17)", fontsize=14, pad=15)
    plt.tight_layout()
    beeswarm_path_17 = os.path.join(RESULTS_DIR, "shap_summary_cresci17.png")
    plt.savefig(beeswarm_path_17, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved SHAP Beeswarm Plot for Cresci-17: {beeswarm_path_17}")

    print("\n" + "=" * 60)
    print("STEP 3: LEAKAGE-FREE FEATURE PRUNING IMPACT ANALYSIS (CV + HELD-OUT)")
    print("=" * 60)

    subsets = [5, 10, 15, 20, 25, len(common_feats)]
    pruning_results = []

    for k in subsets:
        selected_feats = shap_ranking_df["Feature"].iloc[:k].tolist()
        
        cv_f1s, cv_accs = [], []
        for train_idx, val_idx in skf.split(X_train_imp, y_train):
            X_tr = X_train_imp[selected_feats].iloc[train_idx]
            y_tr = y_train[train_idx]
            X_va = X_train_imp[selected_feats].iloc[val_idx]
            y_va = y_train[val_idx]

            clf_k = LGBMClassifier(**best_lgb.get_params())
            clf_k.fit(X_tr, y_tr)
            preds_va = clf_k.predict(X_va)
            cv_f1s.append(f1_score(y_va, preds_va, zero_division=0))
            cv_accs.append(accuracy_score(y_va, preds_va))

        clf_k_full = LGBMClassifier(**best_lgb.get_params())
        clf_k_full.fit(X_train_imp[selected_feats], y_train)
        test_preds_k = clf_k_full.predict(X_test_imp[selected_feats])
        test_f1_k = f1_score(y_test, test_preds_k, zero_division=0)
        test_acc_k = accuracy_score(y_test, test_preds_k)

        pruning_results.append({
            "Num_Features": k,
            "CV_Accuracy_Mean": round(np.mean(cv_accs), 4),
            "CV_F1_Score_Mean": round(np.mean(cv_f1s), 4),
            "HeldOut_Accuracy": round(test_acc_k, 4),
            "HeldOut_F1_Score": round(test_f1_k, 4)
        })
        print(f"  * Top {k:2d} Features: CV F1 = {np.mean(cv_f1s):.4f} | Held-Out Test F1 = {test_f1_k:.4f} (Acc = {test_acc_k:.4f})")

    pruning_df = pd.DataFrame(pruning_results)
    pruning_csv = os.path.join(RESULTS_DIR, "feature_pruning_performance.csv")
    pruning_df.to_csv(pruning_csv, index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(pruning_df["Num_Features"], pruning_df["CV_F1_Score_Mean"], marker='o', color='navy', lw=2, label="5-Fold CV F1 (Train Split)")
    plt.plot(pruning_df["Num_Features"], pruning_df["HeldOut_F1_Score"], marker='s', color='crimson', lw=2, linestyle='--', label="Held-Out 20% Test F1")
    plt.title("Model Performance vs. Feature Subset Size (Leak-Free Evaluation)", fontsize=13)
    plt.xlabel("Number of Top Features Selected (SHAP Ranking)")
    plt.ylabel("F1-Score")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    pruning_plot_path = os.path.join(RESULTS_DIR, "feature_pruning_curve.png")
    plt.savefig(pruning_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved Leak-Free Feature Pruning Curve: {pruning_plot_path}")

    print("\n" + "=" * 60)
    print("STEP 4: HELD-OUT TEST EVALUATION & PREDICTIONS SERIALIZATION")
    print("=" * 60)

    test_preds = best_lgb.predict(X_test_imp)
    test_probs = best_lgb.predict_proba(X_test_imp)

    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds, zero_division=0)
    test_rec = recall_score(y_test, test_preds, zero_division=0)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)
    test_auc = roc_auc_score(y_test, test_probs[:, 1])
    test_brier = brier_score_loss(y_test, test_probs[:, 1])

    print("GENUINE HELD-OUT (20%) TEST PERFORMANCE:")
    print(f"  * Accuracy   : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  * Precision  : {test_prec:.4f}")
    print(f"  * Recall     : {test_rec:.4f}")
    print(f"  * F1-Score   : {test_f1:.4f}")
    print(f"  * ROC-AUC    : {test_auc:.4f}")
    print(f"  * Brier Score: {test_brier:.5f} (lower is better, 0 = perfect calibration)")

    preds_df = pd.DataFrame({
        "user_id": df_test["user_id"].values,
        "actual_label": y_test,
        "predicted_label": test_preds,
        "bot_probability": np.round(test_probs[:, 1], 5),
        "human_probability": np.round(test_probs[:, 0], 5),
        "correct": (y_test == test_preds).astype(int)
    })
    preds_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"[✓] Saved Held-Out Test Predictions: {PREDICTIONS_CSV} ({len(preds_df)} rows)")


    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Genuine", "Bot"], yticklabels=["Genuine", "Bot"])
    plt.title("Held-Out Test Confusion Matrix (Tuned LightGBM)", fontsize=13)
    plt.ylabel("Actual Label (Held-Out Test)")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix_tuned_lgb.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()


    fpr, tpr, _ = roc_curve(y_test, test_probs[:, 1])
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Held-Out ROC curve (AUC = {test_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) - Held-Out Test Set', fontsize=13)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    roc_path = os.path.join(RESULTS_DIR, "roc_curve_tuned_lgb.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    print(f"[✓] Saved Held-Out Confusion Matrix Heatmap: {cm_path}")
    print(f"[✓] Saved Held-Out ROC Curve: {roc_path}")

    print("\n" + "=" * 60)
    print("STEP 5: LOCAL EXPLAINABILITY VIA LIME (HELD-OUT TEST INSTANCES)")
    print("=" * 60)

    explainer_lime = LimeTabularExplainer(
        training_data=np.array(X_train_imp),
        feature_names=common_feats,
        class_names=["Genuine Human", "Spambot / Fake"],
        mode="classification",
        random_state=42
    )

    def predict_wrapper(arr):
        df_arr = pd.DataFrame(arr, columns=common_feats)
        return best_lgb.predict_proba(df_arr)


    true_bot_idx = np.where((y_test == 1) & (test_preds == 1) & (test_probs[:, 1] > 0.95))[0][0]
    true_human_idx = np.where((y_test == 0) & (test_preds == 0) & (test_probs[:, 0] > 0.95))[0][0]
    fn_indices = np.where((y_test == 1) & (test_preds == 0))[0]
    fn_idx = fn_indices[0] if len(fn_indices) > 0 else None
    fp_indices = np.where((y_test == 0) & (test_preds == 1))[0]
    fp_idx = fp_indices[0] if len(fp_indices) > 0 else None

    cases = [
        ("True_Spambot", true_bot_idx, "Ground Truth: Bot | Pred: Bot (Held-Out Test)"),
        ("True_Genuine_Human", true_human_idx, "Ground Truth: Human | Pred: Human (Held-Out Test)")
    ]
    if fn_idx is not None:
        cases.append(("False_Negative_Case", fn_idx, "Ground Truth: Bot | Pred: Human (Held-Out Test)"))
    if fp_idx is not None:
        cases.append(("False_Positive_Case", fp_idx, "Ground Truth: Human | Pred: Bot (Held-Out Test)"))

    lime_report = []

    for label_case, idx, description in cases:
        row_values = np.array(X_test_imp.iloc[idx].values, dtype=float)
        exp = explainer_lime.explain_instance(
            data_row=row_values,
            predict_fn=predict_wrapper,
            num_features=8
        )
        exp_list = exp.as_list()
        
        html_file = os.path.join(RESULTS_DIR, f"lime_{label_case}.html")
        exp.save_to_file(html_file)
        
        card = [f"=== LIME EXPLANATION: {label_case} ===", f"Description: {description}", f"Held-Out Index: {idx}"]
        card.append(f"Predicted Probability -> Human: {test_probs[idx, 0]:.4f}, Bot: {test_probs[idx, 1]:.4f}")
        card.append("Top Contributing Factors to Prediction:")
        for feat_rule, weight in exp_list:
            direction = "Towards BOT" if weight > 0 else "Towards HUMAN"
            card.append(f"  * {feat_rule:35s} | Weight: {weight:+.4f} ({direction})")
        card_str = "\n".join(card)
        lime_report.append(card_str)
        print(f"\n{card_str}")

    lime_summary_file = os.path.join(RESULTS_DIR, "lime_case_studies.txt")
    with open(lime_summary_file, "w", encoding="utf-8") as f:
        f.write("\n\n" + "-" * 60 + "\n\n".join(lime_report))
    print(f"\n[✓] Saved LIME Case Studies to: {lime_summary_file}")

    print("\n" + "=" * 80)
    print("PHASE 3 ALL EXPERIMENTS, AUDIT FIXES & ARTIFACTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
