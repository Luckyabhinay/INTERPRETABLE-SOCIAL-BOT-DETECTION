import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss
)

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(BASE_DIR, "cresci15_train_split.csv")
TEST_FILE = os.path.join(BASE_DIR, "cresci15_heldout_test_split.csv")
OUTPUT_RESULTS = os.path.join(BASE_DIR, "cresci15_benchmark_results.csv")

def main():
    print("=" * 80)
    print("PHASE 1: LEAKAGE-FREE 5-FOLD STRATIFIED CV + HELD-OUT EVALUATION (CRESCI-15)")
    print("=" * 80)

    if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
        print("Error: Train/Test splits do not exist. Running create_heldout_split.py first...")
        import create_heldout_split
        create_heldout_split.main()

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print(f"Loaded 80% Training Set  : {len(train_df)} accounts ({train_df['label'].value_counts().to_dict()})")
    print(f"Loaded 20% Held-Out Test : {len(test_df)} accounts ({test_df['label'].value_counts().to_dict()})")

    metadata_cols = ["user_id", "dataset", "label"]
    feature_cols = sorted([c for c in train_df.columns if c not in metadata_cols])
    print(f"\nCanonical Features ({len(feature_cols)}): {feature_cols[:5]} ...")

    X_train_full = train_df[feature_cols].copy()
    y_train_full = train_df["label"].values

    X_test_full = test_df[feature_cols].copy()
    y_test_full = test_df["label"].values

    models = {
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": lambda: DecisionTreeClassifier(
            max_depth=10, class_weight="balanced", random_state=42
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=15, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "LightGBM": lambda: LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
        ),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            scale_pos_weight=1.0, random_state=42, n_jobs=-1, eval_metric="logloss"
        )
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_model_results = []

    for name, model_factory in models.items():
        print(f"\n[+] Evaluating {name} across 5-Fold Stratified CV on 80% Training Split...")
        
        fold_accuracies, fold_precisions, fold_recalls, fold_f1s, fold_aucs, fold_cms = [], [], [], [], [], []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
            X_tr = X_train_full.iloc[train_idx].copy()
            y_tr = y_train_full[train_idx]
            X_va = X_train_full.iloc[val_idx].copy()
            y_va = y_train_full[val_idx]

            imputer = SimpleImputer(strategy="median")
            X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=feature_cols)
            X_va_imp = pd.DataFrame(imputer.transform(X_va), columns=feature_cols)

            if name in ["Logistic Regression"]:
                scaler = StandardScaler()
                X_tr_proc = pd.DataFrame(scaler.fit_transform(X_tr_imp), columns=feature_cols)
                X_va_proc = pd.DataFrame(scaler.transform(X_va_imp), columns=feature_cols)
            else:
                X_tr_proc = X_tr_imp
                X_va_proc = X_va_imp

            clf = model_factory()
            clf.fit(X_tr_proc, y_tr)

            y_pred = clf.predict(X_va_proc)
            y_prob = clf.predict_proba(X_va_proc)[:, 1] if hasattr(clf, "predict_proba") else y_pred

            fold_accuracies.append(accuracy_score(y_va, y_pred))
            fold_precisions.append(precision_score(y_va, y_pred, zero_division=0))
            fold_recalls.append(recall_score(y_va, y_pred, zero_division=0))
            fold_f1s.append(f1_score(y_va, y_pred, zero_division=0))
            fold_aucs.append(roc_auc_score(y_va, y_prob))
            fold_cms.append(confusion_matrix(y_va, y_pred))

        # Train on 100% of 80% Train split, evaluate ONCE on held-out test split
        imputer_full = SimpleImputer(strategy="median")
        X_train_imp = pd.DataFrame(imputer_full.fit_transform(X_train_full), columns=feature_cols)
        X_test_imp = pd.DataFrame(imputer_full.transform(X_test_full), columns=feature_cols)

        if name in ["Logistic Regression"]:
            scaler_full = StandardScaler()
            X_train_final = pd.DataFrame(scaler_full.fit_transform(X_train_imp), columns=feature_cols)
            X_test_final = pd.DataFrame(scaler_full.transform(X_test_imp), columns=feature_cols)
        else:
            X_train_final = X_train_imp
            X_test_final = X_test_imp

        full_clf = model_factory()
        full_clf.fit(X_train_final, y_train_full)

        test_pred = full_clf.predict(X_test_final)
        test_prob = full_clf.predict_proba(X_test_final)[:, 1] if hasattr(full_clf, "predict_proba") else test_pred

        test_acc = accuracy_score(y_test_full, test_pred)
        test_prec = precision_score(y_test_full, test_pred, zero_division=0)
        test_rec = recall_score(y_test_full, test_pred, zero_division=0)
        test_f1 = f1_score(y_test_full, test_pred, zero_division=0)
        test_auc = roc_auc_score(y_test_full, test_prob)
        test_brier = brier_score_loss(y_test_full, test_prob)

        mean_acc, std_acc = np.mean(fold_accuracies), np.std(fold_accuracies)
        mean_prec, std_prec = np.mean(fold_precisions), np.std(fold_precisions)
        mean_rec, std_rec = np.mean(fold_recalls), np.std(fold_recalls)
        mean_f1, std_f1 = np.mean(fold_f1s), np.std(fold_f1s)
        mean_auc, std_auc = np.mean(fold_aucs), np.std(fold_aucs)
        
        total_cm = sum(fold_cms)
        tn, fp, fn, tp = total_cm.ravel()

        print(f"    5-Fold CV F1: {mean_f1:.4f} (+/- {std_f1:.4f}) | AUC: {mean_auc:.4f}")
        print(f"    [★] 20% Held-Out Test -> Acc: {test_acc:.4f} | Prec: {test_prec:.4f} | Rec: {test_rec:.4f} | F1: {test_f1:.4f} | AUC: {test_auc:.4f} | Brier: {test_brier:.5f}")

        all_model_results.append({
            "Model": name,
            "CV_Accuracy_Mean": round(mean_acc, 5),
            "CV_Accuracy_Std": round(std_acc, 5),
            "CV_Precision_Mean": round(mean_prec, 5),
            "CV_Recall_Mean": round(mean_rec, 5),
            "CV_F1_Score_Mean": round(mean_f1, 5),
            "CV_F1_Score_Std": round(std_f1, 5),
            "CV_ROC_AUC_Mean": round(mean_auc, 5),
            "HeldOut_Accuracy": round(test_acc, 5),
            "HeldOut_Precision": round(test_prec, 5),
            "HeldOut_Recall": round(test_rec, 5),
            "HeldOut_F1_Score": round(test_f1, 5),
            "HeldOut_ROC_AUC": round(test_auc, 5),
            "HeldOut_Brier_Score": round(test_brier, 5),
            "CV_TN": tn,
            "CV_FP": fp,
            "CV_FN": fn,
            "CV_TP": tp
        })

    results_df = pd.DataFrame(all_model_results)
    results_df.to_csv(OUTPUT_RESULTS, index=False)
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK SUMMARY (5-FOLD CV vs. 20% HELD-OUT TEST):")
    print("=" * 80)
    summary_cols = ["Model", "CV_F1_Score_Mean", "HeldOut_Accuracy", "HeldOut_Precision", "HeldOut_Recall", "HeldOut_F1_Score", "HeldOut_ROC_AUC", "HeldOut_Brier_Score"]
    print(results_df[summary_cols].to_string(index=False))
    print(f"\n[✓] Saved leak-free benchmark results to: {OUTPUT_RESULTS}")

if __name__ == "__main__":
    main()
