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
    confusion_matrix
)

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "cresci17_full_features.csv")
OUTPUT_RESULTS = os.path.join(BASE_DIR, "cresci17_benchmark_results.csv")

def main():
    print("=" * 75)
    print("PHASE 2: LEAKAGE-FREE 5-FOLD STRATIFIED CV BENCHMARK (CRESCI-17)")
    print("=" * 75)

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} does not exist yet. Please run extraction first.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded Cresci-17: {df.shape[0]} accounts, {df.shape[1]} columns.")
    print(f"Target distribution (0=Genuine, 1=Bot):")
    print(df["label"].value_counts().to_dict())

    metadata_cols = ["user_id", "dataset", "subtype", "label"]
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    print(f"Feature count: {len(feature_cols)} canonical features")

    X = df[feature_cols].copy()
    y = df["label"].values

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
        print(f"\n[+] Evaluating {name} on Cresci-17 across 5 Stratified Folds...")
        
        fold_accuracies, fold_precisions, fold_recalls, fold_f1s, fold_aucs, fold_cms = [], [], [], [], [], []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y[train_idx], y[val_idx]

            imputer = SimpleImputer(strategy="median")
            X_train_imp = imputer.fit_transform(X_train)
            X_val_imp = imputer.transform(X_val)

            if name in ["Logistic Regression"]:
                scaler = StandardScaler()
                X_train_proc = scaler.fit_transform(X_train_imp)
                X_val_proc = scaler.transform(X_val_imp)
            else:
                X_train_proc = X_train_imp
                X_val_proc = X_val_imp

            clf = model_factory()
            clf.fit(X_train_proc, y_train)

            y_pred = clf.predict(X_val_proc)
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_val_proc)[:, 1]
            elif hasattr(clf, "decision_function"):
                y_prob = clf.decision_function(X_val_proc)
            else:
                y_prob = y_pred

            fold_accuracies.append(accuracy_score(y_val, y_pred))
            fold_precisions.append(precision_score(y_val, y_pred, zero_division=0))
            fold_recalls.append(recall_score(y_val, y_pred, zero_division=0))
            fold_f1s.append(f1_score(y_val, y_pred, zero_division=0))
            fold_aucs.append(roc_auc_score(y_val, y_prob))
            fold_cms.append(confusion_matrix(y_val, y_pred))

        mean_acc, std_acc = np.mean(fold_accuracies), np.std(fold_accuracies)
        mean_prec, std_prec = np.mean(fold_precisions), np.std(fold_precisions)
        mean_rec, std_rec = np.mean(fold_recalls), np.std(fold_recalls)
        mean_f1, std_f1 = np.mean(fold_f1s), np.std(fold_f1s)
        mean_auc, std_auc = np.mean(fold_aucs), np.std(fold_aucs)
        
        total_cm = sum(fold_cms)
        tn, fp, fn, tp = total_cm.ravel()

        print(f"    Accuracy : {mean_acc:.4f} (+/- {std_acc:.4f})")
        print(f"    Precision: {mean_prec:.4f} (+/- {std_prec:.4f})")
        print(f"    Recall   : {mean_rec:.4f} (+/- {std_rec:.4f})")
        print(f"    F1-Score : {mean_f1:.4f} (+/- {std_f1:.4f})")
        print(f"    ROC-AUC  : {mean_auc:.4f} (+/- {std_auc:.4f})")
        print(f"    Aggregated Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        all_model_results.append({
            "Model": name,
            "Accuracy_Mean": round(mean_acc, 5),
            "Accuracy_Std": round(std_acc, 5),
            "Precision_Mean": round(mean_prec, 5),
            "Precision_Std": round(std_prec, 5),
            "Recall_Mean": round(mean_rec, 5),
            "Recall_Std": round(std_rec, 5),
            "F1_Score_Mean": round(mean_f1, 5),
            "F1_Score_Std": round(std_f1, 5),
            "ROC_AUC_Mean": round(mean_auc, 5),
            "ROC_AUC_Std": round(std_auc, 5),
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        })

    results_df = pd.DataFrame(all_model_results)
    results_df.to_csv(OUTPUT_RESULTS, index=False)
    print("\n" + "=" * 75)
    print("CRESCI-17 5-FOLD BENCHMARK SUMMARY TABLE:")
    print("=" * 75)
    summary_cols = ["Model", "Accuracy_Mean", "Precision_Mean", "Recall_Mean", "F1_Score_Mean", "ROC_AUC_Mean"]
    print(results_df[summary_cols].to_string(index=False))
    print(f"\nSaved Cresci-17 benchmark results to: {OUTPUT_RESULTS}")

if __name__ == "__main__":
    main()

