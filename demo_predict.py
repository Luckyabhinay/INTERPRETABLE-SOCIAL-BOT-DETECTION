import os
import sys
import numpy as np
import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "tuned_lightgbm_cresci15.joblib")
CRESCI15_DATA = os.path.join(BASE_DIR, "cresci15_full_features.csv")

def main():
    print("=" * 75)
    print("INTERPRETABLE SOCIAL BOT DETECTION - LIVE INTERACTIVE DEMONSTRATION")
    print("=" * 75)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    model = joblib.load(MODEL_PATH)
    print("[✓] Loaded Tuned LightGBM Champion Model.")

    TRAIN_FILE = os.path.join(BASE_DIR, "cresci15_train_split.csv")
    TEST_FILE = os.path.join(BASE_DIR, "cresci15_heldout_test_split.csv")
    if os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE):
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
    else:
        df = pd.read_csv(CRESCI15_DATA)
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df["label"])
    
    ignore_cols = ["user_id", "dataset", "label"]
    raw_cols = [c for c in train_df.columns if c not in ignore_cols]
    if hasattr(model, "feature_name_"):
        feature_cols = list(model.feature_name_)
    else:
        feature_cols = sorted(raw_cols)
    
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df["label"].values

    def predict_wrapper(arr):
        df_arr = pd.DataFrame(arr, columns=feature_cols)
        return model.predict_proba(df_arr)

    explainer = LimeTabularExplainer(
        training_data=np.array(X_train.values, dtype=float),
        feature_names=feature_cols,
        class_names=["Genuine Human", "Spambot / Fake"],
        mode="classification",
        random_state=42
    )
    print(f"[✓] Initialized LIME Tabular Explainer on training fold ({len(X_train)} samples).")
    print(f"[✓] Selecting strictly UNSEEN samples from held-out test fold ({len(X_test)} samples)...")

    # Select random test samples from each class from the UNSEEN test set
    human_test_indices = np.where(y_test == 0)[0]
    bot_test_indices = np.where(y_test == 1)[0]
    
    sample_human_rel_idx = human_test_indices[np.random.randint(0, len(human_test_indices))]
    sample_bot_rel_idx = bot_test_indices[np.random.randint(0, len(bot_test_indices))]

    test_indices = [
        ("Genuine Human Sample (Unseen Test Set)", sample_human_rel_idx, 0),
        ("Automated Bot Sample (Unseen Test Set)", sample_bot_rel_idx, 1)
    ]

    for label_desc, idx, true_label in test_indices:
        user_id = test_df.iloc[idx]["user_id"]
        dataset_src = test_df.iloc[idx]["dataset"]
        row_vec = np.array(X_test.iloc[idx].values, dtype=float)
        row_df = pd.DataFrame([row_vec], columns=feature_cols)

        probs = model.predict_proba(row_df)[0]
        pred_label = int(np.argmax(probs))
        pred_text = "Spambot / Fake" if pred_label == 1 else "Genuine Human"
        actual_text = "Spambot / Fake" if true_label == 1 else "Genuine Human"

        print("\n" + "=" * 65)
        print(f"TEST CASE: {label_desc} (User ID: {user_id}, Source: {dataset_src})")
        print("=" * 65)
        print(f"Actual Ground Truth : {actual_text}")
        print(f"Model Prediction    : {pred_text}")
        print(f"Prediction Confidence: Human = {probs[0]:.2%}, Bot = {probs[1]:.2%}")
        
        # Key Metadata Values
        print("\nKey Profile & Engagement Attributes:")
        print(f"  * Followers Count : {int(X_test.iloc[idx]['followers_count']):,}")
        print(f"  * Friends Count   : {int(X_test.iloc[idx]['friends_count']):,}")
        print(f"  * F-F Ratio       : {X_test.iloc[idx]['ffratio']:.4f}")
        print(f"  * Has Description : {'Yes' if X_test.iloc[idx]['description_binary'] == 1 else 'No'}")
        print(f"  * Avg Replies/Tw  : {X_test.iloc[idx]['avg_replies']:.4f}")
        print(f"  * Avg Mentions/Tw : {X_test.iloc[idx]['avg_mentions']:.4f}")
        print(f"  * Avg Favorites/Tw: {X_test.iloc[idx]['avg_favorites']:.4f}")

        # LIME Explanation
        exp = explainer.explain_instance(
            data_row=row_vec,
            predict_fn=predict_wrapper,
            num_features=6
        )

        print("\nLIME Interpretability Breakdown (Top Influential Attributes):")
        for feat_rule, weight in exp.as_list():
            direction = "Towards BOT" if weight > 0 else "Towards HUMAN"
            print(f"  * {feat_rule:35s} | Impact: {weight:+.4f} ({direction})")

    print("\n" + "=" * 75)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY.")
    print("=" * 75)

if __name__ == "__main__":
    main()

