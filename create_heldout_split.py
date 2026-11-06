"""
Step 1: Create Genuine 80/20 Stratified Train/Held-Out Test Split for Cresci-15
"""
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "cresci15_full_features.csv")
TRAIN_FILE = os.path.join(BASE_DIR, "cresci15_train_split.csv")
TEST_FILE = os.path.join(BASE_DIR, "cresci15_heldout_test_split.csv")

def main():
    print("=" * 80)
    print("STEP 1: CREATING LEAK-FREE 80/20 STRATIFIED TRAIN / HELD-OUT TEST SPLIT")
    print("=" * 80)

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded full dataset: {len(df)} accounts ({df['label'].value_counts().to_dict()})")

    # Stratified 80/20 split with fixed random seed
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )

    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print(f"\n[✓] Saved 80% Training Set: {TRAIN_FILE}")
    print(f"    Rows: {len(train_df)} | Genuine (0): {sum(train_df['label']==0)}, Bots (1): {sum(train_df['label']==1)}")

    print(f"\n[✓] Saved 20% Held-Out Test Set: {TEST_FILE}")
    print(f"    Rows: {len(test_df)} | Genuine (0): {sum(test_df['label']==0)}, Bots (1): {sum(test_df['label']==1)}")

    # Sanity check: Ensure zero overlap of user_id
    train_uids = set(train_df["user_id"])
    test_uids = set(test_df["user_id"])
    assert len(train_uids.intersection(test_uids)) == 0, "FATAL: Overlap detected between train and test splits!"
    print("\n[✓] Overlap Check: Exactly 0 overlapping user_ids between train and held-out test splits.")

if __name__ == "__main__":
    main()

