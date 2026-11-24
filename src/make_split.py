"""
make_split.py

Creates a reproducible, stratified 80/20 train-test split of the cleaned
AgriPak data and saves both halves to disk. Stratifying on the target
keeps the low_yield_risk class balance the same in both splits. The fixed
random_state means re-running this script always produces the exact same
split.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from load_and_clean import load_and_clean, CLEAN_PATH, TARGET_COL

RANDOM_SEED = 42
TEST_SIZE = 0.20

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"


def make_split(df: pd.DataFrame):
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df[TARGET_COL],
    )
    return train_df, test_df


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH) if __import__("os").path.exists(CLEAN_PATH) \
        else load_and_clean()

    train_df, test_df = make_split(df)

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    print("Train target balance:")
    print(train_df[TARGET_COL].value_counts(normalize=True))
    print("Test target balance:")
    print(test_df[TARGET_COL].value_counts(normalize=True))
