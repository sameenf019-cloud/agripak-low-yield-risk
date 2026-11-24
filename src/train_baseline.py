"""
train_baseline.py

Trains a Logistic Regression baseline on data/train.csv and evaluates it
on data/test.csv. Categorical columns are one-hot encoded and numeric
columns are standard-scaled inside a single sklearn Pipeline, so the exact
same transformation (fit only on the training data) is applied to the
test set — this avoids leaking test-set statistics into training.
"""

import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from load_and_clean import NUMERIC_COLS, BINARY_COLS, CATEGORICAL_COLS, TARGET_COL

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
METRICS_PATH = "data/baseline_metrics.json"

RANDOM_SEED = 42


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS + BINARY_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


if __name__ == "__main__":
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    feature_cols = NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
    }

    print("Baseline Logistic Regression — test set metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")
