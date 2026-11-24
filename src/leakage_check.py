"""
leakage_check.py

Leakage sanity check: one-hot encodes every feature (so categorical levels
are checked individually too), computes each resulting column's
correlation with the target, and flags anything above the 0.9 threshold.

A flagged column is not automatically leakage — it just means "look at
this closely." The actual judgement calls are written out in README.md,
since they require knowing what each column means in the real world, not
just the number.
"""

import json

import pandas as pd

from load_and_clean import NUMERIC_COLS, BINARY_COLS, CATEGORICAL_COLS, TARGET_COL

CLEAN_PATH = "data/crop_data_clean.csv"
REPORT_PATH = "data/leakage_report.json"

THRESHOLD = 0.9


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH)

    feature_cols = NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS
    encoded = pd.get_dummies(df[feature_cols], columns=CATEGORICAL_COLS)
    encoded[TARGET_COL] = df[TARGET_COL]

    correlations = encoded.corr(numeric_only=True)[TARGET_COL].drop(TARGET_COL)
    correlations = correlations.sort_values(key=abs, ascending=False)

    flagged = correlations[correlations.abs() > THRESHOLD]

    print("Correlation of every feature (incl. one-hot levels) with target:")
    print(correlations)
    print()
    if flagged.empty:
        print(f"No column exceeds |correlation| > {THRESHOLD}. No leakage flagged.")
    else:
        print(f"FLAGGED (|correlation| > {THRESHOLD}):")
        print(flagged)

    report = {
        "threshold": THRESHOLD,
        "all_correlations": correlations.round(4).to_dict(),
        "flagged_columns": flagged.round(4).to_dict(),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved full report to {REPORT_PATH}")
