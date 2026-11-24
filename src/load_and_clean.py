"""
load_and_clean.py

Loads the raw AgriPak crop dataset, applies cleaning rules, and writes
a cleaned copy to data/crop_data_clean.csv.

Cleaning decisions (see README for the full writeup):
- Numeric columns (rainfall_mm, temperature_c, days_to_harvest) are cast to
  numeric dtype; any value that fails to parse becomes NaN and is then
  imputed with the column median.
- Binary flag columns (fertilizer_used, irrigation_used) are cast to int
  (0/1); missing values are imputed with the column mode (most common flag).
- Categorical columns (region, soil_type, crop, weather_condition) have
  whitespace stripped and casing standardised (title case); missing values
  are filled with the literal string "Unknown" rather than dropped, so we
  don't lose rows.
- Rows missing the target column (low_yield_risk) are dropped, since we
  cannot train or evaluate on a row with no label.
- Exact duplicate rows are dropped.
"""

import pandas as pd

RAW_PATH = "data/crop_data.csv"
CLEAN_PATH = "data/crop_data_clean.csv"

NUMERIC_COLS = ["rainfall_mm", "temperature_c", "days_to_harvest"]
BINARY_COLS = ["fertilizer_used", "irrigation_used"]
CATEGORICAL_COLS = ["region", "soil_type", "crop", "weather_condition"]
TARGET_COL = "low_yield_risk"


def load_and_clean(raw_path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    n_start = len(df)

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Drop rows with no label — can't train or evaluate without a target
    df = df.dropna(subset=[TARGET_COL])

    # Numeric columns: coerce to numeric, impute missing with median
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Binary flag columns: coerce to int, impute missing with mode
    for col in BINARY_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = df[col].astype(int)

    # Categorical columns: strip whitespace, title-case, fill missing
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str).str.strip().str.title()
        df[col] = df[col].replace({"Nan": "Unknown", "": "Unknown"})

    df[TARGET_COL] = df[TARGET_COL].astype(int)

    n_end = len(df)
    print(f"Loaded {n_start} rows, {n_end} rows after cleaning "
          f"({n_start - n_end} dropped).")

    return df


if __name__ == "__main__":
    cleaned = load_and_clean()
    cleaned.to_csv(CLEAN_PATH, index=False)
    print(f"Saved cleaned data to {CLEAN_PATH}")
