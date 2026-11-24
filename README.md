# AgriPak Low-Yield Risk Prediction

Machine learning internship project for **AgriPak Farms**: a model that flags
fields at risk of a low yield *before* harvest, so the client can act in time.
This repo is built incrementally over a 7-week internship — each week adds on
top of the previous week's work.

## Week 1 — Repo setup, data cleaning, baseline, leakage check

### Project structure

```
.
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   ├── crop_data.csv           # raw data (provided)
│   ├── crop_data_clean.csv     # after cleaning (generated)
│   ├── train.csv                # 80% stratified split (generated)
│   ├── test.csv                  # 20% stratified split (generated)
│   ├── baseline_metrics.json   # baseline test-set scores (generated)
│   └── leakage_report.json     # full correlation report (generated)
├── src/
│   ├── load_and_clean.py
│   ├── make_split.py
│   ├── train_baseline.py
│   └── leakage_check.py
└── notebooks/
```

### How to run the pipeline

From the repo root, with `pandas` and `scikit-learn` installed:

```bash
pip install pandas scikit-learn

python src/load_and_clean.py      # data/crop_data.csv -> data/crop_data_clean.csv
python src/make_split.py          # -> data/train.csv, data/test.csv
python src/train_baseline.py      # trains Logistic Regression, prints + saves metrics
python src/leakage_check.py       # correlation-based leakage scan
```

Each script can also be re-run independently as long as the file it depends
on already exists (e.g. `make_split.py` will call `load_and_clean.py`
automatically if `crop_data_clean.csv` isn't there yet).

### Data cleaning decisions

The raw dataset (`crop_data.csv`, 50,000 rows, 10 columns) had **no missing
values and no duplicate rows** to begin with. The cleaning script still
applies the following rules defensively, so the pipeline is safe to re-run
on future weeks' data even if it isn't as clean:

- **Numeric columns** (`rainfall_mm`, `temperature_c`, `days_to_harvest`):
  coerced to numeric; any unparseable value becomes `NaN` and is imputed
  with the column median.
- **Binary flag columns** (`fertilizer_used`, `irrigation_used`): coerced to
  `int` (0/1); missing values imputed with the column mode.
- **Categorical columns** (`region`, `soil_type`, `crop`,
  `weather_condition`): whitespace-stripped, title-cased for consistency;
  missing values filled with `"Unknown"` rather than dropping the row.
- Rows with no label (`low_yield_risk`) are dropped — we can't train or
  evaluate on an unlabeled row.
- Exact duplicate rows are dropped.

### Train-test split

- Stratified 80/20 split on `low_yield_risk`, `random_state=42`.
- Train: 40,000 rows, Test: 10,000 rows.
- Class balance preserved in both splits: 65% class 0 / 35% class 1.
- Reproducible: re-running `make_split.py` always produces identical
  `train.csv` / `test.csv`.

### Baseline model — Logistic Regression

One-hot encoding for categorical features, standard scaling for numeric
features, both fit only on the training set (via a single sklearn
`Pipeline`, so no test-set information leaks into preprocessing).

Test-set results:

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.9187 |
| Precision | 0.8884 |
| Recall    | 0.8780 |
| F1        | 0.8832 |

### Leakage sanity check

Correlation of every feature — including each one-hot-encoded categorical
level — against the target, threshold **0.9**.

**No column exceeds 0.9.** Full ranked list is in
`data/leakage_report.json`; the notable ones:

| Column             | Correlation | Leakage or genuine predictor? |
|--------------------|-------------|-------------------------------|
| `rainfall_mm`       | -0.636      | **Genuine predictor.** Low rainfall plausibly drives low yield; this is the kind of pre-harvest signal the model is supposed to use, and it's well below the 0.9 leakage threshold. |
| `fertilizer_used`   | -0.334      | **Genuine predictor.** Known agronomic input, measured before harvest — not derived from the outcome. |
| `irrigation_used`   | -0.266      | **Genuine predictor.** Same reasoning as fertilizer — a pre-harvest decision, not a post-harvest measurement. |
| `temperature_c`     | -0.069      | Genuine predictor, weak signal. |
| all soil/region/crop/weather one-hot levels | all \|r\| < 0.01 | Negligible signal on their own; kept in the model, nothing to flag. |
| `days_to_harvest`   | 0.003       | Negligible, genuine (planning input, not an outcome). |

No column looks like it was computed *from* the target (e.g. no
post-harvest measurement, no ID-like column perfectly separating classes),
and nothing clears the 0.9 bar, so **no leakage found in Week 1 data**.

### Submission summary (for the form)

> Baseline Logistic Regression scored **0.92 accuracy / 0.88 F1** on the held-out
> test set. Leakage scan found no feature above the 0.9 correlation threshold —
> the strongest signal (`rainfall_mm`, r = -0.64) is a genuine pre-harvest
> predictor, not leakage.

## Notes

This repo is reused for every week of the internship — do not start a new
repo for future tasks.
