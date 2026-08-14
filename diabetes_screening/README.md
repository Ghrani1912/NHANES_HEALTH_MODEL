# NHANES Diabetes Screening Module

A machine learning screening tool that predicts **undiagnosed diabetes and
prediabetes** using only non-lab, easily-collectible features — age, BMI,
waist circumference, blood pressure, family history, physical activity, and
smoking status.

This is a separate module from the CKD mortality model. It solves a
**binary classification** problem, not a survival analysis problem.

---

## Clinical Motivation

Approximately 1 in 5 diabetics in the US are undiagnosed. Standard screening
tools like FINDRISC and the ADA risk test use simple questionnaire + physical
measurement features to identify who should be sent for an HbA1c test. This
model mirrors that paradigm but learns the risk weights from data rather than
using hand-crafted scoring rules.

**The key constraint:** no lab values as features. The model must work with
what a GP can collect before ordering any blood tests.

---

## Structure

```
diabetes_screening/
├── scripts/
│   ├── download_components.py      # Step 0: download new XPT components
│   ├── merge_diabetes_cycle.py     # Step 1: merge components per cycle
│   ├── pool_diabetes_cycles.py     # Step 2: stack all cycles
│   ├── build_diabetes_label.py     # Step 3: build labels + engineer features
│   └── train_diabetes_model.py     # Step 4: train + evaluate + save models
├── processed/                      # Intermediate parquet files (not committed)
├── models/                         # Saved model artifacts (not committed)
└── README.md
```

---

## Additional Data Required

This module needs 6 new NHANES component files per cycle that are **not**
used in the CKD module. Run the download script first:

```bash
cd diabetes_screening/scripts
python download_components.py
```

This downloads the following into the existing `raw/<cycle>/` folders:

| Component | Key variable(s) | Role |
|---|---|---|
| `GHB` | `LBXGH` | HbA1c % — **outcome label only** |
| `GLU` | `LBXGLU` | Fasting glucose mg/dL — **outcome label only** |
| `BPX` | `BPXSY1`, `BPXSY2`, `BPXDI1`, `BPXDI2` | Measured blood pressure readings |
| `MCQ` | `MCQ300C` | Close relative had diabetes (family history) |
| `PAQ` | `PAQ605`, `PAQ620`, `PAQ650`, `PAQ665` | Physical activity (work + recreational) |
| `SMQ` | `SMQ020`, `SMQ040` | Smoking history and current status |

**BMX** (`BMXWAIST`) is already downloaded as part of the CKD module.

---

## Run Order

```bash
# Step 0: download new components (one-time)
python download_components.py

# Step 1: merge per cycle
python merge_diabetes_cycle.py 2007-2008 E
python merge_diabetes_cycle.py 2009-2010 F
python merge_diabetes_cycle.py 2011-2012 G
python merge_diabetes_cycle.py 2013-2014 H
python merge_diabetes_cycle.py 2015-2016 I
python merge_diabetes_cycle.py 2017-2018 J

# Step 2: pool cycles
python pool_diabetes_cycles.py

# Step 3: build labels and features
python build_diabetes_label.py

# Step 4: train models
python train_diabetes_model.py
```

---

## Features (Non-Lab Only)

| Feature | Source | Description |
|---|---|---|
| `RIDAGEYR` | DEMO | Age in years |
| `BMXBMI` | BMX | Body mass index |
| `BMXWAIST` | BMX | Waist circumference (cm) |
| `systolic_bp` | BPX (mean of 2 readings) | Systolic blood pressure |
| `diastolic_bp` | BPX (mean of 2 readings) | Diastolic blood pressure |
| `INDFMPIR` | DEMO | Income-to-poverty ratio |
| `female` | DEMO (RIAGENDR) | Sex |
| `family_hx_dm` | MCQ (MCQ300C) | Close relative with diabetes |
| `physically_active` | PAQ | Any moderate/vigorous activity |
| `current_smoker` | SMQ (SMQ040) | Current smoker |
| `race_eth` | DEMO (RIDRETH1) | Race/ethnicity (one-hot encoded) |

**Strictly excluded from features:** `LBXGH` (HbA1c), `LBXGLU` (glucose),
`LBXSCR` (creatinine), or any other lab value.

---

## Outcome Labels

| Label | Definition |
|---|---|
| `diabetes_lab` | HbA1c ≥ 6.5% OR fasting glucose ≥ 126 mg/dL |
| `prediabetes_lab` | HbA1c 5.7–6.4% OR glucose 100–125 mg/dL |
| `diabetes_selfreport` | DIQ010 == 1 (told by doctor they have diabetes) |
| `undiagnosed_dm` | `diabetes_lab=True` AND `diabetes_selfreport=False` |
| `screen_positive` | `undiagnosed_dm` OR `prediabetes_lab` — **training target** |

The model trains on `screen_positive`. The bonus analysis reports sensitivity
specifically within the `undiagnosed_dm` subgroup.

---

## Models

**Logistic Regression** — interpretable clinical baseline. Mirrors how FINDRISC
and similar tools are validated. Coefficients map directly to clinical intuition.

**XGBoost (binary:logistic)** — primary model. Captures non-linear relationships
(e.g. BMI × age interaction) that logistic regression cannot.

Both evaluated with 5-fold stratified CV. Metrics: AUC (primary), precision,
recall, F1.

**SHAP values** computed for XGBoost to show which features drive individual
predictions — important for clinical trust and interpretability.

---

## Bonus Analysis: Sensitivity Among Undiagnosed Diabetics

The most clinically meaningful metric for a screening tool:

> "Of the people who have diabetes but don't know it, what fraction does
> the model flag as high-risk?"

This is reported separately from overall recall because the screening population
(self-reported non-diabetics) has a very different base rate than the general
population.

---

## Saved Models

| File | Format | Load with |
|---|---|---|
| `models/diabetes_logistic.joblib` | joblib | `joblib.load(path)` |
| `models/diabetes_xgboost.json` | XGBoost native | `xgb.Booster(); model.load_model(path)` |
| `models/diabetes_xgb_preprocessor.joblib` | joblib | `joblib.load(path)` |
| `models/diabetes_feature_meta.joblib` | joblib | `joblib.load(path)` |

Models are also versioned as wandb artifacts under project
`nhanes-diabetes-screening`, artifact name `diabetes_screening_models`.

---

## Difference from the CKD Module

| Aspect | CKD Module | Diabetes Screening Module |
|---|---|---|
| Problem type | Survival analysis (time to death) | Binary classification (present condition) |
| Outcome | All-cause / renal death | Undiagnosed DM / prediabetes |
| Features | Includes lab values (eGFR, ACR) | Non-lab only by design |
| Primary metric | C-index | AUC |
| Key model | Cox PH + XGBoost survival | XGBoost classifier + Logistic Regression |
| Interpretability | Hazard ratios | SHAP values |
| Clinical use case | Risk stratification | Population screening |
