# NHANES CKD Risk Prediction Pipeline

A data pipeline and machine learning system for predicting **Chronic Kidney Disease (CKD) progression risk** using 12 years of US population health data (NHANES 2007–2018).

## Project Goal

Predict which patients with CKD are at high risk of mortality using routine clinical measurements — kidney function (eGFR), urine protein leakage (ACR), age, sex, diabetes, hypertension, and BMI. The pipeline cleans raw survey data, computes clinical features, and trains three survival/classification models.

---

## Project Structure

```
nhanes_ckd/
├── raw/                        # Raw NHANES XPT + mortality .dat files (not committed)
│   ├── 2007-2008/
│   ├── 2009-2010/
│   ├── 2011-2012/
│   ├── 2013-2014/
│   ├── 2015-2016/
│   └── 2017-2018/
├── processed/                  # Intermediate + final parquet files (not committed)
├── models/                     # Saved model artifacts (not committed, versioned in wandb)
│   ├── logistic_regression.joblib
│   ├── cox_allcause.pkl
│   ├── cox_renal.pkl
│   ├── xgboost_survival.json
│   └── feature_meta.joblib
├── scripts/
│   ├── merge_cycle.py          # Step 1: merge XPT files per cycle
│   ├── pool_cycles.py          # Step 2: stack all cycles into one dataset
│   ├── parse_mortality.py      # Step 3a: parse NCHS mortality .dat files
│   ├── merge_mortality.py      # Step 3b: link mortality to NHANES data
│   ├── build_features.py       # Step 4: compute eGFR, ACR, KDIGO staging
│   └── train_model_v2.py       # Step 5: train + evaluate + save models
├── requirements.txt
├── .gitignore
├── MODEL_REPORT.md             # Detailed model results and interpretation
└── README.md
```

---

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Log in to Weights & Biases (first time only)
wandb login
```

---

## Pipeline — Run Order

### Step 1 — Merge each NHANES cycle

```bash
cd scripts
python merge_cycle.py 2007-2008 E
python merge_cycle.py 2009-2010 F
python merge_cycle.py 2011-2012 G
python merge_cycle.py 2013-2014 H
python merge_cycle.py 2015-2016 I
python merge_cycle.py 2017-2018 J
```

Reads `raw/<cycle>/*.xpt`, joins 7 component files on SEQN (participant ID),
writes `processed/cycle_<cycle>.parquet`.

### Step 2 — Pool all cycles

```bash
python pool_cycles.py
```

Stacks all 6 cycle files into `processed/pooled.parquet`. Creates a globally
unique `respondent_id = cycle_SEQN` to prevent ID collisions across cycles.

### Step 3 — Parse and link mortality data

```bash
python parse_mortality.py
python merge_mortality.py
```

Parses the NCHS linked mortality fixed-width `.dat` files (one per cycle),
links them to NHANES participants, writes `processed/final_dataset.parquet`.
Adds `outcome_allcause_death`, `outcome_renal_death`, and `followup_months`.

### Step 4 — Build clinical features

```bash
python build_features.py
```

Computes:
- **eGFR** using CKD-EPI 2021 (race-free creatinine equation)
- **ACR** (albumin-to-creatinine ratio, mg/g)
- **KDIGO staging** (GFR stages G1–G5, albuminuria stages A1–A3)
- Dialysis exclusion flag
- Placeholder renal risk outcome (superseded by real mortality in final dataset)

### Step 5 — Train models

```bash
python train_model_v2.py
```

Trains three models on the full CKD spectrum cohort (G1–G5, n=7,044):
- Logistic Regression (5-fold CV, AUC)
- Cox Proportional Hazards — all-cause death (5-fold CV, C-index)
- Cox Proportional Hazards — renal-specific death (cause-specific, 5-fold CV)
- XGBoost with Cox survival objective (5-fold CV, C-index)

Saves all models to `models/`. Logs all metrics, coefficients, and feature
importances to [Weights & Biases](https://wandb.ai) project `nhanes-ckd`.

---

## Model Results Summary

| Model | Metric | Score |
|---|---|---|
| Logistic Regression | AUC | 0.817 ± 0.010 |
| Cox PH (all-cause) | C-index | 0.785 ± 0.004 |
| Cox PH (renal-specific) | C-index | 0.899 ± 0.082 |
| XGBoost Survival | C-index | 0.793 ± 0.002 |

Full details, coefficient tables, and interpretation in [MODEL_REPORT.md](MODEL_REPORT.md).

---

## Key Clinical Features

| Variable | Source | Description |
|---|---|---|
| `eGFR` | BIOPRO (LBXSCR) | Kidney filtration rate — CKD-EPI 2021 |
| `log_ACR` | ALB_CR (URXUMA, URXUCR) | Log-transformed albumin-creatinine ratio |
| `RIDAGEYR` | DEMO | Age in years |
| `female` | DEMO (RIAGENDR) | Sex (1=female) |
| `diabetes` | DIQ (DIQ010) | Diabetes diagnosis |
| `htn_treated` | BPQ (BPQ020, BPQ040A) | Hypertension + on medication |
| `BMXBMI` | BMX | Body mass index |

---

## Notes on Generalizability

The clinical feature engineering (CKD-EPI, KDIGO) is internationally standard.
The training data is US-specific (NHANES). Models trained here require external
validation before clinical use in other populations.

## Known Limitations

- 34 renal-specific death events — cause-specific Cox results are underpowered
- PH assumption violated for `RIDAGEYR`, `log_ACR`, `BMXBMI` in all-cause model
- No glycemic control marker (HbA1c) — planned next step
- No blood pressure readings (BPX) — planned next step
