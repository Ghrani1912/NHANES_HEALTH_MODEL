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

## Data Acquisition

The raw data is not committed to this repository. You must download it manually
from the CDC before running the pipeline. All files are free and publicly available.

### Part A — NHANES Component Files (XPT format)

For each cycle, go to the NHANES data page and download the following 7 component
files. Place them in `raw/<cycle_years>/` exactly as shown.

**Base URL:** `https://wwwn.cdc.gov/nchs/nhanes/`

| Cycle | Letter | DEMO | BIOPRO | ALB_CR | KIQ_U | DIQ | BPQ | BMX |
|---|---|---|---|---|---|---|---|---|
| 2007-2008 | E | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2007-2008/DEMO_E.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2007-2008/BIOPRO_E.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2007-2008/ALB_CR_E.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2007-2008/KIQ_U_E.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2007-2008/DIQ_E.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2007-2008/BPQ_E.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2007-2008/BMX_E.XPT) |
| 2009-2010 | F | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/DEMO_F.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/BIOPRO_F.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/ALB_CR_F.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/KIQ_U_F.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/DIQ_F.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/BPQ_F.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/BMX_F.XPT) |
| 2011-2012 | G | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/DEMO_G.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/BIOPRO_G.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/ALB_CR_G.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/KIQ_U_G.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/DIQ_G.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/BPQ_G.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/BMX_G.XPT) |
| 2013-2014 | H | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/DEMO_H.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/BIOPRO_H.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/ALB_CR_H.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/KIQ_U_H.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/DIQ_H.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/BPQ_H.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/BMX_H.XPT) |
| 2015-2016 | I | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/BIOPRO_I.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/ALB_CR_I.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/KIQ_U_I.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DIQ_I.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/BPQ_I.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/BMX_I.XPT) |
| 2017-2018 | J | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BIOPRO_J.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/ALB_CR_J.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/KIQ_U_J.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DIQ_J.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BPQ_J.XPT) | [link](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BMX_J.XPT) |

**What each component contains:**

| Component | Variables used | Description |
|---|---|---|
| `DEMO` | RIDAGEYR, RIAGENDR, RIDRETH1, INDFMPIR, WTMEC2YR, WTINT2YR, SDMVPSU, SDMVSTRA | Demographics, survey weights, design variables |
| `BIOPRO` | LBXSCR | Standard biochemistry panel — serum creatinine (mg/dL) |
| `ALB_CR` | URXUMA, URXUCR | Urine albumin (ug/mL) and urine creatinine (mg/dL) |
| `KIQ_U` | KIQ022, KIQ025 | Kidney questionnaire — self-reported kidney failure, dialysis history |
| `DIQ` | DIQ010 | Diabetes questionnaire — diabetes diagnosis (1=yes, 2=no) |
| `BPQ` | BPQ020, BPQ040A | Blood pressure questionnaire — HTN diagnosis, taking BP medication |
| `BMX` | BMXBMI | Body measurement exam — BMI |

### Part B — NCHS Linked Mortality Files (DAT format)

These fixed-width `.dat` files link each NHANES participant to their death
record (if any) from the National Death Index, with follow-up through 2019.

**Source:** [https://www.cdc.gov/nchs/data-linkage/mortality-public.htm](https://www.cdc.gov/nchs/data-linkage/mortality-public.htm)

Download one file per cycle and place it in the corresponding `raw/<cycle>/` folder:

| Cycle | Filename |
|---|---|
| 2007-2008 | `NHANES_2007_2008_MORT_2019_PUBLIC.dat` |
| 2009-2010 | `NHANES_2009_2010_MORT_2019_PUBLIC.dat` |
| 2011-2012 | `NHANES_2011_2012_MORT_2019_PUBLIC.dat` |
| 2013-2014 | `NHANES_2013_2014_MORT_2019_PUBLIC.dat` |
| 2015-2016 | `NHANES_2015_2016_MORT_2019_PUBLIC.dat` |
| 2017-2018 | `NHANES_2017_2018_MORT_2019_PUBLIC.dat` |

**Key variables extracted from mortality files:**

| Variable | Description |
|---|---|
| `eligstat` | Eligibility status for mortality follow-up (1=eligible) |
| `mortstat` | Final mortality status (1=deceased, 0=assumed alive) |
| `ucod_leading` | Underlying cause of death (1=heart disease, 10=renal disease, etc.) |
| `permth_int` | Months from interview to death or end of follow-up |
| `permth_exm` | Months from exam to death or end of follow-up |

**Cause of death codes (ucod_leading):**

| Code | Cause |
|---|---|
| 1 | Heart disease |
| 2 | Malignant neoplasm (cancer) |
| 3 | Chronic lower respiratory disease |
| 4 | Accidents/unintentional injuries |
| 5 | Cerebrovascular disease (stroke) |
| 6 | Alzheimer's disease |
| 7 | Diabetes mellitus |
| 8 | Influenza and pneumonia |
| 9 | Nephritis/nephrotic syndrome/nephrosis |
| 10 | All other causes |

> Note: in this pipeline `renal_death` is flagged as `ucod_leading == 10`
> based on NCHS coding conventions for the public-use mortality files.

### Expected folder structure after downloading

```
raw/
├── 2007-2008/
│   ├── DEMO_E.xpt
│   ├── BIOPRO_E.xpt
│   ├── ALB_CR_E.xpt
│   ├── KIQ_U_E.xpt
│   ├── DIQ_E.xpt
│   ├── BPQ_E.xpt
│   ├── BMX_E.xpt
│   └── NHANES_2007_2008_MORT_2019_PUBLIC.dat
├── 2009-2010/
│   └── ... (same pattern, suffix _F)
... (same for G, H, I, J)
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
