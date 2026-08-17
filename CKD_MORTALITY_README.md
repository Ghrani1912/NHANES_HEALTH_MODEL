# CKD Mortality Risk Module

Predicts all-cause and renal-specific mortality risk in patients with
Chronic Kidney Disease (CKD) using 12 years of US population health survey
data (NHANES 2007–2018).

---

## Purpose

Risk-stratify CKD patients (G1–G5) using routine clinical measurements
that any physician already has — kidney function (eGFR), urine protein
leakage (ACR), age, sex, diabetes, hypertension, and BMI. The output is
a survival risk score that identifies which patients are most likely to
die and should receive closer monitoring or earlier intervention.

---

## Scripts (`scripts/`)

| Script | Step | What it does |
|---|---|---|
| `merge_cycle.py` | 1 | Merges 13 NHANES component files per cycle into one wide table |
| `pool_cycles.py` | 2 | Stacks all 6 cycles into one pooled dataset |
| `parse_mortality.py` | 3a | Parses NCHS linked mortality fixed-width `.dat` files |
| `merge_mortality.py` | 3b | Links mortality records to NHANES participants |
| `build_features.py` | 4 | Computes eGFR (CKD-EPI 2021), ACR, KDIGO staging |
| `train_model_v2.py` | 5 | Trains and evaluates all models, saves artifacts |

---

## Data Required

### NHANES Component Files (XPT)

For each cycle, download from `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/<year>/DataFiles/`
and place in `raw/<cycle>/`:

| Component | Variables | Description |
|---|---|---|
| `DEMO` | RIDAGEYR, RIAGENDR, RIDRETH1, INDFMPIR, WTMEC2YR, WTINT2YR, SDMVPSU, SDMVSTRA | Demographics + survey weights |
| `BIOPRO` | LBXSCR | Serum creatinine (mg/dL) |
| `ALB_CR` | URXUMA, URXUCR | Urine albumin + creatinine |
| `KIQ_U` | KIQ022, KIQ025 | Kidney questionnaire |
| `DIQ` | DIQ010 | Diabetes self-report |
| `BPQ` | BPQ020, BPQ040A | Hypertension + medication |
| `BMX` | BMXBMI, BMXWAIST | BMI + waist circumference |

### NCHS Linked Mortality Files (DAT)

Download from `https://www.cdc.gov/nchs/data-linkage/mortality-public.htm`
and place in `raw/<cycle>/`:

| Cycle | Filename |
|---|---|
| 2007-2008 | `NHANES_2007_2008_MORT_2019_PUBLIC.dat` |
| 2009-2010 | `NHANES_2009_2010_MORT_2019_PUBLIC.dat` |
| 2011-2012 | `NHANES_2011_2012_MORT_2019_PUBLIC.dat` |
| 2013-2014 | `NHANES_2013_2014_MORT_2019_PUBLIC.dat` |
| 2015-2016 | `NHANES_2015_2016_MORT_2019_PUBLIC.dat` |
| 2017-2018 | `NHANES_2017_2018_MORT_2019_PUBLIC.dat` |

---

## Run Order

```bash
cd scripts

# Step 1: merge per cycle
python merge_cycle.py 2007-2008 E
python merge_cycle.py 2009-2010 F
python merge_cycle.py 2011-2012 G
python merge_cycle.py 2013-2014 H
python merge_cycle.py 2015-2016 I
python merge_cycle.py 2017-2018 J

# Step 2: pool cycles
python pool_cycles.py

# Step 3: parse and link mortality
python parse_mortality.py
python merge_mortality.py

# Step 4: build clinical features
python build_features.py

# Step 5: train models
python train_model_v2.py
```

---

## Cohort

- **Definition:** Adults with CKD across the full GFR spectrum (G1–G5)
- **Size:** 7,044 rows (complete cases)
- **Outcome:** All-cause death and renal-specific death, follow-up through 2019

---

## Models and Results

| Model | Metric | Score |
|---|---|---|
| Logistic Regression | AUC | 0.817 ± 0.010 |
| Cox PH — all-cause death | C-index | 0.785 ± 0.004 |
| Cox PH — renal-specific death | C-index | 0.899 ± 0.082 (underpowered: 34 events) |
| XGBoost Survival | C-index | 0.793 ± 0.002 |

All models use 5-fold cross-validation. Cox models are calibrated for
time-to-event analysis using lifelines. XGBoost uses the Cox partial
likelihood objective.

### Key Features (by importance)

1. `RIDAGEYR` — age (dominant signal)
2. `log_ACR` — log-transformed albumin-creatinine ratio
3. `eGFR` — estimated glomerular filtration rate (CKD-EPI 2021)
4. `female` — sex (protective)
5. `hypertension` — treated hypertension

### Important Finding: Hypertension Paradox Resolved

Early analysis showed hypertension appearing "protective" (HR 0.73). Root
cause was **survivorship bias**: restricting to advanced CKD only excluded
people who died early from hypertension before reaching CKD. Fixing this by
broadening to the full G1–G5 spectrum eliminated the paradox (HR now ~0.99).

---

## Saved Models (`models/`)

| File | Format | Load with |
|---|---|---|
| `logistic_regression.joblib` | joblib | `joblib.load(path)` |
| `cox_allcause.pkl` | pickle | `pickle.load(f)` |
| `cox_renal.pkl` | pickle | `pickle.load(f)` |
| `xgboost_survival.json` | XGBoost native | `xgb.Booster(); model.load_model(path)` |
| `feature_meta.joblib` | joblib | `joblib.load(path)` |

Models also versioned as wandb artifacts — project: `nhanes-ckd`.

---

## Known Limitations

- **Renal-specific Cox model underpowered** — only 34 renal death events;
  wide confidence intervals, interpret directionally only
- **PH assumption violated** for `RIDAGEYR`, `log_ACR`, `BMXBMI` in all-cause
  model — effects are not constant over follow-up time; stratification needed
- **US population only** — CKD-EPI 2021 and KDIGO staging are internationally
  standard, but model weights reflect US demographics
- **No HbA1c or measured BP severity** — only binary diagnosis flags used;
  adding these would likely improve performance

---

## Experiment Tracking

All runs logged to Weights & Biases:
- Project: `nhanes-ckd`
- Metrics: per-fold C-index/AUC, coefficients, hazard ratios, feature importance
- Run with: `wandb login` before first use
