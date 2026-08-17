# Diabetes Screening Module

Predicts undiagnosed diabetes and prediabetes using only non-lab,
easily-collectible features — age, BMI, waist circumference, family
history, physical activity, diet, sleep, and smoking. No blood draw required.

This mirrors real clinical screening tools (FINDRISC, ADA risk test) but
learns risk weights from data rather than using hand-crafted scoring rules.

---

## Purpose

Flag people who are likely to have undiagnosed diabetes so a GP can order
a confirmatory HbA1c test. The model is designed for the pre-diagnosis
screening context: it only applies to people who report they do not have
diabetes (DIQ010 ≠ 1).

**The key clinical metric** is sensitivity at a fixed specificity — not
accuracy. A GP cares about "how many undiagnosed diabetics would I miss,"
not "what percentage of predictions are correct."

---

## Scripts (`scripts/`)

| Script | Step | What it does |
|---|---|---|
| `merge_diabetes_cycle.py` | 1 | Merges components per cycle (reads from `diabetes_screening/raw/`) |
| `pool_diabetes_cycles.py` | 2 | Stacks all 6 cycles |
| `build_diabetes_label.py` | 3 | Builds ADA-threshold labels + engineers features |
| `train_diabetes_model.py` | 4 | Trains LR + XGBoost with calibration, reports clinical metrics |
| `diagnose_diabetes_model.py` | 5 | Calibration plots, PR curve, subgroup analysis, missingness check |
| `evaluate_model.py` | 6 | Bootstrap CIs, leave-one-cycle-out validation, subgroup performance |
| `assay_era_analysis.py` | 7 | Investigates G8 assay era performance gap |

---

## Data Required

### From `diabetes_screening/raw/<cycle>/`

These files are separate from the CKD module. Download and place in
`diabetes_screening/raw/<cycle>/`:

| Component | Variables | Description |
|---|---|---|
| `GHB` | LBXGH | HbA1c % — **outcome label only, never a feature** |
| `GLU` | LBXGLU, WTSAF2YR | Fasting glucose — **outcome label only** |
| `MCQ` | MCQ300C | Close relative had diabetes |
| `PAQ` | PAQ605–PAQ670, PAD615–PAD680 | Physical activity (vigorous/moderate, days, minutes) |
| `SMQ` | SMQ020, SMQ040 | Smoking history and current status |
| `SLQ` | SLD010H / SLD012 | Sleep hours (variable renamed in 2015) |
| `DR1TOT` | DR1TKCAL, DR1TSUGR, DR1TFIBE, DR1TCARB | 24hr dietary recall |

Plus `DEMO`, `BMX`, `BPQ`, `DIQ` which are shared with the CKD module and
also needed here.

Download URL pattern:
`https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/<start_year>/DataFiles/<FILE>.XPT`

---

## Run Order

```bash
cd diabetes_screening/scripts

# Step 1: merge per cycle (or use main merge_cycle.py which now includes all components)
python merge_diabetes_cycle.py 2007-2008 E
python merge_diabetes_cycle.py 2009-2010 F
python merge_diabetes_cycle.py 2011-2012 G
python merge_diabetes_cycle.py 2013-2014 H
python merge_diabetes_cycle.py 2015-2016 I
python merge_diabetes_cycle.py 2017-2018 J

# Step 2: pool
python pool_diabetes_cycles.py

# Step 3: build labels and features
python build_diabetes_label.py

# Step 4: train
python train_diabetes_model.py

# Step 5 (optional): detailed diagnostics
python diagnose_diabetes_model.py

# Step 6 (optional): full evaluation
python evaluate_model.py

# Step 7 (optional): assay era investigation
python assay_era_analysis.py
```

> **Note:** The main `scripts/merge_cycle.py` now pulls all components including
> GHB, GLU, MCQ, PAQ, SMQ, SLQ, DR1TOT. After running the main merge/pool
> pipeline, `build_diabetes_label.py` can read directly from
> `processed/pooled.parquet` instead.

---

## Cohort

- **Definition:** Adults (age ≥ 18) who self-report no diabetes diagnosis
  (DIQ010 = 2 or 3) and have at least one lab measurement (HbA1c or glucose)
- **Size:** 22,175 rows (complete cases including diet/sleep/income)
- **Undiagnosed diabetes prevalence:** 5.0%
- **Prediabetes prevalence:** 39.6%

---

## Features (Non-Lab Only)

| Feature | Source | Description |
|---|---|---|
| `age` | DEMO | Age in years |
| `female` | DEMO | Sex (1=female) |
| `race_eth` | DEMO | Race/ethnicity (one-hot encoded) |
| `bmi` | BMX | Body mass index |
| `waist_cm` | BMX | Waist circumference (cm) |
| `hypertension` | BPQ | Diagnosed hypertension |
| `family_history_diabetes` | MCQ | Close relative with diabetes |
| `ever_smoker` | SMQ | Smoked ≥100 cigarettes ever |
| `current_smoker` | SMQ | Currently smoking |
| `met_minutes_total` | PAQ | Total MET-minutes/week (GPAQ method) |
| `sedentary_minutes` | PAQ | Sedentary minutes per day |
| `calories` | DR1TOT | Daily caloric intake |
| `sugar_g` | DR1TOT | Daily sugar intake (g) |
| `fiber_g` | DR1TOT | Daily fiber intake (g) |
| `carbs_g` | DR1TOT | Daily carbohydrate intake (g) |
| `sleep_hours` | SLQ | Sleep hours per night |
| `income_ratio` | DEMO | Income-to-poverty ratio |

**Strictly excluded:** HbA1c (LBXGH), fasting glucose (LBXGLU), serum
creatinine (LBXSCR), or any other lab value.

---

## Models and Results

### Deployment-relevant numbers (G8 era, 2015–2018)

These are the honest numbers for a tool deployed today, since any
real-world use faces a G8-era-equivalent population:

| Model | AUC | Sensitivity @ 80% specificity |
|---|---|---|
| Logistic Regression | **0.736 ± 0.007** | **~49%** |

### Historical / all-eras numbers (2007–2018 pooled)

Useful for comparison with older literature, but not the deployment number:

| Model | AUC | 95% CI | Sensitivity @ 80% specificity |
|---|---|---|---|
| Logistic Regression | 0.786 | [0.774, 0.797] | 56.9% [53.9–59.7%] |
| XGBoost | 0.780 | — | 57.1% |

---

## Why the G8-era Performance is Lower

Three tests were run to diagnose the 2015–2018 performance gap:

1. **HbA1c assay check** — No systematic G8 calibration drift found.
   The Tosoh G8 assay does not read consistently lower/higher.

2. **Within-era validation** — G8-era cycles score 0.736 even when
   trained within G8 data. The problem is not the assay — the population
   is genuinely harder to screen.

3. **Root cause** — Self-reported diabetes diagnosis rate rose from
   **11.8% to 14.5%** between eras (controlled for age). More people
   know their status in 2015–2018, leaving a harder-to-detect residual
   undiagnosed population — likely people who have slipped through
   multiple screening opportunities.

---

## Key Diagnostic Findings

**Lean subgroup blind spot:** The model catches only 18% of lean
diabetics (normal BMI + normal waist) at the 80% specificity threshold,
vs 60% of non-lean. The model leans heavily on waist and BMI — a
documented limitation of all anthropometric T2D screening tools.

**Missingness is unbiased:** Dropped rows (diet/income missing) have
5.4% diabetes prevalence vs 5.0% in kept rows — the complete-case
cohort is representative.

**Age 65+ weakest subgroup:** AUC 0.677 vs 0.810 for age 18-44. The
model is designed for early/mid-life screening; older adults have more
complex comorbidity patterns that the non-lab features cannot capture.

---

## Saved Models (`models/`)

| File | Format | Load with |
|---|---|---|
| `diabetes_logistic.joblib` | joblib | `joblib.load(path)` |
| `diabetes_xgboost.json` | XGBoost native | `xgb.Booster(); model.load_model(path)` |
| `diabetes_xgb_preprocessor.joblib` | joblib | `joblib.load(path)` |
| `diabetes_feature_meta.joblib` | joblib | `joblib.load(path)` |

Models also versioned as wandb artifacts — project: `nhanes-diabetes-screening`.

---

## Experiment Tracking

All runs logged to Weights & Biases:
- Project: `nhanes-diabetes-screening`
- Run with: `wandb login` before first use

---

## Difference from CKD Module

| Aspect | CKD Mortality | Diabetes Screening |
|---|---|---|
| Problem type | Survival analysis | Binary classification |
| Outcome | All-cause / renal death | Undiagnosed DM / prediabetes |
| Features | Includes lab values | Non-lab only by design |
| Primary metric | C-index | AUC + sensitivity@specificity |
| Key model | Cox PH + XGBoost survival | Logistic Regression + XGBoost |
| Interpretability | Hazard ratios | Coefficients + SHAP values |
| Clinical use | Risk stratification | Population screening |
