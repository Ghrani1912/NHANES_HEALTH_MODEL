# NHANES CKD — Model Report

**Last updated:** August 2, 2026  
**Script:** `scripts/train_model_v2.py`  
**Dataset:** `processed/final_dataset.parquet`  
**wandb project:** `nhanes-ckd`

---

## Cohort

| Property | Value |
|---|---|
| Source | NHANES 2007–2018 (6 cycles pooled) |
| Cohort definition | Full CKD spectrum: GFR stages G1–G5 |
| Total rows | 7,044 |
| All-cause deaths | 1,463 (20.8%) |
| Renal-specific deaths | 34 (0.5%) |
| Median follow-up | 98 months |
| Hypertension (on meds) | 6,063 (86%) |
| Hypertension (not on meds) | 981 (14%) |
| Diabetes | 1,721 (24%) |

### GFR Stage Breakdown

| Stage | n | eGFR range | Clinical meaning |
|---|---|---|---|
| G1 | 3,052 | ≥ 90 | Normal/high |
| G2 | 2,825 | 60–89 | Mildly decreased |
| G3a | 697 | 45–59 | Mildly-moderately decreased |
| G3b | 317 | 30–44 | Moderately-severely decreased |
| G4 | 107 | 15–29 | Severely decreased |
| G5 | 46 | < 15 | Kidney failure |

---

## Features

| Feature | Derivation | Notes |
|---|---|---|
| `eGFR` | CKD-EPI 2021 from serum creatinine (LBXSCR) | Race-free equation |
| `log_ACR` | log(URXUMA / URXUCR × 100 + 1) | ACR right-skewed; log-transform applied |
| `RIDAGEYR` | Direct from DEMO | Age in years |
| `female` | RIAGENDR == 2 | Binary sex indicator |
| `diabetes` | DIQ010 == 1 | Self-reported diabetes diagnosis |
| `htn_treated` | BPQ020==1 AND BPQ040A==1 | Hypertension diagnosed + on medication |
| `BMXBMI` | Direct from BMX | Body mass index |

**Note on hypertension encoding:** Every participant in the G1–G5 cohort with
complete data has a hypertension diagnosis (BPQ020==1). The variable therefore
captures treatment status (on meds vs not), not diagnosis status.

---

## Model 1 — Logistic Regression

**Purpose:** Binary classification baseline (died vs survived).  
**Evaluation:** 5-fold stratified CV, AUC.

### Performance

| Fold | AUC |
|---|---|
| 1 | 0.807 |
| 2 | 0.835 |
| 3 | 0.818 |
| 4 | 0.813 |
| 5 | 0.812 |
| **Mean ± SD** | **0.817 ± 0.010** |

### Coefficients (fit on full cohort)

| Feature | Coefficient | Direction |
|---|---|---|
| `eGFR` | −0.0074 | Lower kidney function → higher risk ✓ |
| `log_ACR` | +0.3284 | More albumin in urine → higher risk ✓ |
| `RIDAGEYR` | +0.0819 | Older → higher risk ✓ |
| `female` | −0.4618 | Female sex protective ✓ |
| `diabetes` | +0.0976 | Diabetes → higher risk ✓ |
| `htn_treated` | +0.0499 | Treated HTN slightly elevated risk (near zero) |
| `BMXBMI` | −0.0070 | Near-zero effect |

**Interpretation:** log_ACR and age are the dominant signals. Diabetes now
points in the correct (harmful) direction after broadening the cohort from
advanced CKD to full spectrum. The hypertension paradox (HR < 1) has
been resolved — htn_treated is now near zero rather than appearing protective.

---

## Model 2 — Cox Proportional Hazards (All-Cause Death)

**Purpose:** Survival model for time-to-death from any cause.  
**Evaluation:** 5-fold CV, concordance index (C-index).

### Performance

| Fold | C-index |
|---|---|
| 1 | 0.780 |
| 2 | 0.788 |
| 3 | 0.790 |
| 4 | 0.788 |
| 5 | 0.780 |
| **Mean ± SD** | **0.785 ± 0.004** |

Full cohort concordance: **0.79**  
Partial AIC: 23,166.44  
Log-likelihood ratio test: 1,652.39 on 7 df (p ≈ 0)

### Coefficients (full cohort)

| Feature | Coef | HR (exp coef) | 95% CI | p | Significant |
|---|---|---|---|---|---|
| `eGFR` | −0.01 | 0.99 | [0.99, 1.00] | < 0.005 | ✓ |
| `log_ACR` | +0.26 | 1.29 | [1.25, 1.34] | < 0.005 | ✓ |
| `RIDAGEYR` | +0.07 | 1.07 | [1.07, 1.08] | < 0.005 | ✓ |
| `female` | −0.35 | 0.70 | [0.63, 0.78] | < 0.005 | ✓ |
| `diabetes` | +0.08 | 1.08 | [0.96, 1.22] | 0.18 | ✗ |
| `htn_treated` | −0.01 | 0.99 | [0.78, 1.26] | 0.95 | ✗ |
| `BMXBMI` | −0.01 | 0.99 | [0.98, 1.00] | 0.04 | ✓ |

### Proportional Hazards Assumption

Violations detected (p < 0.05):

| Variable | p-value | Recommended fix |
|---|---|---|
| `RIDAGEYR` | 0.011 | Bin age into categories, use as strata |
| `log_ACR` | 0.036 | Add log_ACR² term or bin + stratify |
| `BMXBMI` | 0.029 | Bin BMI or add time interaction |

These violations mean the effect of age, ACR, and BMI changes over follow-up
time. The C-index is still valid; coefficient estimates are slightly biased.
Fixing these is a planned next step.

---

## Model 3 — Cox Proportional Hazards (Renal-Specific Death)

**Purpose:** Cause-specific survival model — isolates renal mortality from
competing causes (cardiovascular, cancer, etc.).  
**Evaluation:** 5-fold CV, C-index.

> ⚠️ **Only 34 renal death events.** Results are directionally informative
> but statistically underpowered. Wide confidence intervals throughout.

### Performance

| Fold | C-index |
|---|---|
| 1 | 0.829 |
| 2 | 0.913 |
| 3 | 0.783 |
| 4 | 0.985 |
| 5 | 0.984 |
| **Mean ± SD** | **0.899 ± 0.082** |

The high mean C-index and wide SD reflect small-sample instability — some
folds happen to get most of the 34 events in training. Interpret with caution.

### Coefficients (full cohort)

| Feature | Coef | HR | 95% CI | p |
|---|---|---|---|---|
| `eGFR` | −0.03 | 0.97 | [0.95, 0.98] | < 0.005 |
| `log_ACR` | +0.56 | 1.76 | [1.45, 2.12] | < 0.005 |
| `RIDAGEYR` | +0.09 | 1.09 | [1.05, 1.14] | < 0.005 |
| `female` | +0.00 | 1.00 | [0.50, 1.99] | 1.00 |
| `diabetes` | +0.57 | 1.77 | [0.83, 3.80] | 0.14 |
| `htn_treated` | −0.99 | 0.37 | [0.09, 1.61] | 0.19 |
| `BMXBMI` | −0.02 | 0.98 | [0.93, 1.04] | 0.59 |

**Key finding:** Diabetes HR = 1.77 (trending harmful for renal death, p=0.14).
This is the correct direction and would likely reach significance with more
renal death events. eGFR and log_ACR are the strongest renal-specific predictors.

### Proportional Hazards Assumption

Only `female` failed (p=0.029). Fix: stratify on sex in `.fit()`.

---

## Model 4 — XGBoost Survival (Cox Objective)

**Purpose:** Nonlinear survival model to capture interactions (e.g., age × eGFR)
that the linear Cox model cannot learn.  
**Evaluation:** 5-fold CV, C-index.

### Performance

| Fold | C-index |
|---|---|
| 1 | 0.793 |
| 2 | 0.796 |
| 3 | 0.789 |
| 4 | 0.793 |
| 5 | 0.792 |
| **Mean ± SD** | **0.793 ± 0.002** |

Extremely tight variance — most stable model across folds.

### Hyperparameters

| Parameter | Value |
|---|---|
| objective | survival:cox |
| max_depth | 3 |
| eta (learning rate) | 0.05 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| num_boost_round | 200 |

### Feature Importance (Gain)

| Rank | Feature | Gain |
|---|---|---|
| 1 | `RIDAGEYR` | 37.91 |
| 2 | `log_ACR` | 11.24 |
| 3 | `eGFR` | 10.26 |
| 4 | `female` | 8.07 |
| 5 | `BMXBMI` | 5.85 |
| 6 | `htn_treated` | 4.82 |
| 7 | `diabetes` | 4.22 |

Age dominates because in a cohort where everyone has CKD, age is the
strongest differentiator of who dies first. The kidney-specific markers
(log_ACR, eGFR) rank 2nd and 3rd.

---

## Model Comparison

| Model | Metric | Score | Notes |
|---|---|---|---|
| Logistic Regression | AUC | 0.817 ± 0.010 | Best discriminator for binary outcome |
| Cox PH (all-cause) | C-index | 0.785 ± 0.004 | Most stable survival model |
| Cox PH (renal-specific) | C-index | 0.899 ± 0.082 | Underpowered (34 events) |
| XGBoost Survival | C-index | 0.793 ± 0.002 | Most stable, slight edge over Cox |

All models improved substantially over the previous version (which used only
advanced CKD cases, n=775, C-index ~0.68). Broadening to the full CKD spectrum
(n=7,044) resolved the survivorship bias and raised all scores by ~0.11.

---

## What Changed vs Previous Version

| Issue | Previous version | This version |
|---|---|---|
| Cohort | Advanced CKD only (G3b+, n=775) | Full spectrum G1–G5 (n=7,044) |
| Hypertension paradox | HR 0.73 (appeared protective) | HR 0.99 (null — bias removed) |
| Diabetes effect | p=0.61, null | Trending positive in renal model (HR 1.77) |
| C-index range | 0.663–0.680 | 0.785–0.793 |
| Cause-specific model | None | Added renal-specific Cox |
| Model saving | Not implemented | All models saved to models/ + wandb |

---

## Saved Model Files

| File | Format | Load with |
|---|---|---|
| `models/logistic_regression.joblib` | joblib | `joblib.load(path)` |
| `models/cox_allcause.pkl` | pickle | `pickle.load(f)` |
| `models/cox_renal.pkl` | pickle | `pickle.load(f)` |
| `models/xgboost_survival.json` | XGBoost native | `xgb.Booster(); model.load_model(path)` |
| `models/feature_meta.joblib` | joblib | `joblib.load(path)` → dict with `feature_cols` |

Models are also versioned as wandb artifacts under project `nhanes-ckd`,
artifact name `ckd_models`.

---

## Known Issues and Next Steps

### PH assumption violations (Cox all-cause)
`RIDAGEYR`, `log_ACR`, and `BMXBMI` violate the proportional hazards assumption.
**Fix:** bin age into categories (e.g. <50, 50–65, 65–75, 75+) and use as strata.

### Underpowered renal-specific model
Only 34 renal death events. Wide confidence intervals on all coefficients.
**Fix:** cannot be fixed with this dataset alone. Would need a larger dataset
or pooling with other CKD cohorts (CRIC, CKD-BC).

### Missing glycemic control marker
Diabetes is a binary "ever diagnosed" flag. HbA1c (continuous glycemic control)
is available in NHANES `GHB` component files but not yet pulled.
**Fix:** add `parse_ghb.py` step and include `LBXGH` as a feature.

### Missing blood pressure readings
Hypertension is a binary flag. Actual measured BP values are in the NHANES
`BPX` component files and would capture severity and control.
**Fix:** add BPX to `merge_cycle.py` and engineer systolic BP feature.
