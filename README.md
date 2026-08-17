# NHANES Clinical ML Pipeline

Two ML models built on 12 years of US population health survey data
(NHANES 2007–2018), targeting two distinct clinical problems.

---

## Modules

### [CKD Mortality Risk](CKD_MORTALITY_README.md)

Predicts all-cause and renal-specific mortality in CKD patients using
survival analysis (Cox PH + XGBoost survival). Input: eGFR, ACR, age,
sex, diabetes, hypertension, BMI.

- Scripts: `scripts/`
- Best model: Cox PH C-index **0.785**, Logistic Regression AUC **0.817**
- Cohort: 7,044 CKD patients (G1–G5), 2007–2018

### [Diabetes Screening](diabetes_screening/README.md)

Predicts undiagnosed diabetes using only non-lab features (no blood draw
required). Input: age, BMI, waist, family history, activity, diet, sleep,
smoking. Mirrors ADA/FINDRISC risk score methodology.

- Scripts: `diabetes_screening/scripts/`
- Deployment number (G8 era): AUC **0.736**, sensitivity ~49% @ 80% specificity
- Historical/pooled: AUC 0.786
- Cohort: 22,175 adults, 2007–2018

---

## Project Structure

```
nhanes_ckd/
├── scripts/                    CKD mortality module scripts
├── models/                     Saved CKD model artifacts
├── raw/                        Raw NHANES XPT + mortality .dat files
│   └── <cycle>/                One folder per 2-year cycle
├── processed/                  CKD processed parquet files
├── diabetes_screening/
│   ├── scripts/                Diabetes screening module scripts
│   ├── raw/                    Additional XPT files for diabetes module
│   ├── processed/              Diabetes processed parquets + diagnostic plots
│   └── models/                 Saved diabetes model artifacts
├── CKD_MORTALITY_README.md     CKD module documentation
├── MODEL_REPORT.md             Detailed CKD model results
├── requirements.txt            All pinned dependencies
├── .gitignore
└── README.md                   This file
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac/Linux

pip install -r requirements.txt
wandb login                     # first time only
```

---

## Data

Raw data is not committed. See each module's README for download links
and folder placement instructions.

- **NHANES XPT files:** `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/<year>/DataFiles/`
- **NCHS Mortality files:** `https://www.cdc.gov/nchs/data-linkage/mortality-public.htm`

---

## Experiment Tracking

- CKD module: wandb project `nhanes-ckd`
- Diabetes module: wandb project `nhanes-diabetes-screening`
