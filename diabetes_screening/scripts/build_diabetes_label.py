"""
build_diabetes_label.py

Builds the analytic dataset for the diabetes screening project.

Cohort definition (this matches the real deployment scenario -- a GP only
needs this tool for patients who don't already know they have diabetes):
    modeling cohort = DIQ010 in {2, 3}  ("No" or "Borderline")
i.e. excludes people already clearly told "Yes, you have diabetes"
(DIQ010 == 1), since the screening question doesn't apply to them.

Label (ground truth, from labs -- NOT used as a model input):
    diabetes_lab   = 1 if LBXGH >= 6.5  (HbA1c-based ADA diabetes threshold)
                       OR LBXGLU >= 126 (fasting glucose threshold, where available)
    prediabetes_lab = 1 if 5.7 <= LBXGH < 6.5, or 100 <= LBXGLU < 126,
                       and not already diabetes_lab

HbA1c (LBXGH) is used as the PRIMARY criterion since it's measured on
nearly the full exam sample. Fasting glucose (LBXGLU) is only available
for the fasting-morning subsample and is used as a supplementary check
where present, not required.

Predictors (all non-lab, all collectible in a 5-minute GP visit):
    age, sex, race, BMI, waist circumference, hypertension diagnosis,
    family history of diabetes, physical activity indicators, smoking status

Usage:
    python build_diabetes_label.py

Reads:  ../../processed/pooled.parquet   (main CKD pipeline output)
Writes: ../processed/diabetes_analytic_dataset.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

# reads from main CKD pipeline's pooled.parquet (has all components now)
BASE          = Path(__file__).resolve().parent.parent.parent   # nhanes_ckd root
PROCESSED_IN  = BASE / "processed"
PROCESSED_OUT = Path(__file__).resolve().parent.parent / "processed"
PROCESSED_OUT.mkdir(parents=True, exist_ok=True)


def build_label(df: pd.DataFrame) -> pd.DataFrame:
    hba1c_diabetes   = df["LBXGH"]  >= 6.5
    glucose_diabetes = df["LBXGLU"] >= 126   # NaN-safe: comparisons with NaN are False

    df["diabetes_lab"] = (hba1c_diabetes.fillna(False) | glucose_diabetes.fillna(False))

    hba1c_prediabetes   = df["LBXGH"].between(5.7, 6.499, inclusive="both")
    glucose_prediabetes = df["LBXGLU"].between(100, 125.999, inclusive="both")
    df["prediabetes_lab"] = (
        (hba1c_prediabetes.fillna(False) | glucose_prediabetes.fillna(False))
        & ~df["diabetes_lab"]
    )

    # need at least one lab value to have a valid label at all
    df["has_lab_label"] = df["LBXGH"].notna() | df["LBXGLU"].notna()
    return df


def met_minutes(df: pd.DataFrame, yn_col: str, days_col: str,
                min_col: str, met_value: float) -> pd.Series:
    """
    Standard GPAQ-style MET-minutes/week for one activity type:
    METs x days/week x minutes/day, but only counted when the yes/no flag
    is 'yes' -- when it's 'no', the days/minutes questions are skip-pattern
    NaN by design and should contribute 0, not be treated as missing data.
    """
    did_activity = df[yn_col] == 1
    days    = df[days_col].where(did_activity, 0).fillna(0)
    minutes = df[min_col].where(did_activity, 0).fillna(0)
    return met_value * days * minutes


def clean_predictors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age"]      = df["RIDAGEYR"]
    df["female"]   = (df["RIAGENDR"] == 2).astype(int)
    df["race_eth"] = df["RIDRETH1"]   # keep as categorical code, one-hot later
    df["bmi"]      = df["BMXBMI"]
    df["waist_cm"] = df["BMXWAIST"]

    n0 = df.shape[0]
    print(f"    [funnel] start: {n0}")

    # hypertension: valid responses only
    df = df[df["BPQ020"].isin([1, 2])].copy()
    print(f"    [funnel] after BPQ020 filter: {df.shape[0]} (-{n0 - df.shape[0]})")
    n1 = df.shape[0]
    df["hypertension"] = (df["BPQ020"] == 1).astype(int)

    # family history: 1=yes, 2=no; drop refused/DK/missing
    df = df[df["MCQ300C"].isin([1, 2])].copy()
    print(f"    [funnel] after MCQ300C filter: {df.shape[0]} (-{n1 - df.shape[0]})")
    n2 = df.shape[0]
    df["family_history_diabetes"] = (df["MCQ300C"] == 1).astype(int)

    # smoking: SMQ020 ever smoked 100 cigs (1=yes, 2=no)
    df = df[df["SMQ020"].isin([1, 2])].copy()
    print(f"    [funnel] after SMQ020 filter: {df.shape[0]} (-{n2 - df.shape[0]})")
    df["ever_smoker"]    = (df["SMQ020"] == 1).astype(int)
    df["current_smoker"] = df["SMQ040"].isin([1, 2]).astype(int)   # NaN-safe

    # --- physical activity: real MET-minutes/week (replaces crude yes/no count) ---
    df["met_min_vigorous_work"]    = met_minutes(df, "PAQ605", "PAQ610", "PAD615", 8)
    df["met_min_moderate_work"]    = met_minutes(df, "PAQ620", "PAQ625", "PAD630", 4)
    df["met_min_active_transport"] = met_minutes(df, "PAQ635", "PAQ640", "PAD645", 4)
    df["met_min_vigorous_rec"]     = met_minutes(df, "PAQ650", "PAQ655", "PAD660", 8)
    df["met_min_moderate_rec"]     = met_minutes(df, "PAQ665", "PAQ670", "PAD675", 4)
    df["met_minutes_total"] = (
        df["met_min_vigorous_work"]    + df["met_min_moderate_work"] +
        df["met_min_active_transport"] + df["met_min_vigorous_rec"] +
        df["met_min_moderate_rec"]
    )
    df["sedentary_minutes"] = df["PAD680"]

    # --- diet (single-day 24hr recall — noisy but real signal in aggregate) ---
    df["calories"] = df["DR1TKCAL"]
    df["sugar_g"]  = df["DR1TSUGR"]
    df["fiber_g"]  = df["DR1TFIBE"]
    df["carbs_g"]  = df["DR1TCARB"]

    # --- sleep (unified column from merge_cycle.py: handles SLD010H/SLD012 rename) ---
    df["sleep_hours"] = df["sleep_hours"] if "sleep_hours" in df.columns else np.nan

    # --- income (already in DEMO, just unused until now) ---
    df["income_ratio"] = df["INDFMPIR"]

    n3 = df.shape[0]
    check_cols = ["bmi", "waist_cm", "sedentary_minutes", "calories",
                  "sugar_g", "fiber_g", "carbs_g", "sleep_hours", "income_ratio"]
    print(f"    [funnel] before final dropna, {n3} rows remain. "
          f"Missingness on columns that CAN still be NaN here:")
    for col in check_cols:
        n_miss = df[col].isna().sum()
        print(f"      {col}: {n_miss} ({n_miss/n3:.1%})")

    return df


PREDICTOR_COLS = [
    "age", "female", "race_eth", "bmi", "waist_cm", "hypertension",
    "family_history_diabetes", "ever_smoker", "current_smoker",
    "met_minutes_total", "sedentary_minutes",
    "calories", "sugar_g", "fiber_g", "carbs_g",
    "sleep_hours", "income_ratio",
]


def main():
    df = pd.read_parquet(PROCESSED_IN / "pooled.parquet")
    print(f"Loaded pooled dataset: {df.shape[0]} rows")

    # Adults only. This is both a scope decision (a FINDRISC/ADA-style
    # screening tool is meant for adults) AND the fix for most of the
    # missingness in BPQ020/MCQ300C/SMQ020: those questions have hard age
    # eligibility floors -- minors were never asked them.
    n_before_age = df.shape[0]
    df = df[df["RIDAGEYR"] >= 18].copy()
    print(f"Adults only (RIDAGEYR >= 18): {df.shape[0]} (-{n_before_age - df.shape[0]})")

    df = build_label(df)

    # deployment-matching cohort: people who do NOT already know they have diabetes
    df = df[df["DIQ010"].isin([2, 3])].copy()
    df["borderline_selfreport"] = (df["DIQ010"] == 3).astype(int)
    print(f"Cohort excluding self-reported diagnosed diabetics: {df.shape[0]} rows")

    df = df[df["has_lab_label"]]
    print(f"Cohort with a valid lab label (HbA1c or glucose present): {df.shape[0]} rows")

    df = clean_predictors(df)
    df = df.dropna(subset=PREDICTOR_COLS)
    print(f"Cohort with complete predictors: {df.shape[0]} rows")

    out_path = PROCESSED_OUT / "diabetes_analytic_dataset.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    print(f"\nLab-confirmed diabetes prevalence (undiagnosed, by definition): "
          f"{df['diabetes_lab'].mean():.3f}  (n={int(df['diabetes_lab'].sum())})")
    print(f"Lab-confirmed prediabetes prevalence: "
          f"{df['prediabetes_lab'].mean():.3f}  (n={int(df['prediabetes_lab'].sum())})")


if __name__ == "__main__":
    main()
