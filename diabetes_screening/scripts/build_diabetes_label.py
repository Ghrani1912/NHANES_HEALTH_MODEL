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

Reads:  ../processed/pooled.parquet
Writes: ../processed/diabetes_analytic_dataset.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE      = Path(__file__).resolve().parent.parent   # diabetes_screening/
PROCESSED = BASE / "processed"


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


def impute_with_missingness_flag(df: pd.DataFrame, col: str,
                                  new_col_name: str = None) -> pd.DataFrame:
    """
    Impute a binary (0/1) column's NaNs at the observed prevalence among
    non-missing rows, and add a companion *_missing indicator.

    Rationale: flat 0.5 fill assumes unknown == 50/50, but the actual base
    rate may differ (MCQ300C ~42%). Imputing at prevalence is more honest.
    The missingness flag lets models learn whether 'unknown' itself carries
    a risk signal.
    """
    col_out    = new_col_name or col
    is_missing = df[col].isna()
    prevalence = df.loc[~is_missing, col].mean()
    df[f"{col_out}_missing"] = is_missing.astype(int)
    df[col_out] = df[col].fillna(prevalence)
    print(f"    imputed {col}: {is_missing.sum()} NaNs at prevalence={prevalence:.3f}")
    return df


def clean_predictors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # avoid SettingWithCopyWarning
    df["age"]      = df["RIDAGEYR"]
    df["female"]   = (df["RIAGENDR"] == 2).astype(int)
    df["race_eth"] = df["RIDRETH1"]   # keep as categorical code, one-hot later
    df["bmi"]      = df["BMXBMI"]
    df["waist_cm"] = df["BMXWAIST"]

    n0 = df.shape[0]
    print(f"    [funnel] start: {n0}")

    # hypertension: valid responses only (9=don't know → only 32 adults, drop)
    df = df[df["BPQ020"].isin([1, 2])].copy()
    print(f"    [funnel] after BPQ020 filter: {df.shape[0]} (-{n0 - df.shape[0]})")
    df["hypertension"] = (df["BPQ020"] == 1).astype(int)

    # family history: map 1→1, 2→0; impute residual NaN at observed prevalence
    # + add missingness flag (residual NaN ~1580 adults, evenly spread ~250/cycle)
    df["_mcq_raw"] = df["MCQ300C"].map({1: 1.0, 2: 0.0})  # refused/DK → NaN
    df = impute_with_missingness_flag(df, "_mcq_raw",
                                      new_col_name="family_history_diabetes")
    df = df.drop(columns=["_mcq_raw"])

    # smoking: map SMQ020 1→1, 2→0; impute residual NaN at observed prevalence
    # NOTE: residual NaN in SMQ020 (n=823) is entirely age 18-19 in cycles
    # 2007-2011 — NHANES changed SMQ routing after 2012 so all adults get asked.
    # ever_smoker_missing flag was tested in ablation (PR-AUC diff=0.0001) and
    # found REDUNDANT with RIDAGEYR — dropped. Age already encodes this signal.
    df["_smq_raw"] = df["SMQ020"].map({1: 1.0, 2: 0.0})
    df = impute_with_missingness_flag(df, "_smq_raw",
                                      new_col_name="ever_smoker")
    df = df.drop(columns=["_smq_raw", "ever_smoker_missing"])  # flag redundant with age
    df["current_smoker"] = df["SMQ040"].isin([1, 2]).astype(int)  # NaN-safe

    # physical activity: PAQ605/620/635/650/665 are 1=yes, 2=no
    for col, name in [("PAQ605", "vigorous_work"),  ("PAQ620", "moderate_work"),
                      ("PAQ635", "active_transport"), ("PAQ650", "vigorous_rec"),
                      ("PAQ665", "moderate_rec")]:
        df[name] = (df[col] == 1).astype(int)   # NaN-safe

    df["activity_score"] = (
        df["vigorous_work"] + df["moderate_work"] +
        df["active_transport"] + df["vigorous_rec"] +
        df["moderate_rec"]
    )
    df["sedentary_minutes"] = df["PAD680"]

    n3 = df.shape[0]
    n_bmi_missing      = df["bmi"].isna().sum()
    n_waist_missing    = df["waist_cm"].isna().sum()
    n_sedentary_missing = df["sedentary_minutes"].isna().sum()
    print(f"    [funnel] before final dropna, {n3} rows remain. "
          f"Missingness on columns that CAN still be NaN here: "
          f"bmi={n_bmi_missing} ({n_bmi_missing/n3:.1%}), "
          f"waist_cm={n_waist_missing} ({n_waist_missing/n3:.1%}), "
          f"sedentary_minutes={n_sedentary_missing} ({n_sedentary_missing/n3:.1%})")
    return df


PREDICTOR_COLS = [
    "age", "female", "race_eth", "bmi", "waist_cm", "hypertension",
    "family_history_diabetes", "family_history_diabetes_missing",
    "ever_smoker",   # ever_smoker_missing dropped — redundant with RIDAGEYR (ablation PR-AUC diff=0.0001)
    "current_smoker", "activity_score", "sedentary_minutes",
]


def main():
    df = pd.read_parquet(PROCESSED / "pooled.parquet")
    print(f"Loaded pooled dataset: {df.shape[0]} rows")

    df = build_label(df)

    # Step 0: adults only — MCQ300C/BPQ020/SMQ020 are not asked to children,
    # so including under-18s creates spurious "missing" counts that inflate
    # apparent data loss. Filter first, then apply cohort criteria.
    df = df[df["RIDAGEYR"] >= 18].copy()
    print(f"Adults only (age >= 18): {df.shape[0]} rows")

    # deployment-matching cohort: people who do NOT already know they have diabetes
    df = df[df["DIQ010"].isin([2, 3])].copy()
    df["borderline_selfreport"] = (df["DIQ010"] == 3).astype(int)
    print(f"Cohort excluding self-reported diagnosed diabetics: {df.shape[0]} rows")
    df = df[df["has_lab_label"]]
    print(f"Cohort with a valid lab label (HbA1c or glucose present): {df.shape[0]} rows")

    df = clean_predictors(df)
    df = df.dropna(subset=PREDICTOR_COLS)
    print(f"Cohort with complete predictors: {df.shape[0]} rows")

    out_path = PROCESSED / "diabetes_analytic_dataset.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    print(f"\nLab-confirmed diabetes prevalence (undiagnosed, by definition): "
          f"{df['diabetes_lab'].mean():.3f}  (n={int(df['diabetes_lab'].sum())})")
    print(f"Lab-confirmed prediabetes prevalence: "
          f"{df['prediabetes_lab'].mean():.3f}  (n={int(df['prediabetes_lab'].sum())})")


if __name__ == "__main__":
    main()
