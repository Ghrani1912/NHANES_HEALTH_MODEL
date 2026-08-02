"""
train_model.py

Trains two models on the advanced-CKD cohort:
  1. Baseline: logistic regression predicting all-cause death within
     follow-up (binary, ignores exact timing).
  2. Main model: Cox proportional hazards model using actual follow-up
     time (followup_months) and event indicator (outcome_allcause_death) --
     this is the more clinically correct model since it uses time-to-event,
     not just a yes/no label.

Usage:
    python train_model.py

Reads:  processed/final_dataset.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "processed"

FEATURES = ["eGFR", "ACR", "RIDAGEYR", "RIAGENDR", "DIQ010", "BPQ020", "BMXBMI"]


def prepare_cohort(df: pd.DataFrame) -> pd.DataFrame:
    cohort = df[df["cohort_advanced_ckd"]].copy()
    cohort = cohort.dropna(subset=FEATURES + ["outcome_allcause_death", "followup_months"])

    # RIAGENDR (1/2) and DIQ010/BPQ020 (1=yes,2=no,7/9=refused/unknown) --
    # recode DIQ010/BPQ020 to clean binary, drop refused/unknown/missing
    cohort = cohort[cohort["DIQ010"].isin([1, 2])]
    cohort = cohort[cohort["BPQ020"].isin([1, 2])]
    cohort["diabetes"] = (cohort["DIQ010"] == 1).astype(int)
    cohort["hypertension"] = (cohort["BPQ020"] == 1).astype(int)
    cohort["female"] = (cohort["RIAGENDR"] == 2).astype(int)

    return cohort


def run_logistic_baseline(cohort: pd.DataFrame):
    X = cohort[["eGFR", "ACR", "RIDAGEYR", "female", "diabetes", "hypertension", "BMXBMI"]]
    y = cohort["outcome_allcause_death"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    print("\n=== Logistic Regression Baseline ===")
    print(f"Test AUC: {auc:.3f}")
    print("Coefficients:")
    for name, coef in zip(X.columns, model.coef_[0]):
        print(f"  {name}: {coef:.4f}")

    return model, auc


def run_cox_model(cohort: pd.DataFrame):
    cox_df = cohort[["followup_months", "outcome_allcause_death",
                      "eGFR", "ACR", "RIDAGEYR", "female",
                      "diabetes", "hypertension", "BMXBMI"]].copy()
    cox_df = cox_df.rename(columns={
        "followup_months": "duration",
        "outcome_allcause_death": "event"
    })

    train_df, test_df = train_test_split(cox_df, test_size=0.25, random_state=42)

    cph = CoxPHFitter()
    cph.fit(train_df, duration_col="duration", event_col="event")

    print("\n=== Cox Proportional Hazards Model ===")
    cph.print_summary()

    c_index = concordance_index(
        test_df["duration"],
        -cph.predict_partial_hazard(test_df),
        test_df["event"]
    )
    print(f"\nTest concordance index (C-index): {c_index:.3f}")

    return cph, c_index


def main():
    df = pd.read_parquet(PROCESSED / "final_dataset.parquet")
    cohort = prepare_cohort(df)
    print(f"Modeling cohort (advanced CKD, complete cases): {cohort.shape[0]} rows")

    if cohort.shape[0] < 50:
        print("[warn] very small cohort -- results will be unstable. "
              "Check that eGFR/ACR are populating correctly before trusting metrics.")

    run_logistic_baseline(cohort)
    run_cox_model(cohort)


if __name__ == "__main__":
    main()
