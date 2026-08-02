"""
build_features.py

Takes the pooled NHANES dataset and derives the clinical features needed
for the CKD project:
  - eGFR (CKD-EPI 2021 race-free creatinine equation)
  - Urine ACR (mg/g)
  - KDIGO CKD stage (G1-G5) and albuminuria category (A1-A3)
  - Prevalent-dialysis exclusion flag
  - A baseline "renal risk" outcome placeholder

NOTE ON MORTALITY LINKAGE: this script does NOT yet merge the NCHS linked
mortality files, since those come as fixed-width .dat files, not XPT, and
need a separate parser. Until that's added, the outcome here is built from
self-report only (KIQ022/KIQ025). Treat this as scaffolding -- swap in the
real time-to-event outcome once mortality files are merged in.

Usage:
    python build_features.py

Reads:  processed/pooled.parquet
Writes: processed/analytic_dataset.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "processed"


def compute_egfr_ckdepi2021(scr_mgdl: pd.Series, age: pd.Series, sex: pd.Series) -> pd.Series:
    """
    CKD-EPI 2021 race-free creatinine equation.
    sex: NHANES RIAGENDR coding, 1 = Male, 2 = Female
    scr_mgdl: serum creatinine in mg/dL
    """
    is_female = sex == 2

    kappa = np.where(is_female, 0.7, 0.9)
    alpha = np.where(is_female, -0.241, -0.302)
    sex_mult = np.where(is_female, 1.012, 1.0)

    scr_over_kappa = scr_mgdl / kappa
    min_term = np.minimum(scr_over_kappa, 1) ** alpha
    max_term = np.maximum(scr_over_kappa, 1) ** -1.200

    egfr = 142 * min_term * max_term * (0.9938 ** age) * sex_mult
    return egfr


def kdigo_gfr_stage(egfr: float) -> str:
    if pd.isna(egfr):
        return np.nan
    if egfr >= 90:
        return "G1"
    elif egfr >= 60:
        return "G2"
    elif egfr >= 45:
        return "G3a"
    elif egfr >= 30:
        return "G3b"
    elif egfr >= 15:
        return "G4"
    else:
        return "G5"


def kdigo_albuminuria_stage(acr: float) -> str:
    if pd.isna(acr):
        return np.nan
    if acr < 30:
        return "A1"
    elif acr <= 300:
        return "A2"
    else:
        return "A3"


def main():
    df = pd.read_parquet(PROCESSED / "pooled.parquet")
    print(f"Loaded pooled dataset: {df.shape[0]} rows")

    # --- eGFR ---
    df["eGFR"] = compute_egfr_ckdepi2021(df["LBXSCR"], df["RIDAGEYR"], df["RIAGENDR"])

    # --- ACR (mg/g) ---
    # URXUMA: urine albumin, ug/mL  |  URXUCR: urine creatinine, mg/dL
    df["ACR"] = (df["URXUMA"] / df["URXUCR"]) * 100

    # --- KDIGO staging ---
    df["GFR_stage"] = df["eGFR"].apply(kdigo_gfr_stage)
    df["albuminuria_stage"] = df["ACR"].apply(kdigo_albuminuria_stage)

    # --- Prevalent dialysis flag (exclude these from "incident risk" analysis) ---
    # KIQ025 == 1 means "yes, received dialysis in past 12 months"
    df["on_dialysis_baseline"] = df["KIQ025"] == 1

    # --- Self-reported kidney failure (weak/failing kidneys ever told by doctor) ---
    # KIQ022 == 1 means "yes"
    df["self_report_kidney_failure"] = df["KIQ022"] == 1

    # --- Placeholder outcome until mortality linkage is merged in ---
    # A crude proxy: prevalent CKD stage 3b+ OR self-reported kidney failure,
    # among those NOT already on dialysis at baseline.
    eligible = ~df["on_dialysis_baseline"].fillna(False)
    high_risk = df["GFR_stage"].isin(["G3b", "G4", "G5"]) | df["self_report_kidney_failure"].fillna(False)
    df["renal_risk_outcome_placeholder"] = np.where(eligible, high_risk, np.nan)

    out_path = PROCESSED / "analytic_dataset.parquet"
    df.to_parquet(out_path, index=False)

    print(f"Saved analytic dataset: {df.shape[0]} rows x {df.shape[1]} cols -> {out_path}")
    print("\nGFR stage distribution:")
    print(df["GFR_stage"].value_counts(dropna=False))
    print("\nBaseline dialysis prevalence:", df["on_dialysis_baseline"].sum())
    print("Placeholder outcome positive rate (eligible only):",
          df.loc[eligible, "renal_risk_outcome_placeholder"].mean())


if __name__ == "__main__":
    main()
