"""
evaluate_model.py

Produces a comprehensive evaluation report covering:
1. Calibration  -- Brier score + calibration plot (already saved by diagnose script)
2. Confidence intervals -- bootstrap 95% CI on AUC and sensitivity@80%spec
3. External validation  -- leave-one-cycle-out (each cycle as holdout)
4. Subgroup performance -- age groups, sex, race/ethnicity, lean vs non-lean

Usage:
    python evaluate_model.py

Reads: ../processed/diabetes_analytic_dataset.parquet
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                              average_precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

BASE      = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "processed"
DIAG_DIR  = PROCESSED / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_COLS = ["age", "bmi", "waist_cm", "met_minutes_total", "sedentary_minutes",
                "calories", "sugar_g", "fiber_g", "carbs_g", "sleep_hours", "income_ratio"]
BINARY_COLS  = ["female", "hypertension", "family_history_diabetes",
                "ever_smoker", "current_smoker"]


def build_X(df: pd.DataFrame) -> pd.DataFrame:
    race_dummies = pd.get_dummies(df["race_eth"], prefix="race", drop_first=True)
    return pd.concat([df[NUMERIC_COLS + BINARY_COLS], race_dummies], axis=1)


def fit_lr(X_train, y_train):
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train)
    base = LogisticRegression(max_iter=1000, class_weight="balanced")
    cal  = CalibratedClassifierCV(base, method="isotonic", cv=3)
    cal.fit(Xt, y_train)
    return cal, scaler


def predict_lr(model, scaler, X_test):
    return model.predict_proba(scaler.transform(X_test))[:, 1]


def sensitivity_at_specificity(y_true, y_score, target_spec=0.80):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    spec = 1 - fpr
    valid = spec >= target_spec
    if not valid.any():
        return np.nan
    idx  = np.where(valid)[0]
    return tpr[idx[np.argmax(tpr[idx])]]


# ── 1. Calibration ────────────────────────────────────────────────────────────

def calibration_report(y, probs, model_name):
    brier = brier_score_loss(y, probs)
    # baseline brier (always predict prevalence)
    prev  = y.mean()
    brier_baseline = brier_score_loss(y, np.full_like(probs, prev))
    skill = 1 - brier / brier_baseline
    print(f"\n--- Calibration: {model_name} ---")
    print(f"  Brier score:          {brier:.4f}  (lower = better)")
    print(f"  Baseline Brier:       {brier_baseline:.4f}")
    print(f"  Brier skill score:    {skill:.4f}  (>0 = better than naive)")
    return brier, skill


# ── 2. Bootstrap confidence intervals ────────────────────────────────────────

def bootstrap_ci(y, probs, n_boot=1000, seed=42):
    rng  = np.random.default_rng(seed)
    aucs, sens = [], []
    for _ in range(n_boot):
        idx   = rng.integers(0, len(y), len(y))
        y_b   = y[idx]
        p_b   = probs[idx]
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            continue
        aucs.append(roc_auc_score(y_b, p_b))
        sens.append(sensitivity_at_specificity(y_b, p_b, 0.80))
    auc_ci  = (np.percentile(aucs, 2.5), np.percentile(aucs, 97.5))
    sens_ci = (np.nanpercentile(sens, 2.5), np.nanpercentile(sens, 97.5))
    return auc_ci, sens_ci


# ── 3. External validation: leave-one-cycle-out ───────────────────────────────

def leave_one_cycle_out(df):
    print("\n--- External validation: leave-one-cycle-out ---")
    print(f"{'Holdout cycle':<14} {'n_test':>7} {'events':>7} {'AUC':>7} {'Sens@80%':>10}")
    print("-" * 50)

    cycles = sorted(df["cycle"].unique())
    results = []
    for holdout in cycles:
        train_df = df[df["cycle"] != holdout]
        test_df  = df[df["cycle"] == holdout]

        X_train = build_X(train_df).values
        y_train = train_df["diabetes_lab"].astype(int).values
        X_test  = build_X(test_df).values
        y_test  = test_df["diabetes_lab"].astype(int).values

        # align columns (one-hot may differ between splits)
        train_cols = list(build_X(train_df).columns)
        test_cols  = list(build_X(test_df).columns)
        all_cols   = sorted(set(train_cols) | set(test_cols))
        def align(df_sub, cols):
            X = build_X(df_sub)
            for c in cols:
                if c not in X.columns:
                    X[c] = 0
            return X[cols].values

        X_train = align(train_df, all_cols)
        X_test  = align(test_df,  all_cols)

        model, scaler = fit_lr(X_train, y_train)
        probs = predict_lr(model, scaler, X_test)

        if y_test.sum() == 0:
            print(f"{holdout:<14} {len(y_test):>7} {int(y_test.sum()):>7}  {'N/A':>7}  {'N/A':>10}")
            continue

        auc  = roc_auc_score(y_test, probs)
        sens = sensitivity_at_specificity(y_test, probs, 0.80)
        results.append({"cycle": holdout, "auc": auc, "sens_80": sens})
        print(f"{holdout:<14} {len(y_test):>7} {int(y_test.sum()):>7} {auc:>7.3f} {sens:>10.3f}")

    if results:
        aucs = [r["auc"] for r in results]
        print(f"\n  Mean AUC across cycles: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
        print(f"  Range: [{min(aucs):.3f}, {max(aucs):.3f}]")
    return results


# ── 4. Subgroup performance ───────────────────────────────────────────────────

def subgroup_performance(df, probs, y):
    print("\n--- Subgroup performance (Logistic Regression OOF predictions) ---")

    subgroups = {
        "Age 18-44":           df["age"] < 45,
        "Age 45-64":           (df["age"] >= 45) & (df["age"] < 65),
        "Age 65+":             df["age"] >= 65,
        "Male":                df["female"] == 0,
        "Female":              df["female"] == 1,
        "Non-Hispanic White":  df["race_eth"] == 3,
        "Non-Hispanic Black":  df["race_eth"] == 4,
        "Mexican American":    df["race_eth"] == 1,
        "Other Hispanic":      df["race_eth"] == 2,
        "Other/Multi-racial":  df["race_eth"] == 5,
        "Lean (BMI<25 + normal waist)": (
            (df["bmi"] < 25) &
            np.where(df["female"] == 0,
                     df["waist_cm"] < 94,
                     df["waist_cm"] < 80)
        ),
        "Non-lean":            ~(
            (df["bmi"] < 25) &
            np.where(df["female"] == 0,
                     df["waist_cm"] < 94,
                     df["waist_cm"] < 80)
        ),
    }

    print(f"{'Subgroup':<30} {'n':>6} {'events':>7} {'AUC':>7} {'Sens@80%':>10} {'Sens@70%':>10}")
    print("-" * 72)
    for label, mask in subgroups.items():
        mask = np.array(mask)
        y_s  = y[mask]
        p_s  = probs[mask]
        if y_s.sum() < 5:
            print(f"{label:<30} {mask.sum():>6} {int(y_s.sum()):>7}  {'<5 events':>18}")
            continue
        try:
            auc  = roc_auc_score(y_s, p_s)
        except Exception:
            auc = np.nan
        s80  = sensitivity_at_specificity(y_s, p_s, 0.80)
        s70  = sensitivity_at_specificity(y_s, p_s, 0.70)
        print(f"{label:<30} {mask.sum():>6} {int(y_s.sum()):>7} {auc:>7.3f} "
              f"{s80 if s80 is not np.nan else float('nan'):>10.3f} "
              f"{s70 if s70 is not np.nan else float('nan'):>10.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_parquet(PROCESSED / "diabetes_analytic_dataset.parquet")
    print(f"Cohort: {len(df)} rows  |  diabetes prevalence: {df['diabetes_lab'].mean():.3f}")

    # get OOF predictions from 5-fold CV (LR only — primary model)
    X  = build_X(df)
    y  = df["diabetes_lab"].astype(int).values
    Xv = X.values

    skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y))
    for tr, te in skf.split(Xv, y):
        model, scaler = fit_lr(Xv[tr], y[tr])
        oof_probs[te] = predict_lr(model, scaler, Xv[te])

    overall_auc = roc_auc_score(y, oof_probs)
    overall_ap  = average_precision_score(y, oof_probs)
    print(f"\nOverall OOF AUC: {overall_auc:.3f}")
    print(f"Overall OOF PR-AUC (avg precision): {overall_ap:.3f}")

    # ── 1. Calibration ────────────────────────────────────────────────────────
    print("\n=== 1. CALIBRATION ===")
    calibration_report(y, oof_probs, "Logistic Regression (OOF, calibrated)")

    # ── 2. Bootstrap CIs ──────────────────────────────────────────────────────
    print("\n=== 2. CONFIDENCE INTERVALS (bootstrap n=1000) ===")
    auc_ci, sens_ci = bootstrap_ci(y, oof_probs)
    print(f"  AUC:              {overall_auc:.3f}  95% CI [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")
    s80 = sensitivity_at_specificity(y, oof_probs, 0.80)
    print(f"  Sensitivity@80%:  {s80:.3f}  95% CI [{sens_ci[0]:.3f}, {sens_ci[1]:.3f}]")

    # ── 3. External validation ────────────────────────────────────────────────
    print("\n=== 3. EXTERNAL VALIDATION (leave-one-cycle-out) ===")
    leave_one_cycle_out(df)

    # ── 4. Subgroup performance ───────────────────────────────────────────────
    print("\n=== 4. SUBGROUP PERFORMANCE ===")
    subgroup_performance(df, oof_probs, y)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
