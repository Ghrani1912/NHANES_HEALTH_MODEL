"""
diagnose_diabetes_model.py

Goes beyond AUC/sensitivity to find WHERE the model fails, not just how
well it scores overall. Produces:

1. Calibration plot -- does a "70% risk" score mean 70% real risk?
2. Precision-Recall curve -- how many people need an HbA1c test per
   true case caught, at your chosen operating point?
3. Subgroup analysis -- is the model systematically blind to "lean"
   diabetics (normal BMI/waist), since it leans heavily on those
   features? This is a well-known hard subgroup in T2D screening.
4. Missingness check -- how many rows get dropped by dropna on
   predictors, and does that missingness correlate with the outcome
   (i.e. are we silently excluding a biased subset)?

Usage:
    python diagnose_diabetes_model.py

Reads:  ../processed/pooled.parquet (from main CKD pipeline),
        ../processed/diabetes_analytic_dataset.parquet
Writes: ../processed/diagnostics/*.png (plots), prints report to console
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
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

BASE           = Path(__file__).resolve().parent.parent
PROCESSED      = BASE / "processed"
PROCESSED_IN   = BASE.parent / "processed"   # main CKD pipeline pooled.parquet
DIAG_DIR       = PROCESSED / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_COLS = ["age", "bmi", "waist_cm", "met_minutes_total", "sedentary_minutes",
                "calories", "sugar_g", "fiber_g", "carbs_g", "sleep_hours", "income_ratio"]
BINARY_COLS  = ["female", "hypertension", "family_history_diabetes",
                "ever_smoker", "current_smoker"]


def build_design_matrix(df: pd.DataFrame):
    race_dummies = pd.get_dummies(df["race_eth"], prefix="race", drop_first=True)
    X = pd.concat([df[NUMERIC_COLS + BINARY_COLS], race_dummies], axis=1)
    return X


def get_oof_predictions(df: pd.DataFrame, n_splits=5):
    X  = build_design_matrix(df)
    y  = df["diabetes_lab"].astype(int).values
    Xv = X.values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    log_oof = np.zeros(len(y))
    xgb_oof = np.zeros(len(y))
    log_fold_aucs, xgb_fold_aucs = [], []

    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    for train_idx, test_idx in skf.split(Xv, y):
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(Xv[train_idx])
        X_test  = scaler.transform(Xv[test_idx])

        lr_base = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr      = CalibratedClassifierCV(lr_base, method="isotonic", cv=3)
        lr.fit(X_train, y[train_idx])
        preds = lr.predict_proba(X_test)[:, 1]
        log_oof[test_idx] = preds
        log_fold_aucs.append(roc_auc_score(y[test_idx], preds))

        xgb_base  = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="auc",
            max_depth=4, learning_rate=0.05, n_estimators=200,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight, verbosity=0,
        )
        xgb_model = CalibratedClassifierCV(xgb_base, method="isotonic", cv=3)
        xgb_model.fit(Xv[train_idx], y[train_idx])
        preds = xgb_model.predict_proba(Xv[test_idx])[:, 1]
        xgb_oof[test_idx] = preds
        xgb_fold_aucs.append(roc_auc_score(y[test_idx], preds))

    print("\n=== Per-fold AUC stability ===")
    print(f"Logistic Regression: {np.mean(log_fold_aucs):.3f} +/- {np.std(log_fold_aucs):.3f}  "
          f"(folds: {[round(a, 3) for a in log_fold_aucs]})")
    print(f"XGBoost:             {np.mean(xgb_fold_aucs):.3f} +/- {np.std(xgb_fold_aucs):.3f}  "
          f"(folds: {[round(a, 3) for a in xgb_fold_aucs]})")

    return y, log_oof, xgb_oof


def plot_calibration(y, log_oof, xgb_oof):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, preds in [("Logistic Regression", log_oof), ("XGBoost", xgb_oof)]:
        frac_pos, mean_pred = calibration_curve(y, preds, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted risk (per bin)")
    ax.set_ylabel("Observed fraction with diabetes")
    ax.set_title("Calibration: predicted risk vs. observed frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DIAG_DIR / "calibration.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved calibration.png")


def plot_pr_curve(y, log_oof, xgb_oof):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, preds in [("Logistic Regression", log_oof), ("XGBoost", xgb_oof)]:
        precision, recall, _ = precision_recall_curve(y, preds)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, label=f"{name} (AUC={pr_auc:.3f})")
    baseline = y.mean()
    ax.axhline(baseline, linestyle="--", color="gray",
               label=f"Baseline prevalence ({baseline:.3f})")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DIAG_DIR / "precision_recall.png", dpi=150)
    plt.close(fig)
    print(f"Saved precision_recall.png")


def subgroup_analysis(df: pd.DataFrame, y, log_oof):
    is_male      = df["female"].values == 0
    normal_waist = np.where(is_male,
                            df["waist_cm"].values < 94,
                            df["waist_cm"].values < 80)
    lean = (df["bmi"].values < 25) & normal_waist

    fpr, tpr, thresholds = roc_curve(y, log_oof)
    specificity = 1 - fpr
    valid = specificity >= 0.80
    idx   = np.where(valid)[0]
    best  = idx[np.argmax(tpr[idx])]
    threshold = thresholds[best]
    flagged   = log_oof >= threshold

    print("\n=== Subgroup analysis: lean vs. non-lean diabetics ===")
    print(f"Operating threshold (80% specificity on full cohort): {threshold:.3f}")
    print(f"Lean subgroup size: {lean.sum()}  |  Non-lean subgroup size: {(~lean).sum()}")

    for label, mask in [("Lean (normal BMI + normal waist)", lean), ("Non-lean", ~lean)]:
        sub_y       = y[mask]
        sub_flagged = flagged[mask]
        n_diabetic  = sub_y.sum()
        if n_diabetic == 0:
            print(f"{label}: no diabetic cases, skipping")
            continue
        sensitivity = sub_flagged[sub_y == 1].mean()
        print(f"{label}: n_diabetic={n_diabetic}, sensitivity={sensitivity:.3f}")

    print("\nIf lean sensitivity is meaningfully lower than non-lean, the model "
          "is leaning on obesity-adjacent features and missing 'metabolically "
          "obese, normal weight' cases -- a real, reportable limitation, not a bug.")


def missingness_check(pooled_path: Path):
    df = pd.read_parquet(pooled_path)
    df = df[df["RIDAGEYR"] >= 18].copy()
    df = df[df["DIQ010"].isin([2, 3])].copy()
    has_label = df["LBXGH"].notna() | df["LBXGLU"].notna()
    df = df[has_label].copy()

    hba1c_diabetes   = df["LBXGH"]  >= 6.5
    glucose_diabetes = df["LBXGLU"] >= 126
    df["diabetes_lab"] = (hba1c_diabetes.fillna(False) | glucose_diabetes.fillna(False))

    n_total = df.shape[0]
    print("\n=== Missingness check (mirrors actual pipeline logic) ===")
    print(f"Eligible adult cohort (has label, not self-reported diabetic): {n_total}")

    for col, lbl in [("BPQ020", "hypertension"), ("MCQ300C", "family history"),
                     ("SMQ020", "ever smoked")]:
        n_valid = df[col].isin([1, 2]).sum()
        print(f"  Valid {lbl} ({col}) response: {n_valid} ({n_total - n_valid} dropped)")

    for col, lbl in [("BMXBMI", "bmi"), ("BMXWAIST", "waist_cm"),
                     ("PAD680", "sedentary_minutes"), ("DR1TKCAL", "calories"),
                     ("INDFMPIR", "income_ratio")]:
        miss = df[col].isna().sum() if col in df.columns else n_total
        print(f"  {lbl} ({col}) missing: {miss} ({miss/n_total:.1%})")

    eligible = (
        df["BPQ020"].isin([1, 2]) & df["MCQ300C"].isin([1, 2]) &
        df["SMQ020"].isin([1, 2]) & df["BMXBMI"].notna() &
        df["BMXWAIST"].notna() & df["PAD680"].notna() &
        (df["DR1TKCAL"].notna() if "DR1TKCAL" in df.columns else True) &
        (df["INDFMPIR"].notna() if "INDFMPIR" in df.columns else True)
    )
    n_final = eligible.sum()
    print(f"\n  TRUE final modeling cohort size: {n_final} ({n_final/n_total:.1%} of eligible)")

    if 0 < n_final < n_total:
        rate_dropped = df.loc[~eligible, "diabetes_lab"].mean()
        rate_kept    = df.loc[eligible,  "diabetes_lab"].mean()
        print(f"  Diabetes prevalence among dropped rows: {rate_dropped:.3f}")
        print(f"  Diabetes prevalence among kept rows:    {rate_kept:.3f}")
        diff = abs(rate_dropped - rate_kept)
        if diff > 0.02:
            print(f"  [flag] {diff:.3f} absolute difference -- missingness may not be random.")
        else:
            print("  Difference is small -- missingness looks close to random.")


def main():
    df = pd.read_parquet(PROCESSED / "diabetes_analytic_dataset.parquet")
    print(f"Modeling cohort: {df.shape[0]} rows")

    y, log_oof, xgb_oof = get_oof_predictions(df)
    plot_calibration(y, log_oof, xgb_oof)
    plot_pr_curve(y, log_oof, xgb_oof)
    subgroup_analysis(df, y, log_oof)
    missingness_check(PROCESSED_IN / "pooled.parquet")

    print(f"\nAll plots saved to {DIAG_DIR}/")


if __name__ == "__main__":
    main()
