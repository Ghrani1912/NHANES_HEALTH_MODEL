"""
train_diabetes_model.py

Trains logistic regression + XGBoost classifiers predicting lab-confirmed
diabetes (undiagnosed, by cohort construction) from non-lab features only.

Beyond AUC, reports the metric that actually matters for a screening tool:
sensitivity (fraction of true lab-confirmed diabetics correctly flagged
high-risk) at several fixed specificity operating points, plus at the
Youden-optimal threshold. A GP using this tool cares about "how many sick
people would I miss," not overall accuracy.

CALIBRATION NOTE: both models use class-weighting (class_weight="balanced"
for LR, scale_pos_weight for XGBoost) to help with the class imbalance in
undiagnosed-diabetes prevalence (~5%). This helps ranking/discrimination
(AUC, sensitivity/specificity are unaffected) but systematically inflates
the raw predicted probabilities away from the true base rate -- a 0.8
predicted score does NOT mean an 80% real chance. Both models are wrapped
in CalibratedClassifierCV (isotonic regression, fit inside each CV fold to
avoid leakage) to correct this before any probability is reported.

Usage:
    python train_diabetes_model.py

Reads: ../processed/diabetes_analytic_dataset.parquet
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline

BASE      = Path(__file__).resolve().parent.parent   # diabetes_screening/
PROCESSED = BASE / "processed"

NUMERIC_COLS = ["age", "bmi", "waist_cm", "activity_score", "sedentary_minutes"]
BINARY_COLS  = ["female", "hypertension", "family_history_diabetes",
                "family_history_diabetes_missing",
                # ever_smoker_missing dropped — redundant with RIDAGEYR
                # (ablation: PR-AUC diff=0.0001, r=0.69 with age_18_19 indicator)
                "ever_smoker", "current_smoker"]


def build_design_matrix(df: pd.DataFrame):
    race_dummies = pd.get_dummies(df["race_eth"], prefix="race", drop_first=True)
    X = pd.concat([df[NUMERIC_COLS + BINARY_COLS], race_dummies], axis=1)
    return X


def sensitivity_at_specificity(y_true, y_score, target_specificity):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    specificity = 1 - fpr
    valid = specificity >= target_specificity
    if not valid.any():
        return np.nan
    idx  = np.where(valid)[0]
    best = idx[np.argmax(tpr[idx])]
    return tpr[best]


def youden_optimal_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j        = tpr - fpr
    best_idx = np.argmax(j)
    return thresholds[best_idx], tpr[best_idx], 1 - fpr[best_idx]


def evaluate_model(name, y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    print(f"\n--- {name} ---")
    print(f"AUC: {auc:.3f}")
    for spec_target in [0.70, 0.80, 0.90]:
        sens = sensitivity_at_specificity(y_true, y_score, spec_target)
        print(f"  Sensitivity at >= {spec_target:.0%} specificity: {sens:.3f}")
    thresh, sens_y, spec_y = youden_optimal_threshold(y_true, y_score)
    print(f"  Youden-optimal threshold: {thresh:.3f} "
          f"(sensitivity={sens_y:.3f}, specificity={spec_y:.3f})")
    print(f"  NOTE: 'threshold' here is a CALIBRATED probability -- unlike "
          f"before, this number is meant to be read as an actual risk %.")
    return auc


def cv_logistic(X: pd.DataFrame, y: pd.Series, n_splits=5):
    Xv = X.values
    yv = y.values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(yv))

    for train_idx, test_idx in skf.split(Xv, yv):
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(Xv[train_idx])
        X_test  = scaler.transform(Xv[test_idx])

        base       = LogisticRegression(max_iter=1000, class_weight="balanced")
        calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
        calibrated.fit(X_train, yv[train_idx])
        oof_preds[test_idx] = calibrated.predict_proba(X_test)[:, 1]

    evaluate_model("Logistic Regression (5-fold OOF, calibrated)", yv, oof_preds)

    # fit on full data for coefficient inspection (uncalibrated base model)
    scaler     = StandardScaler()
    X_full     = scaler.fit_transform(Xv)
    full_model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_full, yv)
    print("\nCoefficients (standardized, fit on full cohort, pre-calibration model):")
    for fname, coef in sorted(zip(X.columns, full_model.coef_[0]),
                               key=lambda x: -abs(x[1])):
        print(f"  {fname}: {coef:.4f}")

    return oof_preds


def cv_xgboost(X: pd.DataFrame, y: pd.Series, n_splits=5):
    Xv = X.values
    yv = y.values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(yv))

    scale_pos_weight = (yv == 0).sum() / max((yv == 1).sum(), 1)

    for train_idx, test_idx in skf.split(Xv, yv):
        base = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="auc",
            max_depth=4, learning_rate=0.05, n_estimators=200,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            verbosity=0,
        )
        calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
        calibrated.fit(Xv[train_idx], yv[train_idx])
        oof_preds[test_idx] = calibrated.predict_proba(Xv[test_idx])[:, 1]

    evaluate_model("XGBoost (5-fold OOF, calibrated)", yv, oof_preds)

    # fit on full data for feature importance (uncalibrated base model)
    full_model = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="auc",
        max_depth=4, learning_rate=0.05, n_estimators=200,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        verbosity=0,
    )
    full_model.fit(Xv, yv)
    importance = dict(zip(X.columns, full_model.feature_importances_))
    print("\nFeature importance (gain-based, pre-calibration model):")
    for fname, score in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {fname}: {score:.4f}")

    return oof_preds


def main():
    df = pd.read_parquet(PROCESSED / "diabetes_analytic_dataset.parquet")
    print(f"Modeling cohort: {df.shape[0]} rows")
    print(f"Lab-confirmed (undiagnosed) diabetes prevalence: {df['diabetes_lab'].mean():.3f}")

    X = build_design_matrix(df)
    y = df["diabetes_lab"].astype(int)

    log_preds = cv_logistic(X, y)
    xgb_preds = cv_xgboost(X, y)

    print("\n=== Summary ===")
    print(f"Logistic Regression AUC: {roc_auc_score(y, log_preds):.3f}")
    print(f"XGBoost AUC:             {roc_auc_score(y, xgb_preds):.3f}")
    print("\nThe sensitivity-at-fixed-specificity numbers above are the ones "
          "that answer the actual clinical question: of people who don't "
          "know they have diabetes, what fraction would this tool have "
          "flagged for a confirmatory HbA1c test? These predicted "
          "probabilities are now calibrated -- a score of 0.3 should mean "
          "roughly a 30% real chance, check this against the updated "
          "calibration plot from diagnose_diabetes_model.py.")


if __name__ == "__main__":
    main()
