"""
train_model_v2.py

Changes from previous version:
  - Option 1: Broadened cohort from advanced CKD only (G3b+) to full CKD
    spectrum (G1-G5). This removes the survivorship bias that was causing
    hypertension and diabetes to appear protective.
  - Option 2: Cause-specific Cox models — one for all-cause death and one
    for renal-specific death (ucod_leading == 10). Comparing the two reveals
    whether HTN/diabetes have different effects on renal vs non-renal mortality.
  - Retains logistic regression and XGBoost as comparison models.
  - All metrics logged to Weights & Biases.

Usage:
    python train_model_v2.py

Reads: processed/final_dataset.parquet
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import wandb
import joblib
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

BASE      = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "processed"
MODELS    = BASE / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# ── config ────────────────────────────────────────────────────────────────────
WANDB_PROJECT = "nhanes-ckd"
WANDB_ENTITY  = None

N_SPLITS  = 5
XGB_PARAMS = {
    "objective":        "survival:cox",
    "eval_metric":      "cox-nloglik",
    "max_depth":        3,
    "eta":              0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
}
XGB_ROUNDS = 200

RAW_FEATURES = ["eGFR", "ACR", "RIDAGEYR", "RIAGENDR", "DIQ010", "BPQ020", "BPQ040A", "BMXBMI"]

# htn_untreated is perfectly collinear with htn_treated in this cohort (everyone
# is diagnosed, so treated + untreated = 1 always). Use only htn_treated in Cox
# and XGBoost. Logistic regression uses both since sklearn handles it via
# regularization, but we keep feature sets consistent for clarity.
FEATURE_COLS     = ["eGFR", "log_ACR", "RIDAGEYR", "female",
                    "diabetes", "htn_treated", "BMXBMI"]
# kept separate for reference in logistic regression output only
FEATURE_COLS_LR  = ["eGFR", "log_ACR", "RIDAGEYR", "female",
                    "diabetes", "htn_treated", "BMXBMI"]
# ──────────────────────────────────────────────────────────────────────────────


def prepare_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Option 1: Full CKD spectrum (G1-G5) instead of advanced CKD only.

    Rationale: restricting to advanced CKD creates survivorship bias where
    hypertension/diabetes appear protective because people who died from those
    conditions early never reached advanced CKD and are absent from the cohort.
    Using the full spectrum avoids this truncation.

    Note on hypertension encoding: in this dataset everyone with complete CKD
    data has a hypertension diagnosis (BPQ020==1), so a never-diagnosed
    reference group does not exist. We therefore encode treatment status as a
    single binary: htn_treated=1 (on meds) vs htn_treated=0 (diagnosed but
    not on meds). This is still more informative than the old single flag.
    """
    # --- full CKD spectrum: G1 through G5 ---
    cohort = df[df["GFR_stage"].isin(["G1", "G2", "G3a", "G3b", "G4", "G5"])].copy()

    # drop rows missing any required feature or outcome
    required = RAW_FEATURES + ["outcome_allcause_death", "outcome_renal_death", "followup_months"]
    cohort = cohort.dropna(subset=required)

    # keep only clean yes/no diabetes responses
    cohort = cohort[cohort["DIQ010"].isin([1, 2])]

    # drop the small number of diagnosed-hypertensive rows with unknown med status
    # BPQ040A: 1=yes taking meds, 2=no, 7=refused, 9=don't know
    htn_diagnosed   = cohort["BPQ020"] == 1
    htn_unknown_med = htn_diagnosed & (~cohort["BPQ040A"].isin([1, 2]))
    n_dropped = htn_unknown_med.sum()
    if n_dropped > 0:
        print(f"Dropping {n_dropped} diagnosed-hypertensive rows with unknown/refused med status")
    cohort = cohort[~htn_unknown_med]

    cohort["diabetes"]      = (cohort["DIQ010"] == 1).astype(int)
    # htn_treated: 1 = diagnosed + on medication, 0 = diagnosed but not on meds
    # (everyone in this cohort is diagnosed, so this captures treatment status)
    cohort["htn_treated"]   = ((cohort["BPQ020"] == 1) & (cohort["BPQ040A"] == 1)).astype(int)
    # keep htn_untreated for logistic regression (collinear with htn_treated here
    # since htn_treated + htn_untreated = 1 for all rows; drop from Cox)
    cohort["htn_untreated"] = ((cohort["BPQ020"] == 1) & (cohort["BPQ040A"] == 2)).astype(int)
    cohort["female"]        = (cohort["RIAGENDR"] == 2).astype(int)
    cohort["log_ACR"]       = np.log(cohort["ACR"] + 1)

    # outcome_renal_death may be None for ineligible rows — coerce
    cohort["outcome_renal_death"]    = cohort["outcome_renal_death"].fillna(False).infer_objects(copy=False).astype(int)
    cohort["outcome_allcause_death"] = cohort["outcome_allcause_death"].astype(int)

    return cohort


def print_subgroup_summary(cohort: pd.DataFrame):
    print(f"\nModeling cohort: {len(cohort)} rows  (full CKD spectrum G1-G5)")
    print("\nGFR stage breakdown:")
    print(cohort["GFR_stage"].value_counts().sort_index().to_string())
    print("\nHypertension treatment status (all diagnosed):")
    print(f"  On medication (htn_treated=1): {cohort['htn_treated'].sum()}")
    print(f"  Not on meds   (htn_treated=0): {cohort['htn_untreated'].sum()}")
    print(f"\nAll-cause deaths:  {cohort['outcome_allcause_death'].sum()}")
    print(f"Renal deaths:      {cohort['outcome_renal_death'].sum()}")
    if cohort["outcome_renal_death"].sum() < 30:
        print("[warn] few renal death events — cause-specific Cox will be unstable.")


# ── save helpers ─────────────────────────────────────────────────────────────

def save_models(logistic_model, cox_allcause_model, cox_renal_model, xgb_model,
                feature_cols: list, feature_cols_lr: list):
    """
    Save all fitted models to the models/ directory.

    Formats:
      - Logistic regression : joblib  (.joblib)  — best for sklearn objects
      - Cox PH models       : pickle  (.pkl)      — lifelines is pickle-safe
      - XGBoost             : native  (.json)     — portable, version-stable
      - Feature column list : joblib  (.joblib)   — needed to reconstruct input
    """
    # logistic regression
    lr_path = MODELS / "logistic_regression.joblib"
    joblib.dump(logistic_model, lr_path)
    print(f"  saved logistic regression  -> {lr_path}")

    # cox all-cause
    cox_all_path = MODELS / "cox_allcause.pkl"
    with open(cox_all_path, "wb") as f:
        pickle.dump(cox_allcause_model, f)
    print(f"  saved cox all-cause        -> {cox_all_path}")

    # cox renal-specific
    cox_renal_path = MODELS / "cox_renal.pkl"
    with open(cox_renal_path, "wb") as f:
        pickle.dump(cox_renal_model, f)
    print(f"  saved cox renal-specific   -> {cox_renal_path}")

    # xgboost — native json format
    xgb_path = MODELS / "xgboost_survival.json"
    xgb_model.save_model(str(xgb_path))
    print(f"  saved xgboost              -> {xgb_path}")

    # feature column lists so downstream code knows input order
    meta = {"feature_cols": feature_cols, "feature_cols_lr": feature_cols_lr}
    meta_path = MODELS / "feature_meta.joblib"
    joblib.dump(meta, meta_path)
    print(f"  saved feature metadata     -> {meta_path}")

    # log model files as wandb artifacts so they're versioned alongside the run
    artifact = wandb.Artifact("ckd_models", type="model")
    for p in [lr_path, cox_all_path, cox_renal_path, xgb_path, meta_path]:
        artifact.add_file(str(p))
    wandb.log_artifact(artifact)
    print("  uploaded models artifact to wandb")




def cv_logistic(cohort: pd.DataFrame, n_splits=N_SPLITS):
    X = cohort[FEATURE_COLS_LR].values
    y = cohort["outcome_allcause_death"].values

    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        mdl = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        auc = roc_auc_score(y[te], mdl.predict_proba(X[te])[:, 1])
        aucs.append(auc)
        wandb.log({f"logistic/fold_{fold+1}_auc": auc})

    mean_auc, std_auc = float(np.mean(aucs)), float(np.std(aucs))
    print(f"\n=== Logistic Regression, {n_splits}-fold CV ===")
    print(f"AUC: {mean_auc:.3f} +/- {std_auc:.3f}  (folds: {[round(a,3) for a in aucs]})")

    full = LogisticRegression(max_iter=1000).fit(X, y)
    coef_dict = dict(zip(FEATURE_COLS_LR, full.coef_[0]))
    print("Coefficients (full cohort):")
    for k, v in coef_dict.items():
        print(f"  {k}: {v:.4f}")

    wandb.log({"logistic/mean_auc": mean_auc, "logistic/std_auc": std_auc})
    wandb.log({f"logistic/coef/{k}": float(v) for k, v in coef_dict.items()})
    return aucs, full


# ── Cox helpers ───────────────────────────────────────────────────────────────

def _run_cox(cox_df: pd.DataFrame, label: str, n_splits=N_SPLITS):
    """Fit and CV a CoxPH model. cox_df must have 'duration' and 'event' cols."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    c_indices = []
    for fold, (tr, te) in enumerate(kf.split(cox_df)):
        cph = CoxPHFitter()
        cph.fit(cox_df.iloc[tr], duration_col="duration", event_col="event",
                show_progress=False)
        ci = float(concordance_index(
            cox_df.iloc[te]["duration"],
            -cph.predict_partial_hazard(cox_df.iloc[te]),
            cox_df.iloc[te]["event"],
        ))
        c_indices.append(ci)
        wandb.log({f"{label}/fold_{fold+1}_cindex": ci})

    mean_c, std_c = float(np.mean(c_indices)), float(np.std(c_indices))
    print(f"\n=== Cox PH — {label}, {n_splits}-fold CV ===")
    print(f"C-index: {mean_c:.3f} +/- {std_c:.3f}  (folds: {[round(c,3) for c in c_indices]})")

    # full-data fit for coefficient table
    full_cph = CoxPHFitter()
    full_cph.fit(cox_df, duration_col="duration", event_col="event",
                 show_progress=False)
    full_cph.print_summary()

    wandb.log({
        f"{label}/mean_cindex":          mean_c,
        f"{label}/std_cindex":           std_c,
        f"{label}/concordance_full":     float(full_cph.concordance_index_),
        f"{label}/partial_aic":          float(full_cph.AIC_partial_),
        f"{label}/llr_test_stat":        float(full_cph.log_likelihood_ratio_test().test_statistic),
    })
    wandb.log({f"{label}/coef/{k}":         float(v) for k, v in full_cph.params_.items()})
    wandb.log({f"{label}/hazard_ratio/{k}": float(v) for k, v in full_cph.hazard_ratios_.items()})

    print(f"\n--- PH assumption check ({label}) ---")
    try:
        full_cph.check_assumptions(cox_df, p_value_threshold=0.05, show_plots=False)
        wandb.log({f"{label}/ph_assumption_passed": True})
    except Exception as e:
        print(f"[note] {e}")
        wandb.log({f"{label}/ph_assumption_passed": False})

    return c_indices, full_cph


def cv_cox_allcause(cohort: pd.DataFrame, n_splits=N_SPLITS):
    cox_df = cohort[["followup_months", "outcome_allcause_death"] + FEATURE_COLS].copy()
    cox_df = cox_df.rename(columns={
        "followup_months": "duration",
        "outcome_allcause_death": "event",
    })
    return _run_cox(cox_df, label="cox_allcause", n_splits=n_splits)


def cv_cox_renal(cohort: pd.DataFrame, n_splits=N_SPLITS):
    """
    Option 2: Cause-specific Cox for renal death.

    People who died of non-renal causes are treated as censored at their
    time of death (standard cause-specific approach). This isolates the
    effect of each covariate on renal-specific mortality, separate from
    competing causes like cardiovascular disease or cancer.
    """
    cox_df = cohort[["followup_months", "outcome_renal_death"] + FEATURE_COLS].copy()
    cox_df = cox_df.rename(columns={
        "followup_months":    "duration",
        "outcome_renal_death": "event",
    })
    n_renal_events = int(cox_df["event"].sum())
    print(f"\n[cause-specific Cox] renal death events: {n_renal_events}")
    if n_renal_events < 30:
        print("[warn] very few renal death events — cause-specific model will be unstable.")
    return _run_cox(cox_df, label="cox_renal", n_splits=n_splits)


# ── XGBoost ───────────────────────────────────────────────────────────────────

def cv_xgboost_survival(cohort: pd.DataFrame, n_splits=N_SPLITS):
    X        = cohort[FEATURE_COLS].values
    duration = cohort["followup_months"].values
    event    = cohort["outcome_allcause_death"].values
    y_xgb    = np.where(event == 1, duration, -duration)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    c_indices = []
    for fold, (tr, te) in enumerate(kf.split(X)):
        mdl = xgb.train(XGB_PARAMS,
                        xgb.DMatrix(X[tr], label=y_xgb[tr]),
                        num_boost_round=XGB_ROUNDS,
                        verbose_eval=False)
        ci = float(concordance_index(
            duration[te], -mdl.predict(xgb.DMatrix(X[te])), event[te]
        ))
        c_indices.append(ci)
        wandb.log({f"xgboost/fold_{fold+1}_cindex": ci})

    mean_c, std_c = float(np.mean(c_indices)), float(np.std(c_indices))
    print(f"\n=== XGBoost Survival (Cox objective), {n_splits}-fold CV ===")
    print(f"C-index: {mean_c:.3f} +/- {std_c:.3f}  (folds: {[round(c,3) for c in c_indices]})")

    full_mdl   = xgb.train(XGB_PARAMS,
                           xgb.DMatrix(X, label=y_xgb, feature_names=FEATURE_COLS),
                           num_boost_round=XGB_ROUNDS, verbose_eval=False)
    importance = full_mdl.get_score(importance_type="gain")
    print("\nFeature importance (gain):")
    for k, v in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.2f}")

    wandb.log({"xgboost/mean_cindex": mean_c, "xgboost/std_cindex": std_c})
    wandb.log({f"xgboost/importance/{k}": float(v) for k, v in importance.items()})

    imp_table = wandb.Table(
        columns=["feature", "importance_gain"],
        data=[[k, float(v)] for k, v in sorted(importance.items(), key=lambda x: -x[1])],
    )
    wandb.log({"xgboost/feature_importance_chart":
               wandb.plot.bar(imp_table, "feature", "importance_gain",
                              title="XGBoost Feature Importance (Gain)")})
    return c_indices, full_mdl


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df     = pd.read_parquet(PROCESSED / "final_dataset.parquet")
    cohort = prepare_cohort(df)
    print_subgroup_summary(cohort)

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="train_v2_fullspectrum",
        config={
            "cohort":         "full_ckd_spectrum_G1_G5",
            "n_rows":         len(cohort),
            "n_splits":       N_SPLITS,
            "features":       FEATURE_COLS,
            "xgb_params":     XGB_PARAMS,
            "xgb_rounds":     XGB_ROUNDS,
            "htn_encoding":   "treated_vs_untreated_vs_never",
            "acr_transform":  "log(ACR+1)",
        },
    )

    # cohort descriptives
    wandb.log({
        "cohort/n_rows":                 len(cohort),
        "cohort/allcause_event_rate":    float(cohort["outcome_allcause_death"].mean()),
        "cohort/renal_event_rate":       float(cohort["outcome_renal_death"].mean()),
        "cohort/n_allcause_deaths":      int(cohort["outcome_allcause_death"].sum()),
        "cohort/n_renal_deaths":         int(cohort["outcome_renal_death"].sum()),
        "cohort/median_followup_months": float(cohort["followup_months"].median()),
        "cohort/pct_htn_treated":        float(cohort["htn_treated"].mean()),
        "cohort/pct_htn_untreated":      float(cohort["htn_untreated"].mean()),
        "cohort/pct_diabetes":           float(cohort["diabetes"].mean()),
    })

    # run all models
    log_aucs, lr_model        = cv_logistic(cohort)
    cox_all_cidx, cox_all_mdl = cv_cox_allcause(cohort)
    cox_renal_cidx, cox_renal_mdl = cv_cox_renal(cohort)
    xgb_cidx, xgb_mdl         = cv_xgboost_survival(cohort)

    # summary
    print("\n=== Summary ===")
    print(f"Logistic regression AUC:        {np.mean(log_aucs):.3f} +/- {np.std(log_aucs):.3f}")
    print(f"Cox PH (all-cause) C-index:     {np.mean(cox_all_cidx):.3f} +/- {np.std(cox_all_cidx):.3f}")
    print(f"Cox PH (renal-specific) C-index:{np.mean(cox_renal_cidx):.3f} +/- {np.std(cox_renal_cidx):.3f}")
    print(f"XGBoost survival C-index:       {np.mean(xgb_cidx):.3f} +/- {np.std(xgb_cidx):.3f}")

    summary_table = wandb.Table(
        columns=["model", "metric", "mean", "std"],
        data=[
            ["Logistic Regression",   "AUC",     round(float(np.mean(log_aucs)), 4),        round(float(np.std(log_aucs)), 4)],
            ["Cox PH (all-cause)",    "C-index",  round(float(np.mean(cox_all_cidx)), 4),    round(float(np.std(cox_all_cidx)), 4)],
            ["Cox PH (renal-specific)","C-index", round(float(np.mean(cox_renal_cidx)), 4),  round(float(np.std(cox_renal_cidx)), 4)],
            ["XGBoost Survival",      "C-index",  round(float(np.mean(xgb_cidx)), 4),        round(float(np.std(xgb_cidx)), 4)],
        ],
    )
    wandb.log({"summary/model_comparison": summary_table})

    wandb.summary["logistic_auc_mean"]       = float(np.mean(log_aucs))
    wandb.summary["cox_allcause_cindex_mean"]= float(np.mean(cox_all_cidx))
    wandb.summary["cox_renal_cindex_mean"]   = float(np.mean(cox_renal_cidx))
    wandb.summary["xgboost_cindex_mean"]     = float(np.mean(xgb_cidx))

    # save all models to disk and wandb artifacts
    print("\n=== Saving models ===")
    save_models(lr_model, cox_all_mdl, cox_renal_mdl, xgb_mdl,
                FEATURE_COLS, FEATURE_COLS_LR)

    wandb.finish()
    print(f"\nwandb run complete. View at: {run.url}")


if __name__ == "__main__":
    main()
