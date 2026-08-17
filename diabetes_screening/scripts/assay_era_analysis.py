"""
assay_era_analysis.py

Investigates whether the 2015-2016/2017-2018 performance drop is caused by
the NHANES HbA1c assay change (Tosoh G7 -> G8 in 2015) rather than a real
population change.

Three tests:
1. HbA1c distribution comparison across the assay boundary, holding
   demographics (age decile + BMI band) constant.
2. Within-era vs across-era leave-one-cycle-out validation.
3. Practical fix: add assay_era indicator as a feature and report
   G8-era-only headline numbers.

Usage:
    python assay_era_analysis.py

Reads: ../../processed/pooled.parquet,
       ../processed/diabetes_analytic_dataset.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve

BASE         = Path(__file__).resolve().parent.parent
PROCESSED    = BASE / "processed"
PROCESSED_IN = BASE.parent / "processed"   # main CKD pipeline pooled.parquet

NUMERIC_COLS = ["age", "bmi", "waist_cm", "met_minutes_total", "sedentary_minutes",
                "calories", "sugar_g", "fiber_g", "carbs_g", "sleep_hours", "income_ratio"]
BINARY_COLS  = ["female", "hypertension", "family_history_diabetes",
                "ever_smoker", "current_smoker"]

PRE_G8_CYCLES = ["2007-2008", "2009-2010", "2011-2012", "2013-2014"]
G8_CYCLES     = ["2015-2016", "2017-2018"]


def build_X(df, extra_cols=None):
    race_dummies = pd.get_dummies(df["race_eth"], prefix="race", drop_first=True)
    cols = NUMERIC_COLS + BINARY_COLS + (extra_cols or [])
    available = [c for c in cols if c in df.columns]
    return pd.concat([df[available], race_dummies], axis=1)


def fit_lr(X_tr, y_tr):
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_tr)
    base = LogisticRegression(max_iter=1000, class_weight="balanced")
    cal  = CalibratedClassifierCV(base, method="isotonic", cv=3)
    cal.fit(Xt, y_tr)
    return cal, scaler


def eval_lr(model, scaler, X_te, y_te):
    probs = model.predict_proba(scaler.transform(X_te))[:, 1]
    if y_te.sum() == 0:
        return np.nan, np.nan
    auc  = roc_auc_score(y_te, probs)
    fpr, tpr, _ = roc_curve(y_te, probs)
    spec = 1 - fpr
    valid = spec >= 0.80
    sens = tpr[np.where(valid)[0][np.argmax(tpr[np.where(valid)[0]])]] if valid.any() else np.nan
    return auc, sens


# ── Part 1: HbA1c distribution across assay boundary ─────────────────────────

def part1_hba1c_distribution():
    print("\n" + "="*60)
    print("PART 1: HbA1c distribution across assay boundary")
    print("="*60)

    pooled = pd.read_parquet(PROCESSED_IN / "pooled.parquet")
    pooled = pooled[pooled["RIDAGEYR"] >= 18].copy()
    pooled = pooled[pooled["DIQ010"].isin([2, 3])].copy()
    pooled = pooled[pooled["LBXGH"].notna()].copy()
    pooled["assay_era"]  = pooled["cycle"].isin(G8_CYCLES).astype(int)
    pooled["age_decile"] = pd.qcut(pooled["RIDAGEYR"], q=10, labels=False)
    pooled["bmi_band"]   = pd.qcut(pooled["BMXBMI"],   q=5,  labels=False,
                                    duplicates="drop")
    matched = pooled.dropna(subset=["age_decile", "bmi_band", "BMXBMI"])
    print(f"Matched pool (adults, non-diagnosed, HbA1c present): {len(matched)}")

    for era, label in [(0, "Pre-G8 (2007-2014)"), (1, "G8 (2015-2018)")]:
        sub = matched[matched["assay_era"] == era]
        dm_prev = (sub["LBXGH"] >= 6.5).mean()
        print(f"  {label}: n={len(sub):5d}, "
              f"mean HbA1c={sub['LBXGH'].mean():.3f}, "
              f"diabetes_prev={dm_prev:.3f}")

    # within matched cells
    cell_hba1c = (matched
                  .groupby(["age_decile", "bmi_band", "assay_era"])["LBXGH"]
                  .mean()
                  .unstack("assay_era")
                  .rename(columns={0: "pre_G8", 1: "G8"})
                  .dropna())
    cell_hba1c["diff"] = cell_hba1c["G8"] - cell_hba1c["pre_G8"]

    cell_dm = (matched
               .groupby(["age_decile", "bmi_band", "assay_era"])
               .apply(lambda x: (x["LBXGH"] >= 6.5).mean(), include_groups=False)
               .unstack("assay_era")
               .rename(columns={0: "pre_G8", 1: "G8"})
               .dropna())
    cell_dm["diff"] = cell_dm["G8"] - cell_dm["pre_G8"]

    print(f"\n  HbA1c difference (G8 - pre-G8) within matched age/BMI cells:")
    print(f"    Mean diff : {cell_hba1c['diff'].mean():+.4f}  "
          f"(negative = G8 reads lower on average)")
    print(f"    Std  diff : {cell_hba1c['diff'].std():.4f}")
    print(f"    % cells where G8 reads lower: {(cell_hba1c['diff'] < 0).mean():.1%}")

    print(f"\n  Diabetes prevalence difference (G8 - pre-G8) within matched cells:")
    print(f"    Mean diff : {cell_dm['diff'].mean():+.4f}")
    print(f"    % cells with lower G8 prevalence: {(cell_dm['diff'] < 0).mean():.1%}")

    if cell_hba1c["diff"].mean() < -0.05:
        print("\n  [FINDING] G8 assay reads systematically LOWER than pre-G8 in "
              "matched cells — consistent with documented Tosoh G8 calibration "
              "shift. This likely explains lower diabetes prevalence in 2015+ "
              "and the performance drop when pre-G8 models are applied to G8 data.")
    else:
        print("\n  [FINDING] No strong systematic HbA1c shift detected — "
              "performance drop may reflect a real population trend rather "
              "than assay drift.")


# ── Part 2: Within-era vs across-era validation ───────────────────────────────

def part2_within_vs_across_era(df):
    print("\n" + "="*60)
    print("PART 2: Within-era vs across-era leave-one-cycle-out validation")
    print("="*60)

    cycles_pre = [c for c in PRE_G8_CYCLES if c in df["cycle"].values]
    cycles_g8  = [c for c in G8_CYCLES     if c in df["cycle"].values]

    def loco(cycles, label):
        print(f"\n  --- {label} ---")
        print(f"  {'Holdout':<14} {'n_test':>7} {'AUC':>7} {'Sens@80%':>10}")
        aucs = []
        for holdout in cycles:
            train_cycles = [c for c in cycles if c != holdout]
            if not train_cycles:
                continue
            tr = df[df["cycle"].isin(train_cycles)]
            te = df[df["cycle"] == holdout]
            X_tr = build_X(tr).values
            y_tr = tr["diabetes_lab"].astype(int).values
            X_te = build_X(te).values
            y_te = te["diabetes_lab"].astype(int).values
            # align columns
            tr_cols = list(build_X(tr).columns)
            te_cols = list(build_X(te).columns)
            all_cols = sorted(set(tr_cols) | set(te_cols))
            def align(subset):
                X = build_X(subset)
                for c in all_cols:
                    if c not in X.columns:
                        X[c] = 0
                return X[all_cols].values
            X_tr = align(tr)
            X_te = align(te)
            model, scaler = fit_lr(X_tr, y_tr)
            auc, sens = eval_lr(model, scaler, X_te, y_te)
            aucs.append(auc)
            print(f"  {holdout:<14} {len(y_te):>7} {auc:>7.3f} {sens:>10.3f}")
        if aucs:
            print(f"  Mean AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
        return aucs

    pre_aucs = loco(cycles_pre, "Within pre-G8 era (2007-2014)")
    g8_aucs  = loco(cycles_g8,  "Within G8 era (2015-2018)")

    # Cross-era: train on all pre-G8, test each G8 cycle
    print(f"\n  --- Cross-era: train on ALL pre-G8, test each G8 cycle ---")
    print(f"  {'Holdout':<14} {'n_test':>7} {'AUC':>7} {'Sens@80%':>10}")
    cross_aucs = []
    tr = df[df["cycle"].isin(cycles_pre)]
    for holdout in cycles_g8:
        te = df[df["cycle"] == holdout]
        tr_cols  = list(build_X(tr).columns)
        te_cols  = list(build_X(te).columns)
        all_cols = sorted(set(tr_cols) | set(te_cols))
        def align2(subset):
            X = build_X(subset)
            for c in all_cols:
                if c not in X.columns:
                    X[c] = 0
            return X[all_cols].values
        model, scaler = fit_lr(align2(tr),
                               tr["diabetes_lab"].astype(int).values)
        auc, sens = eval_lr(model, scaler,
                            align2(te),
                            te["diabetes_lab"].astype(int).values)
        cross_aucs.append(auc)
        print(f"  {holdout:<14} {len(te):>7} {auc:>7.3f} {sens:>10.3f}")
    print(f"  Mean cross-era AUC: {np.mean(cross_aucs):.3f} ± {np.std(cross_aucs):.3f}")

    if pre_aucs and g8_aucs and cross_aucs:
        within_mean = np.mean(pre_aucs + g8_aucs)
        cross_mean  = np.mean(cross_aucs)
        drop = within_mean - cross_mean
        print(f"\n  Within-era mean AUC:    {within_mean:.3f}")
        print(f"  Cross-era mean AUC:     {cross_mean:.3f}")
        print(f"  Performance drop:       {drop:+.3f}")
        if drop > 0.02:
            print("  [FINDING] Performance drops meaningfully when crossing the "
                  "assay boundary — supports assay drift as the primary cause.")
        else:
            print("  [FINDING] No significant cross-era drop — performance "
                  "degradation is not boundary-specific.")


# ── Part 3: Practical fix — assay_era feature + G8-only numbers ──────────────

def part3_practical_fix(df):
    print("\n" + "="*60)
    print("PART 3: Practical fix — assay_era feature + G8-only numbers")
    print("="*60)

    df = df.copy()
    df["assay_era"] = df["cycle"].isin(G8_CYCLES).astype(int)

    # 3a: add assay_era as a feature
    print("\n  3a. Model WITH assay_era indicator feature:")
    X_era = build_X(df, extra_cols=["assay_era"])
    y     = df["diabetes_lab"].astype(int).values
    from sklearn.model_selection import StratifiedKFold
    skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof   = np.zeros(len(y))
    Xv    = X_era.values
    for tr, te in skf.split(Xv, y):
        model, scaler = fit_lr(Xv[tr], y[tr])
        oof[te] = model.predict_proba(scaler.transform(Xv[te]))[:, 1]
    auc_with_era = roc_auc_score(y, oof)
    print(f"    Pooled AUC with assay_era feature: {auc_with_era:.3f}")

    # 3b: G8-era-only numbers
    print("\n  3b. G8-era-only (2015-2018) performance — most relevant for deployment:")
    g8_df = df[df["assay_era"] == 1]
    print(f"    Cohort: {len(g8_df)} rows  |  "
          f"diabetes prevalence: {g8_df['diabetes_lab'].mean():.3f}")

    # LOCO within G8 only
    g8_cycles = [c for c in G8_CYCLES if c in g8_df["cycle"].values]
    g8_aucs   = []
    for holdout in g8_cycles:
        tr = g8_df[g8_df["cycle"] != holdout]
        te = g8_df[g8_df["cycle"] == holdout]
        if len(tr) == 0 or len(te) == 0:
            continue
        all_cols = sorted(set(build_X(tr).columns) | set(build_X(te).columns))
        def align3(sub):
            X = build_X(sub)
            for c in all_cols:
                if c not in X.columns:
                    X[c] = 0
            return X[all_cols].values
        model, scaler = fit_lr(align3(tr), tr["diabetes_lab"].astype(int).values)
        auc, sens = eval_lr(model, scaler,
                            align3(te), te["diabetes_lab"].astype(int).values)
        g8_aucs.append(auc)
        print(f"    {holdout} holdout: AUC={auc:.3f}, Sens@80%={sens:.3f}")

    if g8_aucs:
        print(f"\n  G8-era-only mean AUC: {np.mean(g8_aucs):.3f} ± {np.std(g8_aucs):.3f}")
        print("  These are the headline numbers to lead with for real-world deployment.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_parquet(PROCESSED / "diabetes_analytic_dataset.parquet")
    print(f"Dataset: {len(df)} rows, cycles: {sorted(df['cycle'].unique())}")

    part1_hba1c_distribution()
    part2_within_vs_across_era(df)
    part3_practical_fix(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
