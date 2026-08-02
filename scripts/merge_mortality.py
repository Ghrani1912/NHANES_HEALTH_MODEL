"""
merge_mortality.py

- Parses/loads mortality data for all 6 cycles (run parse_mortality.py first
  for each cycle).
- Merges into analytic_dataset.parquet on SEQN + cycle.
- Builds the REAL outcome: renal-cause death OR all-cause death among people
  with advanced CKD at baseline, with permth_exm as follow-up time.
- Applies correct pooled MEC exam weight (WTMEC2YR / n_cycles), per NCHS
  guidance for combining multiple continuous-NHANES cycles.

Usage:
    python merge_mortality.py

Reads:  processed/analytic_dataset.parquet, processed/mortality_*.parquet
Writes: processed/final_dataset.parquet
"""

import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "processed"

CYCLES = [
    "2007-2008", "2009-2010", "2011-2012",
    "2013-2014", "2015-2016", "2017-2018",
]
N_CYCLES = len(CYCLES)

# cycles where renal-specific UCOD_LEADING is reliably available
RENAL_CAUSE_AVAILABLE_CYCLES = {
    "2007-2008", "2009-2010", "2011-2012", "2013-2014"
}


def main():
    analytic = pd.read_parquet(PROCESSED / "analytic_dataset.parquet")
    print(f"Loaded analytic dataset: {analytic.shape[0]} rows")

    mort_frames = []
    for cycle in CYCLES:
        fpath = PROCESSED / f"mortality_{cycle}.parquet"
        if not fpath.exists():
            print(f"[warn] missing mortality file for {cycle}, run parse_mortality.py {cycle} first")
            continue
        mort_frames.append(pd.read_parquet(fpath))
    mortality = pd.concat(mort_frames, ignore_index=True)

    merged = analytic.merge(
        mortality, on=["SEQN", "cycle"], how="left", validate="one_to_one"
    )

    # eligibility: NCHS excludes under-18 and ineligible from mortality follow-up
    merged["mortality_eligible"] = merged["eligstat"] == 1

    # primary outcome (robust, all 6 cycles): all-cause death among people
    # who had advanced CKD (G3b/G4/G5) at baseline exam, not already on dialysis
    advanced_ckd = merged["GFR_stage"].isin(["G3b", "G4", "G5"])
    eligible = merged["mortality_eligible"] & (~merged["on_dialysis_baseline"].fillna(False))
    merged["cohort_advanced_ckd"] = advanced_ckd & eligible
    merged["outcome_allcause_death"] = merged["mortstat"]  # 0/1, NaN if ineligible

    # secondary outcome (only valid for 2007-2014 cycles): renal-cause death
    merged["renal_cause_available"] = merged["cycle"].isin(RENAL_CAUSE_AVAILABLE_CYCLES)
    merged["outcome_renal_death"] = merged["renal_death"].where(
        merged["renal_cause_available"], other=pd.NA
    )

    # follow-up time in months from exam date
    merged["followup_months"] = merged["permth_exm"]

    # --- pooled survey weight ---
    # NCHS guidance for combining multiple 2-year cycles of continuous NHANES:
    # divide each cycle's MEC exam weight by the number of cycles pooled.
    merged["pooled_wtmec"] = merged["WTMEC2YR"] / N_CYCLES

    out_path = PROCESSED / "final_dataset.parquet"
    merged.to_parquet(out_path, index=False)

    print(f"\nSaved final dataset: {merged.shape[0]} rows -> {out_path}")
    print("\nAdvanced-CKD cohort size:", merged["cohort_advanced_ckd"].sum())
    print("All-cause deaths in that cohort:",
          merged.loc[merged["cohort_advanced_ckd"], "outcome_allcause_death"].sum())
    print("Renal-cause deaths (2007-2014 subset only):",
          merged.loc[merged["cohort_advanced_ckd"] & merged["renal_cause_available"],
                      "outcome_renal_death"].sum())


if __name__ == "__main__":
    main()
