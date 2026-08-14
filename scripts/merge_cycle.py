"""
merge_cycle.py

Merges the NHANES component files for ONE cycle into a single wide table,
joined on SEQN. Covers both the CKD-mortality project (DEMO, BIOPRO,
ALB_CR, KIQ_U, DIQ, BPQ, BMX) and the diabetes-screening project (adds
MCQ, PAQ, SMQ, GHB, GLU) -- unused components for a given project can just
be ignored downstream.

Usage:
    python merge_cycle.py 2007-2008 E
    python merge_cycle.py 2009-2010 F
    ... etc for each cycle

Reads from:  raw/<cycle_years>/<COMPONENT>_<letter>.XPT
Writes to:   processed/cycle_<cycle_years>.parquet

NOTE: if you already ran this script for the CKD project and only need
to add the new diabetes-screening columns, just re-run it per cycle --
it re-merges everything from scratch and overwrites the existing
cycle_<cycle_years>.parquet with the fuller version (safe to re-run,
nothing downstream depends on the old narrower file).
"""

import sys
import pandas as pd
from pathlib import Path

BASE      = Path(__file__).resolve().parent.parent
RAW       = BASE / "raw"
PROCESSED = BASE / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# component file prefixes we need, and which columns to keep from each
# (SEQN is always kept automatically as the join key)
COMPONENTS = {
    # --- shared / CKD project ---
    "DEMO":   ["RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR",
               "WTMEC2YR", "WTINT2YR", "SDMVPSU", "SDMVSTRA"],
    "BIOPRO": ["LBXSCR"],                      # serum creatinine, mg/dL
    "ALB_CR": ["URXUMA", "URXUCR"],             # urine albumin (ug/mL), urine creatinine (mg/dL)
    "KIQ_U":  ["KIQ022", "KIQ025"],             # weak/failing kidneys; dialysis in past 12mo
    "DIQ":    ["DIQ010"],                        # diabetes diagnosis (self-report)
    "BPQ":    ["BPQ020", "BPQ040A"],             # hypertension diagnosis, taking meds
    "BMX":    ["BMXBMI", "BMXWAIST"],            # BMI, waist circumference

    # --- diabetes screening project (new) ---
    "MCQ":    ["MCQ300C"],                       # close relative had diabetes
    "PAQ":    ["PAQ605", "PAQ610", "PAD615",     # vigorous work: y/n, days/wk, min/day
               "PAQ620", "PAQ625", "PAD630",     # moderate work
               "PAQ635", "PAQ640", "PAD645",     # active transport
               "PAQ650", "PAQ655", "PAD660",     # vigorous recreation
               "PAQ665", "PAQ670", "PAD675",     # moderate recreation
               "PAD680"],                        # sedentary minutes/day
    "SMQ":    ["SMQ020", "SMQ040"],              # ever smoked 100 cigs; current smoking status
    "GHB":    ["LBXGH"],                         # HbA1c (%) -- primary label source
    "GLU":    ["LBXGLU", "WTSAF2YR"],            # fasting glucose (mg/dL) + subsample weight
    "SLQ":    ["SLD010H", "SLD012"],             # sleep hours -- variable RENAMED across cycles,
                                                 # SLD010H used 2007-2014, SLD012 used 2015-2018.
                                                 # Both pulled here; unified into sleep_hours below.
    "DR1TOT": ["DR1TKCAL", "DR1TSUGR",
               "DR1TFIBE", "DR1TCARB"],          # energy, sugar, fiber, carbs
}


def load_xpt(path: Path, keep_cols: list) -> pd.DataFrame:
    """Load a .XPT file and keep only SEQN + requested columns that exist."""
    df = pd.read_sas(path, format="xport")
    # SEQN sometimes comes back as float; normalize to int for clean joins
    df["SEQN"] = df["SEQN"].astype(int)
    available = [c for c in keep_cols if c in df.columns]
    missing   = set(keep_cols) - set(available)
    if missing:
        print(f"  [warn] {path.name}: missing columns {missing}")
    return df[["SEQN"] + available]


def merge_cycle(cycle_years: str, letter: str) -> pd.DataFrame:
    cycle_dir = RAW / cycle_years
    merged    = None

    for prefix, cols in COMPONENTS.items():
        # try lowercase extension first, then uppercase
        fpath = cycle_dir / f"{prefix}_{letter}.xpt"
        if not fpath.exists():
            fpath = cycle_dir / f"{prefix}_{letter}.XPT"
        if not fpath.exists():
            print(f"  [warn] missing file: {prefix}_{letter}.xpt")
            continue

        df = load_xpt(fpath, cols)
        print(f"  loaded {fpath.name}: {df.shape[0]} rows, cols={list(df.columns)}")
        merged = df if merged is None else merged.merge(df, on="SEQN", how="outer")

    merged["cycle"] = cycle_years

    # unify the renamed sleep-hours variable (SLD010H pre-2015, SLD012 2015+)
    # into a single column so downstream code doesn't need to know the split
    if "SLD010H" in merged.columns or "SLD012" in merged.columns:
        sld010h = merged["SLD010H"] if "SLD010H" in merged.columns else pd.Series(
            pd.NA, index=merged.index)
        sld012  = merged["SLD012"] if "SLD012" in merged.columns else pd.Series(
            pd.NA, index=merged.index)
        merged["sleep_hours"] = sld010h.combine_first(sld012)

    return merged


def main():
    if len(sys.argv) != 3:
        print("Usage: python merge_cycle.py <cycle_years> <letter>")
        print("Example: python merge_cycle.py 2007-2008 E")
        sys.exit(1)

    cycle_years, letter = sys.argv[1], sys.argv[2].upper()
    print(f"Merging cycle {cycle_years} (suffix _{letter})...")

    merged   = merge_cycle(cycle_years, letter)
    out_path = PROCESSED / f"cycle_{cycle_years}.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"Saved {merged.shape[0]} rows x {merged.shape[1]} cols -> {out_path}")


if __name__ == "__main__":
    main()
