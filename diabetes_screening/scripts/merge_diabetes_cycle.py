"""
merge_diabetes_cycle.py

Merges the NHANES component files for ONE cycle into a single wide table
for the diabetes screening module. All XPT files live in
diabetes_screening/raw/<cycle_years>/.

Components covered:
  DEMO  - demographics
  BMX   - body measurements (BMI, waist)
  BPX   - blood pressure exam readings      [in diabetes_screening/raw only]
  DIQ   - diabetes self-report (for label construction)
  BPQ   - hypertension questionnaire
  MCQ   - medical conditions (family history of diabetes)
  PAQ   - physical activity questionnaire
  SMQ   - smoking questionnaire
  GHB   - HbA1c lab  (OUTCOME LABEL ONLY — never a feature)
  GLU   - fasting glucose lab (OUTCOME LABEL ONLY — never a feature)

Usage:
    python merge_diabetes_cycle.py 2007-2008 E
    python merge_diabetes_cycle.py 2009-2010 F
    ... etc for each cycle

Reads from:  ../raw/<cycle_years>/<COMPONENT>_<letter>.xpt
Writes to:   ../processed/cycle_<cycle_years>.parquet
"""

import sys
import pandas as pd
from pathlib import Path

# diabetes_screening/ is the base for this module
BASE      = Path(__file__).resolve().parent.parent
RAW       = BASE / "raw"
PROCESSED = BASE / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

COMPONENTS = {
    "DEMO": ["RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR",
             "WTMEC2YR", "WTINT2YR", "SDMVPSU", "SDMVSTRA"],
    "BMX":  ["BMXBMI", "BMXWAIST"],
    "DIQ":  ["DIQ010"],                                    # self-reported diabetes
    "BPQ":  ["BPQ020", "BPQ040A"],                        # hypertension + meds
    "MCQ":  ["MCQ300C"],                                   # close relative had diabetes
    "PAQ":  ["PAQ605", "PAQ620", "PAQ635",
             "PAQ650", "PAQ665", "PAD680"],
    "SMQ":  ["SMQ020", "SMQ040"],
    "GHB":  ["LBXGH"],                                    # HbA1c — label only
    "GLU":  ["LBXGLU", "WTSAF2YR"],                       # fasting glucose — label only
}


def load_xpt(path: Path, keep_cols: list) -> pd.DataFrame:
    """Load a .xpt file and keep only SEQN + requested columns that exist."""
    df = pd.read_sas(path, format="xport")
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
        # try lowercase first (how files were saved), then uppercase
        fpath = cycle_dir / f"{prefix}_{letter}.xpt"
        if not fpath.exists():
            fpath = cycle_dir / f"{prefix}_{letter}.XPT"
        if not fpath.exists():
            print(f"  [warn] missing file: {prefix}_{letter}.xpt — skipping")
            continue

        df = load_xpt(fpath, cols)
        print(f"  loaded {fpath.name}: {df.shape[0]} rows, cols={list(df.columns)}")
        merged = df if merged is None else merged.merge(df, on="SEQN", how="outer")

    if merged is None:
        raise RuntimeError(f"No component files found in {cycle_dir}")

    merged["cycle"] = cycle_years
    return merged


def main():
    if len(sys.argv) != 3:
        print("Usage: python merge_diabetes_cycle.py <cycle_years> <letter>")
        print("Example: python merge_diabetes_cycle.py 2007-2008 E")
        sys.exit(1)

    cycle_years, letter = sys.argv[1], sys.argv[2].upper()
    print(f"\nMerging diabetes cycle {cycle_years} (suffix _{letter})...")

    merged   = merge_cycle(cycle_years, letter)
    out_path = PROCESSED / f"cycle_{cycle_years}.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"Saved {merged.shape[0]} rows x {merged.shape[1]} cols -> {out_path}")


if __name__ == "__main__":
    main()
