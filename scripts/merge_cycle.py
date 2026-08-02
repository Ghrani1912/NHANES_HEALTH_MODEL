"""
merge_cycle.py

Merges the NHANES component files for ONE cycle (DEMO, BIOPRO, ALB_CR,
KIQ_U, DIQ, BPQ, BMX) into a single wide table, joined on SEQN.

Usage:
    python merge_cycle.py 2007-2008 E
    python merge_cycle.py 2009-2010 F
    ... etc for each cycle

Reads from:  raw/<cycle_years>/<COMPONENT>_<letter>.XPT
Writes to:   processed/cycle_<cycle_years>.parquet
"""

import sys
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "raw"
PROCESSED = BASE / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# component file prefixes we need, and which columns to keep from each
# (SEQN is always kept automatically as the join key)
COMPONENTS = {
    "DEMO": ["RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR",
             "WTMEC2YR", "WTINT2YR", "SDMVPSU", "SDMVSTRA"],
    "BIOPRO": ["LBXSCR"],                     # serum creatinine, mg/dL
    "ALB_CR": ["URXUMA", "URXUCR"],            # urine albumin (ug/mL), urine creatinine (mg/dL)
    "KIQ_U": ["KIQ022", "KIQ025"],             # weak/failing kidneys; dialysis in past 12mo
    "DIQ": ["DIQ010"],                         # diabetes diagnosis
    "BPQ": ["BPQ020", "BPQ040A"],              # hypertension diagnosis, taking meds
    "BMX": ["BMXBMI"],                         # BMI
}


def load_xpt(path: Path, keep_cols: list[str]) -> pd.DataFrame:
    """Load a .XPT file and keep only SEQN + requested columns that exist."""
    df = pd.read_sas(path, format="xport")
    # SEQN sometimes comes back as float; normalize to int for clean joins
    df["SEQN"] = df["SEQN"].astype(int)
    available = [c for c in keep_cols if c in df.columns]
    missing = set(keep_cols) - set(available)
    if missing:
        print(f"  [warn] {path.name}: missing columns {missing}")
    return df[["SEQN"] + available]


def merge_cycle(cycle_years: str, letter: str) -> pd.DataFrame:
    cycle_dir = RAW / cycle_years
    merged = None

    for prefix, cols in COMPONENTS.items():
        fname = f"{prefix}_{letter}.XPT"
        fpath = cycle_dir / fname
        if not fpath.exists():
            print(f"  [warn] missing file: {fpath}")
            continue

        df = load_xpt(fpath, cols)
        print(f"  loaded {fname}: {df.shape[0]} rows, cols={list(df.columns)}")

        merged = df if merged is None else merged.merge(df, on="SEQN", how="outer")

    merged["cycle"] = cycle_years
    return merged


def main():
    if len(sys.argv) != 3:
        print("Usage: python merge_cycle.py <cycle_years> <letter>")
        print("Example: python merge_cycle.py 2007-2008 E")
        sys.exit(1)

    cycle_years, letter = sys.argv[1], sys.argv[2]
    print(f"Merging cycle {cycle_years} (suffix _{letter})...")

    merged = merge_cycle(cycle_years, letter)

    out_path = PROCESSED / f"cycle_{cycle_years}.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"Saved {merged.shape[0]} rows x {merged.shape[1]} cols -> {out_path}")


if __name__ == "__main__":
    main()
