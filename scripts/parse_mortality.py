"""
parse_mortality.py

Parses NCHS Public-Use Linked Mortality Files (fixed-width .dat) for one
cycle and extracts the fields needed for a renal-outcome survival analysis.

Column spec (1-indexed, from NCHS documentation):
    SEQN          1-6
    ELIGSTAT      15
    MORTSTAT      16
    UCOD_LEADING  17-19
    DIABETES      20
    HYPERTEN      21
    PERMTH_INT    43-45
    PERMTH_EXM    46-48

Usage:
    python parse_mortality.py 2007-2008

Reads:  raw/<cycle_years>/NHANES_<cycle_years_us>_MORT_2019_PUBLIC.dat
Writes: processed/mortality_<cycle_years>.parquet
"""

import sys
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "raw"
PROCESSED = BASE / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# pandas read_fwf colspecs are 0-indexed, [start, end) half-open
COLSPECS = [
    (0, 6),    # SEQN
    (14, 15),  # ELIGSTAT
    (15, 16),  # MORTSTAT
    (16, 19),  # UCOD_LEADING
    (19, 20),  # DIABETES flag
    (20, 21),  # HYPERTEN flag
    (42, 45),  # PERMTH_INT
    (45, 48),  # PERMTH_EXM
]
COLNAMES = ["SEQN", "eligstat", "mortstat", "ucod_leading",
            "diabetes_flag", "hyperten_flag", "permth_int", "permth_exm"]

# UCOD_LEADING code for kidney disease (Nephritis, nephrotic syndrome,
# nephrosis) per NCHS leading-cause recode
UCOD_KIDNEY = 9


def main():
    if len(sys.argv) != 2:
        print("Usage: python parse_mortality.py <cycle_years>")
        print("Example: python parse_mortality.py 2007-2008")
        sys.exit(1)

    cycle_years = sys.argv[1]
    cycle_dir = RAW / cycle_years

    # find the .dat file in this cycle's folder (name can vary slightly)
    dat_files = list(cycle_dir.glob("*MORT*.dat")) + list(cycle_dir.glob("*mort*.dat"))
    if not dat_files:
        print(f"[error] no mortality .dat file found in {cycle_dir}")
        sys.exit(1)
    dat_path = dat_files[0]
    print(f"Parsing {dat_path.name}...")

    df = pd.read_fwf(dat_path, colspecs=COLSPECS, names=COLNAMES,
                      na_values=[".", ""])

    df["SEQN"] = df["SEQN"].astype(int)
    df["cycle"] = cycle_years

    # renal-cause death flag: died AND underlying cause was kidney disease
    df["renal_death"] = (df["mortstat"] == 1) & (df["ucod_leading"] == UCOD_KIDNEY)

    out_path = PROCESSED / f"mortality_{cycle_years}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {df.shape[0]} rows -> {out_path}")
    print("Mortality status counts:")
    print(df["mortstat"].value_counts(dropna=False))
    print("Renal deaths:", df["renal_death"].sum())


if __name__ == "__main__":
    main()
