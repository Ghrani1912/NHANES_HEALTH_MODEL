"""
pool_cycles.py

Stacks all per-cycle merged parquet files (from merge_cycle.py) into one
pooled dataset. Adds a composite ID so that SEQN (which repeats across
cycles for DIFFERENT people) is never ambiguous.

Usage:
    python pool_cycles.py

Reads from:  processed/cycle_*.parquet
Writes to:   processed/pooled.parquet
"""

import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "processed"

CYCLES = [
    "2007-2008", "2009-2010", "2011-2012",
    "2013-2014", "2015-2016", "2017-2018",
]


def main():
    frames = []
    for cycle in CYCLES:
        fpath = PROCESSED / f"cycle_{cycle}.parquet"
        if not fpath.exists():
            print(f"[warn] missing {fpath}, skipping. Run merge_cycle.py for this cycle first.")
            continue
        df = pd.read_parquet(fpath)
        frames.append(df)
        print(f"  {cycle}: {df.shape[0]} rows")

    if not frames:
        print("No cycle files found. Run merge_cycle.py first.")
        return

    pooled = pd.concat(frames, ignore_index=True)

    # SEQN is reused across cycles for different people -- build a globally
    # unique respondent id so nothing downstream ever confuses two people.
    pooled["respondent_id"] = pooled["cycle"] + "_" + pooled["SEQN"].astype(str)

    out_path = PROCESSED / "pooled.parquet"
    pooled.to_parquet(out_path, index=False)
    print(f"\nPooled dataset: {pooled.shape[0]} rows x {pooled.shape[1]} cols -> {out_path}")
    print(f"Cycles included: {sorted(pooled['cycle'].unique())}")


if __name__ == "__main__":
    main()
