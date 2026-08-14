"""
download_components.py

Downloads the additional NHANES component XPT files needed for the
diabetes screening module. Saves them into the existing raw/<cycle>/
folders alongside the CKD components.

New components:
  GHB   - HbA1c (LBXGH)                  <- outcome label
  GLU   - Fasting plasma glucose (LBXGLU) <- outcome label (alternative)
  BPX   - Blood pressure exam readings    <- systolic/diastolic BP values
  MCQ   - Medical conditions questionnaire <- family history of diabetes
  PAQ   - Physical activity questionnaire  <- activity level
  SMQ   - Smoking questionnaire            <- smoking status

Usage:
    python download_components.py

Downloads to: ../raw/<cycle>/
"""

import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RAW  = BASE / "raw"

# Cycle year → NHANES letter suffix
CYCLES = {
    "2007-2008": "E",
    "2009-2010": "F",
    "2011-2012": "G",
    "2013-2014": "H",
    "2015-2016": "I",
    "2017-2018": "J",
}

# New components to download (prefix → NHANES URL path segment)
NEW_COMPONENTS = ["GHB", "GLU", "BPX", "MCQ", "PAQ", "SMQ"]

NHANES_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"


def build_url(cycle: str, filename: str) -> str:
    """
    CDC canonical URL pattern for all cycles 2007+:
      https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/<start_year>/DataFiles/<FILE>.XPT
    """
    year = cycle.split("-")[0]
    return f"{NHANES_BASE}/{year}/DataFiles/{filename}"


def download_file(url: str, dest: Path):
    if dest.exists():
        # check it's not an HTML error page (CDC returns 200 with HTML for missing files)
        with open(dest, "rb") as f:
            header = f.read(15)
        if header.startswith(b"<!DOCTYPE") or header.startswith(b"<html"):
            print(f"  [stale] removing bad cached file: {dest.name}")
            dest.unlink()
        else:
            print(f"  [skip] already exists: {dest.name}")
            return

    print(f"  downloading {dest.name} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        # validate — CDC returns HTML error pages with 200 status
        with open(dest, "rb") as f:
            header = f.read(15)
        if header.startswith(b"<!DOCTYPE") or header.startswith(b"<html"):
            dest.unlink()
            print(f"FAILED (CDC returned HTML — file may not exist at this URL)")
            print(f"    tried: {url}")
        else:
            size_kb = dest.stat().st_size // 1024
            print(f"done ({size_kb} KB)")
    except Exception as e:
        print(f"FAILED: {e}")


def main():
    for cycle, letter in CYCLES.items():
        print(f"\n--- {cycle} ---")
        cycle_dir = RAW / cycle
        cycle_dir.mkdir(parents=True, exist_ok=True)

        for comp in NEW_COMPONENTS:
            filename = f"{comp}_{letter}.XPT"
            url      = build_url(cycle, filename)
            dest     = cycle_dir / f"{comp}_{letter}.xpt"
            download_file(url, dest)

    print("\nAll downloads complete.")
    print("Run merge_diabetes_cycle.py next to build the pooled dataset.")


if __name__ == "__main__":
    main()
