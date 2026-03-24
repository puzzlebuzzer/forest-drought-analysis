#!/usr/bin/env python3
"""
build_anomaly_rasters.py
------------------------
Annual anomaly rasters for NDVI and NDMI.  (Axis G3)

Reads the annual maximum composites produced by build_annual_composites.py
and subtracts a multi-year baseline composite to produce per-year deviation
rasters.  Positive values = above-average year, negative = below average.

Baseline: pixel-wise nanmean across all available annual composites.
The baseline raster itself is also saved for reference.

To use neutral-year-only baseline instead, set NEUTRAL_YEARS_ONLY = True
and ensure config/wet_dry_years.csv is current.

Dependency: run build_annual_composites.py first.

Output structure:
  Results/rasters/anomaly/
    ndvi_north/
      baseline.tif          multi-year mean baseline
      2017.tif  2018.tif    anomaly = year_max - baseline
      ...
    ndvi_south/  ndmi_north/  ndmi_south/  (same structure)

Run from project root:
  python Analysis/Indices/build_anomaly_rasters.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from src.paths import project_path

# ── Configuration ──────────────────────────────────────────────────────────────

AOIS    = ["north", "south"]
INDICES = ["NDVI", "NDMI", "EVI"]

# Set True to compute baseline from neutral years only (avoids wet/dry bias).
# Requires config/wet_dry_years.csv.  False = all available years.
NEUTRAL_YEARS_ONLY = False

ANNUAL_MAX_DIR = project_path("results_rasters_dir") / "annual_max"
ANOMALY_DIR    = project_path("results_rasters_dir") / "anomaly"


# ── Neutral year filter (used only if NEUTRAL_YEARS_ONLY = True) ──────────────

def neutral_years_for_aoi(aoi: str) -> set[int]:
    wdy_path = project_path("config_dir") / "wet_dry_years.csv"
    df = pd.read_csv(wdy_path)
    return set(
        df[(df["aoi"] == aoi) & (df["classification"] == "neutral")]["year"].tolist()
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_composite(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32), src.profile.copy()


def write_raster(path: Path, data: np.ndarray, profile: dict) -> None:
    p = profile.copy()
    p.update(
        count=1, dtype=np.float32, nodata=np.nan,
        compress="lzw", tiled=True, blockxsize=256, blockysize=256,
    )
    with rasterio.open(path, "w", **p) as dst:
        dst.write(data, 1)


# ── Main loop ──────────────────────────────────────────────────────────────────

total_written = 0

for aoi in AOIS:
    neutral_yrs = neutral_years_for_aoi(aoi) if NEUTRAL_YEARS_ONLY else None

    for index_name in INDICES:
        src_dir = ANNUAL_MAX_DIR / f"{index_name.lower()}_{aoi}"
        out_dir = ANOMALY_DIR    / f"{index_name.lower()}_{aoi}"
        out_dir.mkdir(parents=True, exist_ok=True)

        tif_paths = sorted(src_dir.glob("*.tif"))
        if not tif_paths:
            print(f"  SKIP  {index_name}/{aoi} — no composites found in {src_dir}")
            print(f"        Run build_annual_composites.py first.")
            continue

        # Filter to baseline years if needed
        if NEUTRAL_YEARS_ONLY:
            baseline_paths = [
                p for p in tif_paths
                if p.stem.isdigit() and int(p.stem) in neutral_yrs
            ]
            label = f"neutral years ({sorted(neutral_yrs)})"
        else:
            baseline_paths = [p for p in tif_paths if p.stem.isdigit()]
            label = f"all years ({[p.stem for p in baseline_paths]})"

        print(f"\n  [{index_name} / {aoi}]  baseline from {label}")

        # Build baseline: pixel-wise nanmean across baseline years
        stack   = np.stack([load_composite(p)[0] for p in baseline_paths])
        profile = load_composite(baseline_paths[0])[1]   # borrow profile
        baseline = np.nanmean(stack, axis=0).astype(np.float32)

        baseline_path = out_dir / "baseline.tif"
        write_raster(baseline_path, baseline, profile)
        b_mean = float(np.nanmean(baseline))
        print(f"    Baseline saved  (mean={b_mean:.4f})  →  baseline.tif")
        total_written += 1

        # Anomaly for every available year composite
        year_paths = [p for p in tif_paths if p.stem.isdigit()]
        for yp in year_paths:
            year    = int(yp.stem)
            out_path = out_dir / f"{year}.tif"

            if out_path.exists():
                print(f"    {year}  skipped — already exists")
                continue

            composite, _ = load_composite(yp)
            anomaly = (composite - baseline).astype(np.float32)

            write_raster(out_path, anomaly, profile)

            # Stats
            finite = anomaly[np.isfinite(anomaly)]
            mean_a = float(np.mean(finite)) if len(finite) else np.nan
            print(
                f"    {year}  mean anomaly={mean_a:+.4f}"
                f"  min={float(np.nanmin(anomaly)):+.4f}"
                f"  max={float(np.nanmax(anomaly)):+.4f}"
                f"  →  {year}.tif"
            )
            total_written += 1

print(f"\n  Total rasters written: {total_written}")
print(f"  Output: {ANOMALY_DIR}")
print("\nAnomaly rasters complete.")
