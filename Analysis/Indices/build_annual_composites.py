#!/usr/bin/env python3
"""
build_annual_composites.py
--------------------------
Annual maximum composite rasters for NDVI, NDMI, and EVI.

For each year, each pixel receives the maximum value across all valid
(non-NaN) scenes in that calendar year.  Produces one GeoTIFF per year
per index per AOI.

These composites are the primary spatial deliverable:
  - Load into ArcGIS Pro as a time-enabled mosaic dataset
  - Show actual pixel-level vegetation/moisture variation
  - Foundation for anomaly analysis (year minus multi-year baseline)

Output structure:
  Results/rasters/annual_max/
    ndvi_north/   2017.tif  2018.tif  ...
    ndvi_south/   ...
    ndmi_north/   ...
    ndmi_south/   ...

Point each ArcGIS mosaic dataset at one of these folders.
Each file is tagged with year, index, AOI, and contributing scene count.

If a composite already exists it is skipped — safe to re-run after interruption.

Run from project root:
  python Analysis/Indices/build_annual_composites.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio

from src.aoi import get_aoi_config
from src.paths import project_path

# ── Configuration ──────────────────────────────────────────────────────────────

AOIS       = ["north", "south"]
INDICES    = ["NDVI", "NDMI", "EVI"]
YEAR_FLOOR = 2017   # ignore any scenes older than this

ANNUAL_MAX_DIR = project_path("results_rasters_dir") / "annual_max"


# ── Helpers ────────────────────────────────────────────────────────────────────

def index_scenes_by_year(
    index_dir: Path,
    manifest_path: Path,
) -> dict[int, list[Path]]:
    """
    Returns {year: [sorted scene file paths]} for all on-disk scenes
    at or after YEAR_FLOOR.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    by_year: dict[int, list[Path]] = {}
    for _, meta in manifest.items():
        fp   = index_dir / meta["filename"]
        year = datetime.fromisoformat(meta["date"]).year
        if fp.exists() and year >= YEAR_FLOOR:
            by_year.setdefault(year, []).append(fp)

    return {yr: sorted(paths) for yr, paths in sorted(by_year.items())}


def build_annual_max(
    scene_paths: list[Path],
) -> tuple[np.ndarray, dict]:
    """
    Reads all scenes for one year and returns the element-wise nanmax
    array plus the rasterio profile from the first scene.

    Uses np.fmax for in-place accumulation:
      - NaN-safe: if one value is NaN the other is returned
      - No stacking: constant memory regardless of scene count
    """
    max_arr: np.ndarray | None = None
    profile: dict | None       = None

    for fp in scene_paths:
        with rasterio.open(fp) as src:
            data = src.read(1).astype(np.float32)
            if profile is None:
                profile = src.profile.copy()

        if max_arr is None:
            max_arr = data.copy()
        else:
            np.fmax(max_arr, data, out=max_arr)   # NaN-safe element-wise max

    return max_arr, profile


# ── Main loop ──────────────────────────────────────────────────────────────────

total_written  = 0
total_skipped  = 0

for aoi in AOIS:
    cfg = get_aoi_config(aoi)
    print(f"\n{'='*64}")
    print(f"  {aoi.upper()} AOI")
    print(f"{'='*64}")

    for index_name in INDICES:
        index_dir     = cfg.index_cache_root / index_name
        manifest_path = index_dir / "cache_manifest.json"

        out_dir = ANNUAL_MAX_DIR / f"{index_name.lower()}_{aoi}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{index_name}] Indexing scenes by year...")
        by_year = index_scenes_by_year(index_dir, manifest_path)
        years   = list(by_year.keys())
        print(f"    {len(years)} years: {years}")

        for year, scene_paths in by_year.items():
            out_path = out_dir / f"{year}.tif"

            if out_path.exists():
                print(f"    {year}  ({len(scene_paths):>3} scenes)  skipped — already exists")
                total_skipped += 1
                continue

            print(
                f"    {year}  ({len(scene_paths):>3} scenes)  building...",
                end="", flush=True,
            )

            max_arr, profile = build_annual_max(scene_paths)

            profile.update(
                count      = 1,
                dtype      = np.float32,
                nodata     = np.nan,
                compress   = "lzw",
                tiled      = True,
                blockxsize = 256,
                blockysize = 256,
            )

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(max_arr, 1)
                dst.update_tags(
                    year        = str(year),
                    index       = index_name,
                    aoi         = aoi,
                    scene_count = str(len(scene_paths)),
                    description = (
                        f"Annual maximum {index_name} — {year} — {aoi} AOI  "
                        f"({len(scene_paths)} scenes)"
                    ),
                )

            valid_pct = float(np.isfinite(max_arr).mean()) * 100
            print(f"  {valid_pct:.1f}% valid  →  {out_path.name}")
            total_written += 1

print(f"\n{'='*64}")
print(f"  Written : {total_written} rasters")
print(f"  Skipped : {total_skipped} (already existed)")
print(f"  Output  : {ANNUAL_MAX_DIR}")
print(f"{'='*64}")
