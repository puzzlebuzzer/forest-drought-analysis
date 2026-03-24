#!/usr/bin/env python3
"""
build_landsat_annual_composites.py
----------------------------------
Annual maximum composite rasters for Landsat Collection 2 NDVI, NDMI, and EVI.

For each year, each pixel receives the maximum value across all valid
(non-NaN) scenes in that calendar year. Produces one GeoTIFF per year
per index per AOI.

These composites are the primary spatial deliverable:
  - Load into ArcGIS Pro as a time-enabled mosaic dataset
  - Show actual pixel-level vegetation/moisture variation
  - Foundation for anomaly analysis (year minus multi-year baseline)

Output structure:
  Results/rasters/annual_max/
    ndvi_north_landsat/   1984.tif  1985.tif  ...
    ndvi_south_landsat/   ...
    ndmi_north_landsat/   ...
    ndmi_south_landsat/   ...
    evi_north_landsat/    ...
    evi_south_landsat/    ...

Point each ArcGIS mosaic dataset at one of these folders.
Each file is tagged with year, index, AOI, source platform, and contributing scene count.

If a composite already exists it is skipped — safe to re-run after interruption.

Run from project root:
  python Analysis/Indices/build_landsat_annual_composites.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio

from src.landsat import get_landsat_index_root, load_landsat_scenes
from src.paths import project_path

# ── Configuration ──────────────────────────────────────────────────────────────

AOIS       = ["north", "south"]
INDICES    = ["NDVI", "NDMI", "EVI"]
YEAR_FLOOR = 1984   # Landsat Collection 2 starts at 1984

ANNUAL_MAX_DIR = project_path("landsat_annual_max_dir")


# ── Helpers ────────────────────────────────────────────────────────────────────

def index_scenes_by_year(
    scenes: list[dict],
) -> dict[int, list[dict]]:
    """
    Returns {year: [sorted scene dicts]} for all scenes at or after YEAR_FLOOR.

    Each scene dict contains: date (datetime), filepath (Path), platform, path_row
    """
    by_year: dict[int, list[dict]] = {}
    for scene in scenes:
        year = scene["date"].year
        if year >= YEAR_FLOOR:
            by_year.setdefault(year, []).append(scene)

    return {yr: sorted(by_year[yr], key=lambda s: s["date"])
            for yr in sorted(by_year.keys())}


def build_annual_max(
    scene_dicts: list[dict],
) -> tuple[np.ndarray, dict, set]:
    """
    Reads all scenes for one year and returns the element-wise nanmax
    array plus the rasterio profile from the first scene, and the set of
    platforms that contributed.

    Uses np.fmax for in-place accumulation:
      - NaN-safe: if one value is NaN the other is returned
      - No stacking: constant memory regardless of scene count
    """
    max_arr: np.ndarray | None = None
    profile: dict | None       = None
    platforms: set             = set()

    for scene_dict in scene_dicts:
        fp       = scene_dict["filepath"]
        platform = scene_dict.get("platform", "unknown")
        platforms.add(platform)

        with rasterio.open(fp) as src:
            data = src.read(1).astype(np.float32)
            if profile is None:
                profile = src.profile.copy()

        if max_arr is None:
            max_arr = data.copy()
        else:
            np.fmax(max_arr, data, out=max_arr)   # NaN-safe element-wise max

    return max_arr, profile, platforms


# ── Main loop ──────────────────────────────────────────────────────────────────

print(f"\n{'='*64}")
print(f"  Landsat C2 Annual Maximum Composites (1984–present)")
print(f"{'='*64}")

total_written  = 0
total_skipped  = 0

for aoi in AOIS:
    print(f"\n{'='*64}")
    print(f"  {aoi.upper()} AOI")
    print(f"{'='*64}")

    for index_name in INDICES:
        print(f"\n  [{index_name}] Loading Landsat scenes...")
        scenes = load_landsat_scenes(aoi, index_name)

        if not scenes:
            print(f"    No scenes found — skipping {index_name}")
            continue

        out_dir = ANNUAL_MAX_DIR / f"{index_name.lower()}_{aoi}_landsat"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [{index_name}] Indexing scenes by year...")
        by_year = index_scenes_by_year(scenes)
        years   = list(by_year.keys())
        print(f"    {len(years)} years: {min(years)}–{max(years)}")

        for year, scene_dicts in by_year.items():
            out_path = out_dir / f"{year}.tif"

            if out_path.exists():
                print(f"    {year}  ({len(scene_dicts):>3} scenes)  skipped — already exists")
                total_skipped += 1
                continue

            print(
                f"    {year}  ({len(scene_dicts):>3} scenes)  building...",
                end="", flush=True,
            )

            max_arr, profile, platforms = build_annual_max(scene_dicts)

            profile.update(
                count      = 1,
                dtype      = np.float32,
                nodata     = np.nan,
                compress   = "lzw",
                tiled      = True,
                blockxsize = 256,
                blockysize = 256,
            )

            platform_str = "/".join(sorted(platforms)) if platforms else "unknown"

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(max_arr, 1)
                dst.update_tags(
                    year        = str(year),
                    index       = index_name,
                    aoi         = aoi,
                    scene_count = str(len(scene_dicts)),
                    source      = "Landsat C2",
                    platform    = platform_str,
                    description = (
                        f"Annual maximum {index_name} — {year} — {aoi} AOI  "
                        f"({len(scene_dicts)} scenes, platforms: {platform_str})"
                    ),
                )

            valid_pct = float(np.isfinite(max_arr).mean()) * 100
            print(f"  {valid_pct:.1f}% valid  →  {out_path.name}")
            total_written += 1

print(f"\n{'='*64}")
print(f"  Written : {total_written} rasters")
print(f"  Skipped : {total_skipped} (already existed)")
print(f"  Output  : {ANNUAL_MAX_DIR}")
print(f"{'='*64}\n")
