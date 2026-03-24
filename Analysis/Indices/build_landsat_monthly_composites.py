#!/usr/bin/env python3
"""
build_landsat_monthly_composites.py
------------------------------------
Monthly maximum composite rasters for Landsat Collection 2 NDVI, NDMI, and EVI.

For each calendar month, each pixel receives the maximum value across all
valid (non-NaN) scenes in that month.  Produces one GeoTIFF per month per
index per AOI.

Months with no usable scenes are skipped rather than written as all-NaN rasters.
Landsat 5/7 era (pre-2000) may have sparser monthly coverage than L8/L9.

Output structure:
  Results/rasters/landsat_monthly_max/
    ndvi_north_landsat/   1984_07.tif  1984_08.tif  ...
    ndvi_south_landsat/   ...
    ndmi_north_landsat/   ...
    ndmi_south_landsat/   ...
    evi_north_landsat/    ...
    evi_south_landsat/    ...

If a composite already exists it is skipped — safe to re-run after interruption.

Run from project root:
  python Analysis/Indices/build_landsat_monthly_composites.py
"""

from pathlib import Path

import numpy as np
import rasterio

from src.landsat import load_landsat_scenes
from src.paths import project_path

# ── Configuration ──────────────────────────────────────────────────────────────

AOIS       = ["north", "south"]
INDICES    = ["NDVI", "NDMI", "EVI"]
YEAR_FLOOR = 1984   # Landsat Collection 2 starts at 1984

MONTHLY_MAX_DIR = project_path("landsat_monthly_max_dir")


# ── Helpers ────────────────────────────────────────────────────────────────────

def index_scenes_by_month(
    scenes: list[dict],
) -> dict[tuple[int, int], list[dict]]:
    """
    Returns {(year, month): [sorted scene dicts]} for all scenes at or
    after YEAR_FLOOR.
    """
    by_month: dict[tuple[int, int], list[dict]] = {}
    for scene in scenes:
        dt   = scene["date"]
        year = dt.year
        if year >= YEAR_FLOOR:
            key = (year, dt.month)
            by_month.setdefault(key, []).append(scene)

    return {
        k: sorted(by_month[k], key=lambda s: s["date"])
        for k in sorted(by_month.keys())
    }


def build_monthly_max(
    scene_dicts: list[dict],
) -> tuple[np.ndarray, dict, set]:
    """
    Reads all scenes for one month and returns the element-wise nanmax
    array, the rasterio profile from the first scene, and the set of
    contributing platforms.

    Uses np.fmax for in-place accumulation (NaN-safe, constant memory).
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
            np.fmax(max_arr, data, out=max_arr)

    return max_arr, profile, platforms


# ── Main loop ──────────────────────────────────────────────────────────────────

print(f"\n{'='*64}")
print(f"  Landsat C2 Monthly Maximum Composites (1984–present)")
print(f"{'='*64}")

total_written  = 0
total_skipped  = 0
total_empty    = 0

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

        out_dir = MONTHLY_MAX_DIR / f"{index_name.lower()}_{aoi}_landsat"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [{index_name}] Indexing scenes by month...")
        by_month = index_scenes_by_month(scenes)
        print(f"    {len(by_month)} months with scenes")

        for (year, month), scene_dicts in by_month.items():
            out_path = out_dir / f"{year}_{month:02d}.tif"

            if out_path.exists():
                print(
                    f"    {year}-{month:02d}  ({len(scene_dicts):>3} scenes)"
                    f"  skipped — already exists"
                )
                total_skipped += 1
                continue

            print(
                f"    {year}-{month:02d}  ({len(scene_dicts):>3} scenes)"
                f"  building...",
                end="", flush=True,
            )

            max_arr, profile, platforms = build_monthly_max(scene_dicts)

            # Skip months where the composite is entirely NaN
            if max_arr is None or not np.any(np.isfinite(max_arr)):
                print(f"  no valid pixels — skipped")
                total_empty += 1
                continue

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
                    month       = str(month),
                    index       = index_name,
                    aoi         = aoi,
                    scene_count = str(len(scene_dicts)),
                    source      = "Landsat C2",
                    platform    = platform_str,
                    description = (
                        f"Monthly maximum {index_name} — {year}-{month:02d}"
                        f" — {aoi} AOI  ({len(scene_dicts)} scenes,"
                        f" platforms: {platform_str})"
                    ),
                )

            valid_pct = float(np.isfinite(max_arr).mean()) * 100
            print(f"  {valid_pct:.1f}% valid  →  {out_path.name}")
            total_written += 1

print(f"\n{'='*64}")
print(f"  Written : {total_written} rasters")
print(f"  Skipped : {total_skipped} (already existed)")
print(f"  Empty   : {total_empty} (no valid pixels, not written)")
print(f"  Output  : {MONTHLY_MAX_DIR}")
print(f"{'='*64}\n")
