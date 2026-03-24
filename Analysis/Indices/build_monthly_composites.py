#!/usr/bin/env python3
"""
build_monthly_composites.py
---------------------------
Monthly maximum composite rasters for Sentinel-2 NDVI, NDMI, and EVI.

For each calendar month, each pixel receives the maximum value across all
valid (non-NaN) scenes in that month.  Produces one GeoTIFF per month per
index per AOI.

Months with no usable scenes (common Nov–Mar due to cloud cover / leaf-off)
are skipped rather than written as all-NaN rasters.

Output structure:
  Results/rasters/monthly_max/
    ndvi_north/   2017_04.tif  2017_05.tif  ...  2025_10.tif
    ndvi_south/   ...
    ndmi_north/   ...
    ndmi_south/   ...
    evi_north/    ...
    evi_south/    ...

Use build_mosaic_lpkx.py (or re-run it) to include these in the ArcGIS
time-enabled layer package.  Set the time slider step to 1 Month.

If a composite already exists it is skipped — safe to re-run after interruption.

Run from project root:
  python Analysis/Indices/build_monthly_composites.py
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

MONTHLY_MAX_DIR = project_path("monthly_max_dir")


# ── Helpers ────────────────────────────────────────────────────────────────────

def index_scenes_by_month(
    index_dir: Path,
    manifest_path: Path,
) -> dict[tuple[int, int], list[Path]]:
    """
    Returns {(year, month): [sorted scene file paths]} for all on-disk scenes
    at or after YEAR_FLOOR.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    by_month: dict[tuple[int, int], list[Path]] = {}
    for _, meta in manifest.items():
        fp   = index_dir / meta["filename"]
        dt   = datetime.fromisoformat(meta["date"])
        if fp.exists() and dt.year >= YEAR_FLOOR:
            key = (dt.year, dt.month)
            by_month.setdefault(key, []).append(fp)

    return {k: sorted(v) for k, v in sorted(by_month.items())}


def build_monthly_max(
    scene_paths: list[Path],
) -> tuple[np.ndarray, dict]:
    """
    Reads all scenes for one month and returns the element-wise nanmax
    array plus the rasterio profile from the first scene.

    Uses np.fmax for in-place accumulation (NaN-safe, constant memory).
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
            np.fmax(max_arr, data, out=max_arr)

    return max_arr, profile


# ── Main loop ──────────────────────────────────────────────────────────────────

total_written  = 0
total_skipped  = 0
total_empty    = 0

for aoi in AOIS:
    cfg = get_aoi_config(aoi)
    print(f"\n{'='*64}")
    print(f"  {aoi.upper()} AOI")
    print(f"{'='*64}")

    for index_name in INDICES:
        index_dir     = cfg.index_cache_root / index_name
        manifest_path = index_dir / "cache_manifest.json"

        out_dir = MONTHLY_MAX_DIR / f"{index_name.lower()}_{aoi}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{index_name}] Indexing scenes by month...")
        by_month = index_scenes_by_month(index_dir, manifest_path)
        print(f"    {len(by_month)} months with scenes")

        for (year, month), scene_paths in by_month.items():
            out_path = out_dir / f"{year}_{month:02d}.tif"

            if out_path.exists():
                print(
                    f"    {year}-{month:02d}  ({len(scene_paths):>3} scenes)"
                    f"  skipped — already exists"
                )
                total_skipped += 1
                continue

            print(
                f"    {year}-{month:02d}  ({len(scene_paths):>3} scenes)"
                f"  building...",
                end="", flush=True,
            )

            max_arr, profile = build_monthly_max(scene_paths)

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

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(max_arr, 1)
                dst.update_tags(
                    year        = str(year),
                    month       = str(month),
                    index       = index_name,
                    aoi         = aoi,
                    scene_count = str(len(scene_paths)),
                    description = (
                        f"Monthly maximum {index_name} — {year}-{month:02d}"
                        f" — {aoi} AOI  ({len(scene_paths)} scenes)"
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
print(f"{'='*64}")
