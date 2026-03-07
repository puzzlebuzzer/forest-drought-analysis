"""
audit_aspect_masks.py
---------------------
Verifies that mask_south_facing.tif and mask_north_facing.tif capture
the correct aspect degree ranges by sampling the raw aspect.tif values
under each mask.

Expected (standard convention):
  South-facing: ~135–225° (centered on 180°)
  North-facing: ~315–360° + 0–45° (centered on 0°/360°)

A swap would show south mask centered near 0° and north mask near 180°.
"""

import rasterio
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
TERRAIN_DIR = Path("/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/"
                   "Project_Appalachia/AOI/GWNF_cache/traits/terrain")

ASPECT_PATH      = TERRAIN_DIR / "aspect.tif"
SOUTH_MASK_PATH  = TERRAIN_DIR / "mask_south_facing.tif"
NORTH_MASK_PATH  = TERRAIN_DIR / "mask_north_facing.tif"

# ── Load rasters ─────────────────────────────────────────────────────────────
print("Loading rasters...")

with rasterio.open(ASPECT_PATH) as src:
    aspect = src.read(1).astype(float)
    nodata = src.nodata
    if nodata is not None:
        aspect[aspect == nodata] = np.nan

with rasterio.open(SOUTH_MASK_PATH) as src:
    south_mask = src.read(1).astype(bool)

with rasterio.open(NORTH_MASK_PATH) as src:
    north_mask = src.read(1).astype(bool)

print(f"  Aspect raster shape:   {aspect.shape}")
print(f"  South-facing pixels:   {south_mask.sum():,}")
print(f"  North-facing pixels:   {north_mask.sum():,}")

# ── Sample aspect values under each mask ─────────────────────────────────────
def describe_aspect(label, mask, aspect_arr):
    vals = aspect_arr[mask & ~np.isnan(aspect_arr)]
    if len(vals) == 0:
        print(f"\n{label}: NO VALID PIXELS — mask may be empty or misaligned")
        return

    print(f"\n{label} ({len(vals):,} pixels):")
    print(f"  Mean:    {np.mean(vals):6.1f}°")
    print(f"  Median:  {np.median(vals):6.1f}°")
    print(f"  Std dev: {np.std(vals):6.1f}°")
    print(f"  P5:      {np.percentile(vals, 5):6.1f}°")
    print(f"  P25:     {np.percentile(vals, 25):6.1f}°")
    print(f"  P75:     {np.percentile(vals, 75):6.1f}°")
    print(f"  P95:     {np.percentile(vals, 95):6.1f}°")
    print(f"  Min:     {np.min(vals):6.1f}°")
    print(f"  Max:     {np.max(vals):6.1f}°")

    # Bin into cardinal quadrants
    n = len(vals)
    q_n  = np.sum((vals <= 45) | (vals > 315))
    q_e  = np.sum((vals > 45)  & (vals <= 135))
    q_s  = np.sum((vals > 135) & (vals <= 225))
    q_w  = np.sum((vals > 225) & (vals <= 315))

    print(f"\n  Quadrant breakdown:")
    print(f"    N  (315–360 / 0–45°):  {q_n:,}  ({100*q_n/n:.1f}%)")
    print(f"    E  (45–135°):          {q_e:,}  ({100*q_e/n:.1f}%)")
    print(f"    S  (135–225°):         {q_s:,}  ({100*q_s/n:.1f}%)")
    print(f"    W  (225–315°):         {q_w:,}  ({100*q_w/n:.1f}%)")

describe_aspect("mask_south_facing.tif", south_mask, aspect)
describe_aspect("mask_north_facing.tif", north_mask, aspect)

# ── Verdict ───────────────────────────────────────────────────────────────────
south_vals = aspect[south_mask & ~np.isnan(aspect)]
north_vals = aspect[north_mask & ~np.isnan(aspect)]

if len(south_vals) > 0 and len(north_vals) > 0:
    south_mean = np.mean(south_vals)
    north_mean = np.mean(north_vals)

    print("\n" + "="*50)
    print("VERDICT")
    print("="*50)

    # North aspect wraps around 0°, so check proximity to 0 or 360
    north_centered = north_mean > 270 or north_mean < 90
    south_centered = 90 < south_mean < 270

    if south_centered and north_centered:
        print("✓ Masks look CORRECT")
        print(f"  south mask mean = {south_mean:.1f}° (expected ~180°)")
        print(f"  north mask mean = {north_mean:.1f}° (expected ~0°/360°)")
    elif not south_centered and not north_centered:
        print("⚠ Masks appear SWAPPED")
        print(f"  south mask mean = {south_mean:.1f}° — this looks like NORTH")
        print(f"  north mask mean = {north_mean:.1f}° — this looks like SOUTH")
        print("  → The counterintuitive finding may be a labeling error.")
    else:
        print("? Masks are ambiguous — review quadrant breakdown above manually")
        print(f"  south mask mean = {south_mean:.1f}°")
        print(f"  north mask mean = {north_mean:.1f}°")