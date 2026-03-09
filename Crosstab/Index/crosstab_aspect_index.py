#!/usr/bin/env python3

"""
crosstab_aspect_index.py
------------------------
Computes south vs north facing slope comparison for any spectral index.
Set INDEX and AOI at the top and run.
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from src.aoi import get_aoi_config
from src.cli import make_parser, add_aoi_arg, add_index_arg

parser = make_parser("Compare aspect-stratified index values")
add_aoi_arg(parser)
add_index_arg(parser)
args = parser.parse_args()

AOI = args.aoi
INDEX = args.index
cfg = get_aoi_config(AOI)

TERRAIN_DIR = cfg.terrain_dir
INDEX_ROOT = cfg.index_cache_root
INDEX_DIR = INDEX_ROOT / INDEX
MANIFEST_PATH = INDEX_DIR / "cache_manifest.json"

print(f"AOI:       {AOI}")
print(f"Index:     {INDEX}")
print(f"Cache:     {INDEX_DIR}")
print(f"Manifest:  {MANIFEST_PATH}")

print("\nLoading terrain masks...")
with rasterio.open(TERRAIN_DIR / "mask_south_facing.tif") as src:
    south_mask = src.read(1) == 1
with rasterio.open(TERRAIN_DIR / "mask_north_facing.tif") as src:
    north_mask = src.read(1) == 1

# --- grid sanity check ---
if south_mask.shape != species.shape or north_mask.shape != species.shape:
    raise ValueError(
        f"Grid mismatch: south={south_mask.shape}, "
        f"north={north_mask.shape}, species={species.shape}"
    )

print(f"  South-facing pixels: {south_mask.sum():,}")
print(f"  North-facing pixels: {north_mask.sum():,}")

print("\nLoading manifest...")
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)

scenes = []
for _, meta in manifest.items():
    filepath = INDEX_DIR / meta["filename"]
    if filepath.exists():
        scenes.append({
            "date": datetime.fromisoformat(meta["date"]),
            "filepath": filepath,
        })

scenes = sorted(scenes, key=lambda x: x["date"])
print(f"  Found {len(scenes)} {INDEX} scenes with files on disk")

print("\nComputing stratified percentiles...")
dates = []
whole_aoi_p75 = []
south_p75 = []
north_p75 = []

for i, scene in enumerate(scenes):
    if i % 50 == 0:
        print(f"  {i}/{len(scenes)}...")

    with rasterio.open(scene["filepath"]) as src:
        data = src.read(1)

    valid = ~np.isnan(data)

    whole_aoi_p75.append(np.percentile(data[valid], 75) if valid.sum() > 0 else np.nan)

    sv = valid & south_mask
    south_p75.append(np.percentile(data[sv], 75) if sv.sum() > 100 else np.nan)

    nv = valid & north_mask
    north_p75.append(np.percentile(data[nv], 75) if nv.sum() > 100 else np.nan)

    dates.append(scene["date"])

print(f"  Done — {len(dates)} scenes processed")

south_arr = np.array(south_p75)
north_arr = np.array(north_p75)
diff = south_arr - north_arr

print(f"\n=== Summary Statistics ({INDEX}, {AOI}) ===")
print(f"\nWhole AOI:")
print(f"  Mean p75: {np.nanmean(whole_aoi_p75):.3f}  Std: {np.nanstd(whole_aoi_p75):.3f}")
print(f"\nSouth-facing slopes:")
print(f"  Mean p75: {np.nanmean(south_arr):.3f}  Std: {np.nanstd(south_arr):.3f}")
print(f"\nNorth-facing slopes:")
print(f"  Mean p75: {np.nanmean(north_arr):.3f}  Std: {np.nanstd(north_arr):.3f}")
print(f"\nDifference (South - North):")
print(f"  Mean: {np.nanmean(diff):.3f}  Std: {np.nanstd(diff):.3f}")

valid_diff = diff[~np.isnan(diff)]
if len(valid_diff) > 0:
    if np.nanmean(diff) < -0.05:
        print(f"\n  → South-facing slopes show LOWER {INDEX}")
    elif np.nanmean(diff) > 0.05:
        print(f"\n  → South-facing slopes show HIGHER {INDEX}")
    else:
        print(f"\n  → No strong difference between aspects")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

ax1.plot(dates, whole_aoi_p75, "o-", color="gray", markersize=3, alpha=0.7, label="Whole AOI")
ax1.plot(dates, south_p75, "o-", color="red", markersize=3, alpha=0.7, label="South-facing slopes")
ax1.plot(dates, north_p75, "o-", color="blue", markersize=3, alpha=0.7, label="North-facing slopes")
ax1.set_xlabel("Date")
ax1.set_ylabel(f"{INDEX} (p75)")
ax1.set_title(f"{INDEX} Time Series by Terrain Aspect ({AOI})")
ax1.legend()
ax1.grid(True, alpha=0.3)
if INDEX == "NDMI":
    ax1.set_ylim(-1, 1)
else:
    ax1.set_ylim(0, 1)

ax2.plot(dates, diff, "o-", color="purple", markersize=3, alpha=0.7)
ax2.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
ax2.set_xlabel("Date")
ax2.set_ylabel(f"{INDEX} Difference (South - North)")
ax2.set_title(f"South vs North Facing Slopes: {INDEX} Difference ({AOI})")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_path = Path.cwd() / f"terrain_stratification_{AOI}_{INDEX.lower()}.png"
plt.savefig(output_path, dpi=150)
print(f"\nPlot saved to {output_path}")
plt.show()