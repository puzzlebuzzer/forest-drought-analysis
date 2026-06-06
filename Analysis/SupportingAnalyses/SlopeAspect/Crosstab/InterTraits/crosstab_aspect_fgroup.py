#!/usr/bin/env python3

"""
crosstab_aspect_fgroup.py
-------------------------
Cross-tabulates aspect masks against forest type groups.
"""

import json

import numpy as np
import rasterio

from src.aoi import get_aoi_config, get_forest_group_inventory_path
from src.cli import make_parser, add_aoi_arg
from src.labels import label_forest_group

parser = make_parser("Cross-tabulate aspect masks against forest type groups")
add_aoi_arg(parser)
args = parser.parse_args()

AOI = args.aoi
cfg = get_aoi_config(AOI)

TERRAIN_DIR = cfg.terrain_dir
FOREST_DIR = cfg.forest_group_dir
INVENTORY_PATH = get_forest_group_inventory_path(AOI)

PIXEL_AREA_KM2 = (10 * 10) / 1_000_000

print("Loading masks...")
with rasterio.open(TERRAIN_DIR / "mask_south_facing.tif") as src:
    south_mask = src.read(1) == 1
with rasterio.open(TERRAIN_DIR / "mask_north_facing.tif") as src:
    north_mask = src.read(1) == 1
with rasterio.open(FOREST_DIR / "forest_type_group.tif") as src:
    forest = src.read(1)

# --- grid sanity check ---
if south_mask.shape != forest.shape or north_mask.shape != forest.shape:
    raise ValueError(
        f"Grid mismatch: south={south_mask.shape}, "
        f"north={north_mask.shape}, forest={forest.shape}"
    )

print(f"  South-facing pixels: {south_mask.sum():,}")
print(f"  North-facing pixels: {north_mask.sum():,}")

print(f"\nLoading forest inventory from {INVENTORY_PATH}...")
with open(INVENTORY_PATH, "r", encoding="utf-8") as f:
    inventory = json.load(f)

known_codes = sorted(int(code) for code in inventory.keys())
all_codes = sorted(np.unique(forest))
other_codes = [c for c in all_codes if c not in known_codes and c != 0]

def stratum_row(label, mask):
    total = int(mask.sum())
    row = {"stratum": label, "total_px": total, "total_km2": total * PIXEL_AREA_KM2}
    for code in all_codes:
        row[int(code)] = int(((forest == code) & mask).sum())
    return row

rows = [
    stratum_row("Whole AOI", np.ones(forest.shape, dtype=bool)),
    stratum_row("South-facing", south_mask),
    stratum_row("North-facing", north_mask),
    stratum_row("Neither (E/W/flat)", ~south_mask & ~north_mask),
]

print(f"\n{'='*70}")
print(f"FOREST TYPE GROUP COMPOSITION BY ASPECT ({AOI})")
print(f"{'='*70}")

for row in rows:
    total = row["total_px"]
    print(f"\n{row['stratum']}  ({total:,} px  |  {row['total_km2']:.1f} km²)")
    print(f"  {'Forest type':<32} {'pixels':>10}  {'km²':>8}  {'%':>6}")
    print(f"  {'-'*64}")
    for code in known_codes:
        n = row[code]
        label = label_forest_group(code)
        pct = 100 * n / total if total > 0 else 0
        print(f"  {label:<32} {n:>10,}  {n*PIXEL_AREA_KM2:>8.1f}  {pct:>6.1f}%")

    if other_codes:
        other_n = sum(row[c] for c in other_codes)
        pct = 100 * other_n / total if total > 0 else 0
        print(f"  {'Other/unknown':<32} {other_n:>10,}  {other_n*PIXEL_AREA_KM2:>8.1f}  {pct:>6.1f}%")

print(f"\n{'='*70}")
print("SOUTH vs NORTH — COMPOSITION DIFFERENCE")
print(f"{'='*70}")
print(f"\n  {'Forest type':<32} {'South %':>8}  {'North %':>8}  {'Δ (S-N)':>8}")
print(f"  {'-'*64}")

south_row = rows[1]
north_row = rows[2]

for code in known_codes:
    label = label_forest_group(code)
    s_pct = 100 * south_row[code] / south_row["total_px"] if south_row["total_px"] > 0 else 0
    n_pct = 100 * north_row[code] / north_row["total_px"] if north_row["total_px"] > 0 else 0
    delta = s_pct - n_pct
    flag = " ◄" if abs(delta) > 5 else ""
    print(f"  {label:<32} {s_pct:>8.1f}%  {n_pct:>8.1f}%  {delta:>+8.1f}%{flag}")

print("\n  ◄ = difference > 5 percentage points")