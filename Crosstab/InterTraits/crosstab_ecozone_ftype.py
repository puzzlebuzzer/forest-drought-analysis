#!/usr/bin/env python3

"""
crosstab_ecozone_ftype.py
-------------------------
Cross-tabulates ecozones with aspect and forest type groups.
"""

import json

import numpy as np
import rasterio

from src.aoi import get_aoi_config, get_forest_group_inventory_path
from src.cli import make_parser, add_aoi_arg
from src.labels import ECOZONE_LABELS, label_forest_group

parser = make_parser("Cross-tabulate ecozones with aspect and forest groups")
add_aoi_arg(parser)
args = parser.parse_args()

AOI = args.aoi
cfg = get_aoi_config(AOI)

TERRAIN_DIR = cfg.terrain_dir
FOREST_DIR = cfg.forest_group_dir
ECOZONE_DIR = cfg.ecozone_dir
INVENTORY_PATH = get_forest_group_inventory_path(AOI)

PIXEL_AREA_KM2 = (10 * 10) / 1_000_000


def ecozone_label(code: int) -> str:
    return ECOZONE_LABELS.get(code, f"Unknown ({code})")


def pct(n: int, total: int) -> float:
    return 100 * n / total if total > 0 else 0.0


print("Loading rasters...")
with rasterio.open(TERRAIN_DIR / "mask_south_facing.tif") as src:
    south_mask = src.read(1) == 1
with rasterio.open(TERRAIN_DIR / "mask_north_facing.tif") as src:
    north_mask = src.read(1) == 1
with rasterio.open(FOREST_DIR / "forest_type_group.tif") as src:
    forest = src.read(1)
with rasterio.open(ECOZONE_DIR / "tnc_ecozone_simplified_snapped.tif") as src:
    ecozone = src.read(1)

if (
    south_mask.shape != north_mask.shape
    or south_mask.shape != forest.shape
    or south_mask.shape != ecozone.shape
):
    raise ValueError(
        f"Grid mismatch: south={south_mask.shape}, north={north_mask.shape}, "
        f"forest={forest.shape}, ecozone={ecozone.shape}"
    )

print(f"\nLoading forest inventory from {INVENTORY_PATH}...")
with open(INVENTORY_PATH, "r", encoding="utf-8") as f:
    inventory = json.load(f)

forest_codes = sorted(int(code) for code in inventory.keys())

VALID_ECOZONE_CODES = [1, 2, 3]
eco_codes = [code for code in VALID_ECOZONE_CODES if code in np.unique(ecozone)]

neither_mask = ~south_mask & ~north_mask

print(f"  South-facing pixels: {south_mask.sum():,}")
print(f"  North-facing pixels: {north_mask.sum():,}")

print(f"\n{'=' * 70}")
print("1. ECOZONE × ASPECT")
print(f"{'=' * 70}")

aspect_strata = [
    ("Whole AOI", np.ones(south_mask.shape, dtype=bool)),
    ("South-facing", south_mask),
    ("North-facing", north_mask),
    ("Neither (E/W/flat)", neither_mask),
]

print(f"\n  {'Stratum':<22}", end="")
for code in eco_codes:
    print(f"  {ecozone_label(code) + ' %':>14}", end="")
print(f"  {'Total px':>12}  {'km²':>8}")
print(f"  {'-' * 80}")

eco_by_aspect = {}
for label, mask in aspect_strata:
    total = int(mask.sum())
    row = {}

    print(f"  {label:<22}", end="")
    for code in eco_codes:
        n = int(((ecozone == code) & mask).sum())
        row[code] = n
        print(f"  {pct(n, total):>13.1f}%", end="")
    eco_by_aspect[label] = (total, row)
    print(f"  {total:>12,}  {total * PIXEL_AREA_KM2:>8.1f}")

print(f"\n  South vs North delta (percentage points):")
print(f"  {'':<22}", end="")
s_total, s_row = eco_by_aspect["South-facing"]
n_total, n_row = eco_by_aspect["North-facing"]
flags = []
for code in eco_codes:
    delta = pct(s_row[code], s_total) - pct(n_row[code], n_total)
    flag = " ◄" if abs(delta) > 5 else ""
    flags.append(flag)
    print(f"  {delta:>+13.1f}%", end="")
print()

for code, flag in zip(eco_codes, flags):
    if flag:
        print(f"  ◄ {ecozone_label(code)} zone differs by >5 pp between aspects")

print(f"\n{'=' * 70}")
print("2. ECOZONE × FOREST TYPE")
print(f"{'=' * 70}")

for eco_code in eco_codes:
    eco_label = ecozone_label(eco_code)
    eco_mask = ecozone == eco_code
    eco_total = int(eco_mask.sum())

    print(f"\n  {eco_label} zone  ({eco_total:,} px  |  {eco_total * PIXEL_AREA_KM2:.1f} km²)")
    print(f"  {'Forest type':<32} {'pixels':>10}  {'km²':>8}  {'%':>6}")
    print(f"  {'-' * 64}")

    known = []
    for fc in forest_codes:
        n = int(((forest == fc) & eco_mask).sum())
        known.append((fc, n))
    known.sort(key=lambda x: -x[1])

    for fc, n in known:
        flabel = label_forest_group(fc)
        print(f"  {flabel:<32} {n:>10,}  {n * PIXEL_AREA_KM2:>8.1f}  {pct(n, eco_total):>6.1f}%")

print(f"\n{'=' * 70}")
print("FOREST TYPE × ECOZONE SUMMARY (% of each forest type in each zone)")
print(f"{'=' * 70}")
print(f"\n  {'Forest type':<32}", end="")
for code in eco_codes:
    print(f"  {ecozone_label(code) + ' %':>14}", end="")
print(f"  {'Total px':>12}")
print(f"  {'-' * 82}")

for fc in forest_codes:
    fc_mask = forest == fc
    fc_total = int(fc_mask.sum())
    flabel = label_forest_group(fc)

    print(f"  {flabel:<32}", end="")
    for eco_code in eco_codes:
        n = int(((ecozone == eco_code) & fc_mask).sum())
        print(f"  {pct(n, fc_total):>13.1f}%", end="")
    print(f"  {fc_total:>12,}")