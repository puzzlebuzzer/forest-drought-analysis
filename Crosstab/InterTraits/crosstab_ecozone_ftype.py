"""
ecozone_crosstab.py
--------------------
Two cross-tabulations:
  1. Ecozone × Aspect   — are south/north slopes unevenly distributed
                          across TNC thermal zones?
  2. Ecozone × Forest   — does species composition vary by ecozone?
"""

import rasterio
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
CACHE_BASE   = Path("/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/AOI/GWNF_cache")
TERRAIN_DIR  = CACHE_BASE / "traits" / "terrain"
FOREST_DIR   = CACHE_BASE / "traits" / "forest" / "forest_type_group"
ECOZONE_DIR  = CACHE_BASE / "traits" / "ecozone"
# ============================================================

PIXEL_AREA_KM2 = (10 * 10) / 1_000_000

ECOZONE_LABELS = {1: "Cool", 2: "Intermediate", 3: "Hot"}
FOREST_LABELS  = {
    0:   "Non-forest",
    400: "Oak/pine",
    500: "Oak/hickory",
    800: "Maple/beech/birch",
}

# ── Load rasters ──────────────────────────────────────────────
print("Loading rasters...")
with rasterio.open(TERRAIN_DIR / "mask_south_facing.tif") as src:
    south_mask = src.read(1).astype(bool)
with rasterio.open(TERRAIN_DIR / "mask_north_facing.tif") as src:
    north_mask = src.read(1).astype(bool)
with rasterio.open(FOREST_DIR / "forest_type_group.tif") as src:
    forest = src.read(1)
with rasterio.open(ECOZONE_DIR / "tnc_ecozone_simplified_snapped.tif") as src:
    ecozone = src.read(1)

# Crop all to minimum shared shape
rows_min = min(south_mask.shape[0], north_mask.shape[0],
               forest.shape[0], ecozone.shape[0])
cols_min = min(south_mask.shape[1], north_mask.shape[1],
               forest.shape[1], ecozone.shape[1])
if any(a.shape != (rows_min, cols_min)
       for a in [south_mask, north_mask, forest, ecozone]):
    print(f"  ⚠ Shape mismatch — cropping all to ({rows_min}, {cols_min})")
    south_mask = south_mask[:rows_min, :cols_min]
    north_mask = north_mask[:rows_min, :cols_min]
    forest     = forest[:rows_min, :cols_min]
    ecozone    = ecozone[:rows_min, :cols_min]

neither_mask = ~south_mask & ~north_mask
eco_codes    = sorted(ECOZONE_LABELS.keys())
forest_codes = sorted(FOREST_LABELS.keys())
total_px     = rows_min * cols_min

print(f"  South-facing pixels: {south_mask.sum():,}")
print(f"  North-facing pixels: {north_mask.sum():,}")

# ── Helper ────────────────────────────────────────────────────
def pct(n, total):
    return 100 * n / total if total > 0 else 0

# ============================================================
# 1. ECOZONE × ASPECT
# ============================================================
print(f"\n{'='*70}")
print("1. ECOZONE × ASPECT")
print("Are south/north slopes unevenly distributed across thermal zones?")
print(f"{'='*70}")

aspect_strata = [
    ("Whole AOI",          np.ones((rows_min, cols_min), dtype=bool)),
    ("South-facing",       south_mask),
    ("North-facing",       north_mask),
    ("Neither (E/W/flat)", neither_mask),
]

# Header
print(f"\n  {'Stratum':<22}", end="")
for code in eco_codes:
    label = ECOZONE_LABELS[code]
    print(f"  {label+' %':>14}", end="")
print(f"  {'Total px':>12}  {'km²':>8}")
print(f"  {'-'*80}")

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
    print(f"  {total:>12,}  {total*PIXEL_AREA_KM2:>8.1f}")

# South vs North delta
print(f"\n  South vs North delta (percentage points):")
print(f"  {'':<22}", end="")
s_total, s_row = eco_by_aspect["South-facing"]
n_total, n_row = eco_by_aspect["North-facing"]
flags = []
for code in eco_codes:
    delta = pct(s_row[code], s_total) - pct(n_row[code], n_total)
    flag  = " ◄" if abs(delta) > 5 else ""
    flags.append(flag)
    print(f"  {delta:>+13.1f}%", end="")
print()
for code, flag in zip(eco_codes, flags):
    if flag:
        label = ECOZONE_LABELS[code]
        print(f"  ◄ {label} zone differs by >5 pp between aspects")

# ============================================================
# 2. ECOZONE × FOREST TYPE
# ============================================================
print(f"\n{'='*70}")
print("2. ECOZONE × FOREST TYPE")
print("Does forest composition vary across thermal zones?")
print(f"{'='*70}")

for eco_code in eco_codes:
    eco_label = ECOZONE_LABELS[eco_code]
    eco_mask  = (ecozone == eco_code)
    eco_total = int(eco_mask.sum())

    print(f"\n  {eco_label} zone  ({eco_total:,} px  |  {eco_total*PIXEL_AREA_KM2:.1f} km²)")
    print(f"  {'Forest type':<25} {'pixels':>10}  {'km²':>8}  {'%':>6}")
    print(f"  {'-'*54}")

    known = []
    for fc in forest_codes:
        n = int(((forest == fc) & eco_mask).sum())
        known.append((fc, n))
    known.sort(key=lambda x: -x[1])

    for fc, n in known:
        flabel = FOREST_LABELS[fc]
        print(f"  {flabel:<25} {n:>10,}  {n*PIXEL_AREA_KM2:>8.1f}  {pct(n, eco_total):>6.1f}%")

    other_n = eco_total - sum(n for _, n in known)
    if other_n > 0:
        print(f"  {'Other/unknown':<25} {other_n:>10,}  {other_n*PIXEL_AREA_KM2:>8.1f}  {pct(other_n, eco_total):>6.1f}%")

# Forest type × ecozone summary table
print(f"\n{'='*70}")
print("FOREST TYPE × ECOZONE SUMMARY (% of each forest type in each zone)")
print(f"{'='*70}")
print(f"\n  {'Forest type':<25}", end="")
for code in eco_codes:
    print(f"  {ECOZONE_LABELS[code]+' %':>14}", end="")
print(f"  {'Total px':>12}")
print(f"  {'-'*80}")

for fc in forest_codes:
    fc_mask = (forest == fc)
    fc_total = int(fc_mask.sum())
    flabel = FOREST_LABELS[fc]
    print(f"  {flabel:<25}", end="")
    for eco_code in eco_codes:
        n = int(((ecozone == eco_code) & fc_mask).sum())
        print(f"  {pct(n, fc_total):>13.1f}%", end="")
    print(f"  {fc_total:>12,}")

print(f"\n{'='*70}")
print("INTERPRETATION GUIDE")
print(f"{'='*70}")
print("""
Ecozone × Aspect:
  If south slopes are disproportionately in the Hot zone:
    → Thermal zone may be confounding the aspect signal.
  If distribution is similar:
    → Aspect effect is independent of TNC thermal zonation.

Ecozone × Forest type:
  If oak/hickory dominates the Hot zone and maple/beech dominates Cool:
    → Ecozone and forest type are correlated stratifiers.
  If forest type is similar across ecozones:
    → Ecozone adds independent information for drought analysis.
""")