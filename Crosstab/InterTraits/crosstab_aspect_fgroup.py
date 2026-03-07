"""
aspect_forest_crosstab.py
--------------------------
Cross-tabulates aspect masks (south/north) against forest type group
to check whether the south vs north NDVI/NDMI signal could be explained
by differences in species composition between aspects.
"""

import rasterio
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
CACHE_BASE  = Path("/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/AOI/GWNF_cache")
TERRAIN_DIR = CACHE_BASE / "traits" / "terrain"
FOREST_DIR  = CACHE_BASE / "traits" / "forest" / "forest_type_group"
# ============================================================

FOREST_LABELS = {
    0:   "Non-forest",
    400: "Oak/pine",
    500: "Oak/hickory",
    800: "Maple/beech/birch",
}
PIXEL_AREA_KM2 = (10 * 10) / 1_000_000

# ── Load masks ────────────────────────────────────────────────
print("Loading masks...")
with rasterio.open(TERRAIN_DIR / "mask_south_facing.tif") as src:
    south_mask = src.read(1).astype(bool)
with rasterio.open(TERRAIN_DIR / "mask_north_facing.tif") as src:
    north_mask = src.read(1).astype(bool)
with rasterio.open(FOREST_DIR / "forest_type_group.tif") as src:
    forest = src.read(1)

print(f"  South-facing pixels: {south_mask.sum():,}")
print(f"  North-facing pixels: {north_mask.sum():,}")

# Crop all arrays to minimum shared shape
rows_min = min(south_mask.shape[0], north_mask.shape[0], forest.shape[0])
cols_min = min(south_mask.shape[1], north_mask.shape[1], forest.shape[1])
if (south_mask.shape != (rows_min, cols_min) or
    north_mask.shape != (rows_min, cols_min) or
    forest.shape     != (rows_min, cols_min)):
    print(f"\n  ⚠ Shape mismatch detected — cropping all to ({rows_min}, {cols_min})")
    south_mask = south_mask[:rows_min, :cols_min]
    north_mask = north_mask[:rows_min, :cols_min]
    forest     = forest[:rows_min, :cols_min]

# ── Cross-tabulation ──────────────────────────────────────────
all_codes = sorted(np.unique(forest))

# Build rows for each stratum
def stratum_row(label, mask):
    total = mask.sum()
    row = {"stratum": label, "total_px": total, "total_km2": total * PIXEL_AREA_KM2}
    for code in all_codes:
        n = int(((forest == code) & mask).sum())
        row[code] = n
    return row

rows = [
    stratum_row("Whole AOI",         np.ones(forest.shape, dtype=bool)),
    stratum_row("South-facing",      south_mask),
    stratum_row("North-facing",      north_mask),
    stratum_row("Neither (E/W/flat)", ~south_mask & ~north_mask),
]

# ── Print results ─────────────────────────────────────────────
print(f"\n{'='*70}")
print("FOREST TYPE COMPOSITION BY ASPECT")
print(f"{'='*70}")

# Header
known_codes = [c for c in all_codes if c in FOREST_LABELS]
other_codes = [c for c in all_codes if c not in FOREST_LABELS]

for row in rows:
    total = row["total_px"]
    print(f"\n{row['stratum']}  ({total:,} px  |  {row['total_km2']:.1f} km²)")
    print(f"  {'Forest type':<25} {'pixels':>10}  {'km²':>8}  {'%':>6}")
    print(f"  {'-'*55}")
    for code in known_codes:
        n = row[code]
        label = FOREST_LABELS[code]
        pct = 100 * n / total if total > 0 else 0
        print(f"  {label:<25} {n:>10,}  {n*PIXEL_AREA_KM2:>8.1f}  {pct:>6.1f}%")
    if other_codes:
        other_n = sum(row[c] for c in other_codes)
        pct = 100 * other_n / total if total > 0 else 0
        print(f"  {'Other/unknown':<25} {other_n:>10,}  {other_n*PIXEL_AREA_KM2:>8.1f}  {pct:>6.1f}%")

# ── South vs North comparison ─────────────────────────────────
print(f"\n{'='*70}")
print("SOUTH vs NORTH — COMPOSITION DIFFERENCE")
print(f"{'='*70}")
print(f"\n  {'Forest type':<25} {'South %':>8}  {'North %':>8}  {'Δ (S-N)':>8}")
print(f"  {'-'*55}")

south_row = rows[1]
north_row = rows[2]

for code in known_codes:
    label = FOREST_LABELS[code]
    s_pct = 100 * south_row[code] / south_row["total_px"]
    n_pct = 100 * north_row[code] / north_row["total_px"]
    delta = s_pct - n_pct
    flag  = " ◄" if abs(delta) > 5 else ""
    print(f"  {label:<25} {s_pct:>8.1f}%  {n_pct:>8.1f}%  {delta:>+8.1f}%{flag}")

print(f"\n  ◄ = difference > 5 percentage points")
print(f"\n{'='*70}")
print("INTERPRETATION GUIDE")
print(f"{'='*70}")
print("""
If oak/hickory (500) is much higher on south-facing slopes:
  → Species composition is likely confounding the aspect signal.
  → Next step: compare index values within the same forest type.

If composition is similar between aspects:
  → The aspect signal is more likely driven by microclimate,
    soil moisture, or canopy structure differences.
""")