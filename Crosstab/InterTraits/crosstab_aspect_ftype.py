"""
aspect_species_crosstab.py
---------------------------
Cross-tabulates aspect masks (south/north) against FIA forest type
(species-level) to check whether within-group species composition
differs between aspects.
"""

import rasterio
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
CACHE_BASE   = Path("/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/AOI/GWNF_cache")
TERRAIN_DIR  = CACHE_BASE / "traits" / "terrain"
SPECIES_PATH = CACHE_BASE / "traits" / "forest" / "forest_type_type" / "forest_type_type.tif"
# ============================================================

PIXEL_AREA_KM2 = (10 * 10) / 1_000_000

# FIA forest type code lookup — add more as needed
SPECIES_LABELS = {
    0:   "Non-forest",
    103: "Pitch pine",
    163: "Table Mountain pine",
    167: "Loblolly pine",
    401: "White/Virginia pine mix",
    409: "Virginia pine",
    502: "Scarlet oak",
    503: "White oak",
    504: "Bur oak",
    505: "Pin oak",
    506: "Northern red oak",
    508: "Chestnut oak",
    520: "Black oak",
    801: "Sugar maple",
    802: "Black cherry",
}

# ── Load masks ────────────────────────────────────────────────
print("Loading masks...")
with rasterio.open(TERRAIN_DIR / "mask_south_facing.tif") as src:
    south_mask = src.read(1).astype(bool)
with rasterio.open(TERRAIN_DIR / "mask_north_facing.tif") as src:
    north_mask = src.read(1).astype(bool)
with rasterio.open(SPECIES_PATH) as src:
    species = src.read(1)

print(f"  South-facing pixels: {south_mask.sum():,}")
print(f"  North-facing pixels: {north_mask.sum():,}")

# Crop to minimum shared shape
rows_min = min(south_mask.shape[0], north_mask.shape[0], species.shape[0])
cols_min = min(south_mask.shape[1], north_mask.shape[1], species.shape[1])
if species.shape != (rows_min, cols_min):
    print(f"  ⚠ Shape mismatch — cropping all to ({rows_min}, {cols_min})")
    south_mask = south_mask[:rows_min, :cols_min]
    north_mask = north_mask[:rows_min, :cols_min]
    species    = species[:rows_min, :cols_min]

all_codes = sorted(np.unique(species))

# ── Cross-tabulation function ─────────────────────────────────
def stratum_counts(mask):
    total = int(mask.sum())
    counts = {}
    for code in all_codes:
        counts[code] = int(((species == code) & mask).sum())
    return total, counts

# ── Compute strata ────────────────────────────────────────────
whole_total, whole_counts = stratum_counts(np.ones(species.shape, dtype=bool))
south_total, south_counts = stratum_counts(south_mask)
north_total, north_counts = stratum_counts(north_mask)

# ── Print full composition tables ─────────────────────────────
def print_stratum(label, total, counts):
    print(f"\n{label}  ({total:,} px  |  {total * PIXEL_AREA_KM2:.1f} km²)")
    print(f"  {'Species/Type':<28} {'pixels':>10}  {'km²':>8}  {'%':>6}")
    print(f"  {'-'*58}")
    # Known codes first, sorted by pixel count descending
    known = [(c, counts[c]) for c in all_codes if c in SPECIES_LABELS and counts[c] > 0]
    known.sort(key=lambda x: -x[1])
    for code, n in known:
        label_str = SPECIES_LABELS[code]
        pct = 100 * n / total if total > 0 else 0
        print(f"  {label_str:<28} {n:>10,}  {n*PIXEL_AREA_KM2:>8.1f}  {pct:>6.1f}%")
    # Unknown codes
    unknown_n = sum(counts[c] for c in all_codes if c not in SPECIES_LABELS)
    if unknown_n > 0:
        pct = 100 * unknown_n / total
        print(f"  {'Other/unknown':<28} {unknown_n:>10,}  {unknown_n*PIXEL_AREA_KM2:>8.1f}  {pct:>6.1f}%")

print(f"\n{'='*70}")
print("SPECIES COMPOSITION BY ASPECT")
print(f"{'='*70}")
print_stratum("Whole AOI",     whole_total, whole_counts)
print_stratum("South-facing",  south_total, south_counts)
print_stratum("North-facing",  north_total, north_counts)

# ── South vs North delta table ────────────────────────────────
print(f"\n{'='*70}")
print("SOUTH vs NORTH — SPECIES COMPOSITION DIFFERENCE")
print(f"{'='*70}")
print(f"\n  {'Species/Type':<28} {'South %':>8}  {'North %':>8}  {'Δ (S-N)':>8}")
print(f"  {'-'*58}")

# Sort by absolute delta descending, excluding non-forest and tiny counts
deltas = []
for code in all_codes:
    if code == 0:
        continue
    s_n = south_counts[code]
    n_n = north_counts[code]
    if s_n + n_n < 1000:
        continue
    s_pct = 100 * s_n / south_total
    n_pct = 100 * n_n / north_total
    deltas.append((code, s_pct, n_pct, s_pct - n_pct))

deltas.sort(key=lambda x: -abs(x[3]))

for code, s_pct, n_pct, delta in deltas:
    label_str = SPECIES_LABELS.get(code, f"Code {code}")
    flag = " ◄" if abs(delta) > 3 else ""
    print(f"  {label_str:<28} {s_pct:>8.1f}%  {n_pct:>8.1f}%  {delta:>+8.1f}%{flag}")

print(f"\n  ◄ = difference > 3 percentage points")

# ── Focus: oak species breakdown ─────────────────────────────
oak_codes = [502, 503, 504, 505, 506, 508, 520]
print(f"\n{'='*70}")
print("OAK SPECIES BREAKDOWN (south vs north, forest pixels only)")
print(f"{'='*70}")

# Exclude non-forest from denominator
south_forest = south_total - south_counts.get(0, 0)
north_forest = north_total - north_counts.get(0, 0)

print(f"\n  {'Oak species':<28} {'South %':>8}  {'North %':>8}  {'Δ (S-N)':>8}")
print(f"  {'-'*58}")
for code in oak_codes:
    label_str = SPECIES_LABELS.get(code, f"Code {code}")
    s_pct = 100 * south_counts.get(code, 0) / south_forest if south_forest > 0 else 0
    n_pct = 100 * north_counts.get(code, 0) / north_forest if north_forest > 0 else 0
    delta = s_pct - n_pct
    flag  = " ◄" if abs(delta) > 3 else ""
    print(f"  {label_str:<28} {s_pct:>8.1f}%  {n_pct:>8.1f}%  {delta:>+8.1f}%{flag}")

print(f"\n  Denominator: forested pixels only")
print(f"  South forested: {south_forest:,} px  |  North forested: {north_forest:,} px")