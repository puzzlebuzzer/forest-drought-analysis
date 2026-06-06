#!/usr/bin/env python3

"""
crosstab_aspect_ftype.py
------------------------
Cross-tabulates aspect masks against FIA forest type codes.
"""

import numpy as np
import rasterio

from src.aoi import get_aoi_config
from src.cli import make_parser, add_aoi_arg
from src.labels import FOREST_TYPE_LABELS

parser = make_parser("Cross-tabulate aspect masks against detailed forest types")
add_aoi_arg(parser)
args = parser.parse_args()

AOI = args.aoi
cfg = get_aoi_config(AOI)

TERRAIN_DIR = cfg.terrain_dir
SPECIES_PATH = cfg.species_raster

PIXEL_AREA_KM2 = (10 * 10) / 1_000_000

print("Loading masks...")
with rasterio.open(TERRAIN_DIR / "mask_south_facing.tif") as src:
    south_mask = src.read(1) == 1
with rasterio.open(TERRAIN_DIR / "mask_north_facing.tif") as src:
    north_mask = src.read(1) == 1
with rasterio.open(SPECIES_PATH) as src:
    species = src.read(1)

# --- grid sanity check ---
if south_mask.shape != species.shape or north_mask.shape != species.shape:
    raise ValueError(
        f"Grid mismatch: south={south_mask.shape}, "
        f"north={north_mask.shape}, species={species.shape}"
    )

print(f"  South-facing pixels: {south_mask.sum():,}")
print(f"  North-facing pixels: {north_mask.sum():,}")

all_codes = sorted(np.unique(species))

def stratum_counts(mask: np.ndarray) -> tuple[int, dict[int, int]]:
    total = int(mask.sum())
    counts = {}
    for code in all_codes:
        counts[int(code)] = int(((species == code) & mask).sum())
    return total, counts

whole_total, whole_counts = stratum_counts(np.ones(species.shape, dtype=bool))
south_total, south_counts = stratum_counts(south_mask)
north_total, north_counts = stratum_counts(north_mask)

def print_stratum(label: str, total: int, counts: dict[int, int]) -> None:
    print(f"\n{label}  ({total:,} px  |  {total * PIXEL_AREA_KM2:.1f} km²)")
    print(f"  {'Species/Type':<42} {'pixels':>10}  {'km²':>8}  {'%':>6}")
    print(f"  {'-' * 72}")

    known = [(c, counts[c]) for c in all_codes if c in FOREST_TYPE_LABELS and counts[c] > 0]
    known.sort(key=lambda x: -x[1])
    for code, n in known:
        label_str = FOREST_TYPE_LABELS[int(code)]
        pct = 100 * n / total if total > 0 else 0
        print(f"  {label_str:<42} {n:>10,}  {n * PIXEL_AREA_KM2:>8.1f}  {pct:>6.1f}%")

    unknown_n = sum(counts[c] for c in all_codes if c not in FOREST_TYPE_LABELS)
    if unknown_n > 0:
        pct = 100 * unknown_n / total if total > 0 else 0
        print(f"  {'Other/unknown':<42} {unknown_n:>10,}  {unknown_n * PIXEL_AREA_KM2:>8.1f}  {pct:>6.1f}%")

print(f"\n{'=' * 70}")
print(f"SPECIES COMPOSITION BY ASPECT ({AOI})")
print(f"{'=' * 70}")
print_stratum("Whole AOI", whole_total, whole_counts)
print_stratum("South-facing", south_total, south_counts)
print_stratum("North-facing", north_total, north_counts)

print(f"\n{'=' * 70}")
print("SOUTH vs NORTH — SPECIES COMPOSITION DIFFERENCE")
print(f"{'=' * 70}")
print(f"\n  {'Species/Type':<42} {'South %':>8}  {'North %':>8}  {'Δ (S-N)':>8}")
print(f"  {'-' * 72}")

deltas = []
for code in all_codes:
    if int(code) == 0:
        continue
    s_n = south_counts[int(code)]
    n_n = north_counts[int(code)]
    if s_n + n_n < 1000:
        continue
    s_pct = 100 * s_n / south_total if south_total > 0 else 0
    n_pct = 100 * n_n / north_total if north_total > 0 else 0
    deltas.append((int(code), s_pct, n_pct, s_pct - n_pct))

deltas.sort(key=lambda x: -abs(x[3]))

for code, s_pct, n_pct, delta in deltas:
    label_str = FOREST_TYPE_LABELS.get(code, f"Code {code}")
    flag = " ◄" if abs(delta) > 3 else ""
    print(f"  {label_str:<42} {s_pct:>8.1f}%  {n_pct:>8.1f}%  {delta:>+8.1f}%{flag}")

print("\n  ◄ = difference > 3 percentage points")

oak_codes = [501, 502, 503, 504, 505, 506, 510, 515]
# {
#     501: "Post oak / blackjack oak",
#     502: "Chestnut oak",
#     503: "White oak / red oak / hickory",
#     504: "White oak",
#     505: "Northern red oak",
#     506: "Yellow-poplar / white oak / N. red oak",
#     510: "Scarlet oak",
#     515: "Chestnut oak / black oak / scarlet oak",
# }

print(f"\n{'=' * 70}")
print("OAK SPECIES BREAKDOWN (south vs north, forest pixels only)")
print(f"{'=' * 70}")

south_forest = south_total - south_counts.get(0, 0)
north_forest = north_total - north_counts.get(0, 0)

print(f"\n  {'Oak species':<28} {'South %':>8}  {'North %':>8}  {'Δ (S-N)':>8}")
print(f"  {'-' * 58}")
for code in oak_codes:
    label_str = FOREST_TYPE_LABELS.get(code, f"Code {code}")
    s_pct = 100 * south_counts.get(code, 0) / south_forest if south_forest > 0 else 0
    n_pct = 100 * north_counts.get(code, 0) / north_forest if north_forest > 0 else 0
    delta = s_pct - n_pct
    flag = " ◄" if abs(delta) > 3 else ""
    print(f"  {label_str:<28} {s_pct:>8.1f}%  {n_pct:>8.1f}%  {delta:>+8.1f}%{flag}")

print("\n  Denominator: forested pixels only")
print(f"  South forested: {south_forest:,} px  |  North forested: {north_forest:,} px")