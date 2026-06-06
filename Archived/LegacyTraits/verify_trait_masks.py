#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import rasterio

from src.aoi import get_aoi_config
from src.cli import make_parser, add_aoi_arg
from src.labels import ECOZONE_LABELS, FOREST_GROUP_LABELS

parser = make_parser("Verify alignment and contents of trait rasters and masks")
add_aoi_arg(parser)
args = parser.parse_args()

AOI = args.aoi
cfg = get_aoi_config(AOI)

CACHE_BASE = cfg.cache_root
TERRAIN_DIR = cfg.terrain_dir
FOREST_DIR = cfg.forest_group_dir
ECOZONE_DIR = cfg.ecozone_dir

CANONICAL_REF = TERRAIN_DIR / "elevation.tif"

with rasterio.open(CANONICAL_REF) as ref:
    CANONICAL_SHAPE = (ref.height, ref.width)
    CANONICAL_BOUNDS = (
        ref.bounds.left,
        ref.bounds.bottom,
        ref.bounds.right,
        ref.bounds.top,
    )
    CANONICAL_CRS = ref.crs
    CANONICAL_TRANSFORM = ref.transform

def check_alignment(src: rasterio.io.DatasetReader) -> list[str]:
    issues = []
    crs_epsg = src.crs.to_epsg() if src.crs else None
    res = (abs(src.res[0]), abs(src.res[1]))
    b = src.bounds

    if crs_epsg != 32617:
        issues.append(f"  CRS: {src.crs}  (expected EPSG:32617)")
    else:
        print("  ✓  CRS: EPSG:32617")

    if abs(res[0] - 10.0) > 0.01 or abs(res[1] - 10.0) > 0.01:
        issues.append(f"  Resolution: {res}  (expected ~10m x 10m)")
    else:
        print("  ✓  Resolution: ~10m x 10m")

    if src.height != CANONICAL_SHAPE[0] or src.width != CANONICAL_SHAPE[1]:
        issues.append(
            f"  Shape: {src.height} x {src.width}  "
            f"(expected {CANONICAL_SHAPE[0]} x {CANONICAL_SHAPE[1]})"
        )
    else:
        print(f"  ✓  Shape: {src.height} rows x {src.width} cols")

    return issues


def print_issues(issues: list[str]) -> None:
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  ✓  No alignment issues")


print(f"\n{'=' * 60}\nELEVATION\n{'=' * 60}")
with rasterio.open(TERRAIN_DIR / "elevation.tif") as src:
    issues = check_alignment(src)
    data = src.read(1, masked=True)
    valid = data.compressed()
    print(f"  ✓  Valid pixels: {len(valid):,}  ({len(valid) / data.size * 100:.1f}%)")
    print(f"  ✓  Range: {valid.min():.0f}m – {valid.max():.0f}m")
    print(
        f"  ✓  P25={np.percentile(valid, 25):.0f}m  "
        f"P50={np.percentile(valid, 50):.0f}m  "
        f"P75={np.percentile(valid, 75):.0f}m  "
        f"Mean={valid.mean():.0f}m"
    )
    if valid.min() < 0:
        issues.append(f"  Negative elevation values (min={valid.min():.0f}m)")
    print_issues(issues)

print(f"\n{'=' * 60}\nTERRAIN MASKS\n{'=' * 60}")
for fname, label in [
    ("mask_south_facing.tif", "South-facing slopes   "),
    ("mask_north_facing.tif", "North-facing slopes   "),
    ("mask_steep_slopes.tif", "Steep slopes          "),
    ("mask_elev_low.tif", "Elevation low  <175m  "),
    ("mask_elev_mid.tif", "Elevation mid  175-300m"),
    ("mask_elev_high.tif", "Elevation high >300m  "),
]:
    fpath = TERRAIN_DIR / fname
    if not fpath.exists():
        print(f"  MISSING: {fname}")
        continue
    with rasterio.open(fpath) as src:
        data = src.read(1)
        count = int((data == 1).sum())
        print(f"  ✓  {label}  {count:>10,} px  ({count * 100 / 1e6:6.1f} km²)")

print(f"\n{'=' * 60}\nFIA FOREST TYPE GROUP RASTER\n{'=' * 60}")
with rasterio.open(FOREST_DIR / "forest_type_group.tif") as src:
    issues = check_alignment(src)
    data = src.read(1, masked=True)
    codes, counts = np.unique(data.compressed(), return_counts=True)
    total = counts.sum()
    for code, count in zip(codes, counts):
        label = FOREST_GROUP_LABELS.get(int(code), "Unknown")
        area = count * 100 / 1e6
        pct = count / total * 100
        mk = "✓" if int(code) in FOREST_GROUP_LABELS else "?"
        print(f"  {mk}  Code {int(code):4d}  {area:7.1f} km²  ({pct:5.1f}%)  {label}")
    print_issues(issues)

print(f"\n{'=' * 60}\nFOREST TYPE MASKS  (traits/forest/forest_type_group/)\n{'=' * 60}")
import json

inventory_path = FOREST_DIR / "forest_group_inventory.json"

with open(inventory_path, "r", encoding="utf-8") as f:
    inventory = json.load(f)

for code_str in sorted(inventory.keys(), key=int):
    code = int(code_str)
    label = FOREST_GROUP_LABELS.get(code, "Unknown")
    fname = f"mask_forest_{code}.tif"
    fpath = FOREST_DIR / fname

    if not fpath.exists():
        print(f"  MISSING: {fname}")
        continue

    with rasterio.open(fpath) as src:
        data = src.read(1)
        count = int((data == 1).sum())

    print(f"  ✓  {fname}  ({label})  {count:,} px  ({count * 100 / 1e6:.1f} km²)")

dupes = sorted(TERRAIN_DIR.glob("mask_forest_*.tif"))
if dupes:
    print("\n  Note: forest type masks also found in traits/terrain/")
    print("  confirm these are identical to the ones in traits/forest/")
    for f in dupes:
        print(f"    {f.name}")

print(f"\n{'=' * 60}\nTNC ECOZONE RASTER\n{'=' * 60}")
for label, path in [
    ("Snapped (use this)", ECOZONE_DIR / "tnc_ecozone_simplified_snapped.tif"),
    ("Raw pre-snap", cfg.raw_ecozone_raster),
]:
    print(f"\n  [{label}]")
    if not path.exists():
        print(f"  NOT FOUND: {path}")
        continue
    with rasterio.open(path) as src:
        issues = check_alignment(src)
        data = src.read(1, masked=True)
        codes, counts = np.unique(data.compressed(), return_counts=True)
        total = counts.sum()
        for code, count in zip(codes, counts):
            lbl = ECOZONE_LABELS.get(int(code), f"Unknown ({int(code)})")
            area = count * 100 / 1e6
            pct = count / total * 100
            print(f"  ✓  Code {int(code)}  {area:7.1f} km²  ({pct:5.1f}%)  {lbl}")
        print_issues(issues)

print(f"\n{'=' * 60}\nECOZONE MASKS\n{'=' * 60}")
for fname in ["mask_ecozone_1.tif", "mask_ecozone_2.tif", "mask_ecozone_3.tif"]:
    code = int(fname.replace("mask_ecozone_", "").replace(".tif", ""))
    label = ECOZONE_LABELS.get(code, "Unknown")
    fpath = ECOZONE_DIR / fname
    if not fpath.exists():
        print(f"  MISSING: {fname}")
        continue
    with rasterio.open(fpath) as src:
        data = src.read(1)
        count = int((data == 1).sum())
        print(f"  ✓  {fname}  ({label:<15})  {count:,} px  ({count * 100 / 1e6:.1f} km²)")

print(f"\n{'=' * 60}\nVERIFICATION COMPLETE\n{'=' * 60}")