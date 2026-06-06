#!/usr/bin/env python3

import subprocess
from pathlib import Path

import numpy as np
import rasterio

from src.aoi import get_aoi_config
from src.labels import FOREST_GROUP_LABELS

from src.aoi import get_aoi_config
from src.cli import make_parser, add_aoi_arg

parser = make_parser("Prepare trait masks")
add_aoi_arg(parser)
args = parser.parse_args()

AOI = args.aoi
cfg = get_aoi_config(AOI)

TERRAIN_DIR = cfg.terrain_dir
FOREST_GROUP_DIR = cfg.forest_group_dir
ECOZONE_DIR = cfg.ecozone_dir

ECOZONE_SRC = cfg.raw_ecozone_raster
ECOZONE_SNAP = cfg.ecozone_dir / "tnc_ecozone_simplified_snapped.tif"

CANONICAL_REF = cfg.terrain_dir / "elevation.tif"

ELEVATION_BREAKS = {"low": (None, 175), "mid": (175, 300), "high": (300, None)}
MIN_FOREST_GROUP_PIXELS = 10000
ECOZONE_CODES = {1: "cool", 2: "intermediate", 3: "hot"}

PIXEL_AREA_KM2 = (10 * 10) / 1_000_000


def write_mask(path: Path, data: np.ndarray, profile: dict) -> None:
    out = np.where(data, np.uint8(1), np.uint8(255))
    profile.update(dtype=rasterio.uint8, count=1, nodata=255)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)
    n = int(data.sum())
    print(f"  {path.name}: {n:,} pixels ({n * PIXEL_AREA_KM2:.1f} km²)")


def check_and_warp(
    src_path: Path,
    snap_path: Path,
    canonical_ref_path: Path,
    label: str,
) -> Path:
    with rasterio.open(canonical_ref_path) as ref:
        canonical_crs = ref.crs
        canonical_transform = ref.transform
        canonical_bounds = ref.bounds
        canonical_width = ref.width
        canonical_height = ref.height

    needs_warp = False

    with rasterio.open(src_path) as src:
        if src.crs != canonical_crs:
            print(f"  ⚠ {label} CRS mismatch — warp needed")
            needs_warp = True
        elif src.width != canonical_width or src.height != canonical_height:
            print(
                f"  ⚠ {label} shape mismatch "
                f"({src.height}x{src.width} vs {canonical_height}x{canonical_width}) — warp needed"
            )
            needs_warp = True
        elif src.transform != canonical_transform:
            print(f"  ⚠ {label} transform mismatch — warp needed")
            needs_warp = True
        else:
            print(f"  ✓ {label} already aligned with canonical grid")

    if not needs_warp:
        return src_path

    print(f"  Running exact-grid gdalwarp for {label}...")

    cmd = [
        "gdalwarp",
        "-overwrite",
        "-t_srs", str(canonical_crs),
        "-te",
        str(canonical_bounds.left),
        str(canonical_bounds.bottom),
        str(canonical_bounds.right),
        str(canonical_bounds.top),
        "-ts",
        str(canonical_width),
        str(canonical_height),
        "-r", "near",
        "-of", "GTiff",
        "-co", "COMPRESS=LZW",
        "-co", "TILED=YES",
        str(src_path),
        str(snap_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ✗ gdalwarp failed:\n{result.stderr}")
        raise RuntimeError(f"gdalwarp failed for {label}")

    print(f"  ✓ Warp complete: {snap_path.name}")
    return snap_path

print("Loading canonical grid reference...")
with rasterio.open(CANONICAL_REF) as ref:
    canonical_crs = ref.crs
    canonical_transform = ref.transform
    canonical_res = (ref.res[0], ref.res[1])
    canonical_shape = (ref.height, ref.width)
    canonical_profile = ref.profile.copy()

print(f"  AOI: {AOI}")
print(f"  CRS: {canonical_crs}")
print(f"  Resolution: {canonical_res[0]}m x {canonical_res[1]}m")
print(f"  Shape: {canonical_shape}")

print("\n--- Elevation band masks ---")
with rasterio.open(TERRAIN_DIR / "elevation.tif") as src:
    elev = src.read(1).astype(float)
    nodata = src.nodata
if nodata is not None:
    elev[elev == nodata] = np.nan

for band, (lo, hi) in ELEVATION_BREAKS.items():
    mask = np.ones(elev.shape, dtype=bool)
    if lo is not None:
        mask &= elev >= lo
    if hi is not None:
        mask &= elev < hi
    mask &= ~np.isnan(elev)
    write_mask(TERRAIN_DIR / f"mask_elev_{band}.tif", mask, canonical_profile.copy())

print("\n--- Aspect masks ---")
with rasterio.open(TERRAIN_DIR / "aspect.tif") as src:
    aspect = src.read(1).astype(float)
    nodata = src.nodata
if nodata is not None:
    aspect[aspect == nodata] = np.nan
valid_aspect = ~np.isnan(aspect)

south_mask = valid_aspect & (aspect >= 135) & (aspect <= 225)
write_mask(TERRAIN_DIR / "mask_south_facing.tif", south_mask, canonical_profile.copy())

north_mask = valid_aspect & ((aspect >= 315) | (aspect <= 45))
write_mask(TERRAIN_DIR / "mask_north_facing.tif", north_mask, canonical_profile.copy())

with rasterio.open(TERRAIN_DIR / "slope.tif") as src:
    slope = src.read(1).astype(float)
    nodata = src.nodata
if nodata is not None:
    slope[slope == nodata] = np.nan

steep_mask = ~np.isnan(slope) & (slope > 15)
write_mask(TERRAIN_DIR / "mask_steep_slopes.tif", steep_mask, canonical_profile.copy())

print("\n--- Forest type masks ---")
forest_path = FOREST_GROUP_DIR / "forest_type_group.tif"
with rasterio.open(forest_path) as src:
    forest = src.read(1)

codes, counts = np.unique(forest, return_counts=True)

non_forest = 0
built = []

for code, count in zip(codes, counts):
    code = int(code)
    count = int(count)

    if code == 0:
        non_forest = count
        continue

    if count < MIN_FOREST_GROUP_PIXELS:
        label = FOREST_GROUP_LABELS.get(code, f"code_{code}")
        print(
            f"  Skipping code {code} ({label}) — only {count:,} px "
            f"({count * PIXEL_AREA_KM2:.2f} km²)"
        )
        continue

    label = FOREST_GROUP_LABELS.get(code, f"code_{code}")
    mask = forest == code
    write_mask(
        FOREST_GROUP_DIR / f"mask_forest_{code}.tif",
        mask,
        canonical_profile.copy(),
    )
    built.append((code, label, count))

print(f"  Non-forest pixels excluded (code 0): {non_forest:,} ({non_forest * PIXEL_AREA_KM2:.1f} km²)")

if built:
    print("\n  Forest group masks created:")
    for code, label, count in built:
        print(f"    {code}: {label}  ({count:,} px, {count * PIXEL_AREA_KM2:.1f} km²)")
else:
    print("\n  No forest group masks were created above threshold.")

import json

summary = {
    str(code): {
        "label": label,
        "pixels": count,
        "km2": count * PIXEL_AREA_KM2,
    }
    for code, label, count in built
}

inventory_path = FOREST_GROUP_DIR / "forest_group_inventory.json"
with open(inventory_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Wrote forest group inventory → {inventory_path}")

print("\n--- Ecozone alignment check ---")
ecozone_path = check_and_warp(
    ECOZONE_SRC,
    ECOZONE_SNAP,
    CANONICAL_REF,
    "tnc_ecozone",
)

print("\n--- Ecozone masks ---")
with rasterio.open(ecozone_path) as src:
    ecozone = src.read(1)

for code, label in ECOZONE_CODES.items():
    mask = ecozone == code
    write_mask(ECOZONE_DIR / f"mask_ecozone_{code}.tif", mask, canonical_profile.copy())

print("\n✓ All masks generated successfully.")
print(f"Output directories:\n  terrain={TERRAIN_DIR}\n  forest={FOREST_GROUP_DIR}\n  ecozone={ECOZONE_DIR}")