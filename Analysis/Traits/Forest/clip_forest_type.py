#!/usr/bin/env python3
"""
Clip and snap the CONUS forest type raster to the canonical AOI grid.

Input:
    CONUS forest type raster (type/species-level codes)

Output:
    traits/forest/forest_type_type/forest_type_type.tif
"""

import subprocess
from pathlib import Path

import numpy as np
import rasterio

from src.aoi import get_aoi_config
from src.cli import make_parser, add_aoi_arg
from src.labels import FOREST_TYPE_LABELS
from src.paths import project_path


parser = make_parser("Clip and snap the CONUS forest type raster to an AOI")
add_aoi_arg(parser)
args = parser.parse_args()

AOI = args.aoi
cfg = get_aoi_config(AOI)

CONUS_SRC = project_path("conus_forest_type_img")
FOREST_TYPE_DIR = cfg.forest_type_dir
CANONICAL_REF = cfg.terrain_dir / "elevation.tif"
OUT_RASTER = FOREST_TYPE_DIR / "forest_type_type.tif"

FOREST_TYPE_DIR.mkdir(parents=True, exist_ok=True)

print(f"AOI:    {AOI}")
print(f"Output: {OUT_RASTER}")

print("\nReading canonical grid...")
with rasterio.open(CANONICAL_REF) as ref:
    crs = ref.crs
    res = ref.res
    shape = (ref.height, ref.width)
    bounds = ref.bounds

print(f"  CRS:        {crs}")
print(f"  Resolution: {res[0]}m x {res[1]}m")
print(f"  Shape:      {shape[0]} x {shape[1]}")
print(f"  Bounds:     {bounds.left:.1f}, {bounds.bottom:.1f}, {bounds.right:.1f}, {bounds.top:.1f}")

print("\nChecking source raster...")
if not CONUS_SRC.exists():
    raise FileNotFoundError(f"Source not found: {CONUS_SRC}")

with rasterio.open(CONUS_SRC) as src:
    print(f"  CRS:        {src.crs}")
    print(f"  Resolution: {src.res[0]:.1f}m x {src.res[1]:.1f}m")
    print(f"  Shape:      {src.height} x {src.width}")
    print(f"  Dtype:      {src.dtypes[0]}")

print("\nRunning gdalwarp...")
cmd = [
    "gdalwarp",
    "-overwrite",
    "-t_srs", f"EPSG:{crs.to_epsg()}",
    "-te",
    str(bounds.left),
    str(bounds.bottom),
    str(bounds.right),
    str(bounds.top),
    "-ts",
    str(shape[1]),   # width
    str(shape[0]),   # height
    "-r", "near",
    "-of", "GTiff",
    "-co", "COMPRESS=LZW",
    "-co", "TILED=YES",
    str(CONUS_SRC),
    str(OUT_RASTER),
]

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"✗ gdalwarp failed:\n{result.stderr}")
    raise RuntimeError("gdalwarp failed — check paths and GDAL installation")

print(f"✓ Warp complete: {OUT_RASTER.name}")

print("\nValidating output...")
with rasterio.open(OUT_RASTER) as src:
    out_shape = (src.height, src.width)
    out_res = src.res
    data = src.read(1)
    nodata = src.nodata

    codes, counts = np.unique(data, return_counts=True)
    total = data.size

    shape_ok = (out_shape == shape)
    res_ok = abs(out_res[0] - res[0]) < 0.01 and abs(out_res[1] - res[1]) < 0.01

    print(f"  Shape:      {out_shape}  {'✓' if shape_ok else '⚠ mismatch'}")
    print(f"  Resolution: {out_res}  {'✓' if res_ok else '⚠ mismatch'}")
    print(f"  Nodata:     {nodata}")
    print(f"  Unique codes: {len(codes)}")

    print(f"\n  Top 15 codes by pixel count:")
    print(f"  {'Code':>6}  {'Pixels':>10}  {'km²':>8}  {'%':>6}  {'Species'}")
    print(f"  {'-' * 71}")

    for code, count in sorted(zip(codes, counts), key=lambda x: -x[1])[:15]:
        area = count * 100 / 1e6
        pct = 100 * count / total
        label = FOREST_TYPE_LABELS.get(int(code), "")
        print(f"  {int(code):>6}  {count:>10,}  {area:>8.1f}  {pct:>6.1f}%  {label}")

if not shape_ok:
    raise RuntimeError(
        f"Output raster shape {out_shape} does not match canonical shape {shape}"
    )

print(f"\n✓ Done. Output: {OUT_RASTER}")