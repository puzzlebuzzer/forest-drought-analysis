"""
clip_forest_type.py
--------------------
Clips the CONUS forest type raster (conus_foresttype.img) to the AOI
and snaps it to the canonical 10m UTM Zone 17N grid.
Output: traits/forest/forest_type_type/forest_type_type.tif
"""

import rasterio
import numpy as np
import subprocess
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
CACHE_BASE   = Path("/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/AOI/GWNF_cache")
TERRAIN_DIR  = CACHE_BASE / "traits" / "terrain"
FOREST_TYPE_DIR = CACHE_BASE / "traits" / "forest" / "forest_type_type"

SRC_RASTER   = FOREST_TYPE_DIR / "conus_foresttype.img"
OUT_RASTER   = FOREST_TYPE_DIR / "forest_type_type.tif"

# Canonical grid reference
CANONICAL_REF = TERRAIN_DIR / "elevation.tif"
# ============================================================

# ── Read canonical grid parameters ───────────────────────────
print("Reading canonical grid...")
with rasterio.open(CANONICAL_REF) as ref:
    crs       = ref.crs
    res       = ref.res
    shape     = (ref.height, ref.width)
    transform = ref.transform
    bounds    = ref.bounds

print(f"  CRS:        {crs}")
print(f"  Resolution: {res[0]}m x {res[1]}m")
print(f"  Shape:      {shape[0]} x {shape[1]}")
print(f"  Bounds:     {bounds.left:.1f}, {bounds.bottom:.1f}, {bounds.right:.1f}, {bounds.top:.1f}")

# ── Check source raster ───────────────────────────────────────
print(f"\nChecking source raster...")
if not SRC_RASTER.exists():
    raise FileNotFoundError(f"Source not found: {SRC_RASTER}")

with rasterio.open(SRC_RASTER) as src:
    print(f"  CRS:        {src.crs}")
    print(f"  Resolution: {src.res[0]:.1f}m x {src.res[1]:.1f}m")
    print(f"  Shape:      {src.height} x {src.width}")
    print(f"  Dtype:      {src.dtypes[0]}")

if OUT_RASTER.exists():
    print(f"\n  Output already exists: {OUT_RASTER.name}")
    print(f"  Delete it to force re-clip.")
else:
    # ── Run gdalwarp ──────────────────────────────────────────
    print(f"\nRunning gdalwarp (this may take a minute on the CONUS file)...")
    cmd = [
        "gdalwarp",
        "-t_srs", f"EPSG:{crs.to_epsg()}",
        "-tr", str(res[0]), str(res[1]),
        "-tap",
        "-r", "near",              # nearest-neighbour for categorical data
        "-te",                     # clip to exact AOI extent
            str(bounds.left),
            str(bounds.bottom),
            str(bounds.right),
            str(bounds.top),
        "-of", "GTiff",
        "-co", "COMPRESS=LZW",
        "-co", "TILED=YES",
        str(SRC_RASTER),
        str(OUT_RASTER),
    ]
    print(f"  Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"✗ gdalwarp failed:\n{result.stderr}")
        raise RuntimeError("gdalwarp failed — check paths and GDAL installation")
    print(f"✓ Warp complete: {OUT_RASTER.name}")

# ── Validate output ───────────────────────────────────────────
print(f"\nValidating output...")
with rasterio.open(OUT_RASTER) as src:
    out_shape = (src.height, src.width)
    out_res   = src.res
    data      = src.read(1)
    nodata    = src.nodata

    codes, counts = np.unique(data, return_counts=True)
    total = data.size

    shape_ok = (abs(out_shape[0] - shape[0]) <= 2 and
                abs(out_shape[1] - shape[1]) <= 2)
    res_ok   = (abs(out_res[0] - res[0]) < 0.01 and
                abs(out_res[1] - res[1]) < 0.01)

    print(f"  Shape:      {out_shape}  {'✓' if shape_ok else '⚠ mismatch'}")
    print(f"  Resolution: {out_res}  {'✓' if res_ok else '⚠ mismatch'}")
    print(f"  Nodata:     {nodata}")
    print(f"  Unique codes: {len(codes)}")
    print(f"\n  Top 15 codes by pixel count:")
    print(f"  {'Code':>6}  {'Pixels':>10}  {'km²':>8}  {'%':>6}")
    print(f"  {'-'*36}")
    for code, count in sorted(zip(codes, counts), key=lambda x: -x[1])[:15]:
        area = count * 100 / 1e6
        pct  = 100 * count / total
        print(f"  {int(code):>6}  {count:>10,}  {area:>8.1f}  {pct:>6.1f}%")

print(f"\n✓ Done. Output: {OUT_RASTER}")
print(f"  Ready for species-level cross-tabulation.")