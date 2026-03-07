# prep_terrain_masks.py
# Run once to generate all terrain stratification masks.
# Outputs: mask_elev_low/mid/high.tif, mask_forest_400/500/800.tif,
#          tnc_ecozone_snapped.tif, mask_ecozone_1/2/3.tif,
#          mask_south_facing.tif, mask_north_facing.tif, mask_steep_slopes.tif

import rasterio
import numpy as np
import subprocess
from pathlib import Path
from rasterio.enums import Resampling

# ============= CONFIGURATION =============
TERRAIN_DIR    = Path("/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/AOI/GWNF_cache/traits/terrain")
FOREST_GROUP_DIR = Path("/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/AOI/GWNF_cache/traits/forest/forest_type_group")
ECOZONE_DIR    = Path("/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/AOI/GWNF_cache/traits/ecozone")
ECOZONE_SRC    = ECOZONE_DIR / "tnc_ecozone_simplified/fortype_aoiN.tif"
ECOZONE_SNAP   = ECOZONE_DIR / "tnc_ecozone_simplified_snapped.tif"

FOREST_SRC     = FOREST_GROUP_DIR / "forest_type_group.tif"
FOREST_SNAP    = FOREST_GROUP_DIR / "forest_type_group_snapped.tif"

# Canonical grid reference — any existing mask on the right grid works
CANONICAL_REF  = TERRAIN_DIR / "elevation.tif"

ELEVATION_BREAKS = {"low": (None, 175), "mid": (175, 300), "high": (300, None)}
FOREST_CODES     = {400: "oak_pine", 500: "oak_hickory", 800: "maple_beech"}
ECOZONE_CODES    = {1: "cool", 2: "intermediate", 3: "hot"}

PIXEL_AREA_KM2 = (10 * 10) / 1_000_000
# =========================================

def write_mask(path, data, profile):
    """Write a uint8 mask: 1=valid, 255=nodata."""
    out = np.where(data, np.uint8(1), np.uint8(255))
    profile.update(dtype=rasterio.uint8, count=1, nodata=255)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(out, 1)
    n = int(data.sum())
    print(f"  {path.name}: {n:,} pixels ({n * PIXEL_AREA_KM2:.1f} km²)")

def check_and_warp(src_path, snap_path, canonical_crs, canonical_res,
                   canonical_shape, canonical_transform, label):
    """Check alignment of a raster and warp to canonical grid if needed.
    Returns the path to use (snapped if warped, source if already aligned)."""
    needs_warp = False
    with rasterio.open(src_path) as src:
        if src.crs != canonical_crs:
            print(f"  ⚠ {label} CRS mismatch — warp needed")
            needs_warp = True
        elif (abs(src.res[0] - canonical_res[0]) > 0.01 or
              abs(src.res[1] - canonical_res[1]) > 0.01):
            print(f"  ⚠ {label} resolution mismatch — warp needed")
            needs_warp = True
        elif (src.height, src.width) != canonical_shape:
            print(f"  ⚠ {label} shape mismatch ({src.height}x{src.width} vs "
                  f"{canonical_shape[0]}x{canonical_shape[1]}) — warp needed")
            needs_warp = True
        elif src.transform != canonical_transform:
            print(f"  ⚠ {label} transform mismatch — warp needed")
            needs_warp = True
        else:
            print(f"  ✓ {label} already aligned with canonical grid")

    if not needs_warp:
        return src_path

    if snap_path.exists():
        print(f"  Snapped file already exists: {snap_path.name} — skipping warp")
        print(f"  (Delete {snap_path.name} to force re-warp)")
        return snap_path

    print(f"  Running gdalwarp for {label}...")
    cmd = [
        "gdalwarp",
        "-t_srs", str(canonical_crs),
        "-tr", str(canonical_res[0]), str(canonical_res[1]),
        "-tap",
        "-r", "near",
        "-te",
            str(canonical_transform.c),
            str(canonical_transform.f + canonical_shape[0] * canonical_transform.e),
            str(canonical_transform.c + canonical_shape[1] * canonical_transform.a),
            str(canonical_transform.f),
        "-of", "GTiff",
        "-co", "COMPRESS=LZW",
        str(src_path),
        str(snap_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ✗ gdalwarp failed:\n{result.stderr}")
        raise RuntimeError(f"gdalwarp failed for {label}")
    print(f"  ✓ Warp complete: {snap_path.name}")

    with rasterio.open(snap_path) as src:
        snapped_shape = (src.height, src.width)
        snapped_res   = src.res
    if snapped_shape == canonical_shape and snapped_res == canonical_res:
        print(f"  ✓ Validated: shape={snapped_shape}, res={snapped_res}")
    else:
        print(f"  ⚠ Snapped file shape/res mismatch — inspect before proceeding")

    return snap_path

# ── Load canonical grid profile ────────────────────────────────
print("Loading canonical grid reference...")
with rasterio.open(CANONICAL_REF) as ref:
    canonical_crs       = ref.crs
    canonical_transform = ref.transform
    canonical_res       = (ref.res[0], ref.res[1])
    canonical_shape     = (ref.height, ref.width)
    canonical_profile   = ref.profile.copy()
print(f"  CRS: {canonical_crs}")
print(f"  Resolution: {canonical_res[0]}m x {canonical_res[1]}m")
print(f"  Shape: {canonical_shape}")

# ============================================================
# 1. ELEVATION BAND MASKS
# ============================================================
print("\n--- Elevation band masks ---")
with rasterio.open(TERRAIN_DIR / "elevation.tif") as src:
    elev = src.read(1).astype(float)
    nodata = src.nodata
if nodata is not None:
    elev[elev == nodata] = np.nan

for band, (lo, hi) in ELEVATION_BREAKS.items():
    mask = np.ones(elev.shape, dtype=bool)
    if lo is not None: mask &= (elev >= lo)
    if hi is not None: mask &= (elev < hi)
    mask &= ~np.isnan(elev)
    write_mask(TERRAIN_DIR / f"mask_elev_{band}.tif", mask, canonical_profile.copy())

# ============================================================
# 2. ASPECT MASKS
# ============================================================
print("\n--- Aspect masks ---")
aspect_path = TERRAIN_DIR / "aspect.tif"
with rasterio.open(aspect_path) as src:
    aspect = src.read(1).astype(float)
    nodata = src.nodata
if nodata is not None:
    aspect[aspect == nodata] = np.nan

valid_aspect = ~np.isnan(aspect)

# South-facing: 135–225° (centered on 180°)
south_mask = valid_aspect & (aspect >= 135) & (aspect <= 225)
write_mask(TERRAIN_DIR / "mask_south_facing.tif", south_mask, canonical_profile.copy())

# North-facing: 315–360° and 0–45° (centered on 0°/360°)
north_mask = valid_aspect & ((aspect >= 315) | (aspect <= 45))
write_mask(TERRAIN_DIR / "mask_north_facing.tif", north_mask, canonical_profile.copy())

# Steep slopes: > 15° (adjust threshold as needed)
slope_path = TERRAIN_DIR / "slope.tif"
with rasterio.open(slope_path) as src:
    slope = src.read(1).astype(float)
    nodata = src.nodata
if nodata is not None:
    slope[slope == nodata] = np.nan
steep_mask = ~np.isnan(slope) & (slope > 15)
write_mask(TERRAIN_DIR / "mask_steep_slopes.tif", steep_mask, canonical_profile.copy())

# ============================================================
# 3. FOREST TYPE MASKS
# ============================================================
print("\n--- Forest type alignment check ---")
forest_path = check_and_warp(
    FOREST_SRC, FOREST_SNAP,
    canonical_crs, canonical_res, canonical_shape, canonical_transform,
    "forest_type_group.tif"
)

print("\n--- Forest type masks ---")
with rasterio.open(forest_path) as src:
    forest = src.read(1)

for code, label in FOREST_CODES.items():
    mask = (forest == code)
    write_mask(FOREST_GROUP_DIR / f"mask_forest_{code}.tif", mask, canonical_profile.copy())

non_forest = int((forest == 0).sum())
print(f"  Non-forest pixels excluded (code 0): {non_forest:,} ({non_forest * PIXEL_AREA_KM2:.1f} km²)")

# ============================================================
# 4. ECOZONE — check alignment, warp if needed, build masks
# ============================================================
print("\n--- Ecozone alignment check ---")
ecozone_path = check_and_warp(
    ECOZONE_SRC, ECOZONE_SNAP,
    canonical_crs, canonical_res, canonical_shape, canonical_transform,
    "tnc_ecozone"
)

print("\n--- Ecozone masks ---")
with rasterio.open(ecozone_path) as src:
    ecozone = src.read(1)

for code, label in ECOZONE_CODES.items():
    mask = (ecozone == code)
    write_mask(ECOZONE_DIR / f"mask_ecozone_{code}.tif", mask, canonical_profile.copy())

print("\n✓ All masks generated successfully.")
print(f"Output directory: {TERRAIN_DIR}")