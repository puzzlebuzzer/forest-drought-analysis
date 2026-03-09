#!/usr/bin/env python3

import numpy as np
import rasterio

from src.aoi import get_aoi_config
from src.cli import make_parser, add_aoi_arg

parser = make_parser("Audit north- and south-facing slope masks")
add_aoi_arg(parser)
args = parser.parse_args()

AOI = args.aoi
cfg = get_aoi_config(AOI)

TERRAIN_DIR = cfg.terrain_dir
LANDSCAPE_ID = cfg.landscape_id
PIXEL_AREA_KM2 = (10 * 10) / 1_000_000

south_path = TERRAIN_DIR / "mask_south_facing.tif"
north_path = TERRAIN_DIR / "mask_north_facing.tif"
slope_path = TERRAIN_DIR / "slope.tif"
aspect_path = TERRAIN_DIR / "aspect.tif"

print(f"AOI: {AOI} ({LANDSCAPE_ID})")
print(f"Terrain dir: {TERRAIN_DIR}")

with rasterio.open(south_path) as src:
    south = src.read(1)
with rasterio.open(north_path) as src:
    north = src.read(1)
with rasterio.open(slope_path) as src:
    slope = src.read(1).astype(float)
    slope_nodata = src.nodata
with rasterio.open(aspect_path) as src:
    aspect = src.read(1).astype(float)
    aspect_nodata = src.nodata

if slope_nodata is not None:
    slope[slope == slope_nodata] = np.nan
if aspect_nodata is not None:
    aspect[aspect == aspect_nodata] = np.nan

south_bool = south == 1
north_bool = north == 1
overlap = south_bool & north_bool

south_count = int(south_bool.sum())
north_count = int(north_bool.sum())
overlap_count = int(overlap.sum())

print("\nMask coverage")
print(f"  South-facing: {south_count:,} px ({south_count * PIXEL_AREA_KM2:.1f} km²)")
print(f"  North-facing: {north_count:,} px ({north_count * PIXEL_AREA_KM2:.1f} km²)")
print(f"  Overlap:      {overlap_count:,} px ({overlap_count * PIXEL_AREA_KM2:.4f} km²)")

if overlap_count > 0:
    print("\n⚠ Overlap detected — inspect aspect thresholds")
else:
    print("\n✓ No overlap between north/south masks")

for label, mask in [("South", south_bool), ("North", north_bool)]:
    if mask.sum() == 0:
        print(f"\n{label}: no pixels")
        continue
    aspect_vals = aspect[mask & np.isfinite(aspect)]
    slope_vals = slope[mask & np.isfinite(slope)]
    print(f"\n{label} summary")
    print(f"  Aspect range: {np.nanmin(aspect_vals):.1f}° – {np.nanmax(aspect_vals):.1f}°")
    print(f"  Slope range:  {np.nanmin(slope_vals):.1f}° – {np.nanmax(slope_vals):.1f}°")
    print(f"  Mean slope:   {np.nanmean(slope_vals):.1f}°")