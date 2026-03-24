#!/usr/bin/env python3

from pathlib import Path

import geopandas as gpd
import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.warp import reproject, Resampling
from shapely.ops import transform

from src.aoi import get_aoi_config, get_aoi_shapefile
from src.cli import make_parser, add_aoi_arg

parser = make_parser("Build elevation, slope, and aspect rasters for an AOI")
add_aoi_arg(parser)
args = parser.parse_args()

AOI = args.aoi
cfg = get_aoi_config(AOI)

TARGET_RESOLUTION = 10  # meters
TARGET_CRS = "EPSG:32617"

AOI_SHAPEFILE = get_aoi_shapefile()
LANDSCAPE_ID = cfg.landscape_id
OUTPUT_DIR = cfg.terrain_dir

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"AOI:        {AOI} ({LANDSCAPE_ID})")
print(f"Output:     {OUTPUT_DIR}")
print(f"CRS:        {TARGET_CRS}")

print("\nLoading AOI shapefile...")
aoi = gpd.read_file(AOI_SHAPEFILE)
aoi = aoi[aoi["LscapeID"] == LANDSCAPE_ID]

if len(aoi) == 0:
    raise ValueError(f"No AOI found with LscapeID='{LANDSCAPE_ID}'")

aoi_wgs84 = aoi.to_crs("EPSG:4326")
aoi_wgs84["geometry"] = aoi_wgs84["geometry"].apply(
    lambda geom: transform(lambda x, y, z=None: (x, y), geom)
)
aoi_utm = aoi.to_crs(TARGET_CRS)
bounds = aoi_utm.total_bounds

print(f"AOI bounds (UTM): {bounds}")
print(f"AOI area: {aoi_utm.area.sum() / 1e6:.2f} km²")

dst_width = int((bounds[2] - bounds[0]) / TARGET_RESOLUTION)
dst_height = int((bounds[3] - bounds[1]) / TARGET_RESOLUTION)
dst_transform = rasterio.transform.from_bounds(
    bounds[0], bounds[1], bounds[2], bounds[3], dst_width, dst_height
)

print(f"Canonical grid: {dst_width} × {dst_height} pixels")

print("\nConnecting to Planetary Computer...")
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

print("Searching for Copernicus DEM GLO-30...")
search = catalog.search(
    collections=["cop-dem-glo-30"],
    intersects=aoi_wgs84.geometry.iloc[0],
)

items = list(search.items())
print(f"Found {len(items)} DEM tiles")

print("\nLoading and mosaicking DEM tiles...")
dem_tiles = []
for idx, item in enumerate(items):
    print(f"  Loading tile {idx + 1}/{len(items)}: {item.id}")
    dem_href = item.assets["data"].href
    with rasterio.open(dem_href) as src:
        dem_tile = np.empty((dst_height, dst_width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dem_tile,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            resampling=Resampling.bilinear,
        )
        dem_tiles.append(dem_tile)

print("Mosaicking tiles...")
if len(dem_tiles) == 1:
    elevation = dem_tiles[0]
else:
    stacked = np.stack(dem_tiles)
    elevation = np.nanmean(stacked, axis=0)


def save_raster(path: Path, data: np.ndarray, dtype=np.float32, nodata=np.nan) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=dst_height,
        width=dst_width,
        count=1,
        dtype=dtype,
        crs=TARGET_CRS,
        transform=dst_transform,
        compress="lzw",
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def save_mask(path: Path, data: np.ndarray) -> None:
    save_raster(path, data.astype(np.uint8), dtype=np.uint8, nodata=255)


elevation_path = OUTPUT_DIR / "elevation.tif"
print(f"\nSaving elevation to {elevation_path}")
save_raster(elevation_path, elevation)
print(f"  Range: {np.nanmin(elevation):.1f}m to {np.nanmax(elevation):.1f}m")
print(
    f"  P25={np.nanpercentile(elevation, 25):.0f}m  "
    f"P50={np.nanpercentile(elevation, 50):.0f}m  "
    f"P75={np.nanpercentile(elevation, 75):.0f}m"
)

print("\nComputing slope...")
dx = np.gradient(elevation, TARGET_RESOLUTION, axis=1)
dy = np.gradient(elevation, TARGET_RESOLUTION, axis=0)
slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

slope_path = OUTPUT_DIR / "slope.tif"
save_raster(slope_path, slope_deg)
print(f"  Range: {np.nanmin(slope_deg):.1f}° to {np.nanmax(slope_deg):.1f}°")

print("\nComputing aspect...")
aspect_deg = np.degrees(np.arctan2(-dy, dx))
aspect_deg = 90 - aspect_deg
aspect_deg = np.where(aspect_deg < 0, aspect_deg + 360, aspect_deg)
aspect_deg = np.where(slope_deg < 2, np.nan, aspect_deg)

aspect_path = OUTPUT_DIR / "aspect.tif"
save_raster(aspect_path, aspect_deg)
print("  Convention: 0°=North, 90°=East, 180°=South, 270°=West")

print("\nCreating terrain masks...")

south = (aspect_deg >= 135) & (aspect_deg <= 225) & (slope_deg >= 5)
save_mask(OUTPUT_DIR / "mask_south_facing.tif", south)
print(f"  South-facing: {south.sum():,} px ({south.sum() / south.size * 100:.1f}% of AOI)")

north = ((aspect_deg >= 315) | (aspect_deg <= 45)) & (slope_deg >= 5)
save_mask(OUTPUT_DIR / "mask_north_facing.tif", north)
print(f"  North-facing: {north.sum():,} px ({north.sum() / north.size * 100:.1f}% of AOI)")

steep = slope_deg > 15
save_mask(OUTPUT_DIR / "mask_steep_slopes.tif", steep)
print(f"  Steep slopes: {steep.sum():,} px ({steep.sum() / steep.size * 100:.1f}% of AOI)")

print(f"\n{'=' * 60}")
print(f"Terrain setup complete — {AOI.upper()} AOI ({LANDSCAPE_ID})")
print(f"{'=' * 60}")
print(f"Output directory: {OUTPUT_DIR}")
print("\nFiles created:")
print("  elevation.tif / slope.tif / aspect.tif")
print("  mask_south_facing.tif / mask_north_facing.tif / mask_steep_slopes.tif")