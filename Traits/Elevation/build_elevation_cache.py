# terrain_setup.py

import planetary_computer
import pystac_client
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
import numpy as np
import geopandas as gpd
from pathlib import Path
from shapely.ops import transform

# Configuration
AOI_SHAPEFILE = r"/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/AOI/TNC_AOIs.shp"
OUTPUT_DIR = r"/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/AOI/GWNF_cache/terrain"
TARGET_CRS = "EPSG:32617"  # UTM Zone 17N
TARGET_RESOLUTION = 10  # meters (match NDVI grid)

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("Loading AOI shapefile...")
aoi = gpd.read_file(AOI_SHAPEFILE)
aoi = aoi[aoi['LscapeID'] == 'UW-GW North']

# Convert to WGS84 for STAC search and UTM for processing
aoi_wgs84 = aoi.to_crs("EPSG:4326")
aoi_wgs84['geometry'] = aoi_wgs84['geometry'].apply(lambda geom: transform(lambda x, y, z=None: (x, y), geom))

aoi_utm = aoi.to_crs(TARGET_CRS)
bounds = aoi_utm.total_bounds  # minx, miny, maxx, maxy

print(f"AOI bounds (UTM): {bounds}")

# Calculate canonical grid dimensions (same as NDVI cache)
dst_width = int((bounds[2] - bounds[0]) / TARGET_RESOLUTION)
dst_height = int((bounds[3] - bounds[1]) / TARGET_RESOLUTION)
dst_transform = rasterio.transform.from_bounds(
    bounds[0], bounds[1], bounds[2], bounds[3],
    dst_width, dst_height
)

print(f"Canonical grid: {dst_width} × {dst_height} pixels")

# Connect to Planetary Computer
print("\nConnecting to Planetary Computer...")
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# Search for Copernicus DEM
print("Searching for Copernicus DEM...")
search = catalog.search(
    collections=["cop-dem-glo-30"],  # 30m global DEM
    intersects=aoi_wgs84.geometry.iloc[0]
)

items = list(search.items())
print(f"Found {len(items)} DEM tiles")

# Load and mosaic DEM tiles
print("\nLoading and mosaicking DEM tiles...")
dem_tiles = []

for idx, item in enumerate(items):
    print(f"  Loading tile {idx+1}/{len(items)}: {item.id}")
    dem_href = item.assets["data"].href
    
    with rasterio.open(dem_href) as src:
        # Reproject to canonical grid
        dem_tile = np.empty((dst_height, dst_width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dem_tile,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            resampling=Resampling.bilinear
        )
        dem_tiles.append(dem_tile)

# Mosaic tiles (take first valid value, or mean if multiple)
print("Mosaicking tiles...")
if len(dem_tiles) == 1:
    elevation = dem_tiles[0]
else:
    # Stack and take mean, ignoring nodata
    stacked = np.stack(dem_tiles)
    elevation = np.nanmean(stacked, axis=0)

# Save elevation
elevation_path = f"{OUTPUT_DIR}/elevation.tif"
print(f"\nSaving elevation to {elevation_path}")
with rasterio.open(
    elevation_path,
    'w',
    driver='GTiff',
    height=dst_height,
    width=dst_width,
    count=1,
    dtype=np.float32,
    crs=TARGET_CRS,
    transform=dst_transform,
    compress='lzw',
    nodata=np.nan
) as dst:
    dst.write(elevation, 1)

print(f"Elevation range: {np.nanmin(elevation):.1f}m to {np.nanmax(elevation):.1f}m")

# Compute slope (in degrees)
print("\nComputing slope...")
# Calculate gradient in meters per pixel
dx = np.gradient(elevation, TARGET_RESOLUTION, axis=1)
dy = np.gradient(elevation, TARGET_RESOLUTION, axis=0)

# Slope = arctan(sqrt(dx² + dy²)) converted to degrees
slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
slope_deg = np.degrees(slope_rad)

# Save slope
slope_path = f"{OUTPUT_DIR}/slope.tif"
print(f"Saving slope to {slope_path}")
with rasterio.open(
    slope_path,
    'w',
    driver='GTiff',
    height=dst_height,
    width=dst_width,
    count=1,
    dtype=np.float32,
    crs=TARGET_CRS,
    transform=dst_transform,
    compress='lzw',
    nodata=np.nan
) as dst:
    dst.write(slope_deg, 1)

print(f"Slope range: {np.nanmin(slope_deg):.1f}° to {np.nanmax(slope_deg):.1f}°")

# Compute aspect (direction slope faces, in degrees from north)
print("\nComputing aspect...")
# Aspect = arctan2(-dy, dx) converted to compass bearing
aspect_rad = np.arctan2(-dy, dx)
aspect_deg = np.degrees(aspect_rad)

# Convert to compass bearing (0 = North, 90 = East, 180 = South, 270 = West)
aspect_deg = 90 - aspect_deg
aspect_deg = np.where(aspect_deg < 0, aspect_deg + 360, aspect_deg)

# Flat areas have undefined aspect - set to NaN
aspect_deg = np.where(slope_deg < 2, np.nan, aspect_deg)

# Save aspect
aspect_path = f"{OUTPUT_DIR}/aspect.tif"
print(f"Saving aspect to {aspect_path}")
with rasterio.open(
    aspect_path,
    'w',
    driver='GTiff',
    height=dst_height,
    width=dst_width,
    count=1,
    dtype=np.float32,
    crs=TARGET_CRS,
    transform=dst_transform,
    compress='lzw',
    nodata=np.nan
) as dst:
    dst.write(aspect_deg, 1)

print(f"Aspect range: 0° (North) to 360°")

# Create useful masks
print("\nCreating terrain masks...")

# South-facing slopes (135° - 225°, slope ≥ 5°)
south_facing = (aspect_deg >= 135) & (aspect_deg <= 225) & (slope_deg >= 5)
south_mask_path = f"{OUTPUT_DIR}/mask_south_facing.tif"
with rasterio.open(
    south_mask_path,
    'w',
    driver='GTiff',
    height=dst_height,
    width=dst_width,
    count=1,
    dtype=np.uint8,
    crs=TARGET_CRS,
    transform=dst_transform,
    compress='lzw'
) as dst:
    dst.write(south_facing.astype(np.uint8), 1)
print(f"South-facing mask: {south_facing.sum() / south_facing.size * 100:.1f}% of AOI")

# North-facing slopes (315° - 45°, slope ≥ 5°)
north_facing = (((aspect_deg >= 315) | (aspect_deg <= 45)) & (slope_deg >= 5))
north_mask_path = f"{OUTPUT_DIR}/mask_north_facing.tif"
with rasterio.open(
    north_mask_path,
    'w',
    driver='GTiff',
    height=dst_height,
    width=dst_width,
    count=1,
    dtype=np.uint8,
    crs=TARGET_CRS,
    transform=dst_transform,
    compress='lzw'
) as dst:
    dst.write(north_facing.astype(np.uint8), 1)
print(f"North-facing mask: {north_facing.sum() / north_facing.size * 100:.1f}% of AOI")

# Steep slopes (>15°)
steep_slopes = slope_deg > 15
steep_mask_path = f"{OUTPUT_DIR}/mask_steep_slopes.tif"
with rasterio.open(
    steep_mask_path,
    'w',
    driver='GTiff',
    height=dst_height,
    width=dst_width,
    count=1,
    dtype=np.uint8,
    crs=TARGET_CRS,
    transform=dst_transform,
    compress='lzw'
) as dst:
    dst.write(steep_slopes.astype(np.uint8), 1)
print(f"Steep slopes mask: {steep_slopes.sum() / steep_slopes.size * 100:.1f}% of AOI")

print("\n" + "="*60)
print("Terrain setup complete!")
print(f"Output directory: {OUTPUT_DIR}")
print("\nFiles created:")
print("  - elevation.tif")
print("  - slope.tif")
print("  - aspect.tif")
print("  - mask_south_facing.tif")
print("  - mask_north_facing.tif")
print("  - mask_steep_slopes.tif")