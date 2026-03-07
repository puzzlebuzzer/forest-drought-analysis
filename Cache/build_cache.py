# build_cache.py

import sys
import planetary_computer
import pystac_client
import rasterio
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
import os
from pathlib import Path
import json
import time
from shapely.ops import transform

# ============= CONFIGURATION =============
AOI = "south"   # Options: "north", "south"

INDICES_TO_RUN = ["NDVI", "NDMI", "EVI"]
START_DATE     = "2017-01-01"
END_DATE       = "2026-03-01"
CLOUD_COVER_MAX   = 40
TARGET_RESOLUTION = 10  # meters

# EVI coefficients (MODIS-standard)
EVI_G  = 2.5
EVI_C1 = 6.0
EVI_C2 = 7.5
EVI_L  = 1.0
# =========================================

# ── AOI config ────────────────────────────────────────────────
AOI_SHAPEFILE = "/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/AOI/TNC_AOI_LayerPkg/TNC_AOIs.shp"

AOI_CONFIG = {
    "north": {
        "landscape_id": "UW-GW North",
        "cache_base":   "/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/AOI/NorthAOI/GWNF_cache/indices",
        "crs":          "EPSG:32617",   # UTM Zone 17N
    },
    "south": {
        "landscape_id": "UW-Smoky",
        "cache_base":   "/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/AOI/SouthAOI/Smoky_cache/indices",
        "crs":          "EPSG:32617",   # UTM Zone 17N — confirm south AOI is still in Zone 17
    },
}

if AOI not in AOI_CONFIG:
    raise ValueError(f"Unknown AOI '{AOI}'. Valid options: {list(AOI_CONFIG.keys())}")

LANDSCAPE_ID   = AOI_CONFIG[AOI]["landscape_id"]
CACHE_BASE_DIR = AOI_CONFIG[AOI]["cache_base"]
TARGET_CRS     = AOI_CONFIG[AOI]["crs"]

# ── Index config — manifests now live inside each index dir ──
INDEX_CONFIG = {
    "NDVI": {"bands": ["B04", "B08"]},
    "NDMI": {"bands": ["B08", "B11"]},
    "EVI":  {"bands": ["B02", "B04", "B08"]},
}

for idx in INDICES_TO_RUN:
    if idx not in INDEX_CONFIG:
        raise ValueError(f"Unknown index '{idx}'. Valid options: {list(INDEX_CONFIG.keys())}")

# Create output directories
for idx in INDICES_TO_RUN:
    Path(CACHE_BASE_DIR, idx).mkdir(parents=True, exist_ok=True)

# Derive minimum band set needed across all requested indices
BANDS_NEEDED = set()
for idx in INDICES_TO_RUN:
    BANDS_NEEDED.update(INDEX_CONFIG[idx]["bands"])
BANDS_NEEDED.add("SCL")

print(f"AOI:            {AOI} ({LANDSCAPE_ID})")
print(f"Cache:          {CACHE_BASE_DIR}")
print(f"CRS:            {TARGET_CRS}")
print(f"Indices to run: {INDICES_TO_RUN}")
print(f"Bands per scene:{sorted(BANDS_NEEDED)}")

# ── Load AOI ──────────────────────────────────────────────────
print("\nLoading AOI shapefile...")
aoi = gpd.read_file(AOI_SHAPEFILE)
aoi = aoi[aoi['LscapeID'] == LANDSCAPE_ID]

if len(aoi) == 0:
    raise ValueError(f"No AOI found with LscapeID='{LANDSCAPE_ID}'")

aoi_wgs84 = aoi.to_crs("EPSG:4326")
aoi_wgs84['geometry'] = aoi_wgs84['geometry'].apply(
    lambda geom: transform(lambda x, y, z=None: (x, y), geom)
)
aoi_utm = aoi.to_crs(TARGET_CRS)
bounds  = aoi_utm.total_bounds

print(f"AOI bounds (UTM): {bounds}")
print(f"AOI area: {aoi_utm.area.sum() / 1e6:.2f} km²")

dst_width  = int((bounds[2] - bounds[0]) / TARGET_RESOLUTION)
dst_height = int((bounds[3] - bounds[1]) / TARGET_RESOLUTION)
dst_transform = rasterio.transform.from_bounds(
    bounds[0], bounds[1], bounds[2], bounds[3],
    dst_width, dst_height
)

print(f"Canonical grid: {dst_width} × {dst_height} pixels")
print(f"Estimated file size per scene: ~{(dst_width * dst_height * 4) / 1e6:.1f} MB")

# ── Connect to Planetary Computer and search in 6-month chunks
print("\nConnecting to Planetary Computer...")

start = datetime.strptime(START_DATE, "%Y-%m-%d")
end   = datetime.strptime(END_DATE,   "%Y-%m-%d")

print(f"Searching for scenes ({START_DATE} to {END_DATE}, cloud cover < {CLOUD_COVER_MAX}%)...")

windows = []
current = start
while current < end:
    month = current.month + 6
    year  = current.year
    if month > 12:
        month -= 12
        year  += 1
    window_end = min(datetime(year, month, 1), end)
    windows.append((current.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")))
    current = window_end

all_raw_items = []
for window_start, window_end in windows:
    print(f"  Searching {window_start}/{window_end}...")
    retries = 3
    for attempt in range(retries):
        try:
            catalog = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=planetary_computer.sign_inplace,
            )
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                intersects=aoi_wgs84.geometry.iloc[0],
                datetime=f"{window_start}/{window_end}",
            )
            window_items = list(search.items())
            all_raw_items.extend(window_items)
            print(f"    Found {len(window_items)} scenes")
            break
        except Exception as e:
            if attempt < retries - 1:
                print(f"    Attempt {attempt+1} failed: {e} — retrying in 10s...")
                time.sleep(10)
            else:
                print(f"    Failed after {retries} attempts: {e}")

# ── Filter by cloud cover ─────────────────────────────────────
items = [i for i in all_raw_items
         if i.properties.get("eo:cloud_cover", 100) < CLOUD_COVER_MAX]
print(f"\nAfter cloud filter (<{CLOUD_COVER_MAX}%): {len(items)} scenes")

# ── Deduplicate by date + tile ────────────────────────────────
seen = set()
deduped_items = []
for i in items:
    date_str = i.datetime.strftime("%Y%m%d")
    tile_id  = 'unknown'
    for part in i.id.split('_'):
        if part.startswith('T') and len(part) == 6:
            tile_id = part
            break
    key = (date_str, tile_id)
    if key not in seen:
        seen.add(key)
        deduped_items.append(i)

items = deduped_items
print(f"After deduplication: {len(items)} unique date+tile scenes\n")

# ── Load all manifests upfront ────────────────────────────────
manifests = {}
for index_name in INDICES_TO_RUN:
    manifest_path = Path(CACHE_BASE_DIR) / index_name / "cache_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifests[index_name] = json.load(f)
        print(f"Loaded {index_name} manifest: {len(manifests[index_name])} scenes")
    else:
        manifests[index_name] = {}

# ── Helpers ───────────────────────────────────────────────────
def reproject_band(href, dtype=np.float32, resampling=Resampling.bilinear):
    arr = np.empty((dst_height, dst_width), dtype=dtype)
    with rasterio.open(href) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            resampling=resampling
        )
    return arr

def compute_index(index_name, scaled):
    if index_name == "NDVI":
        nir, red = scaled["B08"], scaled["B04"]
        with np.errstate(divide='ignore', invalid='ignore'):
            return (nir - red) / (nir + red)
    elif index_name == "NDMI":
        nir, swir = scaled["B08"], scaled["B11"]
        with np.errstate(divide='ignore', invalid='ignore'):
            return (nir - swir) / (nir + swir)
    elif index_name == "EVI":
        nir, red, blue = scaled["B08"], scaled["B04"], scaled["B02"]
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = nir + EVI_C1 * red - EVI_C2 * blue + EVI_L
            return np.clip(EVI_G * (nir - red) / denom, -1.0, 1.0)

# ── Main loop ─────────────────────────────────────────────────
all_results = {idx: {'successful': 0, 'skipped': 0,
                     'failed': 0, 'redownloaded': 0}
               for idx in INDICES_TO_RUN}

for scene_idx, item in enumerate(items):
    scene_id = item.id
    date_str = item.datetime.strftime("%Y%m%d")

    tile_id = 'unknown'
    for part in scene_id.split('_'):
        if part.startswith('T') and len(part) == 6:
            tile_id = part
            break

    indices_needed = []
    for index_name in INDICES_TO_RUN:
        output_filename = f"{index_name}_{date_str}_{tile_id}.tif"
        output_path     = Path(CACHE_BASE_DIR) / index_name / output_filename
        if output_path.exists():
            if scene_id not in manifests[index_name]:
                manifests[index_name][scene_id] = {
                    'filename': output_filename,
                    'date':     item.datetime.isoformat()
                }
            all_results[index_name]['skipped'] += 1
        else:
            indices_needed.append(index_name)

    if not indices_needed:
        if (scene_idx + 1) % 10 == 0:
            print(f"[{scene_idx+1}/{len(items)}] " +
                  " | ".join(f"{k}: {v['successful']}new/{v['skipped']}skip/{v['failed']}fail"
                             for k, v in all_results.items()))
        continue

    print(f"\n[{scene_idx+1}/{len(items)}] {scene_id}")
    print(f"  Date: {item.datetime}  |  Tile: {tile_id}  |  "
          f"Cloud: {item.properties.get('eo:cloud_cover', 'N/A')}%")
    print(f"  Indices needed: {indices_needed}")

    for index_name in indices_needed:
        if scene_id in manifests[index_name]:
            print(f"  ⚠ {index_name}: in manifest but file missing — re-downloading")
            all_results[index_name]['redownloaded'] += 1

    try:
        bands_for_this_scene = set()
        for index_name in indices_needed:
            bands_for_this_scene.update(INDEX_CONFIG[index_name]["bands"])
        bands_for_this_scene.add("SCL")

        print(f"  Fetching {len(bands_for_this_scene)} bands: {sorted(bands_for_this_scene)}")

        fetched = {}
        for band in bands_for_this_scene:
            rs = Resampling.nearest if band == "SCL" else Resampling.bilinear
            dt = np.uint8 if band == "SCL" else np.float32
            fetched[band] = reproject_band(item.assets[band].href, dtype=dt, resampling=rs)

        veg_mask   = (fetched["SCL"] == 4)
        data_bands = {k: v for k, v in fetched.items() if k != "SCL"}
        valid_mask = np.ones((dst_height, dst_width), dtype=bool)
        for arr in data_bands.values():
            valid_mask &= (arr > 0) & (arr < 10000)
        combined_mask = veg_mask & valid_mask

        veg_frac = combined_mask.sum() / combined_mask.size
        print(f"  Vegetation coverage: {veg_frac*100:.1f}%")

        scaled = {k: v / 10000.0 for k, v in data_bands.items()}

        for index_name in indices_needed:
            output_filename = f"{index_name}_{date_str}_{tile_id}.tif"
            output_path     = Path(CACHE_BASE_DIR) / index_name / output_filename
            manifest_path   = Path(CACHE_BASE_DIR) / index_name / "cache_manifest.json"

            try:
                result = compute_index(index_name, scaled)
                result[~combined_mask] = np.nan

                with rasterio.open(
                    output_path, 'w',
                    driver='GTiff',
                    height=dst_height,
                    width=dst_width,
                    count=1,
                    dtype=np.float32,
                    crs=TARGET_CRS,
                    transform=dst_transform,
                    compress='lzw',
                    nodata=np.nan,
                    tiled=True,
                    blockxsize=256,
                    blockysize=256
                ) as dst:
                    dst.write(result, 1)

                manifests[index_name][scene_id] = {
                    'filename':     output_filename,
                    'date':         item.datetime.isoformat(),
                    'cloud_cover':  item.properties.get('eo:cloud_cover'),
                    'veg_coverage': float(veg_frac),
                    'processed':    datetime.now().isoformat(),
                }

                with open(manifest_path, 'w') as f:
                    json.dump(manifests[index_name], f, indent=2)

                all_results[index_name]['successful'] += 1
                print(f"  ✓ {output_filename}")

            except Exception as e:
                all_results[index_name]['failed'] += 1
                print(f"  ✗ {index_name} compute/save error: {e}")

    except Exception as e:
        for index_name in indices_needed:
            all_results[index_name]['failed'] += 1
        print(f"  ✗ Band fetch error: {e}")
        continue

# ── Summary ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"SUMMARY — {AOI.upper()} AOI ({LANDSCAPE_ID})")
print(f"{'='*60}")
for index_name, r in all_results.items():
    print(f"\n{index_name}:")
    print(f"  Successfully processed (new):  {r['successful']}")
    print(f"  Re-downloaded (was missing):   {r['redownloaded']}")
    print(f"  Skipped (already cached):      {r['skipped']}")
    print(f"  Failed:                        {r['failed']}")
print(f"\nCache location: {CACHE_BASE_DIR}")

total_failed = sum(r['failed'] for r in all_results.values())
if total_failed > 0:
    print(f"\n{total_failed} failures detected — exiting with code 1 for restart")
    sys.exit(1)