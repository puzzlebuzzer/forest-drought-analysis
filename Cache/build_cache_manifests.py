#!/usr/bin/env python3

import json
import time
from datetime import datetime

import geopandas as gpd
import numpy as np
import planetary_computer
import pystac_client
import rasterio
from shapely.ops import transform

from src.aoi import get_aoi_config, get_aoi_shapefile
from src.cli import make_parser, add_aoi_arg, add_cloud_arg, add_date_range_args

parser = make_parser("Rebuild cache manifests from existing cached rasters")
add_aoi_arg(parser)
add_cloud_arg(parser, default=40)
add_date_range_args(parser)
args = parser.parse_args()

AOI = args.aoi
CLOUD_THRESHOLD = args.cloud_max
START_DATE = args.start_date
END_DATE = args.end_date

cfg = get_aoi_config(AOI)

INDICES = ["NDVI", "NDMI", "EVI"]

AOI_SHAPEFILE = get_aoi_shapefile()
LANDSCAPE_ID = cfg.landscape_id
CACHE_BASE_DIR = cfg.index_cache_root

print("Loading AOI...")
aoi = gpd.read_file(AOI_SHAPEFILE)
aoi = aoi[aoi["LscapeID"] == LANDSCAPE_ID]
aoi_wgs84 = aoi.to_crs("EPSG:4326")
aoi_wgs84["geometry"] = aoi_wgs84["geometry"].apply(
    lambda geom: transform(lambda x, y, z=None: (x, y), geom)
)
aoi_geom = aoi_wgs84.geometry.iloc[0]

windows = []
for year in range(START_YEAR, END_YEAR + 1):
    windows.append((f"{year}-01-01", f"{year}-07-01"))
    windows.append((f"{year}-07-01", f"{year + 1}-01-01"))
windows = [(s, e) for s, e in windows if s < datetime.now().strftime("%Y-%m-%d")]

print(f"Searching Planetary Computer ({START_YEAR}–{END_YEAR})...")
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
                intersects=aoi_geom,
                datetime=f"{window_start}/{window_end}",
                sortby="datetime",
            )
            window_items = list(search.items())
            all_raw_items.extend(window_items)
            print(f"    Found {len(window_items)} scenes")
            break
        except Exception as e:
            if attempt < retries - 1:
                print(f"    Attempt {attempt + 1} failed: {e} — retrying in 10s...")
                time.sleep(10)
            else:
                print(f"    Failed after {retries} attempts: {e}")

filtered = [
    i for i in all_raw_items
    if i.properties.get("eo:cloud_cover", 100) < CLOUD_THRESHOLD
]

seen = set()
deduped = []
for item in filtered:
    date_str = item.datetime.strftime("%Y%m%d")
    tile_id = "unknown"
    for part in item.id.split("_"):
        if part.startswith("T") and len(part) == 6:
            tile_id = part
            break
    key = (date_str, tile_id)
    if key not in seen:
        seen.add(key)
        deduped.append((item, date_str, tile_id))

print(f"\nTotal unique scenes to process: {len(deduped)}")

manifests = {idx: {} for idx in INDICES}
rebuilt = {idx: 0 for idx in INDICES}
missing_file = {idx: 0 for idx in INDICES}
rebuilt_at = datetime.now().isoformat()

for i, (item, date_str, tile_id) in enumerate(deduped):
    scene_id = item.id
    cloud_cover = item.properties.get("eo:cloud_cover", None)

    for idx in INDICES:
        filename = f"{idx}_{date_str}_{tile_id}.tif"
        filepath = CACHE_BASE_DIR / idx / filename

        if not filepath.exists():
            missing_file[idx] += 1
            continue

        try:
            with rasterio.open(filepath) as src:
                data = src.read(1)
                total_pixels = data.size
                valid_pixels = np.sum(~np.isnan(data))
                veg_coverage = float(valid_pixels / total_pixels)
        except Exception as e:
            print(f"  Could not read {filename}: {e}")
            veg_coverage = None

        manifests[idx][scene_id] = {
            "filename": filename,
            "date": item.datetime.isoformat(),
            "cloud_cover": cloud_cover,
            "veg_coverage": veg_coverage,
            "rebuilt": rebuilt_at,
        }
        rebuilt[idx] += 1

    if (i + 1) % 50 == 0:
        print(
            f"  [{i + 1}/{len(deduped)}] "
            + " | ".join(f"{idx}: {rebuilt[idx]} rebuilt" for idx in INDICES)
        )

print("\nWriting manifests...")
for idx in INDICES:
    manifest_path = CACHE_BASE_DIR / idx / "cache_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifests[idx], f, indent=2)
    print(
        f"  {idx}/cache_manifest.json "
        f"({rebuilt[idx]} entries, {missing_file[idx]} files not found)"
    )

print(f"\n{'=' * 60}")
print(f"MANIFEST REBUILD SUMMARY — {AOI.upper()} ({LANDSCAPE_ID})")
print(f"{'=' * 60}")
for idx in INDICES:
    print(f"\n{idx}:")
    print(f"  Entries written:   {rebuilt[idx]}")
    print(f"  Files not found:   {missing_file[idx]}")
print(f"\nManifests written to: {CACHE_BASE_DIR}/<INDEX>/cache_manifest.json")