#!/usr/bin/env python3

import time
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import planetary_computer
from pystac_client import Client
from shapely.ops import transform

from src.aoi import get_aoi_config, get_aoi_shapefile
from src.cli import make_parser, add_aoi_arg, add_cloud_arg

parser = make_parser("Audit cached index rasters against available Sentinel scenes")
add_aoi_arg(parser)
add_cloud_arg(parser, default=40)
args = parser.parse_args()

AOI = args.aoi
CLOUD_THRESHOLD = args.cloud_max

cfg = get_aoi_config(AOI)

START_YEAR = 2017
END_YEAR = 2026
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
    windows.append((f"{year}-07-01", f"{year+1}-01-01"))

windows = [
    (s, e) for s, e in windows
    if s < datetime.now().strftime("%Y-%m-%d")
]

print(f"Searching Planetary Computer ({START_YEAR}–{END_YEAR})...")
all_raw_items = []

for window_start, window_end in windows:
    print(f"  Searching {window_start}/{window_end}...")
    retries = 3
    for attempt in range(retries):
        try:
            catalog = Client.open(
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
                print(f"    Attempt {attempt+1} failed: {e} — retrying in 10s...")
                time.sleep(10)
            else:
                print(f"    Failed after {retries} attempts: {e}")

print(f"\nTotal scenes retrieved: {len(all_raw_items)}")

filtered_items = [
    i for i in all_raw_items
    if i.properties.get("eo:cloud_cover", 100) < CLOUD_THRESHOLD
]
print(f"After cloud filter (<{CLOUD_THRESHOLD}%): {len(filtered_items)} scenes")

seen = set()
deduped = []
for item in filtered_items:
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

print(f"After deduplication: {len(deduped)} unique date+tile combinations\n")

missing = {idx: [] for idx in INDICES}
present = {idx: [] for idx in INDICES}

for item, date_str, tile_id in deduped:
    cloud = item.properties.get("eo:cloud_cover", 0)
    for idx in INDICES:
        filename = f"{idx}_{date_str}_{tile_id}.tif"
        filepath = CACHE_BASE_DIR / idx / filename
        if filepath.exists():
            present[idx].append((date_str, tile_id, cloud))
        else:
            missing[idx].append((date_str, tile_id, cloud))

print("=" * 60)
print(f"CACHE AUDIT REPORT — {AOI.upper()} ({LANDSCAPE_ID})")
print("=" * 60)

for idx in INDICES:
    print(f"\n{idx}:")
    print(f"  Cached:  {len(present[idx])}")
    print(f"  Missing: {len(missing[idx])}")
    if missing[idx]:
        print("  Missing scenes:")
        for date_str, tile_id, cloud in missing[idx]:
            print(f"    {date_str}  tile={tile_id}  cloud={cloud:.1f}%")

output_file = Path(f"cache_audit_missing_{AOI}.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for idx in INDICES:
        f.write(f"\n{idx} MISSING ({len(missing[idx])}):\n")
        for date_str, tile_id, cloud in missing[idx]:
            f.write(f"  {date_str}  {tile_id}  cloud={cloud:.1f}%\n")

print(f"\nMissing scenes saved to {output_file}")