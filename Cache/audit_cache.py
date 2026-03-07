import planetary_computer
from pystac_client import Client
import geopandas as gpd
from pathlib import Path
from shapely.ops import transform
from datetime import datetime
import time

# ── Config ────────────────────────────────────────────────────
AOI_SHAPEFILE  = "/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/AOI/TNC_AOI_LayerPkg/TNC_AOIs.shp"
CACHE_BASE_DIR = Path("/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/AOI/GWNF_cache/indices")
CLOUD_THRESHOLD = 40
START_YEAR      = 2017
END_YEAR        = 2026
INDICES         = ["NDVI", "NDMI", "EVI"]

# ── Load AOI ──────────────────────────────────────────────────
print("Loading AOI...")
aoi = gpd.read_file(AOI_SHAPEFILE)
aoi = aoi[aoi['LscapeID'] == 'UW-GW North']
aoi_wgs84 = aoi.to_crs("EPSG:4326")
aoi_wgs84['geometry'] = aoi_wgs84['geometry'].apply(
    lambda geom: transform(lambda x, y, z=None: (x, y), geom)
)
aoi_geom = aoi_wgs84.geometry.iloc[0]

# ── Generate 6-month windows ──────────────────────────────────
windows = []
for year in range(START_YEAR, END_YEAR + 1):
    windows.append((f"{year}-01-01", f"{year}-07-01"))
    windows.append((f"{year}-07-01", f"{year+1}-01-01"))

# Trim last window to today
windows = [(s, e) for s, e in windows
           if s < datetime.now().strftime("%Y-%m-%d")]

# ── Search Planetary Computer ─────────────────────────────────
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

# ── Filter by cloud cover ─────────────────────────────────────
filtered_items = [i for i in all_raw_items
                  if i.properties.get("eo:cloud_cover", 100) < CLOUD_THRESHOLD]
print(f"After cloud filter (<{CLOUD_THRESHOLD}%): {len(filtered_items)} scenes")

# ── Deduplicate by date + tile ────────────────────────────────
seen = set()
deduped = []
for item in filtered_items:
    date_str = item.datetime.strftime("%Y%m%d")
    tile_id  = 'unknown'
    for part in item.id.split('_'):
        if part.startswith('T') and len(part) == 6:
            tile_id = part
            break
    key = (date_str, tile_id)
    if key not in seen:
        seen.add(key)
        deduped.append((item, date_str, tile_id))

print(f"After deduplication: {len(deduped)} unique date+tile combinations\n")

# ── Compare against cache ─────────────────────────────────────
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

# ── Report ────────────────────────────────────────────────────
print("=" * 60)
print("CACHE AUDIT REPORT")
print("=" * 60)

for idx in INDICES:
    print(f"\n{idx}:")
    print(f"  Cached:  {len(present[idx])}")
    print(f"  Missing: {len(missing[idx])}")
    if missing[idx]:
        print(f"  Missing scenes:")
        for date_str, tile_id, cloud in missing[idx]:
            print(f"    {date_str}  tile={tile_id}  cloud={cloud:.1f}%")

# ── Save missing list to file ─────────────────────────────────
output_file = Path("cache_audit_missing.txt")
with open(output_file, "w") as f:
    for idx in INDICES:
        f.write(f"\n{idx} MISSING ({len(missing[idx])}):\n")
        for date_str, tile_id, cloud in missing[idx]:
            f.write(f"  {date_str}  {tile_id}  cloud={cloud:.1f}%\n")

print(f"\nMissing scenes saved to {output_file}")