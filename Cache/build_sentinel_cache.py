#!/usr/bin/env python3
"""
build_sentinel_cache.py — Sentinel-2 index cache builder (v2)
--------------------------------------------------------------
Downloads Sentinel-2 L2A scenes from Planetary Computer and writes
per-scene GeoTIFFs for each requested vegetation index plus the raw
SCL classification band.

Quality masking:
  Index rasters retain all pixels where SCL is NOT in {8,9,10}
  (cloud medium, cloud high, thin cirrus). Cloud shadow (3), snow (11),
  water (6), bare soil (5), and all other non-cloud classes are kept.
  The raw SCL raster is saved alongside each scene so any additional
  SCL-based filter can be applied at analysis time without re-downloading.

Spatial masking:
  Output pixels are NaN outside the TNC AOI polygon boundary.
  The bounding-box canonical grid is retained for spatial alignment,
  but only pixels inside the polygon carry valid values.

Rate limiting:
  On a 403 from Planetary Computer the script sleeps 5 minutes and
  retries the same scene, rather than marking it failed and continuing.

SCL class reference:
  0  No data          5  Not vegetated    9  Cloud (high)
  1  Saturated/defect 6  Water           10  Thin cirrus
  2  Dark area pixels 7  Unclassified    11  Snow / ice
  3  Cloud shadow     8  Cloud (medium)
  4  Vegetation

Run from project root:
  python Cache/build_sentinel_cache.py                                        # both AOIs, all indices
  python Cache/build_sentinel_cache.py --aoi north                            # north only
  python Cache/build_sentinel_cache.py --aoi south --indices NDVI NDMI
  python Cache/build_sentinel_cache.py --cache-suffix _3_24                   # writes to GWNF_cache_3_24/ and Smoky_cache_3_24/
"""

import json
import sys
import time
from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import planetary_computer
import pystac_client
import rasterio
import rasterio.features
from rasterio.enums import Resampling
from rasterio.warp import reproject
from shapely.ops import transform

from src.aoi import get_aoi_config, get_aoi_shapefile
from src.cli import (
    make_parser,
    add_aoi_arg,
    add_indices_arg,
    add_date_range_args,
    add_cloud_arg,
    add_cache_suffix_arg,
)

# ── Args ───────────────────────────────────────────────────────────────────────

parser = make_parser("Build Sentinel-2 index cache")
add_aoi_arg(parser, default=None)    # None = run both AOIs in sequence
add_indices_arg(parser)
add_date_range_args(parser)
add_cloud_arg(parser)
add_cache_suffix_arg(parser)
args = parser.parse_args()

INDICES_TO_RUN  = args.indices
START_DATE      = args.start_date
END_DATE        = args.end_date
CLOUD_COVER_MAX = args.cloud_max
AOIS_TO_RUN     = [args.aoi] if args.aoi else ["north", "south"]

# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_CRS        = "EPSG:32617"
TARGET_RESOLUTION = 10
RATE_LIMIT_SLEEP  = 300   # 5 minutes on 403

# SCL values to exclude: cloud medium (8), cloud high (9), thin cirrus (10),
# snow/ice (11). Snow is excluded from index values because it inflates NDMI
# (high NIR, low SWIR → falsely high moisture signal). Snow fraction per scene
# is recorded in the manifest for later analysis.
# Cloud shadow (3), water (6), bare soil (5) are all retained.
QUALITY_EXCLUDE = np.array([8, 9, 10, 11], dtype=np.uint8)

INDEX_CONFIG = {
    "NDVI": {"bands": ["B04", "B08"]},
    "NDMI": {"bands": ["B08", "B11"]},
    "EVI":  {"bands": ["B02", "B04", "B08"]},
}

EVI_G  = 2.5
EVI_C1 = 6.0
EVI_C2 = 7.5
EVI_L  = 1.0

for idx in INDICES_TO_RUN:
    if idx not in INDEX_CONFIG:
        raise ValueError(f"Unknown index '{idx}'. Valid: {list(INDEX_CONFIG.keys())}")

# ── Helpers (AOI-independent) ─────────────────────────────────────────────────

def reproject_band(href, dst_height, dst_width, dst_transform,
                   dtype=np.float32, resampling=Resampling.bilinear):
    arr = np.empty((dst_height, dst_width), dtype=dtype)
    with rasterio.open(href) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            resampling=resampling,
        )
    return arr


def compute_index(index_name, scaled):
    if index_name == "NDVI":
        nir, red = scaled["B08"], scaled["B04"]
        with np.errstate(divide="ignore", invalid="ignore"):
            return (nir - red) / (nir + red)
    if index_name == "NDMI":
        nir, swir = scaled["B08"], scaled["B11"]
        with np.errstate(divide="ignore", invalid="ignore"):
            return (nir - swir) / (nir + swir)
    if index_name == "EVI":
        nir, red, blue = scaled["B08"], scaled["B04"], scaled["B02"]
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = nir + EVI_C1 * red - EVI_C2 * blue + EVI_L
            return np.clip(EVI_G * (nir - red) / denom, -1.0, 1.0)
    raise ValueError(f"Unsupported index: {index_name}")


def is_rate_limited(exc):
    return "403" in str(exc)


def write_tif(path, arr, dst_height, dst_width, dst_transform, dtype, nodata=None):
    profile = dict(
        driver="GTiff",
        height=dst_height,
        width=dst_width,
        count=1,
        dtype=dtype,
        crs=TARGET_CRS,
        transform=dst_transform,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


AOI_SHAPEFILE = get_aoi_shapefile()

# ── Per-AOI loop ──────────────────────────────────────────────────────────────

for AOI in AOIS_TO_RUN:
    print(f"\n{'#'*62}")
    print(f"# PROCESSING AOI: {AOI.upper()}")
    print(f"{'#'*62}")

    cfg            = get_aoi_config(AOI)
    LANDSCAPE_ID   = cfg.landscape_id
    CACHE_BASE_DIR = cfg.index_cache_root

    if args.cache_suffix:
        # cfg.index_cache_root is  .../GWNF_cache/s2/indices
        # parents[1]           is  .../GWNF_cache   (the versioned root)
        _cache_root     = CACHE_BASE_DIR.parents[1]
        _new_cache_root = _cache_root.parent / (_cache_root.name + args.cache_suffix)
        CACHE_BASE_DIR  = _new_cache_root / CACHE_BASE_DIR.relative_to(_cache_root)

    print(f"AOI:            {AOI} ({LANDSCAPE_ID})")
    print(f"Cache:          {CACHE_BASE_DIR}")
    print(f"Indices:        {INDICES_TO_RUN}")
    print(f"Date range:     {START_DATE} → {END_DATE}")
    print(f"Cloud max:      {CLOUD_COVER_MAX}%")

    # ── AOI geometry + canonical grid ─────────────────────────────────────────

    print("\nLoading AOI shapefile...")
    aoi_gdf   = gpd.read_file(AOI_SHAPEFILE)
    aoi_gdf   = aoi_gdf[aoi_gdf["LscapeID"] == LANDSCAPE_ID]
    if len(aoi_gdf) == 0:
        raise ValueError(f"No AOI found with LscapeID='{LANDSCAPE_ID}'")

    aoi_wgs84 = aoi_gdf.to_crs("EPSG:4326")
    aoi_wgs84["geometry"] = aoi_wgs84["geometry"].apply(
        lambda geom: transform(lambda x, y, z=None: (x, y), geom)
    )
    aoi_utm   = aoi_gdf.to_crs(TARGET_CRS)
    bounds    = aoi_utm.total_bounds

    dst_width     = int((bounds[2] - bounds[0]) / TARGET_RESOLUTION)
    dst_height    = int((bounds[3] - bounds[1]) / TARGET_RESOLUTION)
    dst_transform = rasterio.transform.from_bounds(
        bounds[0], bounds[1], bounds[2], bounds[3],
        dst_width, dst_height,
    )

    print(f"Canonical grid: {dst_width} × {dst_height} pixels  ({TARGET_RESOLUTION}m)")
    print(f"~{(dst_width * dst_height * 4) / 1e6:.1f} MB per index scene, "
          f"~{(dst_width * dst_height * 1) / 1e6:.1f} MB per SCL scene")

    # ── AOI polygon mask ───────────────────────────────────────────────────────

    print("Rasterizing AOI polygon mask...")
    aoi_polygon_mask = rasterio.features.geometry_mask(
        [geom.__geo_interface__ for geom in aoi_utm.geometry],
        out_shape=(dst_height, dst_width),
        transform=dst_transform,
        all_touched=False,
        invert=True,   # True = inside AOI polygon
    )
    print(f"  {aoi_polygon_mask.mean()*100:.1f}% of bounding box is inside polygon")

    # ── Directories + manifests ────────────────────────────────────────────────

    for idx in INDICES_TO_RUN:
        (CACHE_BASE_DIR / idx).mkdir(parents=True, exist_ok=True)

    scl_dir = CACHE_BASE_DIR / "SCL"
    scl_dir.mkdir(parents=True, exist_ok=True)

    manifests = {}
    for index_name in INDICES_TO_RUN:
        mp = CACHE_BASE_DIR / index_name / "cache_manifest.json"
        if mp.exists():
            with open(mp, "r", encoding="utf-8") as f:
                manifests[index_name] = json.load(f)
            print(f"Loaded {index_name} manifest: {len(manifests[index_name])} scenes")
        else:
            manifests[index_name] = {}

    scl_manifest_path = scl_dir / "cache_manifest.json"
    if scl_manifest_path.exists():
        with open(scl_manifest_path, "r", encoding="utf-8") as f:
            scl_manifest = json.load(f)
        print(f"Loaded SCL manifest: {len(scl_manifest)} scenes")
    else:
        scl_manifest = {}

    # ── Scene search ──────────────────────────────────────────────────────────

    print(f"\nSearching for scenes ({START_DATE} to {END_DATE}, "
          f"cloud < {CLOUD_COVER_MAX}%)...")

    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end   = datetime.strptime(END_DATE,   "%Y-%m-%d")

    windows, current = [], start
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
    for win_start, win_end in windows:
        print(f"  Searching {win_start}/{win_end}...")
        for attempt in range(3):
            try:
                catalog = pystac_client.Client.open(
                    "https://planetarycomputer.microsoft.com/api/stac/v1",
                    modifier=planetary_computer.sign_inplace,
                )
                search = catalog.search(
                    collections=["sentinel-2-l2a"],
                    intersects=aoi_wgs84.geometry.iloc[0],
                    datetime=f"{win_start}/{win_end}",
                )
                window_items = list(search.items())
                all_raw_items.extend(window_items)
                print(f"    Found {len(window_items)} scenes")
                break
            except Exception as e:
                if attempt < 2:
                    print(f"    Attempt {attempt+1} failed: {e} — retrying in 10s...")
                    time.sleep(10)
                else:
                    print(f"    Failed after 3 attempts: {e}")

    items = [
        i for i in all_raw_items
        if i.properties.get("eo:cloud_cover", 100) < CLOUD_COVER_MAX
    ]
    print(f"\nAfter cloud filter: {len(items)} scenes")

    seen, deduped_items = set(), []
    for i in items:
        d = i.datetime.strftime("%Y%m%d")
        t = next((p for p in i.id.split("_") if p.startswith("T") and len(p) == 6), "unknown")
        key = (d, t)
        if key not in seen:
            seen.add(key)
            deduped_items.append(i)
    items = deduped_items
    print(f"After deduplication: {len(items)} unique date+tile scenes\n")

    # ── Pre-scan: inventory what's already cached ──────────────────────────────

    print("Scanning existing cache...")
    RUN_START = datetime.now()

    pre_cached     = {idx: 0 for idx in INDICES_TO_RUN}
    pre_needed     = {idx: 0 for idx in INDICES_TO_RUN}
    scl_pre_cached = 0
    scl_pre_needed = 0

    for item in items:
        d = item.datetime.strftime("%Y%m%d")
        t = next((p for p in item.id.split("_") if p.startswith("T") and len(p) == 6), "unknown")
        for idx in INDICES_TO_RUN:
            if (CACHE_BASE_DIR / idx / f"{idx}_{d}_{t}.tif").exists():
                pre_cached[idx] += 1
            else:
                pre_needed[idx] += 1
        if (scl_dir / f"SCL_{d}_{t}.tif").exists():
            scl_pre_cached += 1
        else:
            scl_pre_needed += 1

    for idx in INDICES_TO_RUN:
        print(f"  {idx}: {pre_cached[idx]} cached, {pre_needed[idx]} needed")
    print(f"  SCL: {scl_pre_cached} cached, {scl_pre_needed} needed")

    # ── Progress tracking ──────────────────────────────────────────────────────

    run_done           = {idx: 0 for idx in INDICES_TO_RUN}
    run_failed         = {idx: 0 for idx in INDICES_TO_RUN}
    scl_done           = 0
    rate_limit_events  = []
    current_scene_idx  = 0
    SUMMARY_PATH       = CACHE_BASE_DIR / f"build_progress_{AOI}.txt"

    def write_summary(status="running"):
        now     = datetime.now()
        elapsed = now - RUN_START
        elapsed_str = (f"{int(elapsed.total_seconds()//3600)}h "
                       f"{int((elapsed.total_seconds()%3600)//60)}m")
        total_needed = max(pre_needed[INDICES_TO_RUN[0]], 1)
        total_done   = run_done[INDICES_TO_RUN[0]]
        if total_done > 0:
            rate      = elapsed.total_seconds() / total_done
            eta_sec   = rate * (total_needed - total_done)
            eta_str   = (f"{int(eta_sec//3600)}h "
                         f"{int((eta_sec%3600)//60)}m")
        else:
            eta_str = "unknown"

        lines = [
            f"Cache Build Progress — {AOI.upper()} AOI  (Sentinel-2)",
            "=" * 52,
            f"Status:       {status}",
            f"Started:      {RUN_START.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Updated:      {now.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Elapsed:      {elapsed_str}",
            f"Date range:   {START_DATE} to {END_DATE}",
            f"Cloud max:    {CLOUD_COVER_MAX}%",
            "",
            f"Scene inventory  ({len(items)} total after cloud filter + dedup)",
            "",
            f"  {'Index':<8}  {'At start':>10}  {'Needed':>8}  "
            f"{'Done':>6}  {'Failed':>6}  {'Left':>6}",
            f"  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*6}",
        ]
        for idx in INDICES_TO_RUN:
            left = pre_needed[idx] - run_done[idx] - run_failed[idx]
            lines.append(
                f"  {idx:<8}  {pre_cached[idx]:>10}  {pre_needed[idx]:>8}"
                f"  {run_done[idx]:>6}  {run_failed[idx]:>6}  {left:>6}"
            )
        scl_left = scl_pre_needed - scl_done
        lines += [
            f"  {'SCL':<8}  {scl_pre_cached:>10}  {scl_pre_needed:>8}"
            f"  {scl_done:>6}  {'—':>6}  {scl_left:>6}",
            "",
            f"  Progress:  scene {current_scene_idx} / {len(items)}",
            f"  ETA:       {eta_str} remaining",
        ]
        if rate_limit_events:
            lines += ["", "403 rate-limit events:"]
            for ev in rate_limit_events[-10:]:
                lines.append(f"  {ev}")
        lines.append("")
        SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")

    write_summary(status="starting")
    print(f"Summary file: {SUMMARY_PATH}\n")

    # ── Main scene loop ────────────────────────────────────────────────────────

    all_results = {
        idx: {"successful": 0, "skipped": 0, "failed": 0, "redownloaded": 0}
        for idx in INDICES_TO_RUN
    }

    for scene_idx, item in enumerate(items):
        current_scene_idx = scene_idx + 1
        scene_id = item.id
        date_str = item.datetime.strftime("%Y%m%d")
        tile_id  = next(
            (p for p in scene_id.split("_") if p.startswith("T") and len(p) == 6),
            "unknown",
        )

        scl_filename = f"SCL_{date_str}_{tile_id}.tif"
        scl_path     = scl_dir / scl_filename
        scl_needed   = not scl_path.exists()

        indices_needed = []
        for index_name in INDICES_TO_RUN:
            out_filename = f"{index_name}_{date_str}_{tile_id}.tif"
            out_path     = CACHE_BASE_DIR / index_name / out_filename
            if out_path.exists():
                if scene_id not in manifests[index_name]:
                    manifests[index_name][scene_id] = {
                        "filename": out_filename,
                        "date":     item.datetime.isoformat(),
                    }
                all_results[index_name]["skipped"] += 1
            else:
                indices_needed.append(index_name)

        if not indices_needed and not scl_needed:
            if current_scene_idx % 10 == 0:
                print(
                    f"[{current_scene_idx}/{len(items)}] "
                    + " | ".join(
                        f"{k}: {v['successful']}new/{v['skipped']}skip/{v['failed']}fail"
                        for k, v in all_results.items()
                    )
                )
            continue

        print(f"\n[{current_scene_idx}/{len(items)}] {scene_id}")
        print(f"  {item.datetime}  |  Tile: {tile_id}  |  "
              f"Cloud: {item.properties.get('eo:cloud_cover','N/A')}%")
        if indices_needed:
            print(f"  Indices needed: {indices_needed}")
        if scl_needed:
            print(f"  SCL needed:     yes")

        for index_name in indices_needed:
            if scene_id in manifests[index_name]:
                print(f"  ⚠ {index_name}: in manifest but file missing — re-downloading")
                all_results[index_name]["redownloaded"] += 1

        # ── Fetch + write, retry on 403 ────────────────────────────────────────
        while True:
            try:
                bands_to_fetch = set()
                for index_name in indices_needed:
                    bands_to_fetch.update(INDEX_CONFIG[index_name]["bands"])
                bands_to_fetch.add("SCL")

                print(f"  Fetching {len(bands_to_fetch)} bands: {sorted(bands_to_fetch)}")

                fetched = {}
                for band in sorted(bands_to_fetch):
                    rs = Resampling.nearest if band == "SCL" else Resampling.bilinear
                    dt = np.uint8  if band == "SCL" else np.float32
                    fetched[band] = reproject_band(
                        item.assets[band].href,
                        dst_height, dst_width, dst_transform,
                        dtype=dt, resampling=rs,
                    )

                scl_arr    = fetched["SCL"]
                data_bands = {k: v for k, v in fetched.items() if k != "SCL"}

                # Snow fraction recorded before masking — stored in manifest
                # so snow distribution is queryable without re-reading rasters
                snow_frac = float(
                    ((scl_arr == 11) & aoi_polygon_mask).sum()
                    / aoi_polygon_mask.sum()
                )

                quality_mask = ~np.isin(scl_arr, QUALITY_EXCLUDE)
                valid_mask   = np.ones((dst_height, dst_width), dtype=bool)
                for arr in data_bands.values():
                    valid_mask &= (arr > 0) & (arr < 10000)
                combined_mask = quality_mask & valid_mask & aoi_polygon_mask

                clear_frac = combined_mask.sum() / aoi_polygon_mask.sum()
                print(f"  Clear pixels within AOI polygon: {clear_frac*100:.1f}%"
                      + (f"  snow: {snow_frac*100:.1f}%" if snow_frac > 0.001 else ""))

                scaled = {k: v / 10000.0 for k, v in data_bands.items()}

                # Save SCL
                if scl_needed:
                    scl_out = scl_arr.copy()
                    scl_out[~aoi_polygon_mask] = 0   # 0 = no data outside polygon
                    write_tif(scl_path, scl_out, dst_height, dst_width,
                              dst_transform, dtype=np.uint8, nodata=0)
                    scl_manifest[scene_id] = {
                        "filename":    scl_filename,
                        "date":        item.datetime.isoformat(),
                        "cloud_cover": item.properties.get("eo:cloud_cover"),
                        "snow_frac":   round(snow_frac, 4),
                        "processed":   datetime.now().isoformat(),
                    }
                    with open(scl_manifest_path, "w", encoding="utf-8") as f:
                        json.dump(scl_manifest, f, indent=2)
                    scl_done += 1
                    print(f"  ✓ {scl_filename}")

                # Save indices
                for index_name in indices_needed:
                    out_filename = f"{index_name}_{date_str}_{tile_id}.tif"
                    out_path     = CACHE_BASE_DIR / index_name / out_filename
                    mfst_path    = CACHE_BASE_DIR / index_name / "cache_manifest.json"
                    try:
                        result = compute_index(index_name, scaled)
                        result[~combined_mask] = np.nan
                        write_tif(out_path, result, dst_height, dst_width,
                                  dst_transform, dtype=np.float32, nodata=np.nan)
                        manifests[index_name][scene_id] = {
                            "filename":    out_filename,
                            "date":        item.datetime.isoformat(),
                            "cloud_cover": item.properties.get("eo:cloud_cover"),
                            "clear_frac":  round(float(clear_frac), 4),
                            "snow_frac":   round(snow_frac, 4),
                            "processed":   datetime.now().isoformat(),
                        }
                        with open(mfst_path, "w", encoding="utf-8") as f:
                            json.dump(manifests[index_name], f, indent=2)
                        all_results[index_name]["successful"] += 1
                        run_done[index_name] += 1
                        print(f"  ✓ {out_filename}")
                    except Exception as e:
                        all_results[index_name]["failed"] += 1
                        run_failed[index_name] += 1
                        print(f"  ✗ {index_name} compute/save error: {e}")

                break   # success

            except Exception as e:
                if is_rate_limited(e):
                    event = (f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — "
                             f"scene {current_scene_idx}/{len(items)} "
                             f"({scene_id[:40]}...)")
                    rate_limit_events.append(event)
                    wake = (datetime.now() + timedelta(seconds=RATE_LIMIT_SLEEP)
                            ).strftime("%H:%M:%S")
                    print(f"  ⚠ 403 rate limit — sleeping "
                          f"{RATE_LIMIT_SLEEP//60} min, retrying at {wake}...")
                    write_summary(
                        status=f"sleeping (403) — retrying at {wake}")
                    time.sleep(RATE_LIMIT_SLEEP)
                else:
                    for index_name in indices_needed:
                        all_results[index_name]["failed"] += 1
                        run_failed[index_name] += 1
                    print(f"  ✗ Band fetch error: {e}")
                    break

    # ── AOI summary ───────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"SUMMARY — {AOI.upper()} AOI  (Sentinel-2)")
    print(f"{'='*60}")
    for index_name, r in all_results.items():
        print(f"\n{index_name}:")
        print(f"  Successfully processed (new):  {r['successful']}")
        print(f"  Re-downloaded (was missing):   {r['redownloaded']}")
        print(f"  Skipped (already cached):      {r['skipped']}")
        print(f"  Failed:                        {r['failed']}")
    print(f"\nCache location: {CACHE_BASE_DIR}")

    aoi_failed = sum(r["failed"] for r in all_results.values())
    if aoi_failed > 0:
        write_summary(status=f"finished with {aoi_failed} failures — re-run to retry")
    else:
        write_summary(status="complete")

print(f"\n{'#'*62}")
print(f"# ALL AOIs COMPLETE")
print(f"{'#'*62}")

total_failed_all = 0   # individual AOI exits already reported failures
sys.exit(0)
