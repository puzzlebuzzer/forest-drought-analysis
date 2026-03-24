#!/usr/bin/env python3
"""
build_landsat_cache.py — Landsat Collection 2 index cache builder (v2)
-----------------------------------------------------------------------
Downloads NDVI, NDMI, and/or EVI from Landsat 5/7/8/9 (1984–present)
via Microsoft Planetary Computer STAC.

Quality masking:
  Index rasters exclude only cloud-coded pixels: dilated cloud (bit 1),
  cirrus (bit 2), and cloud (bit 3).  Cloud shadow (bit 4), snow (bit 5),
  and water (bit 7) are retained.  The raw QA_PIXEL band is saved
  alongside each scene for any further filtering at analysis time.
  For max compositing this is fine: cloudy pixels have low NDVI and
  will not survive the annual or monthly max.

Spatial masking:
  Output pixels are NaN outside the TNC AOI polygon boundary.

Rate limiting:
  On a 403 the script sleeps 5 minutes and retries the same scene.

QA_PIXEL bit reference (Landsat Collection 2):
  Bit 0  Fill            Bit 4  Cloud Shadow
  Bit 1  Dilated Cloud   Bit 5  Snow
  Bit 2  Cirrus          Bit 6  Clear
  Bit 3  Cloud           Bit 7  Water

Key differences from Sentinel-2:
  Collection : landsat-c2-l2
  Satellites : Landsat 5 (1984–2013), 7 (1999–2022), 8 (2013+), 9 (2021+)
  Resolution : 30 m
  Bands      : red, nir08, swir16, blue  (lowercase STAC asset keys)
  Scale      : raw × 0.0000275 + (−0.2)  (Collection 2 SR)
  QA band    : qa_pixel  (uint16 bitmask)

Run from project root:
  python Cache/build_landsat_cache.py                                     # both AOIs, all indices
  python Cache/build_landsat_cache.py --aoi north
  python Cache/build_landsat_cache.py --aoi south --indices NDVI NDMI --start-date 1984-01-01
  python Cache/build_landsat_cache.py --cache-suffix _3_24                # writes to GWNF_cache_3_24/ and Smoky_cache_3_24/
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
from src.cli import make_parser, add_aoi_arg, add_indices_arg, add_date_range_args, add_cloud_arg, add_cache_suffix_arg
from src.paths import project_path

# ── CLI ────────────────────────────────────────────────────────────────────────

parser = make_parser("Build Landsat Collection 2 index cache")
add_aoi_arg(parser, default=None)
add_indices_arg(parser)
add_date_range_args(parser, "1984-01-01")
add_cloud_arg(parser)
add_cache_suffix_arg(parser)
args = parser.parse_args()

INDICES_TO_RUN = args.indices
START_DATE     = args.start_date
END_DATE       = args.end_date
CLOUD_MAX      = args.cloud_max
AOIS_TO_RUN    = [args.aoi] if args.aoi else ["north", "south"]

# ── Constants ──────────────────────────────────────────────────────────────────

TARGET_CRS        = "EPSG:32617"
TARGET_RESOLUTION = 30
RATE_LIMIT_SLEEP  = 300   # 5 minutes on 403

INDEX_CONFIG = {
    "NDVI": {"bands": ["red", "nir08"]},
    "NDMI": {"bands": ["nir08", "swir16"]},
    "EVI":  {"bands": ["blue", "red", "nir08"]},
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


def apply_scale(raw: np.ndarray) -> np.ndarray:
    """Landsat C2 SR scale: × 0.0000275 + (−0.2). Values outside [0,1] → NaN."""
    scaled = raw.astype(np.float32) * 0.0000275 - 0.2
    return np.where((scaled >= 0) & (scaled <= 1.0), scaled, np.nan)


def compute_index(index_name: str, scaled: dict) -> np.ndarray:
    if index_name == "NDVI":
        nir, red = scaled["nir08"], scaled["red"]
        with np.errstate(divide="ignore", invalid="ignore"):
            return (nir - red) / (nir + red)
    if index_name == "NDMI":
        nir, swir = scaled["nir08"], scaled["swir16"]
        with np.errstate(divide="ignore", invalid="ignore"):
            return (nir - swir) / (nir + swir)
    if index_name == "EVI":
        nir, red, blue = scaled["nir08"], scaled["red"], scaled["blue"]
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
    print(f"# PROCESSING AOI: {AOI.upper()}  (Landsat C2)")
    print(f"{'#'*62}")

    cfg                = get_aoi_config(AOI)
    LANDSCAPE_ID       = cfg.landscape_id
    LANDSAT_INDEX_ROOT = project_path(f"{AOI}_landsat_index_root")

    if args.cache_suffix:
        # project_path returns  .../GWNF_cache/landsat/indices
        # parents[1]        is  .../GWNF_cache   (the versioned root)
        _cache_root        = LANDSAT_INDEX_ROOT.parents[1]
        _new_cache_root    = _cache_root.parent / (_cache_root.name + args.cache_suffix)
        LANDSAT_INDEX_ROOT = _new_cache_root / LANDSAT_INDEX_ROOT.relative_to(_cache_root)

    print(f"AOI:            {AOI} ({LANDSCAPE_ID})")
    print(f"Cache:          {LANDSAT_INDEX_ROOT}")
    print(f"Indices:        {INDICES_TO_RUN}")
    print(f"Date range:     {START_DATE} → {END_DATE}")
    print(f"Cloud max:      {CLOUD_MAX}%")

    # ── AOI geometry + canonical grid ─────────────────────────────────────────

    print("\nLoading AOI shapefile...")
    aoi_gdf   = gpd.read_file(AOI_SHAPEFILE)
    aoi_gdf   = aoi_gdf[aoi_gdf["LscapeID"] == LANDSCAPE_ID]
    if len(aoi_gdf) == 0:
        raise ValueError(f"No AOI found for LscapeID='{LANDSCAPE_ID}'")

    aoi_wgs84 = aoi_gdf.to_crs("EPSG:4326")
    aoi_wgs84["geometry"] = aoi_wgs84["geometry"].apply(
        lambda g: transform(lambda x, y, z=None: (x, y), g)
    )
    aoi_utm   = aoi_gdf.to_crs(TARGET_CRS)
    bounds    = aoi_utm.total_bounds

    dst_width     = int((bounds[2] - bounds[0]) / TARGET_RESOLUTION)
    dst_height    = int((bounds[3] - bounds[1]) / TARGET_RESOLUTION)
    dst_transform = rasterio.transform.from_bounds(
        bounds[0], bounds[1], bounds[2], bounds[3], dst_width, dst_height,
    )

    print(f"Canonical grid: {dst_width} × {dst_height} px  ({TARGET_RESOLUTION}m)")
    print(f"~{(dst_width * dst_height * 4) / 1e6:.1f} MB per index scene, "
          f"~{(dst_width * dst_height * 2) / 1e6:.1f} MB per QA_PIXEL scene (uint16)")

    # ── AOI polygon mask ───────────────────────────────────────────────────────

    print("Rasterizing AOI polygon mask...")
    aoi_polygon_mask = rasterio.features.geometry_mask(
        [geom.__geo_interface__ for geom in aoi_utm.geometry],
        out_shape=(dst_height, dst_width),
        transform=dst_transform,
        all_touched=False,
        invert=True,
    )
    print(f"  {aoi_polygon_mask.mean()*100:.1f}% of bounding box inside polygon")

    # ── Directories + manifests ────────────────────────────────────────────────

    for idx in INDICES_TO_RUN:
        (LANDSAT_INDEX_ROOT / idx).mkdir(parents=True, exist_ok=True)

    qa_dir = LANDSAT_INDEX_ROOT / "QA_PIXEL"
    qa_dir.mkdir(parents=True, exist_ok=True)

    manifests = {}
    for index_name in INDICES_TO_RUN:
        mp = LANDSAT_INDEX_ROOT / index_name / "cache_manifest.json"
        if mp.exists():
            with open(mp, "r", encoding="utf-8") as f:
                manifests[index_name] = json.load(f)
            print(f"Loaded {index_name} manifest: {len(manifests[index_name])} scenes")
        else:
            manifests[index_name] = {}

    qa_manifest_path = qa_dir / "cache_manifest.json"
    if qa_manifest_path.exists():
        with open(qa_manifest_path, "r", encoding="utf-8") as f:
            qa_manifest = json.load(f)
        print(f"Loaded QA_PIXEL manifest: {len(qa_manifest)} scenes")
    else:
        qa_manifest = {}

    # ── Scene search ──────────────────────────────────────────────────────────

    print(f"\nSearching Planetary Computer for Landsat C2 scenes...")

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
                    collections=["landsat-c2-l2"],
                    intersects=aoi_wgs84.geometry.iloc[0],
                    datetime=f"{win_start}/{win_end}",
                    query={"platform": {"in": [
                        "landsat-5", "landsat-7", "landsat-8", "landsat-9"
                    ]}},
                )
                found = list(search.items())
                all_raw_items.extend(found)
                print(f"    {len(found)} scenes")
                break
            except Exception as e:
                if attempt < 2:
                    print(f"    Attempt {attempt+1} failed: {e} — retrying in 10s...")
                    time.sleep(10)
                else:
                    print(f"    Failed after 3 attempts: {e}")

    items = [
        i for i in all_raw_items
        if i.properties.get("eo:cloud_cover", 100) < CLOUD_MAX
    ]
    print(f"\nAfter cloud filter (<{CLOUD_MAX}%): {len(items)} scenes")

    seen, deduped = set(), []
    for i in items:
        date_str = i.datetime.strftime("%Y%m%d")
        path_row = (i.properties.get("landsat:wrs_path", "?")
                    + i.properties.get("landsat:wrs_row", "?"))
        key = (date_str, path_row)
        if key not in seen:
            seen.add(key)
            deduped.append(i)
    items = deduped
    print(f"After deduplication: {len(items)} unique scenes\n")

    # ── Pre-scan ──────────────────────────────────────────────────────────────

    print("Scanning existing cache...")
    RUN_START = datetime.now()

    pre_cached    = {idx: 0 for idx in INDICES_TO_RUN}
    pre_needed    = {idx: 0 for idx in INDICES_TO_RUN}
    qa_pre_cached = 0
    qa_pre_needed = 0

    for item in items:
        d  = item.datetime.strftime("%Y%m%d")
        pr = (item.properties.get("landsat:wrs_path", "?")
              + item.properties.get("landsat:wrs_row", "?"))
        short_id = f"{d}_{pr}"
        for idx in INDICES_TO_RUN:
            if (LANDSAT_INDEX_ROOT / idx / f"{idx}_{short_id}.tif").exists():
                pre_cached[idx] += 1
            else:
                pre_needed[idx] += 1
        if (qa_dir / f"QA_PIXEL_{short_id}.tif").exists():
            qa_pre_cached += 1
        else:
            qa_pre_needed += 1

    for idx in INDICES_TO_RUN:
        print(f"  {idx}: {pre_cached[idx]} cached, {pre_needed[idx]} needed")
    print(f"  QA_PIXEL: {qa_pre_cached} cached, {qa_pre_needed} needed")

    # ── Progress tracking ──────────────────────────────────────────────────────

    run_done          = {idx: 0 for idx in INDICES_TO_RUN}
    run_failed        = {idx: 0 for idx in INDICES_TO_RUN}
    qa_done           = 0
    rate_limit_events = []
    current_scene_idx = 0
    SUMMARY_PATH      = LANDSAT_INDEX_ROOT / f"build_progress_{AOI}.txt"

    def write_summary(status="running"):
        now     = datetime.now()
        elapsed = now - RUN_START
        elapsed_str = (f"{int(elapsed.total_seconds()//3600)}h "
                       f"{int((elapsed.total_seconds()%3600)//60)}m")
        total_needed = max(pre_needed[INDICES_TO_RUN[0]], 1)
        total_done   = run_done[INDICES_TO_RUN[0]]
        if total_done > 0:
            rate    = elapsed.total_seconds() / total_done
            eta_sec = rate * (total_needed - total_done)
            eta_str = (f"{int(eta_sec//3600)}h "
                       f"{int((eta_sec%3600)//60)}m")
        else:
            eta_str = "unknown"

        lines = [
            f"Cache Build Progress — {AOI.upper()} AOI  (Landsat C2)",
            "=" * 52,
            f"Status:       {status}",
            f"Started:      {RUN_START.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Updated:      {now.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Elapsed:      {elapsed_str}",
            f"Date range:   {START_DATE} to {END_DATE}",
            f"Cloud max:    {CLOUD_MAX}%",
            "",
            f"Scene inventory  ({len(items)} total after cloud filter + dedup)",
            "",
            f"  {'Index':<10}  {'At start':>10}  {'Needed':>8}  "
            f"{'Done':>6}  {'Failed':>6}  {'Left':>6}",
            f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*6}",
        ]
        for idx in INDICES_TO_RUN:
            left = pre_needed[idx] - run_done[idx] - run_failed[idx]
            lines.append(
                f"  {idx:<10}  {pre_cached[idx]:>10}  {pre_needed[idx]:>8}"
                f"  {run_done[idx]:>6}  {run_failed[idx]:>6}  {left:>6}"
            )
        qa_left = qa_pre_needed - qa_done
        lines += [
            f"  {'QA_PIXEL':<10}  {qa_pre_cached:>10}  {qa_pre_needed:>8}"
            f"  {qa_done:>6}  {'—':>6}  {qa_left:>6}",
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
        scene_id  = item.id
        date_str  = item.datetime.strftime("%Y%m%d")
        platform  = item.properties.get("platform", "unknown")
        wrs_path  = item.properties.get("landsat:wrs_path", "?")
        wrs_row   = item.properties.get("landsat:wrs_row",  "?")
        path_row  = f"{wrs_path}{wrs_row}"
        short_id  = f"{date_str}_{path_row}"

        qa_filename = f"QA_PIXEL_{short_id}.tif"
        qa_path     = qa_dir / qa_filename
        qa_needed   = not qa_path.exists()

        indices_needed = []
        for index_name in INDICES_TO_RUN:
            out_file = f"{index_name}_{short_id}.tif"
            out_path = LANDSAT_INDEX_ROOT / index_name / out_file
            if out_path.exists():
                if scene_id not in manifests[index_name]:
                    manifests[index_name][scene_id] = {
                        "filename": out_file,
                        "date":     item.datetime.isoformat(),
                    }
                all_results[index_name]["skipped"] += 1
            else:
                indices_needed.append(index_name)

        if not indices_needed and not qa_needed:
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
        print(f"  {item.datetime.date()}  |  {platform}  |  "
              f"p{wrs_path}r{wrs_row}  |  "
              f"cloud: {item.properties.get('eo:cloud_cover','N/A')}%")
        if indices_needed:
            print(f"  Indices needed: {indices_needed}")
        if qa_needed:
            print(f"  QA_PIXEL needed: yes")

        for index_name in indices_needed:
            if scene_id in manifests[index_name]:
                print(f"  ⚠ {index_name}: in manifest but file missing — re-downloading")
                all_results[index_name]["redownloaded"] += 1

        # ── Fetch + write, retry on 403 ────────────────────────────────────────
        while True:
            try:
                # Re-sign item before each fetch attempt to refresh PC URLs
                item = planetary_computer.sign(item)

                bands_to_fetch = set()
                for index_name in indices_needed:
                    bands_to_fetch.update(INDEX_CONFIG[index_name]["bands"])
                bands_to_fetch.add("qa_pixel")

                print(f"  Fetching {len(bands_to_fetch)} bands: {sorted(bands_to_fetch)}")

                fetched = {}
                for band in sorted(bands_to_fetch):
                    if band not in item.assets:
                        raise KeyError(f"Band '{band}' not in scene assets")
                    rs = Resampling.nearest if band == "qa_pixel" else Resampling.bilinear
                    dt = np.uint16 if band == "qa_pixel" else np.float32
                    fetched[band] = reproject_band(
                        item.assets[band].href,
                        dst_height, dst_width, dst_transform,
                        dtype=dt, resampling=rs,
                    )

                qa_arr     = fetched["qa_pixel"]
                refl_bands = {k: apply_scale(v) for k, v in fetched.items()
                              if k != "qa_pixel"}

                # Snow fraction recorded before masking — stored in manifest
                # so snow distribution is queryable without re-reading rasters
                snow_frac = float(
                    (((qa_arr & (1 << 5)) != 0) & aoi_polygon_mask).sum()
                    / aoi_polygon_mask.sum()
                )

                # Exclusion mask: cloud (bits 1,2,3) + snow (bit 5).
                # Snow excluded because it inflates NDMI; fraction recorded above.
                # Cloud shadow (bit 4) and water (bit 7) are retained.
                # QA_PIXEL saved to disk for any further filtering at analysis time.
                exclude_mask = (
                    ((qa_arr & (1 << 1)) != 0) |   # dilated cloud
                    ((qa_arr & (1 << 2)) != 0) |   # cirrus
                    ((qa_arr & (1 << 3)) != 0) |   # cloud
                    ((qa_arr & (1 << 5)) != 0)     # snow
                )
                valid_refl = np.ones((dst_height, dst_width), dtype=bool)
                for arr in refl_bands.values():
                    valid_refl &= np.isfinite(arr)
                combined_mask = valid_refl & ~exclude_mask & aoi_polygon_mask

                valid_frac = combined_mask.sum() / aoi_polygon_mask.sum()
                print(f"  Valid pixels within AOI polygon: {valid_frac*100:.1f}%"
                      + (f"  snow: {snow_frac*100:.1f}%" if snow_frac > 0.001 else ""))

                # Save QA_PIXEL
                if qa_needed:
                    qa_out = qa_arr.copy()
                    qa_out[~aoi_polygon_mask] = 0   # 0 = fill outside polygon
                    write_tif(qa_path, qa_out, dst_height, dst_width,
                              dst_transform, dtype=np.uint16, nodata=0)
                    qa_manifest[scene_id] = {
                        "filename":    qa_filename,
                        "date":        item.datetime.isoformat(),
                        "platform":    platform,
                        "path_row":    f"p{wrs_path}r{wrs_row}",
                        "cloud_cover": item.properties.get("eo:cloud_cover"),
                        "snow_frac":   round(snow_frac, 4),
                        "processed":   datetime.now().isoformat(),
                    }
                    with open(qa_manifest_path, "w", encoding="utf-8") as f:
                        json.dump(qa_manifest, f, indent=2)
                    qa_done += 1
                    print(f"  ✓ {qa_filename}")

                # Save indices
                for index_name in indices_needed:
                    out_file  = f"{index_name}_{short_id}.tif"
                    out_path  = LANDSAT_INDEX_ROOT / index_name / out_file
                    mfst_path = LANDSAT_INDEX_ROOT / index_name / "cache_manifest.json"
                    try:
                        result = compute_index(index_name, refl_bands)
                        result[~combined_mask] = np.nan
                        write_tif(out_path, result, dst_height, dst_width,
                                  dst_transform, dtype=np.float32, nodata=np.nan)
                        manifests[index_name][scene_id] = {
                            "filename":     out_file,
                            "date":         item.datetime.isoformat(),
                            "cloud_cover":  item.properties.get("eo:cloud_cover"),
                            "valid_frac":   round(float(valid_frac), 4),
                            "snow_frac":    round(snow_frac, 4),
                            "platform":     platform,
                            "path_row":     f"p{wrs_path}r{wrs_row}",
                            "processed":    datetime.now().isoformat(),
                        }
                        with open(mfst_path, "w", encoding="utf-8") as f:
                            json.dump(manifests[index_name], f, indent=2)
                        all_results[index_name]["successful"] += 1
                        run_done[index_name] += 1
                        print(f"  ✓ {out_file}")
                    except Exception as e:
                        all_results[index_name]["failed"] += 1
                        run_failed[index_name] += 1
                        print(f"  ✗ {index_name} error: {e}")

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
    print(f"SUMMARY — {AOI.upper()} AOI  (Landsat C2)")
    print(f"{'='*60}")
    for index_name, r in all_results.items():
        print(f"\n{index_name}:")
        print(f"  Successfully processed (new):  {r['successful']}")
        print(f"  Re-downloaded (was missing):   {r['redownloaded']}")
        print(f"  Skipped (already cached):      {r['skipped']}")
        print(f"  Failed:                        {r['failed']}")
    print(f"\nCache location: {LANDSAT_INDEX_ROOT}")

    aoi_failed = sum(r["failed"] for r in all_results.values())
    if aoi_failed > 0:
        write_summary(status=f"finished with {aoi_failed} failures — re-run to retry")
    else:
        write_summary(status="complete")

print(f"\n{'#'*62}")
print(f"# ALL AOIs COMPLETE  (Landsat C2)")
print(f"{'#'*62}")
