"""
src/sentinel.py
---------------
Shared helpers for Sentinel-2 analysis scripts.

Provides:
  load_sentinel_scenes(aoi, index)  -> List of normalized scene records
  load_sentinel_ecozone(aoi)        -> (ecozone_arr, height, width, transform)

Sentinel caches are already aligned to the AOI's S2 grid, so the ecozone
raster is read directly without reprojection.
"""

import json
from datetime import datetime
from pathlib import Path

import rasterio

from src.aoi import get_aoi_config


def _sentinel_tile_from_meta(meta: dict, scene_id: str) -> str:
    filename = meta.get("filename", "")
    parts = Path(filename).stem.split("_")
    if parts:
        last = parts[-1]
        if last.startswith("T") and len(last) >= 2:
            return last

    tokens = scene_id.split("_")
    for token in tokens:
        if token.startswith("T") and len(token) >= 2:
            return token
    return ""


def load_sentinel_scenes(aoi: str, index_name: str) -> list[dict]:
    """
    Load Sentinel-2 scenes from the manifest for one index, sorted by date.
    Returns empty list (with a warning) if the manifest does not exist.
    Each entry: {date, filepath, platform, tile, scene_id}
    """
    cfg = get_aoi_config(aoi)
    index_dir = cfg.index_cache_root / index_name
    manifest_path = index_dir / "cache_manifest.json"

    if not manifest_path.exists():
        print(f"  [{index_name}] No Sentinel manifest found — run build_sentinel_cache.py first.")
        return []

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    scenes = []
    for scene_id, meta in manifest.items():
        fp = index_dir / meta["filename"]
        if fp.exists():
            scenes.append(
                {
                    "date": datetime.fromisoformat(meta["date"]),
                    "filepath": fp,
                    "platform": scene_id.split("_", 1)[0],
                    "tile": _sentinel_tile_from_meta(meta, scene_id),
                    "scene_id": scene_id,
                }
            )
    return sorted(scenes, key=lambda s: s["date"])


def load_sentinel_ecozone(aoi: str) -> tuple:
    """
    Read the S2-snapped ecozone raster for the AOI.
    Returns (ecozone_arr, height, width, transform).
    """
    cfg = get_aoi_config(aoi)
    ecozone_path = cfg.ecozone_dir / "tnc_ecozone_simplified_snapped.tif"

    with rasterio.open(ecozone_path) as src:
        ecozone_arr = src.read(1)
        return ecozone_arr, src.height, src.width, src.transform
