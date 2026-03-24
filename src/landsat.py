"""
src/landsat.py
--------------
Shared helpers for Landsat Collection 2 analysis scripts.

Provides:
  get_landsat_index_root(aoi)       → Path to Landsat index cache
  load_landsat_scenes(aoi, index)   → List of {date, filepath, platform, path_row}
  load_landsat_ecozone(aoi)         → (ecozone_arr, height, width, transform)

The ecozone array is reprojected on-the-fly from the S2-snapped
tnc_ecozone_simplified_snapped.tif to the Landsat 30m canonical grid,
determined by reading the first available scene from the cache.

VALUE field:  1=Cool  2=Intermediate  3=Hot  (0=excluded)
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from src.aoi import get_aoi_config
from src.paths import project_path


def get_landsat_index_root(aoi: str) -> Path:
    """Return the Landsat index cache root for the given AOI."""
    return project_path(f"{aoi}_landsat_index_root")


def load_landsat_scenes(aoi: str, index_name: str) -> list[dict]:
    """
    Load scenes from the Landsat manifest for one index, sorted by date.
    Returns empty list (with a warning) if the manifest does not exist.
    Each entry: {date, filepath, platform, path_row}
    """
    root          = get_landsat_index_root(aoi)
    index_dir     = root / index_name
    manifest_path = index_dir / "cache_manifest.json"

    if not manifest_path.exists():
        print(f"  [{index_name}] No Landsat manifest found — run build_landsat_cache.py first.")
        return []

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    scenes = []
    for _, meta in manifest.items():
        fp = index_dir / meta["filename"]
        if fp.exists():
            scenes.append({
                "date":     datetime.fromisoformat(meta["date"]),
                "filepath": fp,
                "platform": meta.get("platform", "unknown"),
                "path_row": meta.get("path_row", ""),
            })
    return sorted(scenes, key=lambda s: s["date"])


def load_landsat_ecozone(aoi: str) -> tuple[np.ndarray, int, int, object]:
    """
    Reproject the S2-snapped ecozone raster to the Landsat 30m canonical grid.

    Grid spec is read from the first available Landsat scene (any index).
    Returns (ecozone_arr, dst_height, dst_width, dst_transform).

    Raises FileNotFoundError if no Landsat scenes exist yet for the AOI.
    """
    cfg          = get_aoi_config(aoi)
    root         = get_landsat_index_root(aoi)
    ecozone_path = cfg.ecozone_dir / "tnc_ecozone_simplified_snapped.tif"

    # Find any scene to read canonical grid from
    ref_scene = None
    for idx in ["NDVI", "NDMI", "EVI"]:
        candidates = sorted((root / idx).glob("*.tif"))
        if candidates:
            ref_scene = candidates[0]
            break

    if ref_scene is None:
        raise FileNotFoundError(
            f"No Landsat scenes found under {root}. "
            "Run Cache/build_landsat_cache.py first."
        )

    with rasterio.open(ref_scene) as ref:
        dst_transform = ref.transform
        dst_crs       = ref.crs
        dst_height    = ref.height
        dst_width     = ref.width

    eco_arr = np.zeros((dst_height, dst_width), dtype=np.int16)
    with rasterio.open(ecozone_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=eco_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )

    return eco_arr, dst_height, dst_width, dst_transform
