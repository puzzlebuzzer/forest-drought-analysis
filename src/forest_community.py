from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from src.aoi import get_aoi_config
from src.labels import FOREST_TYPE_LABELS, normalize_label
from src.landsat import get_landsat_index_root


COMMUNITY_RASTER_NAME = "forest_community.tif"
COMMUNITY_INVENTORY_NAME = "forest_community_inventory.json"


def forest_community_raster_path(aoi: str) -> Path:
    cfg = get_aoi_config(aoi)
    preferred = cfg.forest_type_dir / COMMUNITY_RASTER_NAME
    if preferred.exists():
        return preferred
    return cfg.species_raster


def forest_community_inventory_path(aoi: str) -> Path:
    cfg = get_aoi_config(aoi)
    return cfg.forest_type_dir / COMMUNITY_INVENTORY_NAME


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metadata_from_inventory_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        label = (
            value.get("label")
            or value.get("name")
            or value.get("forest_community_label")
            or value.get("forest_type_label")
        )
        return {
            "forest_community_label": str(normalize_label(label)) if label else None,
            "forest_community_display_code": value.get("forest_community_display_code"),
            "forest_community_source_dataset": value.get("forest_community_source_dataset") or value.get("source"),
            "forest_community_source_value": _coerce_int(value.get("forest_community_source_value") or value.get("source_value")),
            "forest_community_source_key": value.get("forest_community_source_key"),
            "ecozone_group_code": _coerce_int(value.get("ecozone_group_code")),
            "ecozone_group_label": value.get("ecozone_group_label"),
            "ecozone_group_raw": value.get("ecozone_group_raw") or value.get("source_ecozone_group"),
            "ecozone_code": _coerce_int(value.get("ecozone_code") or value.get("ecozone")),
            "ecozone_label": value.get("ecozone_label"),
            "pixels": _coerce_int(value.get("pixels")),
            "include": bool(value.get("include", True)),
            "source_pixel_count": _coerce_int(value.get("source_pixel_count")),
            "source_score": _coerce_int(value.get("source_score")),
        }
    if value is None:
        return {}
    return {"forest_community_label": str(value)}


@lru_cache(maxsize=8)
def load_forest_community_inventory(aoi: str) -> dict[int, dict[str, Any]]:
    inventory_path = forest_community_inventory_path(aoi)
    if not inventory_path.exists():
        return {}

    with open(inventory_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    records: dict[int, dict[str, Any]] = {}
    if isinstance(raw, dict):
        iterable = raw.items()
    elif isinstance(raw, list):
        iterable = ((item.get("code") or item.get("value"), item) for item in raw if isinstance(item, dict))
    else:
        iterable = []

    for raw_code, raw_meta in iterable:
        code = _coerce_int(raw_code)
        if code is None:
            continue
        records[code] = _metadata_from_inventory_value(raw_meta)
    return records


def forest_community_label(aoi: str, code: int) -> str:
    inventory = load_forest_community_inventory(aoi)
    label = inventory.get(int(code), {}).get("forest_community_label")
    if label:
        return str(normalize_label(label))
    return str(normalize_label(FOREST_TYPE_LABELS.get(int(code), f"Forest community {int(code)}")))


def forest_community_metadata(aoi: str, code: int) -> dict[str, Any]:
    inventory = load_forest_community_inventory(aoi)
    metadata = dict(inventory.get(int(code), {}))
    metadata["forest_community_label"] = metadata.get("forest_community_label") or forest_community_label(aoi, code)
    return metadata


def load_sentinel_forest_community(aoi: str) -> tuple[np.ndarray, int, int, object]:
    raster_path = forest_community_raster_path(aoi)
    with rasterio.open(raster_path) as src:
        community_arr = src.read(1)
        return community_arr, src.height, src.width, src.transform


def load_landsat_forest_community(aoi: str) -> tuple[np.ndarray, int, int, object]:
    raster_path = forest_community_raster_path(aoi)
    root = get_landsat_index_root(aoi)

    ref_scene = None
    for index_name in ["NDVI", "NDMI", "EVI"]:
        candidates = sorted((root / index_name).glob("*.tif"))
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
        dst_crs = ref.crs
        dst_height = ref.height
        dst_width = ref.width

    community_arr = np.zeros((dst_height, dst_width), dtype=np.uint16)
    with rasterio.open(raster_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=community_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )

    return community_arr, dst_height, dst_width, dst_transform
