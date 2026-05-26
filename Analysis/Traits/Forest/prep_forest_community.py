#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from osgeo import gdal, ogr

from src.aoi import get_aoi_config, valid_aois
from src.paths import PROJECT_ROOT


PROJECT_DIR = PROJECT_ROOT.parent
RAW_ROOT = PROJECT_DIR / "AOI" / "TNC_Forest_Communities"
SOUTH_SOURCE = RAW_ROOT / "South_AOI_forest_types" / "Simon_FA.tif"
SOUTH_VAT = RAW_ROOT / "South_AOI_forest_types" / "Simon_FA.tif.vat.dbf"
NORTH_GDB = RAW_ROOT / "North_AOI_forest_types" / "GWNF_Ecological_Model.gdb"
NORTH_SUBDATASETS = ("AppRidges", "NBlueRidge")
COMMUNITY_RASTER_NAME = "forest_community.tif"
COMMUNITY_INVENTORY_NAME = "forest_community_inventory.json"
NORTH_INTERNAL_CODE_OVERRIDES = {
    ("NBlueRidge", 16): 116,
}
NORTH_DISPLAY_CODE_OVERRIDES = {
    ("NBlueRidge", 16): "16a",
}


def _parse_prefixed_label(value) -> tuple[int | None, str | None, str | None]:
    if value is None:
        return None, None, None
    raw = str(value).strip()
    if not raw:
        return None, None, None
    if "-" not in raw:
        return None, raw, raw
    prefix, label = raw.split("-", 1)
    try:
        code = int(prefix.strip().rstrip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    except ValueError:
        code = None
    return code, label.strip(), raw


def _north_internal_code(source_label: str, source_value: int) -> int:
    return NORTH_INTERNAL_CODE_OVERRIDES.get((source_label, source_value), source_value)


def _display_code(source_label: str, source_value: int) -> str:
    return NORTH_DISPLAY_CODE_OVERRIDES.get((source_label, source_value), str(source_value))


def _open_reference(aoi: str):
    cfg = get_aoi_config(aoi)
    ref_path = cfg.terrain_dir / "elevation.tif"
    with rasterio.open(ref_path) as ref:
        return {
            "path": ref_path,
            "crs_wkt": ref.crs.to_wkt(),
            "bounds": tuple(ref.bounds),
            "width": ref.width,
            "height": ref.height,
            "transform": ref.transform,
            "profile": ref.profile.copy(),
        }


def _warp_to_reference(src: str | Path, dst: Path, ref: dict[str, Any]) -> None:
    if isinstance(src, Path) and not src.exists():
        raise FileNotFoundError(f"Source raster not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    bounds = ref["bounds"]
    options = gdal.WarpOptions(
        format="GTiff",
        dstSRS=ref["crs_wkt"],
        outputBounds=(bounds[0], bounds[1], bounds[2], bounds[3]),
        width=ref["width"],
        height=ref["height"],
        resampleAlg="near",
        srcNodata=0,
        dstNodata=0,
        multithread=True,
        creationOptions=[
            "COMPRESS=LZW",
            "TILED=YES",
            "BLOCKXSIZE=256",
            "BLOCKYSIZE=256",
            "BIGTIFF=IF_SAFER",
        ],
    )
    result = gdal.Warp(str(dst), str(src), options=options)
    if result is None:
        raise RuntimeError(f"gdal.Warp failed for {src}")
    result = None


def _field_value(feature, field_names: set[str], *candidates: str):
    for candidate in candidates:
        if candidate in field_names:
            return feature.GetField(candidate)
    return None


def _read_dbf_inventory(path: Path, *, source_label: str) -> dict[int, dict[str, Any]]:
    ds = ogr.Open(str(path))
    if ds is None:
        raise RuntimeError(f"Could not open VAT DBF: {path}")
    layer = ds.GetLayer(0)
    layer_defn = layer.GetLayerDefn()
    field_names = {layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())}
    records: dict[int, dict[str, Any]] = {}
    for feature in layer:
        source_value = int(_field_value(feature, field_names, "Value", "VALUE"))
        label = _field_value(feature, field_names, "Name", "NAME")
        source_ecozone = _field_value(feature, field_names, "Ecozone", "ECOZONE")
        ecozone_group_code, ecozone_group_label, ecozone_group_raw = _parse_prefixed_label(source_ecozone)
        count = _field_value(feature, field_names, "Count", "COUNT")
        score = _field_value(feature, field_names, "Score", "SCORE")
        include = bool(label) and not str(source_ecozone or "").lower().startswith("0-")
        code = source_value
        records[code] = {
            "forest_community_label": str(label) if label else None,
            "forest_community_display_code": str(source_value),
            "forest_community_source_dataset": source_label,
            "forest_community_source_value": source_value,
            "forest_community_source_key": f"south:{source_label}:{source_value}",
            "ecozone_group_code": ecozone_group_code,
            "ecozone_group_label": ecozone_group_label,
            "ecozone_group_raw": ecozone_group_raw,
            "source_pixel_count": int(float(count)) if count is not None else None,
            "source_score": int(score) if score is not None else None,
            "include": include,
        }
    return records


def _rat_to_inventory(dataset_name: str, *, source_label: str) -> dict[int, dict[str, Any]]:
    ds = gdal.Open(dataset_name)
    if ds is None:
        raise RuntimeError(f"Could not open raster for RAT: {dataset_name}")
    rat = ds.GetRasterBand(1).GetDefaultRAT()
    if rat is None:
        return {}
    fields = [rat.GetNameOfCol(i) for i in range(rat.GetColumnCount())]
    records: dict[int, dict[str, Any]] = {}
    for row_idx in range(rat.GetRowCount()):
        row = {field: rat.GetValueAsString(row_idx, col_idx) for col_idx, field in enumerate(fields)}
        source_value = int(row["VALUE"])
        code = _north_internal_code(source_label, source_value)
        source_ecozone = row.get("Ecoz_grp") or row.get("Zonegrp")
        ecozone_group_code, ecozone_group_label, ecozone_group_raw = _parse_prefixed_label(source_ecozone)
        community = row.get("Ecozone") or row.get("Zonegrp") or f"{code}"
        label = community.split("-", 1)[1].strip() if "-" in community else community.strip()
        records[code] = {
            "forest_community_label": label,
            "forest_community_display_code": _display_code(source_label, source_value),
            "forest_community_source_dataset": source_label,
            "forest_community_source_value": source_value,
            "forest_community_source_key": f"north:{source_label}:{source_value}",
            "ecozone_group_code": ecozone_group_code,
            "ecozone_group_label": ecozone_group_label,
            "ecozone_group_raw": ecozone_group_raw,
            "source_pixel_count": int(float(row["COUNT"])) if row.get("COUNT") else None,
            "source": source_label,
            "include": True,
        }
    ds = None
    return records


def _merge_inventory(*inventories: dict[int, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for inventory in inventories:
        for code, metadata in inventory.items():
            if code not in merged:
                merged[code] = dict(metadata)
                continue
            current = merged[code]
            current.setdefault("forest_community_label", metadata.get("forest_community_label"))
            current.setdefault("forest_community_display_code", metadata.get("forest_community_display_code"))
            current.setdefault("forest_community_source_dataset", metadata.get("forest_community_source_dataset"))
            current.setdefault("forest_community_source_value", metadata.get("forest_community_source_value"))
            current.setdefault("forest_community_source_key", metadata.get("forest_community_source_key"))
            current.setdefault("ecozone_group_code", metadata.get("ecozone_group_code"))
            current.setdefault("ecozone_group_label", metadata.get("ecozone_group_label"))
            current.setdefault("ecozone_group_raw", metadata.get("ecozone_group_raw"))
            current["source_pixel_count"] = int(current.get("source_pixel_count") or 0) + int(metadata.get("source_pixel_count") or 0)
            current_dataset = str(current.get("forest_community_source_dataset") or "")
            metadata_dataset = str(metadata.get("forest_community_source_dataset") or "")
            if metadata_dataset and metadata_dataset not in current_dataset:
                current["forest_community_source_dataset"] = (
                    f"{current_dataset}|{metadata_dataset}" if current_dataset else metadata_dataset
                )
                current["forest_community_source_key"] = f"north:combined:{code}"
            if current.get("source") and metadata.get("source") and metadata["source"] not in str(current["source"]):
                current["source"] = f"{current['source']}|{metadata['source']}"
    return {str(code): metadata for code, metadata in sorted(merged.items())}


def _write_inventory(path: Path, inventory: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8")


def _write_output_array(path: Path, data: np.ndarray, ref: dict[str, Any]) -> None:
    profile = ref["profile"].copy()
    profile.update(
        driver="GTiff",
        dtype=rasterio.uint8,
        count=1,
        nodata=0,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        bigtiff="IF_SAFER",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.uint8, copy=False), 1)


def prep_south() -> None:
    cfg = get_aoi_config("south")
    ref = _open_reference("south")
    output_raster = cfg.forest_type_dir / COMMUNITY_RASTER_NAME
    output_inventory = cfg.forest_type_dir / COMMUNITY_INVENTORY_NAME
    print(f"Snapping south forest communities to {ref['path']}...")
    _warp_to_reference(SOUTH_SOURCE, output_raster, ref)
    inventory = {str(code): metadata for code, metadata in sorted(_read_dbf_inventory(SOUTH_VAT, source_label="Simon").items())}
    _write_inventory(output_inventory, inventory)
    print(f"Wrote {output_raster}")
    print(f"Wrote {output_inventory}")


def _north_subdataset(name: str) -> str:
    return f'OpenFileGDB:"{NORTH_GDB}":{name}'


def prep_north() -> None:
    cfg = get_aoi_config("north")
    ref = _open_reference("north")
    output_raster = cfg.forest_type_dir / COMMUNITY_RASTER_NAME
    output_inventory = cfg.forest_type_dir / COMMUNITY_INVENTORY_NAME
    print(f"Snapping north forest communities to {ref['path']}...")
    inventories = []
    with tempfile.TemporaryDirectory(prefix="forest_community_north_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        warped_paths = []
        for subdataset in NORTH_SUBDATASETS:
            source = _north_subdataset(subdataset)
            tmp_path = tmp_dir / f"{subdataset}.tif"
            _warp_to_reference(source, tmp_path, ref)
            warped_paths.append((subdataset, tmp_path))
            inventories.append(_rat_to_inventory(source, source_label=subdataset))

        mosaic = np.zeros((ref["height"], ref["width"]), dtype=np.uint8)
        overlap_pixels = 0
        conflict_pixels = 0
        for subdataset, warped_path in warped_paths:
            with rasterio.open(warped_path) as src:
                data = src.read(1)
            if subdataset == "NBlueRidge":
                for source_value, internal_code in [
                    (source_value, internal_code)
                    for (source_name, source_value), internal_code in NORTH_INTERNAL_CODE_OVERRIDES.items()
                    if source_name == subdataset
                ]:
                    data = data.copy()
                    data[data == source_value] = internal_code
            valid = data > 0
            overlap = valid & (mosaic > 0)
            conflict = overlap & (mosaic != data)
            overlap_pixels += int(overlap.sum())
            conflict_pixels += int(conflict.sum())
            mosaic[valid] = data[valid]
            print(f"  {subdataset}: {int(valid.sum()):,} valid snapped pixels")

    _write_output_array(output_raster, mosaic, ref)
    _write_inventory(output_inventory, _merge_inventory(*inventories))
    print(f"  north mosaic overlap pixels: {overlap_pixels:,}")
    print(f"  north mosaic conflict pixels: {conflict_pixels:,}")
    print(f"Wrote {output_raster}")
    print(f"Wrote {output_inventory}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TNC forest-community rasters for dashboard summaries.")
    parser.add_argument("--aoi", choices=[*valid_aois(), "all"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.aoi in {"south", "all"}:
        prep_south()
    if args.aoi in {"north", "all"}:
        prep_north()


if __name__ == "__main__":
    main()
