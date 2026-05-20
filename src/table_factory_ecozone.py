from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import time

import numpy as np
import pandas as pd
import rasterio

from src.ecozone_scenelevel import ECOZONE_LABELS, MIN_PIXELS, VALID_ECOZONE_CODES
from src.landsat import load_landsat_ecozone
from src.sentinel import load_sentinel_ecozone
from src.table_factory import (
    CLOUD_THRESHOLDS,
    PIXEL_MASKS,
    PERCENTILE_COLUMNS,
    SPATIAL_PERCENTILES,
    TEMPORAL_AGGS,
    TEMPORAL_PERCENTILE_LABELS,
    TEMPORAL_PERCENTILES,
    _growing_season_day,
    _half_month_bucket,
    _month_bucket,
    build_scene_catalog,
    canonical_dashboard_tables_dir,
)


ECOZONE_SCENE_STEM = "scene_summary_ecozone"
ECOZONE_TEMPORAL_STEM = "temporal_summary_ecozone"
ECOZONE_DICTIONARY_NAME = "data_dictionary_ecozone.md"


@lru_cache(maxsize=8)
def _load_ecozone_array(sensor: str, aoi: str) -> np.ndarray:
    if sensor == "s2":
        ecozone_arr, _, _, _ = load_sentinel_ecozone(aoi)
    else:
        ecozone_arr, _, _, _ = load_landsat_ecozone(aoi)
    return np.asarray(ecozone_arr)


def summarize_scene_ecozone(scene_row: pd.Series) -> list[dict]:
    ecozone_arr = _load_ecozone_array(str(scene_row["sensor"]), str(scene_row["aoi"]))
    mask_meta = PIXEL_MASKS[str(scene_row["sensor"])]

    with rasterio.open(Path(scene_row["filepath"])) as src:
        data = src.read(1, masked=True)
        pixel_count = src.height * src.width

    finite_mask = np.isfinite(np.asarray(data.filled(np.nan), dtype=np.float32))
    if not finite_mask.any():
        return []

    rows: list[dict] = []
    timestamp = pd.Timestamp(scene_row["date"])
    for ecozone_code in VALID_ECOZONE_CODES:
        eco_mask = ecozone_arr == ecozone_code
        combined = finite_mask & eco_mask
        valid_pixels = int(combined.sum())
        if valid_pixels < MIN_PIXELS:
            continue

        pixels = np.asarray(data.filled(np.nan), dtype=np.float32)[combined]
        valid_pixel_fraction = float(valid_pixels) / float(pixel_count)
        for percentile in SPATIAL_PERCENTILES:
            rows.append(
                {
                    "sensor": str(scene_row["sensor"]),
                    "aoi": str(scene_row["aoi"]),
                    "index": str(scene_row["index"]),
                    "ecozone_code": int(ecozone_code),
                    "ecozone_label": ECOZONE_LABELS[int(ecozone_code)],
                    "date": timestamp.normalize(),
                    "year": int(scene_row["year"]),
                    "doy": int(scene_row["doy"]),
                    "growing_season_day": _growing_season_day(timestamp),
                    "season_filter": "all",
                    "temporal_agg": "scene",
                    "temporal_percentile": "none",
                    "spatial_percentile": PERCENTILE_COLUMNS[percentile],
                    "cloud_threshold": pd.NA,
                    "cloud_percent": scene_row["cloud_percent"],
                    "pixel_mask_id": mask_meta["pixel_mask_id"],
                    "pixel_mask_description": mask_meta["pixel_mask_description"],
                    "pixel_mask_version": mask_meta["pixel_mask_version"],
                    "n_pixels": valid_pixels,
                    "valid_pixel_fraction": valid_pixel_fraction,
                    "n_scenes": 1,
                    "value": float(np.percentile(pixels, percentile)),
                    "source_file_or_composite_id": str(scene_row["source_file_or_composite_id"]),
                }
            )
    return rows


def build_scene_summary_ecozone(scene_catalog: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    total_scenes = len(scene_catalog)
    for scene_idx, row in enumerate(scene_catalog.itertuples(index=False), start=1):
        if scene_idx == 1 or scene_idx % 25 == 0 or scene_idx == total_scenes:
            print(f"  Ecozone scene summaries: {scene_idx}/{total_scenes}", flush=True)
        row_series = pd.Series(row._asdict())
        records.extend(summarize_scene_ecozone(row_series))

    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records).sort_values(
        ["sensor", "aoi", "index", "ecozone_code", "date", "spatial_percentile"]
    ).reset_index(drop=True)


def _time_bin_columns(frame: pd.DataFrame, temporal_agg: str) -> pd.DataFrame:
    expanded = frame.copy()
    if temporal_agg == "half_month":
        buckets = expanded["date"].map(_half_month_bucket)
    elif temporal_agg == "month":
        buckets = expanded["date"].map(_month_bucket)
    else:
        buckets = expanded["date"].map(lambda value: (value, value, value.strftime("%Y-%m-%d")))
    expanded["time_bin_start"] = buckets.map(lambda item: item[0])
    expanded["time_bin_end"] = buckets.map(lambda item: item[1])
    expanded["time_bin_label"] = buckets.map(lambda item: item[2])
    return expanded


def build_temporal_summary_ecozone(scene_summary: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    total_start = time.perf_counter()
    for temporal_agg in TEMPORAL_AGGS:
        agg_start = time.perf_counter()
        print(f"  Interval={temporal_agg}: deriving time bins...", flush=True)
        expanded = _time_bin_columns(scene_summary, temporal_agg)
        for season_filter in ["all", "growing"]:
            season_frame = expanded if season_filter == "all" else expanded[expanded["growing_season_day"].notna()]
            if season_frame.empty:
                print(f"    season={season_filter}: no rows, skipping", flush=True)
                continue
            print(
                f"    season={season_filter}: source_rows={len(season_frame)}",
                flush=True,
            )
            for cloud_threshold in CLOUD_THRESHOLDS:
                threshold_start = time.perf_counter()
                threshold_frame = season_frame[
                    season_frame["cloud_percent"].isna() | (season_frame["cloud_percent"] <= cloud_threshold)
                ]
                if threshold_frame.empty:
                    print(f"      cloud<={cloud_threshold}: no rows, skipping", flush=True)
                    continue
                group_columns = [
                    "sensor",
                    "aoi",
                    "index",
                    "ecozone_code",
                    "ecozone_label",
                    "spatial_percentile",
                    "time_bin_start",
                    "time_bin_end",
                    "time_bin_label",
                    "year",
                    "pixel_mask_id",
                    "pixel_mask_description",
                    "pixel_mask_version",
                ]
                grouped = threshold_frame.groupby(group_columns, dropna=False)
                total_groups = grouped.ngroups
                print(
                    f"      cloud<={cloud_threshold}: rows={len(threshold_frame)} groups={total_groups}",
                    flush=True,
                )
                for group_idx, (group_key, group) in enumerate(grouped, start=1):
                    if group_idx == 1 or group_idx % 500 == 0 or group_idx == total_groups:
                        elapsed = time.perf_counter() - threshold_start
                        per_group = elapsed / group_idx if group_idx else 0.0
                        remaining = total_groups - group_idx
                        eta_minutes = (per_group * remaining) / 60.0
                        print(
                            "        "
                            f"group {group_idx}/{total_groups} "
                            f"elapsed={elapsed/60:.1f}m eta={eta_minutes:.1f}m "
                            f"{group_key[0]}/{group_key[1]}/{group_key[2]}/ecozone{int(group_key[3])}",
                            flush=True,
                        )
                    values = group["value"].to_numpy(dtype=float)
                    for percentile in TEMPORAL_PERCENTILES:
                        records.append(
                            {
                                "sensor": group_key[0],
                                "aoi": group_key[1],
                                "index": group_key[2],
                                "ecozone_code": int(group_key[3]),
                                "ecozone_label": group_key[4],
                                "date": group_key[6],
                                "year": int(group_key[9]),
                                "doy": pd.Timestamp(group_key[6]).dayofyear,
                                "growing_season_day": (
                                    _growing_season_day(pd.Timestamp(group_key[6])) if season_filter == "growing" else pd.NA
                                ),
                                "season_filter": season_filter,
                                "temporal_agg": temporal_agg,
                                "temporal_percentile": TEMPORAL_PERCENTILE_LABELS[percentile],
                                "spatial_percentile": group_key[5],
                                "cloud_threshold": cloud_threshold,
                                "cloud_percent": float(group["cloud_percent"].max()) if group["cloud_percent"].notna().any() else pd.NA,
                                "pixel_mask_id": group_key[10],
                                "pixel_mask_description": group_key[11],
                                "pixel_mask_version": group_key[12],
                                "n_pixels": int(group["n_pixels"].median()),
                                "valid_pixel_fraction": float(group["valid_pixel_fraction"].median()),
                                "n_scenes": int(group["source_file_or_composite_id"].nunique()),
                                "value": float(np.percentile(values, percentile)),
                                "source_file_or_composite_id": "|".join(group["source_file_or_composite_id"].astype(str).tolist()),
                                "time_bin_label": group_key[8],
                                "time_bin_start": group_key[6],
                                "time_bin_end": group_key[7],
                            }
                        )
                print(
                    f"      cloud<={cloud_threshold}: completed in {(time.perf_counter() - threshold_start)/60:.1f}m",
                    flush=True,
                )
        print(
            f"  Interval={temporal_agg}: completed in {(time.perf_counter() - agg_start)/60:.1f}m",
            flush=True,
        )

    if not records:
        return pd.DataFrame()
    print(
        f"  Ecozone temporal summary total elapsed={(time.perf_counter() - total_start)/60:.1f}m rows={len(records)}",
        flush=True,
    )
    return pd.DataFrame.from_records(records).sort_values(
        [
            "sensor",
            "aoi",
            "index",
            "ecozone_code",
            "temporal_agg",
            "time_bin_start",
            "spatial_percentile",
            "temporal_percentile",
            "cloud_threshold",
        ]
    ).reset_index(drop=True)


def build_ecozone_data_dictionary_markdown(output_dir: Path) -> str:
    return f"""# Dashboard Ecozone Data Dictionary

Generated table products in `{output_dir}`:

- `{ECOZONE_SCENE_STEM}.csv` / `{ECOZONE_SCENE_STEM}.parquet`
- `{ECOZONE_TEMPORAL_STEM}.csv` / `{ECOZONE_TEMPORAL_STEM}.parquet`

## Dataset definitions

- `scene_summary_ecozone`: one row per scene x AOI x sensor x index x ecozone x spatial percentile.
- `temporal_summary_ecozone`: one row per ecozone x temporal bin x cloud threshold x spatial percentile x temporal percentile.

## Added ecozone columns

| Column | Meaning |
|---|---|
| `ecozone_code` | Integer ecozone class code (`1`, `2`, `3`) |
| `ecozone_label` | Ecozone label (`Cool`, `Intermediate`, `Hot`) |

## Notes

- Ecozone summaries use AOI-aligned ecozone rasters already present in the cache lineage.
- Sentinel ecozone masks use the S2-snapped ecozone raster directly.
- Landsat ecozone masks use the same ecozone raster reprojected to the Landsat canonical grid.
- Rows are omitted where an ecozone contributes fewer than `{MIN_PIXELS}` valid pixels for a scene.
"""


def default_ecozone_output_dir(project_root: Path) -> Path:
    return canonical_dashboard_tables_dir(project_root)


def build_filtered_scene_catalog(
    limit_scenes_per_group: int | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    scene_catalog = build_scene_catalog()
    if start_year is not None:
        scene_catalog = scene_catalog[scene_catalog["year"] >= start_year]
    if end_year is not None:
        scene_catalog = scene_catalog[scene_catalog["year"] <= end_year]
    if limit_scenes_per_group is not None:
        scene_catalog = (
            scene_catalog.groupby(["sensor", "aoi", "index"], group_keys=False)
            .head(limit_scenes_per_group)
            .reset_index(drop=True)
        )
    return scene_catalog.reset_index(drop=True)
