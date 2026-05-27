from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
import time

import numpy as np
import pandas as pd
import rasterio

from src.dashboard_data import ECOZONE_LABELS
from src.forest_community import (
    forest_community_metadata,
    load_landsat_forest_community,
    load_sentinel_forest_community,
)
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


FOREST_COMMUNITY_SCENE_STEM = "scene_summary_forest_community"
FOREST_COMMUNITY_TEMPORAL_STEM = "temporal_summary_forest_community"
FOREST_ECOZONE_GROUP_SCENE_STEM = "scene_summary_forest_ecozone_group"
FOREST_ECOZONE_GROUP_TEMPORAL_STEM = "temporal_summary_forest_ecozone_group"
FOREST_COMMUNITY_DICTIONARY_NAME = "data_dictionary_forest_community.md"
MIN_PIXELS = 100


@lru_cache(maxsize=8)
def _load_community_array(sensor: str, aoi: str) -> np.ndarray:
    if sensor == "s2":
        community_arr, _, _, _ = load_sentinel_forest_community(aoi)
    else:
        community_arr, _, _, _ = load_landsat_forest_community(aoi)
    return np.asarray(community_arr)


@lru_cache(maxsize=8)
def _community_catalog(sensor: str, aoi: str) -> dict[int, dict]:
    community_arr = _load_community_array(sensor, aoi)
    codes = [int(code) for code in np.unique(community_arr) if int(code) > 0]
    catalog = {code: forest_community_metadata(aoi, code) for code in codes}
    return {
        code: metadata
        for code, metadata in catalog.items()
        if bool(metadata.get("include", True))
    }


@lru_cache(maxsize=8)
def _ecozone_group_catalog(sensor: str, aoi: str) -> dict[int, dict]:
    groups: dict[int, dict] = {}
    for metadata in _community_catalog(sensor, aoi).values():
        group_code = metadata.get("ecozone_group_code")
        if group_code is None:
            continue
        code = int(group_code)
        groups.setdefault(
            code,
            {
                "ecozone_group_code": code,
                "ecozone_group_label": metadata.get("ecozone_group_label") or f"Forest community group {code}",
                "ecozone_group_raw": metadata.get("ecozone_group_raw") or metadata.get("ecozone_group_label"),
            },
        )
    return groups


@lru_cache(maxsize=8)
def _ecozone_group_array(sensor: str, aoi: str) -> np.ndarray:
    community_arr = _load_community_array(sensor, aoi)
    group_arr = np.zeros(community_arr.shape, dtype=np.int16)
    for community_code, metadata in _community_catalog(sensor, aoi).items():
        group_code = metadata.get("ecozone_group_code")
        if group_code is None:
            continue
        group_arr[community_arr == int(community_code)] = int(group_code)
    return group_arr


def summarize_scene_forest_community(scene_row: pd.Series) -> list[dict]:
    sensor = str(scene_row["sensor"])
    aoi = str(scene_row["aoi"])
    community_arr = _load_community_array(sensor, aoi)
    catalog = _community_catalog(sensor, aoi)
    mask_meta = PIXEL_MASKS[sensor]

    with rasterio.open(Path(scene_row["filepath"])) as src:
        data = np.asarray(src.read(1, masked=True).filled(np.nan), dtype=np.float32)
        pixel_count = src.height * src.width

    finite_mask = np.isfinite(data)
    community_mask = community_arr > 0
    combined_mask = finite_mask & community_mask
    if not combined_mask.any():
        return []

    values = data[combined_mask]
    community_codes = community_arr[combined_mask].astype(np.int32, copy=False)
    present_codes = np.unique(community_codes)

    rows: list[dict] = []
    timestamp = pd.Timestamp(scene_row["date"])
    for community_code in present_codes:
        code = int(community_code)
        metadata = catalog.get(code)
        if metadata is None:
            continue

        code_mask = community_codes == code
        valid_pixels = int(code_mask.sum())
        if valid_pixels < MIN_PIXELS:
            continue

        pixels = values[code_mask]
        valid_pixel_fraction = float(valid_pixels) / float(pixel_count)
        ecozone_code = metadata.get("ecozone_code")
        ecozone_label = metadata.get("ecozone_label")
        if ecozone_code is not None and not ecozone_label:
            ecozone_label = ECOZONE_LABELS.get(int(ecozone_code))
        forest_community_display_code = metadata.get("forest_community_display_code") or str(code)
        forest_community_source_dataset = metadata.get("forest_community_source_dataset")
        forest_community_source_value = metadata.get("forest_community_source_value")
        forest_community_source_key = metadata.get("forest_community_source_key")
        ecozone_group_code = metadata.get("ecozone_group_code")
        ecozone_group_label = metadata.get("ecozone_group_label")
        ecozone_group_raw = metadata.get("ecozone_group_raw")

        for percentile in SPATIAL_PERCENTILES:
            rows.append(
                {
                    "analysis_scope": "forest_community",
                    "sensor": sensor,
                    "aoi": aoi,
                    "index": str(scene_row["index"]),
                    "ecozone_code": int(ecozone_code) if ecozone_code is not None else pd.NA,
                    "ecozone_label": ecozone_label if ecozone_label else pd.NA,
                    "forest_community_code": code,
                    "forest_community_display_code": forest_community_display_code,
                    "forest_community_label": metadata["forest_community_label"],
                    "forest_community_source_dataset": forest_community_source_dataset if forest_community_source_dataset else pd.NA,
                    "forest_community_source_value": int(forest_community_source_value) if forest_community_source_value is not None else pd.NA,
                    "forest_community_source_key": forest_community_source_key if forest_community_source_key else pd.NA,
                    "ecozone_group_code": int(ecozone_group_code) if ecozone_group_code is not None else pd.NA,
                    "ecozone_group_label": ecozone_group_label if ecozone_group_label else pd.NA,
                    "ecozone_group_raw": ecozone_group_raw if ecozone_group_raw else pd.NA,
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


def build_scene_summary_forest_community(scene_catalog: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    total_scenes = len(scene_catalog)
    for scene_idx, row in enumerate(scene_catalog.itertuples(index=False), start=1):
        if scene_idx == 1 or scene_idx % 25 == 0 or scene_idx == total_scenes:
            print(f"  Forest-community scene summaries: {scene_idx}/{total_scenes}", flush=True)
        row_series = pd.Series(row._asdict())
        records.extend(summarize_scene_forest_community(row_series))

    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records).sort_values(
        [
            "sensor",
            "aoi",
            "index",
            "forest_community_code",
            "date",
            "spatial_percentile",
        ]
    ).reset_index(drop=True)


def summarize_scene_ecozone_group(scene_row: pd.Series) -> list[dict]:
    sensor = str(scene_row["sensor"])
    aoi = str(scene_row["aoi"])
    group_arr = _ecozone_group_array(sensor, aoi)
    catalog = _ecozone_group_catalog(sensor, aoi)
    mask_meta = PIXEL_MASKS[sensor]

    with rasterio.open(Path(scene_row["filepath"])) as src:
        data = np.asarray(src.read(1, masked=True).filled(np.nan), dtype=np.float32)
        pixel_count = src.height * src.width

    finite_mask = np.isfinite(data)
    group_mask = group_arr > 0
    combined_mask = finite_mask & group_mask
    if not combined_mask.any():
        return []

    values = data[combined_mask]
    group_codes = group_arr[combined_mask].astype(np.int16, copy=False)
    present_codes = np.unique(group_codes)

    rows: list[dict] = []
    timestamp = pd.Timestamp(scene_row["date"])
    for group_code in present_codes:
        code = int(group_code)
        metadata = catalog.get(code)
        if metadata is None:
            continue

        code_mask = group_codes == code
        valid_pixels = int(code_mask.sum())
        if valid_pixels < MIN_PIXELS:
            continue

        pixels = values[code_mask]
        valid_pixel_fraction = float(valid_pixels) / float(pixel_count)
        for percentile in SPATIAL_PERCENTILES:
            rows.append(
                {
                    "analysis_scope": "forest_ecozone_group",
                    "sensor": sensor,
                    "aoi": aoi,
                    "index": str(scene_row["index"]),
                    "ecozone_code": pd.NA,
                    "ecozone_label": pd.NA,
                    "forest_community_code": pd.NA,
                    "forest_community_display_code": pd.NA,
                    "forest_community_label": pd.NA,
                    "forest_community_source_dataset": pd.NA,
                    "forest_community_source_value": pd.NA,
                    "forest_community_source_key": pd.NA,
                    "ecozone_group_code": code,
                    "ecozone_group_label": metadata["ecozone_group_label"],
                    "ecozone_group_raw": metadata.get("ecozone_group_raw") or pd.NA,
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


def build_scene_summary_ecozone_group(scene_catalog: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    total_scenes = len(scene_catalog)
    for scene_idx, row in enumerate(scene_catalog.itertuples(index=False), start=1):
        if scene_idx == 1 or scene_idx % 25 == 0 or scene_idx == total_scenes:
            print(f"  Forest community-group scene summaries: {scene_idx}/{total_scenes}", flush=True)
        row_series = pd.Series(row._asdict())
        records.extend(summarize_scene_ecozone_group(row_series))

    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records).sort_values(
        [
            "sensor",
            "aoi",
            "index",
            "ecozone_group_code",
            "date",
            "spatial_percentile",
        ]
    ).reset_index(drop=True)


def iter_scene_summary_ecozone_group_chunks(
    scene_catalog: pd.DataFrame,
    *,
    chunk_size: int = 500,
) -> Iterator[pd.DataFrame]:
    total_scenes = len(scene_catalog)
    if total_scenes == 0:
        return
    chunk_size = max(1, int(chunk_size))
    records: list[dict] = []
    for scene_idx, row in enumerate(scene_catalog.itertuples(index=False), start=1):
        if scene_idx == 1 or scene_idx % 25 == 0 or scene_idx == total_scenes:
            print(f"  Forest community-group scene summaries: {scene_idx}/{total_scenes}", flush=True)
        row_series = pd.Series(row._asdict())
        records.extend(summarize_scene_ecozone_group(row_series))
        if scene_idx % chunk_size == 0 and records:
            yield pd.DataFrame.from_records(records).sort_values(
                [
                    "sensor",
                    "aoi",
                    "index",
                    "ecozone_group_code",
                    "date",
                    "spatial_percentile",
                ]
            ).reset_index(drop=True)
            records = []
    if records:
        yield pd.DataFrame.from_records(records).sort_values(
            [
                "sensor",
                "aoi",
                "index",
                "ecozone_group_code",
                "date",
                "spatial_percentile",
            ]
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


def _source_id_summary(group: pd.DataFrame, include_scene_id_list: bool) -> str:
    source_ids = group["source_file_or_composite_id"].astype(str)
    unique_count = int(source_ids.nunique())
    if include_scene_id_list:
        return "|".join(source_ids.tolist())
    if unique_count == 1:
        return source_ids.iloc[0]
    return f"{unique_count} scenes"


def iter_temporal_summary_forest_community_chunks(
    scene_summary: pd.DataFrame,
    *,
    include_scene_id_list: bool = False,
    analysis_scope: str = "forest_community",
) -> Iterator[pd.DataFrame]:
    total_start = time.perf_counter()
    quantile_values = [percentile / 100.0 for percentile in TEMPORAL_PERCENTILES]
    quantile_label_map = {
        percentile / 100.0: TEMPORAL_PERCENTILE_LABELS[percentile]
        for percentile in TEMPORAL_PERCENTILES
    }
    for temporal_agg in TEMPORAL_AGGS:
        agg_start = time.perf_counter()
        print(f"  Interval={temporal_agg}: deriving time bins...", flush=True)
        expanded = _time_bin_columns(scene_summary, temporal_agg)
        for season_filter in ["all", "growing"]:
            season_frame = expanded if season_filter == "all" else expanded[expanded["growing_season_day"].notna()]
            if season_frame.empty:
                print(f"    season={season_filter}: no rows, skipping", flush=True)
                continue
            print(f"    season={season_filter}: source_rows={len(season_frame)}", flush=True)
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
                    "forest_community_code",
                    "forest_community_display_code",
                    "forest_community_label",
                    "forest_community_source_dataset",
                    "forest_community_source_value",
                    "forest_community_source_key",
                    "ecozone_group_code",
                    "ecozone_group_label",
                    "ecozone_group_raw",
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
                base = grouped.agg(
                    cloud_percent=("cloud_percent", "max"),
                    n_pixels=("n_pixels", "median"),
                    valid_pixel_fraction=("valid_pixel_fraction", "median"),
                    n_scenes=("source_file_or_composite_id", "nunique"),
                    first_source_file_or_composite_id=("source_file_or_composite_id", "first"),
                ).reset_index()
                if include_scene_id_list:
                    source_summary = grouped["source_file_or_composite_id"].agg(
                        lambda values: "|".join(values.astype(str).tolist())
                    ).rename("source_file_or_composite_id").reset_index()
                    base = base.drop(columns=["first_source_file_or_composite_id"]).merge(
                        source_summary,
                        on=group_columns,
                        how="left",
                    )
                else:
                    base["source_file_or_composite_id"] = np.where(
                        base["n_scenes"].eq(1),
                        base["first_source_file_or_composite_id"].astype(str),
                        base["n_scenes"].astype(str) + " scenes",
                    )
                    base = base.drop(columns=["first_source_file_or_composite_id"])

                quantiles = grouped["value"].quantile(quantile_values).rename("value").reset_index()
                quantile_column = [column for column in quantiles.columns if column not in group_columns + ["value"]][0]
                quantiles["temporal_percentile"] = quantiles[quantile_column].map(quantile_label_map)
                quantiles = quantiles.drop(columns=[quantile_column])
                chunk = quantiles.merge(base, on=group_columns, how="left")
                chunk["analysis_scope"] = analysis_scope
                chunk["date"] = chunk["time_bin_start"]
                chunk["doy"] = pd.to_datetime(chunk["time_bin_start"]).dt.dayofyear
                chunk["growing_season_day"] = (
                    pd.to_datetime(chunk["time_bin_start"]).map(_growing_season_day)
                    if season_filter == "growing"
                    else pd.NA
                )
                chunk["season_filter"] = season_filter
                chunk["temporal_agg"] = temporal_agg
                chunk["cloud_threshold"] = cloud_threshold
                chunk["time_bin_start"] = chunk["time_bin_start"]
                chunk["time_bin_end"] = chunk["time_bin_end"]
                chunk["n_pixels"] = chunk["n_pixels"].round().astype("int64")
                chunk["n_scenes"] = chunk["n_scenes"].astype("int64")
                if chunk["forest_community_code"].notna().any():
                    chunk["forest_community_code"] = chunk["forest_community_code"].astype("int64")
                chunk = chunk[
                    [
                        "analysis_scope",
                        "sensor",
                        "aoi",
                        "index",
                        "ecozone_code",
                        "ecozone_label",
                        "forest_community_code",
                        "forest_community_display_code",
                        "forest_community_label",
                        "forest_community_source_dataset",
                        "forest_community_source_value",
                        "forest_community_source_key",
                        "ecozone_group_code",
                        "ecozone_group_label",
                        "ecozone_group_raw",
                        "date",
                        "year",
                        "doy",
                        "growing_season_day",
                        "season_filter",
                        "temporal_agg",
                        "temporal_percentile",
                        "spatial_percentile",
                        "cloud_threshold",
                        "cloud_percent",
                        "pixel_mask_id",
                        "pixel_mask_description",
                        "pixel_mask_version",
                        "n_pixels",
                        "valid_pixel_fraction",
                        "n_scenes",
                        "value",
                        "source_file_or_composite_id",
                        "time_bin_label",
                        "time_bin_start",
                        "time_bin_end",
                    ]
                ]
                print(
                    f"      cloud<={cloud_threshold}: completed in {(time.perf_counter() - threshold_start)/60:.1f}m "
                    f"rows={len(chunk)}",
                    flush=True,
                )
                yield chunk
        print(f"  Interval={temporal_agg}: completed in {(time.perf_counter() - agg_start)/60:.1f}m", flush=True)
    print(f"  Forest-community temporal summary chunks elapsed={(time.perf_counter() - total_start)/60:.1f}m", flush=True)


def iter_temporal_summary_ecozone_group_chunks(
    scene_summary: pd.DataFrame,
    *,
    include_scene_id_list: bool = False,
) -> Iterator[pd.DataFrame]:
    yield from iter_temporal_summary_forest_community_chunks(
        scene_summary,
        include_scene_id_list=include_scene_id_list,
        analysis_scope="forest_ecozone_group",
    )


def build_temporal_summary_forest_community(
    scene_summary: pd.DataFrame,
    *,
    include_scene_id_list: bool = False,
) -> pd.DataFrame:
    frame_chunks = list(
        iter_temporal_summary_forest_community_chunks(
            scene_summary,
            include_scene_id_list=include_scene_id_list,
        )
    )
    if not frame_chunks:
        return pd.DataFrame()
    frame = pd.concat(frame_chunks, ignore_index=True)
    for column in ("date", "time_bin_start", "time_bin_end"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce").dt.tz_localize(None)
    print(
        f"  Forest-community temporal summary rows={len(frame)}",
        flush=True,
    )
    return frame.sort_values(
        [
            "sensor",
            "aoi",
            "index",
            "forest_community_code",
            "temporal_agg",
            "time_bin_start",
            "spatial_percentile",
            "temporal_percentile",
            "cloud_threshold",
        ]
    ).reset_index(drop=True)


def build_temporal_summary_ecozone_group(
    scene_summary: pd.DataFrame,
    *,
    include_scene_id_list: bool = False,
) -> pd.DataFrame:
    frame_chunks = list(
        iter_temporal_summary_ecozone_group_chunks(
            scene_summary,
            include_scene_id_list=include_scene_id_list,
        )
    )
    if not frame_chunks:
        return pd.DataFrame()
    frame = pd.concat(frame_chunks, ignore_index=True)
    for column in ("date", "time_bin_start", "time_bin_end"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce").dt.tz_localize(None)
    print(
        f"  Forest community-group temporal summary rows={len(frame)}",
        flush=True,
    )
    return frame.sort_values(
        [
            "sensor",
            "aoi",
            "index",
            "ecozone_group_code",
            "temporal_agg",
            "time_bin_start",
            "spatial_percentile",
            "temporal_percentile",
            "cloud_threshold",
        ]
    ).reset_index(drop=True)


def build_forest_community_data_dictionary_markdown(output_dir: Path) -> str:
    return f"""# Dashboard Forest Community Data Dictionary

Generated table products in `{output_dir}`:

- `{FOREST_COMMUNITY_SCENE_STEM}.parquet`
- `{FOREST_COMMUNITY_TEMPORAL_STEM}.parquet`
- `{FOREST_ECOZONE_GROUP_SCENE_STEM}.parquet`
- `{FOREST_ECOZONE_GROUP_TEMPORAL_STEM}.parquet`

Optional CSVs may also be present when requested from the builder.

## Dataset definitions

- `scene_summary_forest_community`: one row per scene x AOI x sensor x index x forest community x spatial percentile.
- `temporal_summary_forest_community`: one row per forest community x temporal bin x cloud threshold x spatial percentile x temporal percentile.
- `scene_summary_forest_ecozone_group`: one row per scene x AOI x sensor x index x forest community group x spatial percentile.
- `temporal_summary_forest_ecozone_group`: one row per forest community group x temporal bin x cloud threshold x spatial percentile x temporal percentile.

## Added forest-community columns

| Column | Meaning |
|---|---|
| `forest_community_code` | Integer forest-community class code from the snapped categorical raster |
| `forest_community_display_code` | Human-readable AOI-local category code, preserving special source notes such as north `16a` |
| `forest_community_label` | Human-readable forest-community label from inventory metadata, or a fallback label |
| `forest_community_source_dataset` | TNC source raster/table dataset name (`AppRidges`, `NBlueRidge`, `Simon`) |
| `forest_community_source_value` | Original source `VALUE` from the TNC source |
| `forest_community_source_key` | Stable AOI/source/value key used to preserve provenance |
| `ecozone_group_code` | Legacy column name for TNC forest community group code (`0`-`9`) |
| `ecozone_group_label` | Legacy column name for TNC forest community group label |
| `ecozone_group_raw` | Raw TNC forest community group string |
| `ecozone_code` | Reserved for the broader terrain/thermal ecozone lineage; not populated by the forest-community TNC group |
| `ecozone_label` | Reserved for the broader terrain/thermal ecozone lineage; not populated by the forest-community TNC group |

## Notes

- Forest-community summaries use an AOI-aligned categorical raster named `forest_community.tif`, not live raster reads in Streamlit.
- The builder looks first for `forest_community.tif` in each AOI forest-type trait directory and falls back to the existing configured species/forest-type raster.
- Landsat summaries reproject the same categorical raster to the Landsat canonical grid with nearest-neighbor resampling.
- Rows are omitted where a community contributes fewer than `{MIN_PIXELS}` valid pixels for a scene.
- Forest community-group rows are recomputed from the combined pixel population for every included forest community in that group. They are not averages of forest-community summary rows.
- Temporal tables summarize provenance as a scene count by default to avoid very large repeated scene-id strings across the 24-community output.
"""


def default_forest_community_output_dir(project_root: Path) -> Path:
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
