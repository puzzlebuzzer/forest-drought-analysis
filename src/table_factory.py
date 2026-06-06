from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.aoi import get_aoi_config, valid_aois
from src.landsat import get_landsat_index_root

SENSORS = ["s2", "ls"]
INDICES = ["NDVI", "NDMI", "EVI"]
SPATIAL_PERCENTILES = [50, 75, 95, 98, 99, 100]
TEMPORAL_PERCENTILES = [50, 75, 95, 98, 99, 100]
CLOUD_THRESHOLDS = [30, 40, 50]
TEMPORAL_AGGS = ["scene", "half_month", "month"]
GROWING_START = (5, 15)
GROWING_END = (9, 15)

PIXEL_MASKS = {
    "s2": {
        "pixel_mask_id": "s2_scl4_veg_v1",
        "pixel_mask_description": "Sentinel-2 SCL=4 vegetation-only baseline mask baked into aligned index rasters.",
        "pixel_mask_version": "v1",
    },
    "ls": {
        "pixel_mask_id": "ls_clear_terrestrial_v1",
        "pixel_mask_description": (
            "Landsat QA_PIXEL clear terrestrial baseline mask excluding fill, dilated cloud, cirrus, cloud, "
            "cloud shadow, snow, and water."
        ),
        "pixel_mask_version": "v1",
    },
}

PERCENTILE_COLUMNS = {percentile: f"p{percentile}" for percentile in SPATIAL_PERCENTILES}
TEMPORAL_PERCENTILE_LABELS = {percentile: f"p{percentile}" for percentile in TEMPORAL_PERCENTILES}


@dataclass(frozen=True)
class SceneRecord:
    sensor: str
    aoi: str
    index: str
    source_file_or_composite_id: str
    date: pd.Timestamp
    year: int
    doy: int
    cloud_percent: float | None
    filepath: Path
    provenance_fields: dict[str, object]


def canonical_dashboard_tables_dir(project_root: Path) -> Path:
    return project_root / "SummaryTables" / "dashboard_data"


def _manifest_records(sensor: str, aoi: str, index_name: str) -> list[SceneRecord]:
    if sensor == "s2":
        cfg = get_aoi_config(aoi)
        index_dir = cfg.index_cache_root / index_name
    else:
        index_dir = get_landsat_index_root(aoi) / index_name

    manifest_path = index_dir / "cache_manifest.json"
    if not manifest_path.exists():
        return []

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    records: list[SceneRecord] = []
    for source_id, meta in manifest.items():
        filepath = index_dir / meta["filename"]
        if not filepath.exists():
            continue
        timestamp = pd.Timestamp(datetime.fromisoformat(meta["date"])).tz_localize(None)
        records.append(
            SceneRecord(
                sensor=sensor,
                aoi=aoi,
                index=index_name.lower(),
                source_file_or_composite_id=source_id,
                date=timestamp,
                year=timestamp.year,
                doy=timestamp.dayofyear,
                cloud_percent=float(meta["cloud_cover"]) if meta.get("cloud_cover") is not None else None,
                filepath=filepath,
                provenance_fields=meta,
            )
        )
    return sorted(records, key=lambda record: record.date)


def build_scene_catalog() -> pd.DataFrame:
    records: list[dict] = []
    for sensor in SENSORS:
        for aoi in valid_aois():
            for index_name in INDICES:
                for record in _manifest_records(sensor, aoi, index_name):
                    row = {
                        "sensor": record.sensor,
                        "aoi": record.aoi,
                        "index": record.index,
                        "source_file_or_composite_id": record.source_file_or_composite_id,
                        "date": record.date,
                        "year": record.year,
                        "doy": record.doy,
                        "cloud_percent": record.cloud_percent,
                        "filepath": str(record.filepath),
                    }
                    row.update(record.provenance_fields)
                    records.append(row)
    return pd.DataFrame.from_records(records).sort_values(["sensor", "aoi", "index", "date"]).reset_index(drop=True)


def _season_filter(timestamp: pd.Timestamp) -> str:
    current = (timestamp.month, timestamp.day)
    return "growing" if GROWING_START <= current <= GROWING_END else "all"


def _growing_season_day(timestamp: pd.Timestamp) -> int | None:
    season_start = date(timestamp.year, GROWING_START[0], GROWING_START[1])
    season_end = date(timestamp.year, GROWING_END[0], GROWING_END[1])
    current = timestamp.date()
    if current < season_start or current > season_end:
        return None
    return (current - season_start).days + 1


def summarize_scene(record: SceneRecord) -> list[dict]:
    import rasterio

    with rasterio.open(record.filepath) as src:
        data = src.read(1, masked=True)
        pixel_count = src.height * src.width

    finite = np.asarray(data.compressed(), dtype=np.float32)
    if finite.size == 0:
        return []

    valid_pixel_fraction = float(finite.size) / float(pixel_count)
    growing_day = _growing_season_day(record.date)
    mask_meta = PIXEL_MASKS[record.sensor]

    rows: list[dict] = []
    for percentile in SPATIAL_PERCENTILES:
        rows.append(
            {
                "sensor": record.sensor,
                "aoi": record.aoi,
                "index": record.index,
                "date": record.date.normalize().tz_localize(None) if record.date.tzinfo is not None else record.date.normalize(),
                "year": record.year,
                "doy": record.doy,
                "growing_season_day": growing_day,
                "season_filter": "all",
                "temporal_agg": "scene",
                "temporal_percentile": "none",
                "spatial_percentile": PERCENTILE_COLUMNS[percentile],
                "cloud_threshold": pd.NA,
                "cloud_percent": record.cloud_percent,
                "pixel_mask_id": mask_meta["pixel_mask_id"],
                "pixel_mask_description": mask_meta["pixel_mask_description"],
                "pixel_mask_version": mask_meta["pixel_mask_version"],
                "n_pixels": int(finite.size),
                "valid_pixel_fraction": valid_pixel_fraction,
                "n_scenes": 1,
                "value": float(np.percentile(finite, percentile)),
                "source_file_or_composite_id": record.source_file_or_composite_id,
            }
        )
    return rows


def build_scene_summary(scene_catalog: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    total_scenes = len(scene_catalog)
    start_time = time.perf_counter()
    group_counts = scene_catalog.groupby(["sensor", "aoi", "index"]).size().to_dict()
    group_seen: dict[tuple[str, str, str], int] = {}

    for scene_idx, row in enumerate(scene_catalog.itertuples(index=False), start=1):
        group_key = (row.sensor, row.aoi, row.index)
        group_seen[group_key] = group_seen.get(group_key, 0) + 1
        if scene_idx == 1 or scene_idx % 25 == 0 or scene_idx == total_scenes:
            elapsed = time.perf_counter() - start_time
            per_scene = elapsed / scene_idx if scene_idx else 0.0
            remaining = max(total_scenes - scene_idx, 0)
            eta_seconds = per_scene * remaining
            print(
                "  "
                f"[{scene_idx}/{total_scenes}] "
                f"{row.sensor}/{row.aoi}/{row.index} "
                f"(group {group_seen[group_key]}/{group_counts[group_key]}) "
                f"elapsed={elapsed/60:.1f}m "
                f"eta={eta_seconds/60:.1f}m",
                flush=True,
            )

        scene_record = SceneRecord(
            sensor=row.sensor,
            aoi=row.aoi,
            index=row.index.upper(),
            source_file_or_composite_id=row.source_file_or_composite_id,
            date=pd.Timestamp(row.date),
            year=int(row.year),
            doy=int(row.doy),
            cloud_percent=float(row.cloud_percent) if pd.notna(row.cloud_percent) else None,
            filepath=Path(row.filepath),
            provenance_fields={},
        )
        for summary_row in summarize_scene(scene_record):
            summary_row["index"] = summary_row["index"].lower()
            records.append(summary_row)
    return pd.DataFrame.from_records(records).sort_values(
        ["sensor", "aoi", "index", "date", "spatial_percentile"]
    ).reset_index(drop=True)


def _half_month_bucket(timestamp: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    if timestamp.day <= 15:
        start = pd.Timestamp(year=timestamp.year, month=timestamp.month, day=1)
        end = pd.Timestamp(year=timestamp.year, month=timestamp.month, day=15)
        label = f"{timestamp.year}-{timestamp.month:02d}a"
    else:
        start = pd.Timestamp(year=timestamp.year, month=timestamp.month, day=16)
        end = start + pd.offsets.MonthEnd(0)
        label = f"{timestamp.year}-{timestamp.month:02d}b"
    return start, end.normalize(), label


def _month_bucket(timestamp: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    start = pd.Timestamp(year=timestamp.year, month=timestamp.month, day=1)
    end = start + pd.offsets.MonthEnd(0)
    label = f"{timestamp.year}-{timestamp.month:02d}"
    return start, end.normalize(), label


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


def build_temporal_summary(scene_summary: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for temporal_agg in TEMPORAL_AGGS:
        expanded = _time_bin_columns(scene_summary, temporal_agg)
        for season_filter in ["all", "growing"]:
            season_frame = expanded if season_filter == "all" else expanded[expanded["growing_season_day"].notna()]
            if season_frame.empty:
                continue
            for cloud_threshold in CLOUD_THRESHOLDS:
                threshold_frame = season_frame[
                    season_frame["cloud_percent"].isna() | (season_frame["cloud_percent"] <= cloud_threshold)
                ]
                if threshold_frame.empty:
                    continue
                group_columns = [
                    "sensor",
                    "aoi",
                    "index",
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
                for group_key, group in grouped:
                    for percentile in TEMPORAL_PERCENTILES:
                        values = group["value"].to_numpy(dtype=float)
                        records.append(
                            {
                                "sensor": group_key[0],
                                "aoi": group_key[1],
                                "index": group_key[2],
                                "date": group_key[4],
                                "year": group_key[7],
                                "doy": pd.Timestamp(group_key[4]).dayofyear,
                                "growing_season_day": (
                                    _growing_season_day(pd.Timestamp(group_key[4])) if season_filter == "growing" else pd.NA
                                ),
                                "season_filter": season_filter,
                                "temporal_agg": temporal_agg,
                                "temporal_percentile": TEMPORAL_PERCENTILE_LABELS[percentile],
                                "spatial_percentile": group_key[3],
                                "cloud_threshold": cloud_threshold,
                                "cloud_percent": float(group["cloud_percent"].max()) if group["cloud_percent"].notna().any() else pd.NA,
                                "pixel_mask_id": group_key[8],
                                "pixel_mask_description": group_key[9],
                                "pixel_mask_version": group_key[10],
                                "n_pixels": int(group["n_pixels"].median()),
                                "valid_pixel_fraction": float(group["valid_pixel_fraction"].median()),
                                "n_scenes": int(group["source_file_or_composite_id"].nunique()),
                                "value": float(np.percentile(values, percentile)),
                                "source_file_or_composite_id": "|".join(group["source_file_or_composite_id"].astype(str).tolist()),
                                "time_bin_label": group_key[6],
                                "time_bin_start": group_key[4],
                                "time_bin_end": group_key[5],
                            }
                        )
    return pd.DataFrame.from_records(records).sort_values(
        ["sensor", "aoi", "index", "temporal_agg", "time_bin_start", "spatial_percentile", "temporal_percentile", "cloud_threshold"]
    ).reset_index(drop=True)


def build_data_dictionary_markdown(output_dir: Path) -> str:
    return f"""# Dashboard Data Dictionary

Generated table products in `{output_dir}`:

- `scene_summary.csv` / `scene_summary.parquet`
- `temporal_summary.csv` / `temporal_summary.parquet`
- `scene_catalog.csv` / `scene_catalog.parquet`

## Dataset definitions

- `scene_summary`: one row per scene x AOI x sensor x index x spatial percentile.
- `temporal_summary`: one row per temporal bin x cloud threshold x spatial percentile x temporal percentile.

## Shared columns

| Column | Meaning |
|---|---|
| `sensor` | `s2` or `ls` |
| `aoi` | `north` or `south` |
| `index` | `ndvi`, `ndmi`, or `evi` |
| `date` | scene date or representative temporal-bin date |
| `year` | calendar year |
| `doy` | day of year |
| `growing_season_day` | May 15 = 1 through September 15 = 124 |
| `season_filter` | `all` or `growing` |
| `temporal_agg` | `scene`, `half_month`, or `month` |
| `temporal_percentile` | temporal percentile label, or `none` for scene rows |
| `spatial_percentile` | spatial percentile label |
| `cloud_threshold` | applied maximum scene cloud percentage; null for raw scene summary |
| `cloud_percent` | original scene-level cloud metadata from the manifest where available |
| `pixel_mask_id` | canonical fixed mask identifier |
| `pixel_mask_description` | human-readable mask definition |
| `pixel_mask_version` | version tag for mask semantics |
| `n_pixels` | valid pixel count contributing to the row |
| `valid_pixel_fraction` | valid pixels divided by raster grid pixels |
| `n_scenes` | number of scenes aggregated into the row |
| `value` | plotted metric |
| `source_file_or_composite_id` | provenance ID or concatenated scene IDs |

## Growing season view

The dashboard's growing-season explorer is derived directly from `scene_summary` by:

- filtering dates to May 15 through September 15
- using `growing_season_day` as the normalized x-axis
- applying cloud-threshold filters at query time

## Canonical masks

- Sentinel-2: `s2_scl4_veg_v1`
- Landsat: `ls_clear_terrestrial_v1`

These masks are fixed dataset definitions and are not intended to be exposed as dashboard toggles.
"""


def write_table_with_optional_parquet(frame: pd.DataFrame, output_path: Path) -> tuple[Path, Path | None]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    parquet_path = output_path.with_suffix(".parquet")
    try:
        frame.to_parquet(parquet_path, index=False)
    except Exception:
        parquet_path = None
    return output_path, parquet_path
