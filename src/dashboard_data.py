from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib.util
from pathlib import Path

import pandas as pd

from src.dashboard_schema import CANONICAL_COLUMNS, COLUMN_ALIASES, DEFAULT_VALUE_ORDER
from src.labels import normalize_label

ECOZONE_LABELS = {
    1: "Cool",
    2: "Intermediate",
    3: "Hot",
}
ECOZONE_SCENE_STEM = "scene_summary_ecozone"
ECOZONE_TEMPORAL_STEM = "temporal_summary_ecozone"
FOREST_COMMUNITY_SCENE_STEM = "scene_summary_forest_community"
FOREST_COMMUNITY_TEMPORAL_STEM = "temporal_summary_forest_community"
FOREST_COMMUNITY_GROUP_SCENE_STEM = "scene_summary_forest_ecozone_group"
FOREST_COMMUNITY_GROUP_TEMPORAL_STEM = "temporal_summary_forest_ecozone_group"
BASE_SCENE_STEM = "scene_summary"
BASE_TEMPORAL_STEM = "temporal_summary"
OPTIMIZED_PARQUET_DIRNAME = "optimized_parquet"
PARTITIONED_PARQUET_DIRNAME = "partitioned_parquet"
DASHBOARD_READ_COLUMNS = [
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
    "pixel_mask_version",
    "n_pixels",
    "valid_pixel_fraction",
    "n_scenes",
    "value",
    "time_bin_label",
    "time_bin_start",
    "time_bin_end",
    "month_day_label",
]


def _read_parquet_or_none(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except ImportError:
        return None


def _has_parquet_engine() -> bool:
    return importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None


def _has_pyarrow_dataset() -> bool:
    try:
        return importlib.util.find_spec("pyarrow.dataset") is not None
    except ModuleNotFoundError:
        return False


def _is_blank_filter_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value == "":
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _parquet_filter_value(column: str, value):
    if column in {"label", "exclude_below_stddev", "exclude_above_stddev", "analysis_scope", "season_filter"}:
        return None
    if _is_blank_filter_value(value):
        return None
    if value == "all":
        return None
    if column == "cloud_threshold":
        return None
    if column in {"ecozone_code", "forest_community_code", "forest_community_source_value", "ecozone_group_code", "year", "doy"}:
        return int(value)
    return value


PARTITION_COLUMNS = ["sensor", "aoi", "index", "ecozone_code", "forest_community_code", "ecozone_group_code", "temporal_agg"]
HIVE_NULL_PARTITION = "__HIVE_DEFAULT_PARTITION__"
NORTH_NBLUERIDGE_16A_LABEL = "Dry-mesic Oak - NBlueRidge 16a"


def _hive_partition_value(column: str, raw_value) -> str | None:
    value = _parquet_filter_value(column, raw_value)
    if value is None:
        return None
    return str(value)


def _apply_forest_community_display_overrides(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    mask = (
        frame["aoi"].eq("north")
        & frame["forest_community_code"].eq(116)
        & frame["forest_community_display_code"].astype(str).eq("16a")
        & frame["forest_community_source_dataset"].astype(str).eq("NBlueRidge")
        & frame["forest_community_source_value"].eq(16)
    )
    if mask.any():
        frame.loc[mask, "forest_community_label"] = NORTH_NBLUERIDGE_16A_LABEL
    return frame


def _partition_dataset_paths(path: Path, filters: dict) -> list[Path]:
    candidates = [path]
    for column in PARTITION_COLUMNS:
        value = _hive_partition_value(column, filters.get(column))
        next_candidates: list[Path] = []
        for candidate in candidates:
            if value is None:
                next_candidates.extend(child for child in candidate.glob(f"{column}=*") if child.is_dir())
            else:
                child = candidate / f"{column}={value}"
                if child.exists():
                    next_candidates.append(child)
                elif column == "ecozone_code":
                    null_child = candidate / f"{column}={HIVE_NULL_PARTITION}"
                    if null_child.exists():
                        next_candidates.append(null_child)
        if next_candidates:
            candidates = next_candidates
        elif value is not None:
            return []
    return candidates


def _partition_values_from_path(root: Path, path: Path) -> dict[str, object]:
    values: dict[str, object] = {}
    for part in path.relative_to(root).parts:
        if "=" not in part:
            continue
        column, raw_value = part.split("=", 1)
        if column not in PARTITION_COLUMNS:
            continue
        if raw_value == HIVE_NULL_PARTITION:
            values[column] = None
        elif column in {"ecozone_code", "forest_community_code", "ecozone_group_code"}:
            values[column] = int(raw_value)
        else:
            values[column] = raw_value
    return values


def _dataset_schema_for_path(path: Path, pa, pq):
    sample_file = next(path.rglob("*.parquet"), None)
    if sample_file is None:
        return None
    fields = []
    for field in pq.read_schema(sample_file):
        if field.name == "growing_season_day" and pa.types.is_null(field.type):
            fields.append(pa.field(field.name, pa.float64()))
        else:
            fields.append(field)
    return pa.schema(fields)


def _read_parquet_filtered_or_none(path: Path, filters: dict, columns: list[str] | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except ImportError:
        return None

    if path.is_dir():
        dataset_paths = _partition_dataset_paths(path, filters)
        if not dataset_paths:
            return pd.DataFrame()
        frames = []
        for dataset_path in dataset_paths:
            dataset_schema = _dataset_schema_for_path(dataset_path, pa, pq)
            if dataset_schema is None:
                continue
            dataset = ds.dataset(dataset_path, format="parquet", schema=dataset_schema)
            schema_names = set(dataset.schema.names)
            expression = None
            for column, raw_value in filters.items():
                if column not in schema_names:
                    continue
                value = _parquet_filter_value(column, raw_value)
                if value is None:
                    continue
                column_expression = ds.field(column) == value
                expression = column_expression if expression is None else expression & column_expression

            read_columns = None
            if columns is not None:
                read_columns = [column for column in columns if column in schema_names]
            frame = dataset.to_table(columns=read_columns, filter=expression).to_pandas()
            for column, value in _partition_values_from_path(path, dataset_path).items():
                if column not in frame.columns:
                    frame[column] = value
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
    else:
        dataset = ds.dataset(path, format="parquet")
    schema_names = set(dataset.schema.names)
    expression = None
    for column, raw_value in filters.items():
        if column not in schema_names:
            continue
        value = _parquet_filter_value(column, raw_value)
        if value is None:
            continue
        column_expression = ds.field(column) == value
        expression = column_expression if expression is None else expression & column_expression

    read_columns = None
    if columns is not None:
        read_columns = [column for column in columns if column in schema_names]
    table = dataset.to_table(columns=read_columns, filter=expression)
    frame = table.to_pandas()
    if path.is_dir():
        for column in PARTITION_COLUMNS:
            if column not in frame.columns and column in filters:
                frame[column] = _parquet_filter_value(column, filters.get(column))
    return frame


@dataclass
class DashboardDataBundle:
    scene_summary: pd.DataFrame
    temporal_summary: pd.DataFrame
    scene_summary_manifest: pd.DataFrame
    temporal_summary_manifest: pd.DataFrame
    scene_summary_ecozone_manifest: pd.DataFrame
    temporal_summary_ecozone_manifest: pd.DataFrame
    scene_summary_forest_community_manifest: pd.DataFrame
    temporal_summary_forest_community_manifest: pd.DataFrame
    scene_summary_forest_community_group_manifest: pd.DataFrame
    temporal_summary_forest_community_group_manifest: pd.DataFrame
    data_dir: Path

    def _has_segment_tables(self, scene_stem: str) -> bool:
        return any(
            path.exists()
            for path in (
                self.data_dir / PARTITIONED_PARQUET_DIRNAME / scene_stem,
                self.data_dir / OPTIMIZED_PARQUET_DIRNAME / f"{scene_stem}.parquet",
                self.data_dir / f"{scene_stem}.parquet",
                self.data_dir / f"{scene_stem}.csv",
            )
        )

    def has_ecozone_tables(self) -> bool:
        return self._has_segment_tables(ECOZONE_SCENE_STEM)

    def has_forest_community_tables(self) -> bool:
        return self._has_segment_tables(FOREST_COMMUNITY_SCENE_STEM)

    def has_forest_community_group_tables(self) -> bool:
        return self._has_segment_tables(FOREST_COMMUNITY_GROUP_SCENE_STEM)

    def frame_for_temporal_agg(self, temporal_agg: str) -> pd.DataFrame:
        if temporal_agg == "scene":
            return self.scene_summary.copy()
        return self.temporal_summary[self.temporal_summary["temporal_agg"] == temporal_agg].copy()

    def manifest_for_temporal_agg(self, temporal_agg: str) -> pd.DataFrame:
        if temporal_agg == "scene":
            return self.scene_summary_manifest.copy()
        return self.temporal_summary_manifest[self.temporal_summary_manifest["temporal_agg"] == temporal_agg].copy()

    def frame_for_config(self, config) -> pd.DataFrame:
        analysis_scope = getattr(config, "analysis_scope", "overall")
        if analysis_scope == "ecozone":
            stem = ECOZONE_SCENE_STEM if config.temporal_agg == "scene" else ECOZONE_TEMPORAL_STEM
            source = _best_segment_summary_source(self.data_dir, stem)
            return load_summary_filtered(source, stem, filters_for_config(config))
        if analysis_scope == "forest_community":
            stem = FOREST_COMMUNITY_SCENE_STEM if config.temporal_agg == "scene" else FOREST_COMMUNITY_TEMPORAL_STEM
            source = _best_segment_summary_source(self.data_dir, stem)
            return load_summary_filtered(source, stem, filters_for_config(config))

        manifest = self.manifest_for_temporal_agg(config.temporal_agg)
        if not manifest.empty:
            series_match = manifest[
                (manifest["sensor"] == config.sensor)
                & (manifest["aoi"] == config.aoi)
                & (manifest["index"] == config.index)
                & (manifest["spatial_percentile"] == config.spatial_percentile)
                & (manifest["cloud_threshold"] == config.cloud_threshold)
            ].copy()
            if config.temporal_agg == "scene":
                series_match = series_match[series_match["temporal_percentile"] == "none"]
            else:
                series_match = series_match[series_match["temporal_percentile"] == config.temporal_percentile]
            if not series_match.empty:
                manifest_row = series_match.iloc[0]
                dataset_name = BASE_SCENE_STEM if config.temporal_agg == "scene" else BASE_TEMPORAL_STEM
                return load_summary_csv(_resolve_series_path(self.data_dir, str(manifest_row["path"])), dataset_name)
        source = _best_base_summary_source(self.data_dir, config.temporal_agg)
        if source.exists() or source.with_suffix(".parquet").exists() or source.with_suffix(".csv").exists():
            stem = BASE_SCENE_STEM if config.temporal_agg == "scene" else BASE_TEMPORAL_STEM
            return load_summary_filtered(source, stem, filters_for_config(config))
        return self.frame_for_temporal_agg(config.temporal_agg)

    def frame_for_forest_community_group(self, config, group_code: int | None = None) -> pd.DataFrame:
        stem = FOREST_COMMUNITY_GROUP_SCENE_STEM if config.temporal_agg == "scene" else FOREST_COMMUNITY_GROUP_TEMPORAL_STEM
        source = _best_segment_summary_source(self.data_dir, stem)
        filters = filters_for_config(
            config,
            {
                "analysis_scope": "forest_ecozone_group",
                "forest_community_code": None,
                "ecozone_group_code": group_code,
            },
        )
        return load_summary_filtered(source, stem, filters)

    def available_values(self, column: str) -> list:
        if column == "analysis_scope":
            values = ["overall"]
            if self.has_ecozone_tables():
                values.append("ecozone")
            if self.has_forest_community_tables():
                values.append("forest_community")
            return values
        if column == "ecozone_code":
            if self.has_ecozone_tables():
                return list(ECOZONE_LABELS)
        if column == "ecozone_label":
            if self.has_ecozone_tables():
                return list(ECOZONE_LABELS.values())

        values = []
        manifest_frames = [
            self.scene_summary_manifest,
            self.temporal_summary_manifest,
            self.scene_summary_ecozone_manifest,
            self.temporal_summary_ecozone_manifest,
            self.scene_summary_forest_community_manifest,
            self.temporal_summary_forest_community_manifest,
            self.scene_summary_forest_community_group_manifest,
            self.temporal_summary_forest_community_group_manifest,
        ]
        data_frames = [self.scene_summary, self.temporal_summary]
        source_frames = manifest_frames if any(not frame.empty for frame in manifest_frames) else data_frames
        for frame in source_frames:
            if column in frame.columns:
                values.extend(frame[column].dropna().tolist())
        preferred = DEFAULT_VALUE_ORDER.get(column)
        unique_values = list(dict.fromkeys(values))
        if preferred:
            for value in preferred:
                if value not in unique_values:
                    unique_values.append(value)
        if not unique_values:
            return []
        if preferred:
            unique_values = sorted(
                unique_values,
                key=lambda value: (
                    preferred.index(value) if value in preferred else len(preferred),
                    str(value),
                ),
            )
        elif column.endswith("_code"):
            unique_values = sorted(unique_values, key=lambda value: float(value))
        else:
            unique_values = sorted(unique_values, key=lambda value: str(value))
        return unique_values

    def label_for_code(self, code_column: str, label_column: str, code, fallback: str) -> str:
        if _is_blank_filter_value(code):
            return fallback
        try:
            normalized_code = int(code)
        except (TypeError, ValueError):
            normalized_code = code
        source_frames = [
            self.scene_summary_manifest,
            self.temporal_summary_manifest,
            self.scene_summary_ecozone_manifest,
            self.temporal_summary_ecozone_manifest,
            self.scene_summary_forest_community_manifest,
            self.temporal_summary_forest_community_manifest,
            self.scene_summary_forest_community_group_manifest,
            self.temporal_summary_forest_community_group_manifest,
            self.scene_summary,
            self.temporal_summary,
        ]
        for frame in source_frames:
            if code_column not in frame.columns or label_column not in frame.columns:
                continue
            codes = pd.to_numeric(frame[code_column], errors="coerce")
            matches = frame[codes == normalized_code]
            labels = matches[label_column].dropna().astype(str)
            labels = labels[labels != ""]
            if not labels.empty:
                return str(normalize_label(labels.iloc[0]))
        return fallback

    def available_year_range(self) -> tuple[int, int]:
        years: list[int] = []
        for frame in (
            self.scene_summary_manifest,
            self.temporal_summary_manifest,
            self.scene_summary_ecozone_manifest,
            self.temporal_summary_ecozone_manifest,
            self.scene_summary_forest_community_manifest,
            self.temporal_summary_forest_community_manifest,
            self.scene_summary_forest_community_group_manifest,
            self.temporal_summary_forest_community_group_manifest,
        ):
            for column in ("min_year", "max_year"):
                if column in frame.columns:
                    years.extend(frame[column].dropna().astype(int).tolist())
        for frame in (self.scene_summary, self.temporal_summary):
            if "year" in frame.columns:
                years.extend(frame["year"].dropna().astype(int).tolist())
        if not years:
            return (1984, pd.Timestamp.utcnow().year)
        return (min(years), max(years))


def filters_for_config(config, extra_filters: dict | None = None) -> dict:
    analysis_scope = getattr(config, "analysis_scope", "overall")
    filters = {
        "analysis_scope": analysis_scope,
        "label": None,
        "sensor": config.sensor,
        "aoi": config.aoi,
        "index": config.index,
        "ecozone_code": getattr(config, "ecozone_code", None) if analysis_scope == "ecozone" else None,
        "forest_community_code": (
            getattr(config, "forest_community_code", None) if analysis_scope == "forest_community" else None
        ),
        "spatial_percentile": config.spatial_percentile,
        "temporal_agg": config.temporal_agg,
        "temporal_percentile": None if config.temporal_agg == "scene" else config.temporal_percentile,
        "cloud_threshold": config.cloud_threshold,
        "season_filter": config.season_filter,
        "exclude_below_stddev": config.exclude_below_stddev,
        "exclude_above_stddev": config.exclude_above_stddev,
    }
    if extra_filters:
        filters.update(extra_filters)
    return filters


def _normalize_value_strings(series: pd.Series) -> pd.Series:
    normalized = series.where(series.isna(), series.astype(str).str.strip().str.lower())
    return normalized.replace({"<na>": pd.NA, "nan": pd.NA})


def _first_matching_column(columns: list[str], aliases: list[str]) -> str | None:
    for alias in aliases:
        if alias in columns:
            return alias
    return None


def normalize_summary_frame(frame: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    normalized = pd.DataFrame()
    for canonical_name in CANONICAL_COLUMNS:
        source = _first_matching_column(list(frame.columns), COLUMN_ALIASES.get(canonical_name, []))
        normalized[canonical_name] = frame[source] if source else pd.NA

    normalized["dataset_name"] = dataset_name

    for column in (
        "analysis_scope",
        "sensor",
        "aoi",
        "index",
        "season_filter",
        "temporal_agg",
        "temporal_percentile",
        "spatial_percentile",
        "pixel_mask_id",
        "pixel_mask_description",
        "pixel_mask_version",
    ):
        if column in normalized.columns:
            normalized[column] = _normalize_value_strings(normalized[column])

    normalized["sensor"] = normalized["sensor"].replace(
        {
            "sentinel": "s2",
            "sentinel-2": "s2",
            "sentinel2": "s2",
            "landsat": "ls",
            "landsat-8": "ls",
            "landsat-9": "ls",
            "landsat-7": "ls",
            "landsat-5": "ls",
        }
    )
    normalized["aoi"] = normalized["aoi"].replace(
        {
            "gwnf": "north",
            "gw national forest": "north",
            "great smoky mtns": "south",
            "great smoky mountains": "south",
        }
    )

    for date_column in ("date", "time_bin_start", "time_bin_end"):
        normalized[date_column] = pd.to_datetime(normalized[date_column], errors="coerce")

    numeric_columns = [
        "ecozone_code",
        "forest_community_code",
        "forest_community_source_value",
        "ecozone_group_code",
        "year",
        "doy",
        "growing_season_day",
        "cloud_threshold",
        "cloud_percent",
        "n_pixels",
        "valid_pixel_fraction",
        "n_scenes",
        "value",
    ]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if "forest_community" in dataset_name:
        default_scope = "forest_community"
    elif "ecozone" in dataset_name:
        default_scope = "ecozone"
    else:
        default_scope = "overall"
    normalized["analysis_scope"] = normalized["analysis_scope"].fillna(default_scope)
    if normalized["analysis_scope"].eq("overall").all() and normalized["ecozone_code"].notna().any():
        normalized["analysis_scope"] = "ecozone"
    if normalized["analysis_scope"].eq("overall").all() and normalized["forest_community_code"].notna().any():
        normalized["analysis_scope"] = "forest_community"
    if normalized["ecozone_label"].isna().any():
        normalized["ecozone_label"] = normalized["ecozone_label"].fillna(
            normalized["ecozone_code"].map(ECOZONE_LABELS)
        )
    normalized["ecozone_label"] = normalized["ecozone_label"].fillna("overall")
    normalized["forest_community_label"] = normalized["forest_community_label"].fillna(
        normalized["forest_community_code"].map(
            lambda value: f"Forest community {int(value)}" if pd.notna(value) else pd.NA
        )
    )
    normalized["forest_community_label"] = normalized["forest_community_label"].fillna("overall")
    normalized["forest_community_label"] = normalized["forest_community_label"].map(normalize_label)
    normalized["forest_community_display_code"] = normalized["forest_community_display_code"].fillna(
        normalized["forest_community_code"].map(lambda value: str(int(value)) if pd.notna(value) else pd.NA)
    )
    normalized["forest_community_source_value"] = normalized["forest_community_source_value"].fillna(
        normalized["forest_community_code"]
    )
    normalized["forest_community_source_key"] = normalized["forest_community_source_key"].fillna(
        normalized.apply(
            lambda row: (
                f"{row['aoi']}:{row['forest_community_source_dataset']}:{int(row['forest_community_source_value'])}"
                if pd.notna(row["forest_community_source_dataset"]) and pd.notna(row["forest_community_source_value"])
                else pd.NA
            ),
            axis=1,
        )
    )
    normalized = _apply_forest_community_display_overrides(normalized)
    normalized["ecozone_group_raw"] = normalized["ecozone_group_raw"].fillna(normalized["ecozone_group_label"])
    normalized["ecozone_group_label"] = normalized["ecozone_group_label"].fillna(normalized["ecozone_group_raw"])

    if normalized["year"].isna().all() and normalized["date"].notna().any():
        normalized["year"] = normalized["date"].dt.year
    if normalized["doy"].isna().all() and normalized["date"].notna().any():
        normalized["doy"] = normalized["date"].dt.dayofyear
    if normalized["time_bin_start"].isna().all() and normalized["date"].notna().any():
        normalized["time_bin_start"] = normalized["date"]
    if normalized["time_bin_label"].isna().all() and normalized["date"].notna().any():
        normalized["time_bin_label"] = normalized["date"].dt.strftime("%Y-%m-%d")
    if normalized["growing_season_day"].isna().all() and normalized["date"].notna().any():
        season_start = pd.to_datetime(
            normalized["year"].astype("Int64").astype(str) + "-05-15",
            errors="coerce",
        )
        normalized["growing_season_day"] = (normalized["date"] - season_start).dt.days + 1

    normalized["season_filter"] = normalized["season_filter"].fillna("all")
    normalized["temporal_agg"] = normalized["temporal_agg"].fillna(
        "scene" if dataset_name.startswith(BASE_SCENE_STEM) else "month"
    )
    normalized["temporal_percentile"] = normalized["temporal_percentile"].fillna("p95")
    normalized["spatial_percentile"] = normalized["spatial_percentile"].fillna("p95")
    normalized["pixel_mask_id"] = normalized["pixel_mask_id"].fillna("unknown_mask")
    normalized["pixel_mask_description"] = normalized["pixel_mask_description"].fillna("unknown mask")
    normalized["pixel_mask_version"] = normalized["pixel_mask_version"].fillna("unknown")

    return normalized


def load_summary_csv(path: Path, dataset_name: str) -> pd.DataFrame:
    return _load_summary_csv_cached(str(path), dataset_name).copy()


@lru_cache(maxsize=512)
def _load_summary_csv_cached(path_str: str, dataset_name: str) -> pd.DataFrame:
    path = Path(path_str)
    if path.is_dir():
        parquet_path = path
        csv_path = path.with_suffix(".csv")
    elif path.suffix in {".csv", ".parquet"}:
        parquet_path = path if path.suffix == ".parquet" else path.with_suffix(".parquet")
        csv_path = path if path.suffix == ".csv" else path.with_suffix(".csv")
    else:
        parquet_path = path.with_suffix(".parquet")
        csv_path = path
    parquet_frame = _read_parquet_or_none(parquet_path)
    if parquet_frame is not None:
        return normalize_summary_frame(parquet_frame, dataset_name)
    if csv_path.exists():
        return normalize_summary_frame(pd.read_csv(csv_path), dataset_name)
    return normalize_summary_frame(pd.DataFrame(columns=CANONICAL_COLUMNS), dataset_name)


def _filter_value_for_key(column: str, value):
    if _is_blank_filter_value(value):
        return None
    if value == "all" and column != "season_filter":
        return None
    return value


def _filters_to_key(filters: dict) -> tuple[tuple[str, object], ...]:
    items = []
    for key, value in sorted(filters.items()):
        normalized_value = _filter_value_for_key(key, value)
        if normalized_value is not None:
            items.append((key, normalized_value))
    return tuple(items)


def _filters_from_key(filters_key: tuple[tuple[str, object], ...]) -> dict:
    return dict(filters_key)


def load_summary_filtered(path: Path, dataset_name: str, filters: dict) -> pd.DataFrame:
    return _load_summary_filtered_cached(str(path), dataset_name, _filters_to_key(filters)).copy()


@lru_cache(maxsize=32)
def _load_summary_filtered_cached(
    path_str: str,
    dataset_name: str,
    filters_key: tuple[tuple[str, object], ...],
) -> pd.DataFrame:
    path = Path(path_str)
    if path.is_dir():
        parquet_path = path
        csv_path = path.with_suffix(".csv")
    elif path.suffix in {".csv", ".parquet"}:
        parquet_path = path if path.suffix == ".parquet" else path.with_suffix(".parquet")
        csv_path = path if path.suffix == ".csv" else path.with_suffix(".csv")
    else:
        parquet_path = path.with_suffix(".parquet")
        csv_path = path

    filters = _filters_from_key(filters_key)
    parquet_frame = _read_parquet_filtered_or_none(parquet_path, filters, DASHBOARD_READ_COLUMNS)
    if parquet_frame is not None:
        frame = normalize_summary_frame(parquet_frame, dataset_name)
        return filter_frame(frame, filters=filters).reset_index(drop=True)

    if not csv_path.exists():
        return normalize_summary_frame(pd.DataFrame(columns=CANONICAL_COLUMNS), dataset_name)

    matches = []
    for chunk in pd.read_csv(csv_path, chunksize=250_000):
        normalized = normalize_summary_frame(chunk, dataset_name)
        filtered = filter_frame(normalized, filters=filters)
        if not filtered.empty:
            matches.append(filtered)
    if not matches:
        return normalize_summary_frame(pd.DataFrame(columns=CANONICAL_COLUMNS), dataset_name)
    return pd.concat(matches, ignore_index=True)


def load_manifest(path: Path) -> pd.DataFrame:
    return _load_manifest_cached(str(path)).copy()


@lru_cache(maxsize=32)
def _load_manifest_cached(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    parquet_path = path.with_suffix(".parquet")
    parquet_frame = _read_parquet_or_none(parquet_path)
    if parquet_frame is not None:
        if "forest_community_label" in parquet_frame.columns:
            parquet_frame["forest_community_label"] = parquet_frame["forest_community_label"].map(normalize_label)
        return parquet_frame
    if path.exists():
        frame = pd.read_csv(path)
        if "forest_community_label" in frame.columns:
            frame["forest_community_label"] = frame["forest_community_label"].map(normalize_label)
        return frame
    return pd.DataFrame(
        columns=[
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
            "temporal_agg",
            "temporal_percentile",
            "spatial_percentile",
            "cloud_threshold",
            "row_count",
            "min_year",
            "max_year",
            "path",
        ]
    )


def _resolve_series_path(data_dir: Path, relative_path: str) -> Path:
    candidate = data_dir / relative_path
    if candidate.exists() or candidate.with_suffix(".parquet").exists() or candidate.with_suffix(".csv").exists():
        return candidate
    store_candidate = data_dir / "series_store" / relative_path
    return store_candidate


def _best_segment_summary_source(data_dir: Path, stem: str) -> Path:
    partitioned = data_dir / PARTITIONED_PARQUET_DIRNAME / stem
    if partitioned.exists() and _has_pyarrow_dataset():
        return partitioned
    optimized = data_dir / OPTIMIZED_PARQUET_DIRNAME / f"{stem}.parquet"
    if optimized.exists() and _has_parquet_engine():
        return optimized
    parquet = data_dir / f"{stem}.parquet"
    if parquet.exists():
        return parquet
    return data_dir / f"{stem}.csv"


def _best_base_summary_source(data_dir: Path, temporal_agg: str) -> Path:
    stem = BASE_SCENE_STEM if temporal_agg == "scene" else BASE_TEMPORAL_STEM
    optimized = data_dir / OPTIMIZED_PARQUET_DIRNAME / f"{stem}.parquet"
    if optimized.exists() and _has_parquet_engine():
        return optimized
    parquet = data_dir / f"{stem}.parquet"
    if parquet.exists() and _has_parquet_engine():
        return parquet
    return data_dir / f"{stem}.csv"


def _empty_summary(dataset_name: str) -> pd.DataFrame:
    return normalize_summary_frame(pd.DataFrame(columns=CANONICAL_COLUMNS), dataset_name)


def load_dashboard_data(data_dir: str | Path) -> DashboardDataBundle:
    root = Path(data_dir).resolve()
    scene_summary_manifest = load_manifest(root / "scene_summary_manifest.csv")
    temporal_summary_manifest = load_manifest(root / "temporal_summary_manifest.csv")
    scene_summary_ecozone_manifest = load_manifest(root / f"{ECOZONE_SCENE_STEM}_manifest.csv")
    temporal_summary_ecozone_manifest = load_manifest(root / f"{ECOZONE_TEMPORAL_STEM}_manifest.csv")
    scene_summary_forest_community_manifest = load_manifest(root / f"{FOREST_COMMUNITY_SCENE_STEM}_manifest.csv")
    temporal_summary_forest_community_manifest = load_manifest(root / f"{FOREST_COMMUNITY_TEMPORAL_STEM}_manifest.csv")
    scene_summary_forest_community_group_manifest = load_manifest(root / f"{FOREST_COMMUNITY_GROUP_SCENE_STEM}_manifest.csv")
    temporal_summary_forest_community_group_manifest = load_manifest(root / f"{FOREST_COMMUNITY_GROUP_TEMPORAL_STEM}_manifest.csv")
    use_series_store = not scene_summary_manifest.empty or not temporal_summary_manifest.empty
    lazy_base_tables = _has_parquet_engine() and (
        (root / f"{BASE_SCENE_STEM}.parquet").exists()
        or (root / f"{BASE_TEMPORAL_STEM}.parquet").exists()
        or (root / OPTIMIZED_PARQUET_DIRNAME / f"{BASE_SCENE_STEM}.parquet").exists()
        or (root / OPTIMIZED_PARQUET_DIRNAME / f"{BASE_TEMPORAL_STEM}.parquet").exists()
    )
    return DashboardDataBundle(
        scene_summary=(
            _empty_summary("scene_summary")
            if use_series_store or lazy_base_tables
            else load_summary_csv(root / "scene_summary.csv", "scene_summary")
        ),
        temporal_summary=(
            _empty_summary("temporal_summary")
            if use_series_store or lazy_base_tables
            else load_summary_csv(root / "temporal_summary.csv", "temporal_summary")
        ),
        scene_summary_manifest=scene_summary_manifest,
        temporal_summary_manifest=temporal_summary_manifest,
        scene_summary_ecozone_manifest=scene_summary_ecozone_manifest,
        temporal_summary_ecozone_manifest=temporal_summary_ecozone_manifest,
        scene_summary_forest_community_manifest=scene_summary_forest_community_manifest,
        temporal_summary_forest_community_manifest=temporal_summary_forest_community_manifest,
        scene_summary_forest_community_group_manifest=scene_summary_forest_community_group_manifest,
        temporal_summary_forest_community_group_manifest=temporal_summary_forest_community_group_manifest,
        data_dir=root,
    )


def filter_frame(frame: pd.DataFrame, filters: dict, year_range: tuple[int, int] | None = None) -> pd.DataFrame:
    filtered = frame.copy()
    for column, value in filters.items():
        if _is_blank_filter_value(value):
            continue
        if value == "all" and column != "season_filter":
            continue
        if column == "season_filter" and value in {"growing", "stack_growing"} and "growing_season_day" in filtered.columns:
            if column in filtered.columns and filtered[column].eq("growing").any():
                filtered = filtered[filtered[column] == "growing"]
            else:
                filtered = filtered[filtered["growing_season_day"].notna()]
            continue
        if column == "season_filter" and value == "all":
            if column in filtered.columns and filtered[column].eq("all").any():
                filtered = filtered[filtered[column] == "all"]
            continue
        if column == "cloud_threshold" and "cloud_percent" in filtered.columns and (
            column not in filtered.columns or filtered[column].isna().all()
        ):
            filtered = filtered[filtered["cloud_percent"].isna() | (filtered["cloud_percent"] <= value)]
            continue
        if column not in filtered.columns:
            continue
        filtered = filtered[filtered[column] == value]
    if year_range and "year" in filtered.columns:
        start_year, end_year = year_range
        filtered = filtered[filtered["year"].between(start_year, end_year, inclusive="both")]
    return filtered
