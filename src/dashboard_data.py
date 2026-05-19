from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.dashboard_schema import CANONICAL_COLUMNS, COLUMN_ALIASES, DEFAULT_VALUE_ORDER


@dataclass
class DashboardDataBundle:
    scene_summary: pd.DataFrame
    temporal_summary: pd.DataFrame
    data_dir: Path

    def frame_for_temporal_agg(self, temporal_agg: str) -> pd.DataFrame:
        if temporal_agg == "scene":
            return self.scene_summary.copy()
        return self.temporal_summary[self.temporal_summary["temporal_agg"] == temporal_agg].copy()

    def available_values(self, column: str) -> list:
        values = []
        for frame in (self.scene_summary, self.temporal_summary):
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
        else:
            unique_values = sorted(unique_values, key=lambda value: str(value))
        return unique_values

    def available_year_range(self) -> tuple[int, int]:
        years = []
        for frame in (self.scene_summary, self.temporal_summary):
            if "year" in frame.columns:
                years.extend(frame["year"].dropna().astype(int).tolist())
        if not years:
            return (1984, pd.Timestamp.utcnow().year)
        return (min(years), max(years))


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
    normalized["temporal_agg"] = normalized["temporal_agg"].fillna("scene" if dataset_name == "scene_summary" else "month")
    normalized["temporal_percentile"] = normalized["temporal_percentile"].fillna("p95")
    normalized["spatial_percentile"] = normalized["spatial_percentile"].fillna("p95")
    normalized["pixel_mask_id"] = normalized["pixel_mask_id"].fillna("unknown_mask")
    normalized["pixel_mask_description"] = normalized["pixel_mask_description"].fillna("unknown mask")
    normalized["pixel_mask_version"] = normalized["pixel_mask_version"].fillna("unknown")

    return normalized


def load_summary_csv(path: Path, dataset_name: str) -> pd.DataFrame:
    parquet_path = path.with_suffix(".parquet")
    if parquet_path.exists():
        return normalize_summary_frame(pd.read_parquet(parquet_path), dataset_name)
    if path.exists():
        return normalize_summary_frame(pd.read_csv(path), dataset_name)
    return normalize_summary_frame(pd.DataFrame(columns=CANONICAL_COLUMNS), dataset_name)


def load_dashboard_data(data_dir: str | Path) -> DashboardDataBundle:
    root = Path(data_dir).resolve()
    return DashboardDataBundle(
        scene_summary=load_summary_csv(root / "scene_summary.csv", "scene_summary"),
        temporal_summary=load_summary_csv(root / "temporal_summary.csv", "temporal_summary"),
        data_dir=root,
    )


def filter_frame(frame: pd.DataFrame, filters: dict, year_range: tuple[int, int] | None = None) -> pd.DataFrame:
    filtered = frame.copy()
    for column, value in filters.items():
        if value in (None, "", "all"):
            continue
        if column == "season_filter" and value == "growing" and "growing_season_day" in filtered.columns:
            filtered = filtered[filtered["growing_season_day"].notna()]
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
