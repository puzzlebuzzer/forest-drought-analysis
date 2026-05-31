from __future__ import annotations

from dataclasses import dataclass

CANONICAL_COLUMNS = [
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
    "month_day_label",
]

COLUMN_ALIASES = {
    "analysis_scope": ["analysis_scope", "Analysis Scope"],
    "sensor": ["sensor", "Sensor", "platform_group", "Platform Group", "Platform"],
    "aoi": ["aoi", "AOI Key", "AOI", "study_area"],
    "index": ["index", "Index", "veg_index"],
    "ecozone_code": ["ecozone_code", "Ecozone Code", "ecozone_id"],
    "ecozone_label": ["ecozone_label", "Ecozone", "ecozone"],
    "forest_community_code": ["forest_community_code", "Forest Community Code", "community_code", "forest_type_code"],
    "forest_community_display_code": ["forest_community_display_code", "Forest Community Display Code"],
    "forest_community_label": [
        "forest_community_label",
        "Forest Community",
        "community_label",
        "forest_type_label",
        "forest_type",
    ],
    "forest_community_source_dataset": ["forest_community_source_dataset", "Forest Community Source Dataset"],
    "forest_community_source_value": ["forest_community_source_value", "Forest Community Source Value"],
    "forest_community_source_key": ["forest_community_source_key", "Forest Community Source Key"],
    "ecozone_group_code": ["ecozone_group_code", "Ecozone Group Code"],
    "ecozone_group_label": ["ecozone_group_label", "Ecozone Group Label"],
    "ecozone_group_raw": ["ecozone_group_raw", "Ecozone Group"],
    "date": ["date", "Date", "Scene Date", "scene_date", "time_bin_start", "bin_start_date"],
    "year": ["year", "Year"],
    "doy": ["doy", "DOY", "day_of_year"],
    "growing_season_day": ["growing_season_day", "Growing Day", "growing_day", "growing_season_relative_day"],
    "season_filter": ["season_filter", "Season Filter", "season"],
    "temporal_agg": ["temporal_agg", "Temporal Agg", "temporal_aggregation"],
    "temporal_percentile": ["temporal_percentile", "Temporal Percentile", "temporal_pct"],
    "spatial_percentile": ["spatial_percentile", "Spatial Percentile", "spatial_pct", "percentile"],
    "cloud_threshold": ["cloud_threshold", "Cloud Threshold", "cloud_cover_threshold"],
    "cloud_percent": ["cloud_percent", "Cloud Percent", "cloud_cover", "cloud_percentage"],
    "pixel_mask_id": ["pixel_mask_id", "Pixel Mask Id", "mask_id"],
    "pixel_mask_description": ["pixel_mask_description", "Pixel Mask Description", "mask_description"],
    "pixel_mask_version": ["pixel_mask_version", "Pixel Mask Version", "mask_version"],
    "n_pixels": ["n_pixels", "N Pixels", "Valid Pixels", "valid_pixels"],
    "valid_pixel_fraction": ["valid_pixel_fraction", "Valid Pixel Fraction", "valid_frac"],
    "n_scenes": ["n_scenes", "N Scenes", "scene_count"],
    "value": ["value", "Value", "p95", "p75", "p98", "p99", "p100 (max)", "max", "p50"],
    "source_file_or_composite_id": [
        "source_file_or_composite_id",
        "Source File Or Composite Id",
        "Path/Row",
        "source_id",
        "Scene ID",
    ],
    "time_bin_label": ["time_bin_label", "Time Bin Label", "Month Name", "month_name", "bin_label"],
    "time_bin_start": ["time_bin_start", "Time Bin Start", "bin_start_date"],
    "time_bin_end": ["time_bin_end", "Time Bin End", "bin_end_date"],
    "month_day_label": ["month_day_label", "Month Day Label"],
}

DEFAULT_VALUE_ORDER = {
    "analysis_scope": ["overall", "ecozone", "forest_community"],
    "sensor": ["ls", "s2"],
    "aoi": ["north", "south"],
    "index": ["ndvi", "ndmi", "evi"],
    "ecozone_code": [1, 2, 3],
    "ecozone_label": ["Cool", "Intermediate", "Hot"],
    "spatial_percentile": ["p50", "p75", "p95", "p98", "p99", "p100"],
    "temporal_agg": ["month", "half_month", "scene"],
    "temporal_percentile": ["none", "p50", "p75", "p95", "p98", "p99", "p100"],
    "cloud_threshold": [30, 40, 50],
    "season_filter": ["growing", "growing_overlay", "all"],
}


@dataclass(frozen=True)
class ComparisonConfig:
    label: str
    sensor: str
    aoi: str
    index: str
    spatial_percentile: str
    temporal_agg: str
    temporal_percentile: str
    cloud_threshold: int | None
    season_filter: str
    exclude_below_stddev: float | None = None
    exclude_above_stddev: float | None = None
    analysis_scope: str = "overall"
    ecozone_code: int | None = None
    forest_community_code: int | None = None
