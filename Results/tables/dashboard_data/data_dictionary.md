# Dashboard Data Dictionary

Generated table products in `/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Results/tables/dashboard_data`:

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
