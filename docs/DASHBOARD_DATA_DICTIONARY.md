# Dashboard CSV Data Dictionary

The Streamlit dashboard expects a directory containing these public-facing CSVs:

- `scene_summary.csv`
- `temporal_summary.csv`
- `growing_season_summary.csv`

The app normalizes common aliases such as `Scene Date`, `AOI Key`, `Platform`, `Valid Pixels`, and `Path/Row` into a standard internal schema.

## Shared canonical columns

| Column | Type | Meaning |
|---|---|---|
| `sensor` | string | `s2` or `ls` |
| `aoi` | string | `north` or `south` |
| `index` | string | `ndvi`, `ndmi`, or `evi` |
| `date` | date | Scene date or representative date for the time bin |
| `year` | integer | Calendar year |
| `doy` | integer | Day of year |
| `season_filter` | string | Usually `all` or `growing` |
| `temporal_agg` | string | `scene`, `half_month`, or `month` |
| `temporal_percentile` | string | Temporal percentile such as `p75`, `p95`, `p98`, `p99`, `p100` |
| `spatial_percentile` | string | Spatial percentile such as `p75`, `p95`, `p98`, `p99`, `p100` |
| `cloud_threshold` | integer | Scene exclusion threshold, typically `30`, `40`, or `50` |
| `pixel_class_set` | string | Mask/class selection, for example `s2_scl_veg_clear` or `ls_clear_land` |
| `n_pixels` | integer | Number of valid pixels contributing to the summary |
| `valid_pixel_fraction` | float | Fraction of AOI pixels retained after masking |
| `value` | float | The plotted summary statistic |
| `source_file_or_composite_id` | string | Scene ID, path/row, or composite identifier |

## `scene_summary.csv`

One row per scene x AOI x index x mask/configuration.

Recommended extra columns:

- `source_file_or_composite_id`
- `valid_pixel_fraction`
- `n_pixels`

## `temporal_summary.csv`

One row per AOI x sensor x index x time bin x aggregation setup.

Recommended extra columns:

- `time_bin_label`
- `time_bin_start`
- `time_bin_end`

`date` should point at the bin’s representative date. If `time_bin_start` is present, the dashboard uses that on the x-axis for binned series.

## `growing_season_summary.csv`

One row per AOI x sensor x index x year x date or bin inside the May 15 through September 15 growing window.

Recommended extra columns:

- `growing_day`
- `time_bin_label`

If `growing_day` is missing, the dashboard derives it from `date` relative to May 15 of each year.

## Sample fixtures

Runnable example CSVs live in:

- `Results/tables/dashboard_samples/`

They are intentionally small and are meant for app testing, UI development, and export-script validation.
