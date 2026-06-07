# Index-Based Moisture Year Classification

This workflow builds wet / neutral / dry year labels from vegetation-index behavior only.

It does not read PRISM, USDM, SPEI, drought.gov, or any other external climate dataset.

Run:

```bash
python Analysis/DashboardPipeline/Climate/classify_index_based_moisture_years.py
```

## Input Data

The default input is the first available dashboard-ready summary table found under `SummaryTables/dashboard_data/`, currently checked in this order:

1. `temporal_summary`
2. `growing_season_summary`
3. `scene_summary`

For the current dashboard tables, that usually means:

```text
SummaryTables/dashboard_data/temporal_summary.parquet
```

If that table is unavailable, the script can be pointed at another dashboard-ready CSV/parquet table with `--input-table`.

Default filters:

- `season_filter = growing`
- `temporal_agg = half_month`
- `temporal_percentile = p50`
- `spatial_percentile = p50`
- `cloud_threshold = 30`

These defaults use the median vegetation-index signal for each AOI/sensor/index growing-season time bin.

The script classifies each sensor separately. Landsat and Sentinel-2 are not merged into one time series.

## Growing Season

The team-defined growing season is May 15 through September 15.

When the source table has `season_filter = growing`, the script uses that flag. If a future source table lacks that flag but has dates, the script filters dates from May 15 through September 15.

## Annual Aggregation

For each AOI x sensor x index x year, the default annual value is:

```text
annual_index_value = median of growing-season bin values
```

The script also supports `--annual-aggregation p75` and `--annual-aggregation p95` for sensitivity checks.

## Classification Method

Classification is relative within each AOI x sensor x index time series:

- `wet/canopy-moist`: top 20% of years
- `dry/canopy-stressed`: bottom 20% of years
- `neutral`: middle 60%

The output also includes anomaly, z-score, percentile, and rank fields.

These labels are based on relative vegetation-index behavior, not precipitation. Higher NDMI years are labeled `wet/canopy-moist`; lower NDMI years are labeled `dry/canopy-stressed`. NDVI and EVI use the same rank logic but should be interpreted as greenness/productivity response checks.

## Why NDMI Is Primary

NDMI is treated as the primary internal moisture-response classifier because it is more directly tied to canopy water content than NDVI or EVI.

NDVI and EVI are exported as supporting classifications. They should be interpreted as greenness/productivity response checks rather than direct moisture metrics.

## Outputs

Primary output:

```text
Results/tables/index_based_moisture_year_classification.csv
```

Additional outputs:

```text
Results/tables/index_based_moisture_year_classification.parquet
Results/tables/index_based_moisture_annual_summary.csv
Results/tables/index_based_moisture_year_classification.metadata.json
Results/figures/index_based_moisture_year_classification.png
```

If Plotly static PNG export is unavailable, the script writes an HTML fallback and creates a simpler PNG with Pillow.

The metadata JSON records the actual input table and command arguments used for a run.

## Limitations

- This is not an independent meteorological drought classification.
- It reflects observed canopy/index response.
- It may capture phenology, disturbance, sensor differences, cloud/mask artifacts, or vegetation condition, not only moisture.
- Landsat and Sentinel-2 are classified separately and should not be blindly merged unless harmonization is already handled.
- NDVI and EVI can saturate or respond to canopy structure differently from NDMI.
- A high vegetation-index year can reflect favorable moisture, phenology, disturbance recovery, sensor availability, or masking conditions; it is not automatically a meteorological wet year.
- Use this classification alongside PRISM-derived moisture years for comparison, not as a replacement for climate context.
