# Percentile Spread Investigation

## What This Produces

This analysis summarizes monthly distribution spread for each:

- AOI
- ecozone
- index (`NDVI`, `NDMI`, `EVI`)
- month (`Apr-Oct`)
- year-group (`wet`, `neutral`, `dry`)

It computes these spread metrics:

- `p99 - p50`
- `p75 - p25`

## How To Interpret The Metrics

- `p99 - p50`:
  upper-tail spread, showing how far the highest-value portion of the ecozone
  distribution sits above the central tendency

- `p75 - p25`:
  interquartile spread, showing how broad the middle of the ecozone
  distribution is

These are descriptive heterogeneity metrics, not significance tests.

## Outputs

- `percentile_spread_ndvi.png`
- `percentile_spread_ndmi.png`
- `percentile_spread_evi.png`
- `percentile_spread_monthly_percentiles.csv`
- `percentile_spread_metrics.csv`
- `percentile_spread_group_summary.csv`

## How To Run

From the `Python` project root:

```bash
python Analysis/Traits/Ecozone/ecozone_percentile_spread_investigation.py
```

## Input Assumptions

- Reads Sentinel monthly composites from `Results/0-CacheBaseData/monthly_max`
- Reads year-group labels from `config/wet_dry_years.csv`
- Reuses the existing AOI ecozone rasters
- Descriptive only; no inferential statistics are added
