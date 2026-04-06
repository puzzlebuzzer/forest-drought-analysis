# Distribution Shape Investigation

## What This Produces

This analysis summarizes the monthly Sentinel value distribution shape for each:

- AOI
- ecozone
- index (`NDVI`, `NDMI`, `EVI`)
- month (`Apr-Oct`)
- year-group (`wet`, `neutral`, `dry`)

It computes and compares these percentiles:

- `p25`
- `p50`
- `p75`
- `p90`
- `p99`

## Outputs

- `distribution_shape_ndvi.png`
- `distribution_shape_ndmi.png`
- `distribution_shape_evi.png`
- `distribution_shape_monthly_percentiles.csv`
- `distribution_shape_group_summary.csv`

## How To Run

From the `Python` project root:

```bash
python Analysis/Traits/Ecozone/ecozone_distribution_shape_investigation.py
```

## Input Assumptions

- Reads Sentinel monthly composites from `Results/0-CacheBaseData/monthly_max`
- Reads year-group labels from `config/wet_dry_years.csv`
- Reuses the existing AOI ecozone rasters
- Descriptive only; no inferential statistics are added
