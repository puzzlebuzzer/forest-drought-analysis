# Fraction Below Baseline Investigation

## What This Produces

This analysis measures, for each:

- AOI
- ecozone
- index (`NDVI`, `NDMI`, `EVI`)
- month (`Apr-Oct`)
- year-group (`wet`, `dry`, `neutral`)

the fraction of valid ecozone pixels that fall below a neutral-year monthly
baseline raster.

## Baseline Definition

For each AOI / index / calendar month, the script builds a neutral baseline
raster as the pixelwise median across all available `neutral` monthly Sentinel
composites for that month.

Then for each monthly raster:

```text
fraction_below_baseline =
    (# ecozone pixels where monthly_value < neutral_baseline_value)
    /
    (# valid ecozone pixels compared)
```

## Outputs

- `fraction_below_baseline_ndvi.png`
- `fraction_below_baseline_ndmi.png`
- `fraction_below_baseline_evi.png`
- `fraction_below_baseline.csv`
- `neutral_baseline_availability.csv`

## How To Run

From the `Python` project root:

```bash
python Analysis/Traits/Ecozone/ecozone_fraction_below_baseline_investigation.py
```

## Assumptions And Limitations

- Reads Sentinel monthly composites from `Results/0-CacheBaseData/monthly_max`
- Reads year-group labels from `config/wet_dry_years.csv`
- Reuses the existing AOI ecozone rasters
- Uses a neutral-year monthly median raster baseline, not a more complex climatology
- Descriptive only; no inferential statistics are added
