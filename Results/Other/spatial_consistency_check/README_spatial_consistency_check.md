# Spatial Consistency Check

## What This Produces

This script creates lightweight anomaly maps for selected:

- AOIs
- indices
- months
- year-groups or explicit years

It compares each selected monthly Sentinel composite against a neutral-year
monthly baseline raster and saves:

- anomaly PNG maps
- a CSV summary of negative / near-zero / positive anomaly coverage
- a CSV showing baseline availability

## Parameterization

Edit these settings at the top of the script:

- `SELECTED_AOIS`
- `SELECTED_INDICES`
- `SELECTED_MONTHS`
- `SELECTED_YEAR_GROUPS`
- `SELECTED_YEARS`
- `INCLUDE_EVI_IF_EASY`
- `NEAR_ZERO`

Use either:

- `SELECTED_YEAR_GROUPS` with `SELECTED_YEARS = []`

or:

- explicit `SELECTED_YEARS`

## Baseline Definition

For each AOI / index / month, the script builds a neutral baseline raster as
the pixelwise median across all available neutral-year monthly Sentinel
composites for that month.

Anomaly is then:

```text
monthly_composite - neutral_baseline
```

## Coverage Summary

- `negative_fraction`
  fraction of valid AOI pixels with anomaly < `-NEAR_ZERO`

- `near_zero_fraction`
  fraction of valid AOI pixels with `|anomaly| <= NEAR_ZERO`

- `positive_fraction`
  fraction of valid AOI pixels with anomaly > `NEAR_ZERO`

## How To Run

From the `Python` project root:

```bash
python Analysis/Traits/Ecozone/ecozone_spatial_consistency_check.py
```

## Outputs

- `anomaly_<index>_<aoi>_<year>_<month>.png`
- `spatial_consistency_summary.csv`
- `baseline_availability.csv`
