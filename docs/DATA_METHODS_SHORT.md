# Data Methods Short

This short note summarizes the data methods behind the Appalachian Ecozone-Vegetation dashboard. For more detail, see `DATA_PROVENANCE_AND_CACHE_CHARACTERISTICS.md`.

## Study Data

The project uses two Appalachian AOIs:

- north: George Washington / Jefferson National Forest region
- south: Smoky Mountains / Nantahala-region AOI

Primary inputs:

- Sentinel-2 Level-2A from Microsoft Planetary Computer
- Landsat Collection 2 Level 2 from Microsoft Planetary Computer
- PRISM monthly precipitation
- TNC Appalachian forest-community/ecozone source data
- Copernicus DEM GLO-30
- project AOI footprints from the configured TNC AOI shapefile

Vegetation indices:

- NDVI
- NDMI
- EVI

## Spatial Framework

Satellite index rasters are stored in AOI-aligned projected grids using `EPSG:32617`.

Current cache grids:

| Segment | Grid |
| --- | ---: |
| north Sentinel-2 | `6451 x 9812` at 10 m |
| south Sentinel-2 | `9198 x 4289` at 10 m |
| north Landsat | `2150 x 3270` at 30 m |
| south Landsat | `3066 x 1429` at 30 m |

Trait rasters are aligned to the analysis grids and used for stratification:

- thermal ecozone
- forest-community group
- forest community
- terrain/elevation/slope/aspect

## Satellite Cache Baselines

The accepted Sentinel-2 cache used for dashboard summaries has these key characteristics:

- Sentinel-2 L2A source scenes
- scene-level cloud filter: `eo:cloud_cover < 40`
- 10 m output grid in `EPSG:32617`
- canonical AOI bounding-box grid
- no AOI polygon mask during cache generation
- `SCL == 4` vegetation-only mask
- numeric screen requiring needed raw bands in `(0, 10000)`
- reflectance scaling by dividing raw values by `10000`

The current Sentinel rebuild code is broader than the accepted cache:

- it applies an AOI polygon mask
- it excludes cloud/snow SCL classes
- it retains some non-cloud classes that the accepted `SCL == 4` workflow did not retain

For that reason, current Sentinel rebuild code should not be treated as an exact description of the accepted Sentinel cache used for the dashboard summaries.

The current Landsat builder uses:

- Landsat 5, 7, 8, and 9 Collection 2 Level 2
- default scene-level cloud filter: `eo:cloud_cover < 40`
- 30 m output grid in `EPSG:32617`
- AOI polygon mask after reprojection
- reflectance scaling: `raw * 0.0000275 - 0.2`
- QA exclusion of dilated cloud, cirrus, cloud, and snow
- retention of cloud shadow and water in the current builder

Dashboard table metadata contains fixed mask identifiers. Interpret Landsat mask descriptions with the provenance note above if comparing current builder code to generated cache products.

## Temporal Coverage On Disk

Current local cache coverage:

| Segment | Date range on disk |
| --- | --- |
| north Sentinel-2 | `2017-04-09` to `2026-02-21` |
| south Sentinel-2 | `2017-01-25` to `2026-03-01` |
| north Landsat | `1984-03-12` to `2008-12-25` |
| south Landsat | `1984-03-27` to `1988-12-24` |

Landsat coverage on disk is incomplete relative to the intended 1984-2026 analytical period unless the cache is rebuilt.

## Dashboard Summary Tables

The dashboard does not read raw rasters at runtime. It reads precomputed products in:

```text
SummaryTables/dashboard_data/
```

The table factory computes:

- scene-level spatial percentiles
- temporal summaries for `scene`, `half_month`, and `month`
- cloud-threshold-filtered temporal products
- whole-AOI, thermal ecozone, forest-community-group, and forest community summaries
- manifests and partitioned Parquet datasets for fast dashboard filtering

Default dashboard-facing percentiles include:

- spatial percentiles: `p50`, `p75`, `p95`, `p98`, `p99`, `p100`
- temporal percentiles: `p50`, `p75`, `p95`, `p98`, `p99`, `p100`

## Climate Context

PRISM monthly precipitation is used as the external moisture context layer.

Current PRISM classification:

- source: PRISM monthly `ppt`
- preferred resolution: 4 km
- years: 1984-2026
- aggregation: calendar-year total precipitation from January through December
- AOI metric: mean monthly precipitation extracted inside each AOI polygon, summed by year
- table classification: AOI-relative top/bottom 20% rank labels
- dashboard background stripes: annual precipitation z-score relative to that AOI's mean annual precipitation series, with neutral years unshaded within the configured threshold

The PRISM layer is an external precipitation context. It is not derived from canopy response.

## Main Caveats

- The accepted Sentinel cache uses a vegetation-only mask and a bounding-box grid rather than a strict AOI polygon footprint.
- Landsat local cache coverage is incomplete relative to the intended full Landsat-era period.
- Current rebuild code does not exactly match the accepted Sentinel cache behavior used for dashboard summaries.
- Forest-community terminology is canonical for detailed vegetation classes; thermal ecozone refers to the broader cool/intermediate/hot tier.
- A satellite-index-only canopy response classification was used as a sanity check, not as dashboard infrastructure or an independent moisture dataset.

## Reusable Methods Summary

The project uses AOI-aligned Sentinel-2 and Landsat vegetation-index caches in UTM Zone 17N, summarized into precomputed dashboard tables by AOI, sensor, index, temporal bin, spatial percentile, thermal ecozone, forest-community group, and forest community. Sentinel-2, Landsat, PRISM, TNC forest-community data, and DEM-derived terrain traits provide the main data sources. The Streamlit dashboard reads summary tables and PRISM classifications at runtime rather than raw rasters.
