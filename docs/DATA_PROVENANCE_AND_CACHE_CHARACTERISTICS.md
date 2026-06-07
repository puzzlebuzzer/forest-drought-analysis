# Data Provenance And Cache Characteristics

This document summarizes the provenance and interpretation boundaries of the satellite cache products used by the Appalachian Ecozone-Vegetation dashboard.

## Purpose

The dashboard is powered by precomputed summary tables derived from AOI-aligned Sentinel-2 and Landsat raster caches. This document records what is known about those caches, how the accepted Sentinel cache behavior differs from the current rebuild code, and how that affects interpretation.

## Trust Boundary

The accepted Sentinel-2 cache behavior used for dashboard summaries is preserved in this source snapshot:

```text
ScriptSnapshots/cache_build_script.py
```

Current cache-builder scripts are useful for rebuilding and comparison, but the current Sentinel builder does not exactly reproduce that accepted Sentinel cache behavior.

Configured cache branches:

- `AOI/NorthAOI/GWNF_cache`
- `AOI/SouthAOI/Smoky_cache`

Configured sub-branches:

- `s2/indices` for Sentinel-2 NDVI, NDMI, EVI
- `landsat/indices` for Landsat NDVI, NDMI, EVI
- `s2/traits` for terrain and ecological trait rasters aligned to the Sentinel grid

## Segment Summary

| Segment | Sensor family | Grid | CRS | Extent behavior | Date range on disk | Scene count per index | Manifest evidence |
| --- | --- | ---: | --- | --- | --- | ---: | --- |
| North Sentinel | Sentinel-2 L2A | `6451 x 9812` at 10 m | `EPSG:32617` | canonical AOI bounding-box grid; AOI/trait filtering applied downstream | `2017-04-09` to `2026-02-21` | `437` | `veg_coverage`, `cloud_cover`, timestamp |
| South Sentinel | Sentinel-2 L2A | `9198 x 4289` at 10 m | `EPSG:32617` | canonical AOI bounding-box grid; AOI/trait filtering applied downstream | `2017-01-25` to `2026-03-01` | `1618` | `veg_coverage`, `cloud_cover`, timestamp |
| North Landsat | Landsat C2 L2 | `2150 x 3270` at 30 m | `EPSG:32617` | canonical AOI-aligned grid | `1984-03-12` to `2026-02-27` | `2548` | `clear_coverage`, `platform`, `path_row`, `cloud_cover`, timestamp |
| South Landsat | Landsat C2 L2 | `3066 x 1429` at 30 m | `EPSG:32617` | canonical AOI-aligned grid | `1984-03-27` to `2026-02-17` | `2145` | `clear_coverage`, `platform`, `path_row`, `cloud_cover`, timestamp |

Landsat coverage extends from 1984 through the beginning of 2026 in the current cache. The 2026 year is partial in the on-disk cache ranges above.

## Shared Spatial Characteristics

All main cache segments use a canonical projected grid per AOI rather than scene-native geometry.

- projected CRS: `EPSG:32617`
- north extent bounds: `619300.15, 4203684.60, 683810.82, 4301812.35`
- south extent bounds: `225602.86, 3923116.62, 317589.63, 3966007.23`

This design makes later summaries stable and repeatable because scenes are already aligned before table generation.

## Accepted Sentinel-2 Cache Behavior

The accepted Sentinel cache was built from Sentinel-2 L2A scenes using:

- collection: `sentinel-2-l2a`
- date filter: `2017-01-01` to `2026-03-01`
- scene-level cloud filter: `eo:cloud_cover < 40`
- target CRS: `EPSG:32617`
- target resolution: 10 m
- spatial frame: canonical AOI bounding-box grid
- AOI polygon mask during cache generation: none
- pixel inclusion: `SCL == 4` and required bands in `(0, 10000)`
- reflectance scaling: divide raw bands by `10000`
- output indices: NDVI, NDMI, EVI
- output type: one float32 GeoTIFF per scene per index
- nodata: `NaN`

The accepted-cache source script did not:

- save raw SCL sidecars
- compute snow fraction
- apply PB04.00 harmonization
- polygon-clip to the exact AOI footprint

## Sentinel `veg_coverage`

The accepted Sentinel source script defines:

```text
veg_mask = SCL == 4
valid_mask = required raw bands in (0, 10000)
combined_mask = veg_mask & valid_mask
veg_coverage = combined_mask.sum() / combined_mask.size
```

Therefore, `veg_coverage` is the fraction of the full AOI bounding-box grid that survives the vegetation and numeric filters. It is not an AOI-polygon coverage metric.

## Current Sentinel Builder

The current Sentinel builder is documented for rebuild comparison only. It differs from the accepted Sentinel cache.

Current design:

- searches Planetary Computer Sentinel-2 L2A scenes using AOI polygon geometry
- reprojects bands to `EPSG:32617`
- uses a 10 m canonical grid
- rasterizes the AOI polygon and masks outside-footprint pixels
- excludes SCL `8`, `9`, `10`, and `11`
- retains SCL `3`, `4`, `5`, `6`, and `7`
- applies numeric screening of raw bands in `(0, 10000)`
- stores additional metadata such as snow fraction, processing baseline, and harmonization fields

This current behavior should not be projected backward onto the accepted Sentinel cache used for the dashboard summaries.

## Landsat Cache Characteristics

The Landsat cache stores NDVI, NDMI, and EVI rasters with scene metadata:

- file name
- scene date
- scene-level cloud cover
- clear/valid coverage statistic
- platform
- path/row
- processing timestamp

The current Landsat builder uses:

- Landsat Collection 2 Level 2
- Landsat 5, 7, 8, and 9
- scene-level cloud filter: `eo:cloud_cover < 40`
- target CRS: `EPSG:32617`
- target resolution: 30 m
- AOI polygon mask after reprojection
- reflectance scaling: `raw * 0.0000275 - 0.2`
- QA exclusion of dilated cloud, cirrus, cloud, and snow
- retention of cloud shadow and water
- QA_PIXEL sidecar storage for additional filtering if needed

The current dashboard table metadata uses a fixed Landsat mask identifier. If a future methods section needs bit-level precision, use the current builder behavior above rather than relying only on the metadata label.

## Trait Rasters

Trait rasters provide the categorical and terrain strata used by the dashboard table factory.

Active trait branches include:

- `s2/traits/terrain`
- `s2/traits/ecozone`
- `s2/traits/forest`

Current interpretation:

- thermal ecozone is the broad cool/intermediate/hot tier
- forest-community group is the TNC group tier within the forest-community source data
- forest community is the canonical fine ecological class
- terrain/elevation/slope/aspect layers support auxiliary analyses

The dashboard table factory uses AOI-aligned categorical rasters for thermal ecozone, forest-community group, and forest community summaries. Streamlit does not read those rasters at runtime.

## Pixel Inclusion Summary

Accepted Sentinel cache:

- includes only `SCL == 4` vegetation pixels with valid raw bands
- does not exclude pixels solely because they fall outside the AOI polygon if they remain inside the AOI bounding-box grid

Current Sentinel builder:

- excludes cloud/snow SCL classes `8`, `9`, `10`, `11`
- applies an AOI polygon mask
- retains several non-cloud SCL classes that the accepted cache did not retain

Current Landsat builder:

- excludes dilated cloud, cirrus, cloud, and snow
- applies an AOI polygon mask
- retains cloud shadow and water
- stores QA_PIXEL for later inspection/filtering

## Scientific Implications

The accepted Sentinel cache is conservative for vegetated canopy signal. Its cache grid is not a strict AOI-footprint product, but AOI footprint and trait masks are applied downstream during analysis/table generation.

Advantages:

- strong focus on vegetation-class pixels
- reduced contamination from clouds, snow, water, and non-vegetated terrain
- stable AOI-aligned grids for repeated summaries

Limitations:

- transitional canopy states may be underrepresented if stressed vegetation is not classified as `SCL == 4`
- bounding-box leakage is possible at the cache level where vegetation-class pixels occur outside the AOI polygon but inside the bounding-box grid; downstream AOI/trait masks limit the dashboard summaries.
- current rebuild code and accepted Sentinel cache behavior are not interchangeable

## Recommended Language

> The project uses AOI-aligned Sentinel-2 and Landsat vegetation-index caches in UTM Zone 17N for two Appalachian study areas. The accepted Sentinel cache was built on canonical AOI bounding-box grids at 10 m resolution using a strict `SCL == 4` vegetation mask plus numeric validity screening, without AOI polygon masking during cache generation. AOI footprint and trait masks are applied downstream during analysis/table generation. Landsat caches are AOI-aligned at 30 m, cover 1984 through early 2026 in the current on-disk products, and store the same derived indices. Dashboard summary tables are derived from these caches and should be interpreted with these mask and coverage boundaries in mind.

## Reproducibility Note

Dashboard-only users need precomputed `SummaryTables/dashboard_data/` products and do not need the raw satellite caches. Rebuilding dashboard tables from existing caches requires the AOI caches, trait rasters, PRISM products, and table-factory scripts. Full raw-to-dashboard reproducibility requires substantially more storage and should be treated as a separate reproducibility tier.
