# Data Provenance And Cache Characteristics

Last updated: 2026-03-31

This document summarizes the cache segments currently active in project memory and on disk, with emphasis on the characteristics a data scientist, remote sensing analyst, or environmental collaborator would ask about.

Primary historical source of truth for the Sentinel `_3_4` cache:

- [cache_3_4_build_script.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/ScriptSnapshots/cache_3_4_build_script.py)

This document separates:

- verified on-disk characteristics
- confirmed historical `_3_4` Sentinel build behavior from the snapshot script
- current builder behavior in code, included only for comparison

## Scope And Trust Boundary

The authoritative cache baseline is `_3_4`, as configured in [project_paths.yaml](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/config/project_paths.yaml) and described in [PROJECT_OVERVIEW.md](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/docs/PROJECT_OVERVIEW.md).

Active cache branches:

- `AOI/NorthAOI/GWNF_cache_3_4`
- `AOI/SouthAOI/Smoky_cache_3_4`

Sub-branches in current use:

- `s2/indices` for Sentinel-2 NDVI, NDMI, EVI
- `landsat/indices` for Landsat NDVI, NDMI, EVI
- `s2/traits` for terrain and ecological trait rasters aligned to the Sentinel grid

I do not currently find `_3_24` cache manifests on disk, so this document focuses on `_3_4`.

## Important Distinction

There are now two clearly different workflows:

1. The historical Sentinel `_3_4` snapshot script, which is the source of truth for how the trusted Sentinel baseline was built.
2. The current cache-builder code, which describes how a modern rebuild would behave.

Most importantly:

- the historical Sentinel `_3_4` build used a strict `SCL == 4` vegetation mask
- the historical Sentinel `_3_4` build used the AOI bounding-box grid and did not apply an AOI polygon mask
- the current Sentinel builder uses broader cloud/snow exclusion and AOI polygon masking

That difference matters directly for interpretation.

## Segment Summary

| Segment | Sensor family | Grid | CRS | Extent behavior | Date range on disk | Scene count per index | Manifest evidence |
|---|---|---:|---|---|---|---:|---|
| North Sentinel `_3_4` | Sentinel-2 L2A | `6451 x 9812` at 10 m | `EPSG:32617` | canonical AOI bounding-box grid, no polygon mask in historical source script | `2017-04-09` to `2026-02-21` | `437` | `veg_coverage`, `cloud_cover`, timestamp |
| South Sentinel `_3_4` | Sentinel-2 L2A | `9198 x 4289` at 10 m | `EPSG:32617` | canonical AOI bounding-box grid, no polygon mask in historical source script | `2017-01-25` to `2026-03-01` | `1618` | `veg_coverage`, `cloud_cover`, timestamp |
| North Landsat `_3_4` | Landsat C2 L2 | `2150 x 3270` at 30 m | `EPSG:32617` | canonical AOI-aligned bounding-box grid | `1984-03-12` to `2008-12-25` | `1258` | `clear_coverage`, `platform`, `path_row`, `cloud_cover`, timestamp |
| South Landsat `_3_4` | Landsat C2 L2 | `3066 x 1429` at 30 m | `EPSG:32617` | canonical AOI-aligned bounding-box grid | `1984-03-27` to `1988-12-24` | `126` | `clear_coverage`, `platform`, `path_row`, `cloud_cover`, timestamp |

## Shared Spatial Characteristics

### Canonical grid approach

All main cache segments use a canonical projected grid per AOI rather than scene-native geometry.

- north extent bounds: `619300.15, 4203684.60, 683810.82, 4301812.35`
- south extent bounds: `225602.86, 3923116.62, 317589.63, 3966007.23`
- projected CRS: `EPSG:32617`

This means the project works from stable AOI-aligned rasters rather than reprojecting on the fly during analysis.

### Bounding box versus AOI footprint

The historical Sentinel `_3_4` source script constructs a bounding-box grid from the AOI extent and does not rasterize or apply an AOI polygon mask.

Historical `_3_4` Sentinel behavior:

- raster dimensions are derived from AOI bounding-box extents
- no AOI polygon mask is applied during cache generation
- validity comes from vegetation and numeric masks only
- any vegetation-class pixels within the bounding box can survive, even if they fall outside the exact AOI footprint

Current rebuild code differs:

- the modern builder rasterizes the AOI polygon and masks outside-footprint pixels to nodata
- that behavior should not be projected backward onto the trusted `_3_4` Sentinel cache

## Sentinel-2 `_3_4` Cache Characteristics

### What is verified on disk

Observed from current `_3_4` manifests:

- stored indices: `NDVI`, `NDMI`, `EVI`
- all three indices have matching scene counts within each AOI
- rasters are `float32`
- nodata is `NaN`
- manifests store:
  - `filename`
  - `date`
  - `cloud_cover`
  - `veg_coverage`
  - a processing timestamp field, seen as either `rebuilt` or `processed`

Observed range of stored `veg_coverage` values for `NDVI`:

- north: min `0.0000`, median `0.5877`, max `0.9646`
- south: min `0.0013`, median `0.2029`, max `0.8962`

### Historical Sentinel `_3_4` source-script behavior

Confirmed from [cache_3_4_build_script.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/ScriptSnapshots/cache_3_4_build_script.py):

- collection: `sentinel-2-l2a`
- date filter: `2017-01-01` to `2026-03-01`
- scene-level cloud filter: `eo:cloud_cover < 40`
- output indices: `NDVI`, `NDMI`, `EVI`
- required bands fetched per scene: requested spectral bands plus `SCL`
- reprojection target: `EPSG:32617`
- target resolution: `10 m`
- spatial frame: canonical AOI bounding-box grid
- AOI polygon mask: none
- pixel inclusion: `SCL == 4` and required bands in `(0, 10000)`
- reflectance scaling: divide raw bands by `10000.0`
- saved outputs: one float32 GeoTIFF per scene per index
- nodata: `NaN`
- manifest fields: `filename`, `date`, `cloud_cover`, `veg_coverage`, `processed`

What the historical `_3_4` Sentinel script does not do:

- it does not save raw `SCL` sidecars
- it does not compute snow fraction
- it does not apply PB04.00 harmonization
- it does not polygon-clip to the AOI footprint

### What `veg_coverage` means in the source script

The historical script explicitly defines:

- `veg_mask = (fetched["SCL"] == 4)`
- `valid_mask` requires each needed raw band to satisfy `(arr > 0) & (arr < 10000)`
- `combined_mask = veg_mask & valid_mask`

It then records:

- `veg_coverage = combined_mask.sum() / combined_mask.size`

So `veg_coverage` is not an AOI-footprint coverage metric.
It is the fraction of the entire bounding-box grid that survives the vegetation and numeric validity filters.

### Pixel inclusion rule for trusted `_3_4`

Included:

- pixels classified as vegetation by Sentinel SCL
- only where every required raw band is greater than `0` and less than `10000`

Excluded:

- cloud shadow
- cloud medium
- cloud high
- thin cirrus
- snow / ice
- water
- bare or not vegetated surfaces
- unclassified or non-vegetation classes
- saturated or invalid reflectance values outside the numeric screen
- pixels with invalid numeric values

Not excluded by any AOI polygon footprint rule in the `_3_4` Sentinel source script:

- pixels simply because they fall outside the exact AOI polygon but inside the AOI bounding box

### Scientific implication

This is a conservative ecological mask, but not a strict polygon-clipped AOI product.

Advantages:

- focuses analyses on vegetated signal only
- reduces contamination from clouds, snow, water, and non-vegetated terrain

Costs:

- transitional canopy states may be underrepresented
- early drought signal could be muted if stressed vegetation is pushed out of `SCL = 4`
- bounding-box leakage is possible outside the exact AOI footprint if those pixels are classified as vegetation
- comparisons are strongest as relative ecozone comparisons, less so as exact polygon-total summaries

## Current Sentinel Builder Design

Current code: [build_sentinel_cache.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Cache/build_sentinel_cache.py)

This section is included for comparison only. It is not the source of truth for the trusted `_3_4` Sentinel cache.

### Scene discovery

- collection: `sentinel-2-l2a`
- search geometry: AOI polygon in WGS84
- date filtering: CLI date range
- cloud filter: `eo:cloud_cover < cloud_max`
- default `cloud_max`: `40`
- deduplication: one scene per `date + tile`

### Spatial handling

- reproject all bands to `EPSG:32617`
- target resolution `10 m`
- use a canonical AOI bounding-box grid
- rasterize AOI polygon separately
- set pixels outside the AOI polygon to nodata

### Current pixel exclusion rule

Current code excludes only these SCL classes from index rasters:

- `8` cloud medium probability
- `9` cloud high probability
- `10` thin cirrus
- `11` snow / ice

Current code keeps:

- `3` cloud shadow
- `4` vegetation
- `5` non-vegetated / bare
- `6` water
- `7` unclassified

Additional numeric screening:

- raw reflectance bands must satisfy `0 < DN < 10000`
- result pixels outside the combined valid mask become `NaN`

### Current harmonization and metadata

The modern builder applies PB04.00 harmonization and stores more metadata than the historical `_3_4` script, including:

- `clear_frac`
- `snow_frac`
- `processing_baseline`
- `harmonized`

Those fields do not describe how the trusted `_3_4` Sentinel cache itself was built.

## Landsat `_3_4` Cache Characteristics

### What is verified on disk

Observed from current `_3_4` manifests:

- stored indices: `NDVI`, `NDMI`, `EVI`
- all three indices have matching scene counts within each AOI
- rasters are `float32`
- nodata is `NaN`
- manifests store:
  - `filename`
  - `date`
  - `cloud_cover`
  - `clear_coverage`
  - `platform`
  - `path_row`
  - `processed`

Observed range of stored `clear_coverage` values for `NDVI`:

- north: min `0.0000`, median `0.3425`, max `0.9155`
- south: min `0.0000`, median `0.2250`, max `0.9943`

### Temporal coverage caveat

The current Landsat `_3_4` branch is incomplete relative to the code’s nominal 1984-present scope.

On disk today:

- north Landsat runs only through `2008-12-25`
- south Landsat runs only through `1988-12-24`

That is an important limitation for any long-term trend claim.

### Current Landsat builder design

Current code: [build_landsat_cache.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Cache/build_landsat_cache.py)

Verified characteristics of the modern Landsat workflow:

- source: Landsat Collection 2 Level 2 from Planetary Computer
- platforms: Landsat 5, 7, 8, 9
- default scene-level cloud threshold: `eo:cloud_cover < 40`
- output resolution: `30 m`
- output CRS: `EPSG:32617`
- AOI polygon mask applied after reprojection
- reflectance scaling: `raw * 0.0000275 - 0.2`
- excluded QA flags: dilated cloud, cirrus, cloud, snow
- retained QA conditions include cloud shadow and water

Per modern Landsat index manifest entry:

- `filename`
- `date`
- `cloud_cover`
- `valid_frac`
- `snow_frac`
- `platform`
- `path_row`
- `processed`

This section is based on current code because no historical Landsat `_3_4` builder snapshot has been provided yet.

## Trait Rasters And Alignment

The project’s terrain and ecological trait rasters are aligned to the Sentinel AOI grid and used to stratify the index caches.

Trait branches in active use:

- `s2/traits/terrain`
- `s2/traits/ecozone`
- `s2/traits/forest`

Verified terrain alignment:

- north terrain rasters match north Sentinel grid exactly
- south terrain rasters match south Sentinel grid exactly

This means ecozone, aspect, elevation, and forest overlays are all being analyzed on a common AOI-aligned raster framework.

## What Rules Out Pixels

### Historical Sentinel `_3_4`

Excluded:

- every pixel not labeled `SCL == 4`
- any required raw band values not in `(0, 10000)`

Not excluded:

- pixels outside the AOI polygon if they still fall inside the AOI bounding-box grid and meet the vegetation and numeric criteria

### Current Sentinel code

Excluded:

- SCL `8`, `9`, `10`, `11`
- raw DN outside `(0, 10000)`
- pixels outside AOI polygon

### Current Landsat code

Excluded:

- QA cloud bits `1`, `2`, `3`
- snow bit `5`
- scaled reflectance outside `[0, 1]`
- non-finite required bands
- pixels outside AOI polygon

## What Metadata Is Preserved Today

### Preserved in trusted Sentinel `_3_4`

- scene identifier
- date
- file name
- cloud cover
- vegetation-coverage statistic over the full bounding-box grid
- processing timestamp

### Not preserved in trusted Sentinel `_3_4`

- raw `SCL` sidecars
- explicit AOI polygon coverage
- snow fraction
- processing baseline
- harmonization flag

### Preserved in trusted Landsat `_3_4`

- scene identifier
- date
- file name
- cloud cover
- clear-coverage statistic
- platform
- path/row
- processing timestamp

## Main Scientific Caveats

- The trusted Sentinel baseline definitely uses a vegetation-only rule in the historical source script.
- The trusted Sentinel baseline definitely uses bounding-box extents without AOI polygon masking in the historical source script.
- Landsat coverage on disk is incomplete relative to the intended 1984-present scope.
- Current rebuild code should not be assumed to describe the historical trusted `_3_4` Sentinel workflow.
- `veg_coverage` in Sentinel manifests is a bounding-box-grid metric, not an AOI-polygon metric.

## Recommended Language For Explaining The Data

> The project uses AOI-aligned raster caches in UTM Zone 17N for north and south Appalachian landscapes. The trusted Sentinel `_3_4` cache was built on canonical AOI bounding-box grids at 10 m resolution using a strict `SCL == 4` vegetation mask plus numeric validity screening, without AOI polygon masking. It stores scene-level NDVI, NDMI, and EVI rasters and records scene date, cloud cover, and vegetation coverage in the manifest. Landsat `_3_4` caches are also AOI-aligned and store the same derived indices with scene-level cloud and clear-coverage metadata, but their current on-disk temporal coverage is incomplete. Modern rebuild code uses different masking logic and should not be treated as the source of truth for the historical Sentinel `_3_4` baseline.

## Best Next Validation Steps

- Decide whether the historical bounding-box extent is acceptable for current ecological comparisons or should be treated as a core limitation requiring a rebuild.
- If you can recover the historical Landsat `_3_4` builder too, use it to replace the current-code-derived Landsat method section in this document.
- Build a cache-audit table that records per-branch scene counts, date ranges, manifest fields, and trust notes in one place.
- Preserve modern raw-quality sidecars and manifests if future rebuilds are kept.
