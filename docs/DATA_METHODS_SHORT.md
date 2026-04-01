# Data Methods Short

Last updated: 2026-03-31

Primary historical source of truth for the trusted Sentinel `_3_4` build:

- [cache_3_4_build_script.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/ScriptSnapshots/cache_3_4_build_script.py)

## What Data The Project Uses

The project currently relies on `_3_4` cache branches as the authoritative baseline.

Active cache segments:

- north Sentinel-2: `AOI/NorthAOI/GWNF_cache_3_4/s2/indices`
- south Sentinel-2: `AOI/SouthAOI/Smoky_cache_3_4/s2/indices`
- north Landsat: `AOI/NorthAOI/GWNF_cache_3_4/landsat/indices`
- south Landsat: `AOI/SouthAOI/Smoky_cache_3_4/landsat/indices`

Derived products in use:

- `NDVI`
- `NDMI`
- `EVI`

Trait rasters aligned to the Sentinel grid:

- terrain
- ecozone
- forest type and forest group

## Spatial Design

All cache segments are stored on AOI-aligned projected grids in `EPSG:32617`.

- north Sentinel grid: `6451 x 9812` at 10 m
- south Sentinel grid: `9198 x 4289` at 10 m
- north Landsat grid: `2150 x 3270` at 30 m
- south Landsat grid: `3066 x 1429` at 30 m

The trusted Sentinel `_3_4` workflow uses canonical AOI bounding-box grids for alignment and does not apply an AOI polygon mask in the historical source script.

## Temporal Coverage On Disk

- north Sentinel: `2017-04-09` to `2026-02-21`
- south Sentinel: `2017-01-25` to `2026-03-01`
- north Landsat: `1984-03-12` to `2008-12-25`
- south Landsat: `1984-03-27` to `1988-12-24`

## Most Important Data-Handling Point

The trusted Sentinel `_3_4` baseline is more restrictive than the current rebuild code.

Confirmed from the historical source script:

- trusted Sentinel `_3_4` retains only `SCL = 4` vegetation pixels
- it also requires needed raw bands to fall within `(0, 10000)`
- it does not polygon-mask to the exact AOI footprint
- current Sentinel rebuild code is broader and excludes only cloud and snow classes while applying an AOI polygon mask

That means the historical trusted Sentinel baseline is best understood as a vegetation-focused bounding-box-grid analysis dataset, not a strict AOI-footprint condition sample.

## Historical Sentinel `_3_4` Build Logic

The trusted `_3_4` Sentinel cache was built with:

- source: Sentinel-2 L2A from Planetary Computer
- scene-level cloud filter: `eo:cloud_cover < 40`
- 10 m output grid in `EPSG:32617`
- canonical AOI bounding-box grid
- no AOI polygon mask
- `SCL == 4` vegetation-only mask
- numeric screen requiring needed bands in `(0, 10000)`
- reflectance scaling by dividing raw values by `10000`

## Current Landsat Builder Logic

No historical Landsat `_3_4` build snapshot has been provided yet, so the best available method description is from the current code:

- source: Landsat Collection 2 Level 2 from Planetary Computer
- platforms: Landsat 5, 7, 8, 9
- default cloud threshold: `eo:cloud_cover < 40`
- 30 m output grid in `EPSG:32617`
- AOI polygon mask applied after reprojection
- excluded QA flags: dilated cloud, cirrus, cloud, snow
- retained QA conditions include cloud shadow and water
- reflectance scaling: `raw * 0.0000275 - 0.2`

## Metadata Preserved In The Trusted Cache

Sentinel manifests currently preserve:

- file name
- scene date
- scene-level cloud cover
- `veg_coverage`
- processing timestamp

Important:

- `veg_coverage` is the fraction of the bounding-box grid that survives the vegetation and numeric masks
- it is not an AOI-polygon coverage metric

Landsat manifests currently preserve:

- file name
- scene date
- scene-level cloud cover
- `clear_coverage`
- platform
- path/row
- processing timestamp

## Main Caveats

- The trusted Sentinel baseline definitely uses a vegetation-only rule in the historical source script, which reduces contamination but may miss transitional stress states.
- The trusted Sentinel baseline definitely uses bounding-box extents without AOI polygon masking in the historical source script.
- Landsat coverage on disk is incomplete relative to the intended 1984-present scope.
- Current rebuild code should not be assumed to exactly match the historical trusted `_3_4` Sentinel workflow.

## Short Description You Can Reuse

> The project uses AOI-aligned raster caches in UTM Zone 17N for north and south Appalachian study areas. The trusted Sentinel `_3_4` cache was built on canonical AOI bounding-box grids at 10 m resolution using a strict `SCL = 4` vegetation mask plus numeric validity screening, without AOI polygon masking. It stores scene-level NDVI, NDMI, and EVI rasters and records scene date, cloud cover, and vegetation coverage in the manifest. Landsat `_3_4` caches are also AOI-aligned and store the same indices, but current on-disk Landsat coverage is incomplete. Modern rebuild code uses different masking logic and should not be treated as the source of truth for the historical Sentinel `_3_4` baseline.
