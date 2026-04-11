# Derivation: Ecozone Seasonal Curves | NDVI

This note documents how the NDVI ecozone seasonal curves are derived from Sentinel scene caches through the current project workflow, and what an ArcGIS Pro replication path would likely look like.

Relevant outputs:

- [ecozone_ndvi_seasonal.png](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Results/figures/ecozone_ndvi_seasonal.png)
- [ecozone_seasonal_summary.xlsx](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Results/tables/ecozone_seasonal_summary.xlsx)

Relevant script:

- [ecozone_seasonal_curves.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/ecozone_seasonal_curves.py)

## Short Summary

Unlike the anomaly trajectory plot, the seasonal NDVI curves are:

- not year-class anomalies
- not relative to a neutral baseline
- not based on monthly composites

Instead, they are built by:

1. loading all cached Sentinel NDVI scene rasters
2. grouping them by calendar month across all years
3. computing scene-level ecozone `p95` and `p100`
4. averaging those scene-level percentile values within each month
5. plotting the monthly mean `p95` curve, with a shaded `p95` to `p100` band

## Step-By-Step Derivation

### 1. Sentinel-2 scenes are downloaded and converted to per-scene NDVI rasters

The scene cache builder is:

- [build_sentinel_cache.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Cache/build_sentinel_cache.py)

What it does:

- queries Sentinel-2 L2A scenes from Microsoft Planetary Computer STAC
- reprojects scenes to the AOI-aligned canonical grid
- applies the Sentinel quality mask
- applies PB04+ harmonization where needed
- computes NDVI per scene
- writes one AOI-aligned NDVI GeoTIFF per scene into the Sentinel cache

This is the raw input layer for the seasonal curve analysis.

## 2. The seasonal curve script reads the cached NDVI scene rasters directly

The seasonal curve analysis does not use monthly composites.

Instead, it loads all scene rasters listed in the NDVI manifest using:

- [load_scenes()](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/ecozone_seasonal_curves.py#L66)

Each loaded scene contributes:

- its acquisition date
- its raster filepath

The scenes are sorted by date, but the final summaries are grouped by calendar month rather than by year.

## 3. The ecozone raster is loaded and converted into masks

For each AOI, the script reads:

- `tnc_ecozone_simplified_snapped.tif`

from the AOI ecozone directory and builds boolean masks for ecozone codes:

- `1 = Cool`
- `2 = Intermediate`
- `3 = Hot`

Relevant code:

- [ecozone_seasonal_curves.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/ecozone_seasonal_curves.py#L126)

## 4. For each scene, the ecozone p95 and p100 NDVI values are computed

This happens in:

- [monthly_percentiles_by_ecozone()](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/ecozone_seasonal_curves.py#L84)

For each scene:

- the raster is read once
- valid pixels are defined as non-NaN pixels
- for each ecozone, pixels inside the ecozone mask and valid in the raster are selected
- if the ecozone has at least `MIN_PIXELS = 100` valid pixels, the script computes:
  - `p95`
  - `p100`

These are scene-level ecozone summaries, not monthly composite values.

They are grouped into lists by calendar month:

- January values pooled together
- February values pooled together
- ...
- December values pooled together

across all available years.

## 5. The monthly seasonal summary is the mean of those scene-level percentile values

After collecting scene-level `p95` and `p100` values for each month and ecozone, the script computes:

- mean monthly `p95`
- mean monthly `p100`

for each:

- AOI
- index
- month
- ecozone

This happens in:

- [ecozone_seasonal_curves.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/ecozone_seasonal_curves.py#L158)

So the line value is:

`mean across all scenes in that calendar month of scene-level ecozone p95 NDVI`

And the band top is:

`mean across all scenes in that calendar month of scene-level ecozone p100 NDVI`

This is a pooled seasonal climatology across all years on disk, not a year-specific trajectory and not an anomaly.

## 6. Scene counts are also summarized by month

For each month, the script records a simple scene count:

- the maximum number of contributing scene-level `p95` values across ecozones for that month

This count is used only for faint background bars in the figure, to show monthly observation density.

Relevant code:

- [ecozone_seasonal_curves.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/ecozone_seasonal_curves.py#L163)
- [ecozone_seasonal_curves.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/ecozone_seasonal_curves.py#L222)

## 7. The NDVI figure is rendered

The figure helper is:

- [seasonal_figure()](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/ecozone_seasonal_curves.py#L192)

Plot semantics:

- one panel for north
- one panel for south
- one colored line per ecozone
- line = monthly mean `p95 NDVI`
- shaded band = `p95` to `p100`
- faint bars in the background = monthly scene counts

The specific NDVI plot is generated by:

- [ecozone_seasonal_curves.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/ecozone_seasonal_curves.py#L262)

with title:

- `Seasonal Peak Vegetation — Monthly p95 NDVI by Ecozone`
- subtitle note in title: `(shaded band = p95 to p100)`

## Mathematical Interpretation

For a given:

- AOI
- ecozone
- calendar month

the plotted NDVI line value is:

`mean over all scenes in that month across all years of the ecozone-level p95 NDVI`

and the shaded band extends to:

`mean over all scenes in that month across all years of the ecozone-level p100 NDVI`

This is a pooled seasonal shape summary, not an anomaly and not a wet/dry comparison.

## How This Differs From The Monthly Anomaly Trajectories

The NDVI seasonal curves:

- use scene-level cache rasters directly
- pool all years together
- do not use wet / neutral / dry labels
- do not use a neutral baseline
- summarize raw seasonal shape

The monthly anomaly trajectory plot:

- uses monthly ecozone summaries
- groups by wet and dry year class
- subtracts the neutral baseline
- summarizes anomaly trajectories rather than raw seasonal curves

## ArcGIS Pro Replication: Additional Information Needed

For ArcGIS Pro replication, the important implementation details are:

- exact scene-cache input location and naming rules
- canonical AOI grid alignment
- ecozone raster alignment with the scene rasters
- valid pixel rule: non-NaN NDVI only
- minimum pixel rule: at least `100` ecozone pixels before a percentile is computed
- scene grouping rule:
  - grouped by calendar month
  - pooled across all years
- summary rule:
  - compute scene-level ecozone percentiles first
  - then average those scene-level percentiles by month

That last point matters: the figure is not produced by first mosaicking all scenes in a month and then taking a percentile. It is produced by computing percentiles scene by scene and then averaging the resulting monthly lists.

## Likely ArcGIS Pro Tool Sequence

A likely ArcGIS Pro replication path would be:

1. Organize the cached NDVI scene rasters and their dates in a table or catalog.
2. Ensure all scene rasters and the ecozone raster are on the same AOI-aligned grid.
3. If needed, use `Project Raster` so all rasters match the canonical grid.
4. Use the ecozone raster as the zone dataset.
5. For each scene raster, run a zonal summary step by ecozone.
6. Extract percentile values for each scene by ecozone:
   - ideally `p95`
   - ideally `max` for `p100`
7. Record the scene acquisition month for each zonal result.
8. Group all scene-level ecozone results by:
   - AOI
   - month
   - ecozone
9. Use `Summary Statistics` to compute the mean monthly `p95` and mean monthly `p100`.
10. Use the grouped table to build the final seasonal line chart.

## ArcGIS Pro Friction Point

The main difficulty in ArcGIS Pro is the same general issue as elsewhere:

- mean and max zonal summaries are straightforward
- repeated percentile-by-zone extraction across many scene rasters is less convenient

The specific conceptual trap is also important:

- do not replace the intended method with a monthly raster composite followed by zonal percentile extraction

That would be a different analysis.

To match this figure, the workflow must be:

- scene-level ecozone percentiles first
- month-wise averaging second

not:

- month composite first
- percentile second

## Practical Summary

To replicate this figure faithfully, think of it as:

- a scene-level seasonal climatology by ecozone
- summarized by calendar month
- using ecozone `p95` NDVI as the main line
- with `p100` as the upper envelope band

That is the core logic to preserve whether the replication is done in Python, ArcGIS Pro, or a hybrid workflow.
