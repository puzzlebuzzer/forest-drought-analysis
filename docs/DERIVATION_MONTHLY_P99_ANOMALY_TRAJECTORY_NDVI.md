# Derivation: Monthly p99 Anomaly Trajectories vs Neutral Baseline | NDVI

This note documents how the `Monthly p99 Anomaly Trajectories vs Neutral Baseline | NDVI` figure is derived from Sentinel-2 source data through the current project workflow.

Important provenance note:

- the current repository version of the trajectory script produces `p50` and `p75`, not `p99`
- the derivation below is exact for the current trajectory workflow
- a historical `p99` version would use the same workflow, but with percentile `99` included where the current code uses `SUMMARY_PERCENTILES = [50, 75]`

## Short Summary

The plot is derived as:

1. download Sentinel-2 scenes and compute scene-level NDVI rasters
2. build monthly NDVI maximum composites by AOI
3. summarize each monthly composite within each ecozone using the `p99` NDVI value
4. compute a neutral baseline seasonal curve from neutral-year monthly `p99` values
5. compute wet-year and dry-year monthly anomalies as `p99 value - neutral baseline mean`
6. average those anomalies across years within each year class
7. plot the monthly mean anomaly trajectories for wet and dry years by ecozone and AOI

## Step-By-Step Derivation

### 1. Sentinel-2 scenes are downloaded and converted into per-scene NDVI rasters

The Sentinel cache builder is:

- [build_sentinel_cache.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Cache/build_sentinel_cache.py)

What it does:

- queries Sentinel-2 L2A scenes from Microsoft Planetary Computer STAC
- reprojects them to the AOI-aligned canonical grid
- keeps pixels inside the AOI polygon and sets outside pixels to `NaN`
- excludes SCL classes `8, 9, 10, 11`
- applies PB04+ harmonization by subtracting `0.1` reflectance where needed
- computes NDVI as `(B08 - B04) / (B08 + B04)`
- writes one per-scene NDVI GeoTIFF into the Sentinel cache

Relevant code:

- [build_sentinel_cache.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Cache/build_sentinel_cache.py#L9)
- [build_sentinel_cache.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Cache/build_sentinel_cache.py#L16)
- [build_sentinel_cache.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Cache/build_sentinel_cache.py#L138)

## 2. Monthly NDVI composites are built from the scene cache

The monthly composite builder is:

- [build_monthly_composites.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Indices/build_monthly_composites.py)

What it does:

- reads the scene manifest for each AOI and index
- groups cached scenes by `(year, month)`
- computes the monthly composite as the per-pixel maximum across all valid scenes in that month using `np.fmax`
- writes monthly GeoTIFFs like `Results/0-CacheBaseData/monthly_max/ndvi_north/2018_04.tif`

This means the monthly NDVI raster is a monthly maximum composite, not a mean or median composite.

Relevant code:

- [build_monthly_composites.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Indices/build_monthly_composites.py#L5)
- [build_monthly_composites.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Indices/build_monthly_composites.py#L75)
- [build_monthly_composites.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Indices/build_monthly_composites.py#L107)

## 3. Monthly composites are intersected with ecozone masks

Shared investigation helpers are in:

- [investigation_common.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/investigation_common.py)

What this layer loads:

- monthly Sentinel composites from `Results/0-CacheBaseData/monthly_max`
- ecozone masks from the snapped ecozone raster
- year classifications from `wet_dry_years.csv`

Relevant code:

- [investigation_common.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/investigation_common.py#L49)
- [investigation_common.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/investigation_common.py#L61)
- [investigation_common.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/investigation_common.py#L68)

## 4. For each monthly raster, the ecozone percentile value is computed

The function:

- [ecozone_percentiles()](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/investigation_common.py#L88)

does the following for each ecozone:

- keep only finite pixels inside that ecozone mask
- require at least `MIN_PIXELS = 100`
- compute the requested percentile using `np.nanpercentile`

The table builder:

- [build_monthly_ecozone_dataframe()](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/investigation_common.py#L108)

loops over:

- AOI
- index
- year
- month
- ecozone
- percentile

and writes rows with:

- `value`
- `year`
- `month`
- `classification`
- `ecozone_code`
- `summary_percentile`

For a historical `p99` trajectory plot, this is the point where the ecozone summary value would be the monthly `p99` NDVI value for that ecozone.

## 5. The neutral baseline seasonal curve is computed

The trajectory script is:

- [ecozone_monthly_trajectory_investigation.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/ecozone_monthly_trajectory_investigation.py)

It builds the baseline with:

- [baseline_monthly_stats()](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/investigation_common.py#L173)

This groups only `classification == "neutral"` rows by:

- `summary_percentile`
- `aoi`
- `index`
- `ecozone_code`
- `month`

and computes:

- `baseline_mean`
- `baseline_std`
- `baseline_count`

So the neutral baseline is month-specific, AOI-specific, ecozone-specific, and percentile-specific.

## 6. Wet-year and dry-year anomalies are computed relative to the neutral baseline

The trajectory script merges the monthly ecozone table with the neutral baseline table and computes:

- `anomaly = value - baseline_mean`

Relevant code:

- [ecozone_monthly_trajectory_investigation.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/ecozone_monthly_trajectory_investigation.py#L45)
- [ecozone_monthly_trajectory_investigation.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/ecozone_monthly_trajectory_investigation.py#L49)

It then filters to:

- `classification in ["wet", "dry"]`
- `month in GROWING_MONTHS`

which means the trajectory plot only shows wet and dry year anomalies for April through October.

Relevant code:

- [ecozone_monthly_trajectory_investigation.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/ecozone_monthly_trajectory_investigation.py#L55)
- [investigation_common.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/investigation_common.py#L36)

## 7. Monthly anomalies are averaged across wet years and across dry years

The script groups anomaly rows by:

- `summary_percentile`
- `aoi`
- `index`
- `ecozone_code`
- `classification`
- `month`

and computes:

- `mean_anomaly`
- `sd_anomaly`
- `n_year_months`

Relevant code:

- [ecozone_monthly_trajectory_investigation.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/ecozone_monthly_trajectory_investigation.py#L60)

For the historical `p99` NDVI plot, each plotted y-value is therefore:

`mean over wet years or dry years of (monthly ecozone p99 NDVI - neutral-year mean monthly ecozone p99 NDVI)`

for a specific:

- AOI
- ecozone
- month

## 8. The final figure is rendered

For each percentile and index, the script creates a two-panel figure:

- left panel = north AOI
- right panel = south AOI

The figure title is:

- `Monthly p{summary_percentile} Anomaly Trajectories vs Neutral Baseline | {index}`

Relevant code:

- [ecozone_monthly_trajectory_investigation.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/ecozone_monthly_trajectory_investigation.py#L117)

Line semantics:

- ecozone color = cool, intermediate, hot
- wet years = solid lines
- dry years = dashed lines
- x-axis = growing-season months
- y-axis = monthly percentile anomaly versus neutral baseline
- horizontal zero line = no anomaly relative to neutral baseline

The output filename pattern is:

- `trajectories_ndvi_p{summary_percentile}.png`

Relevant code:

- [ecozone_monthly_trajectory_investigation.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/ecozone_monthly_trajectory_investigation.py#L126)
- [ecozone_monthly_trajectory_investigation.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/ecozone_monthly_trajectory_investigation.py#L176)

## Mathematical Interpretation

For a given:

- AOI
- ecozone
- month
- year class

the plotted value is:

`mean_class_years( percentile_value_of_monthly_composite - mean_neutral_percentile_for_same_month )`

For the specific historical plot in question, the percentile would be `p99` and the index would be `NDVI`.

## Current Repository Caveat

The current code defines:

- [SUMMARY_PERCENTILES = [50, 75]](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/investigation_common.py#L29)

So the current repository directly reproduces:

- `Monthly p50 Anomaly Trajectories vs Neutral Baseline | NDVI`
- `Monthly p75 Anomaly Trajectories vs Neutral Baseline | NDVI`

The historical `p99` trajectory figure would follow the same derivation chain, but with percentile `99` included in that percentile list or in the corresponding earlier script version that produced the figure.

## Additional Information Needed For ArcGIS Pro Replication

For someone trying to replicate this process in ArcGIS Pro, the most important missing details are the exact raster handling rules, not the high-level plot idea.

The most relevant implementation details are:

- exact input locations and naming patterns for:
  - Sentinel scene caches
  - monthly composites
  - ecozone rasters
  - [wet_dry_years.csv](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/config/wet_dry_years.csv)
- grid specification:
  - CRS
  - cell size
  - raster dimensions
  - snap/alignment rules for each AOI
- pixel inclusion rules for Sentinel scene rasters:
  - which SCL classes are excluded
  - which classes are retained
- composite rule:
  - monthly raster is a per-pixel monthly maximum, not mean or median
- AOI masking rule:
  - outside-AOI pixels are `NaN`, but the full AOI-aligned bounding-box extent is retained
- ecozone summarization rule:
  - the plot uses ecozone-level percentile summaries, not zonal means
  - only valid ecozone pixels are used
  - ecozones with fewer than `MIN_PIXELS = 100` valid pixels become `NaN`
- baseline rule:
  - neutral baseline is computed separately for each AOI, index, ecozone, month, and percentile
- anomaly rule:
  - `anomaly = monthly percentile value - neutral monthly baseline mean`
- year grouping:
  - exact wet, neutral, and dry year membership by AOI
- month window:
  - trajectory plot uses only `Apr-Oct`
- missing data rule:
  - missing months are skipped, not filled or interpolated
- final aggregation rule:
  - monthly anomalies are averaged across wet years and across dry years
- plotting conventions:
  - ecozone colors
  - wet vs dry line styles
  - zero reference line
  - two-panel AOI layout

## Likely ArcGIS Pro Tool Sequence

A likely ArcGIS Pro replication chain would be:

1. Organize source scene rasters by AOI, index, year, and month.
2. Use `Make Mosaic Dataset` or another scene-management approach if needed.
3. Use `Project Raster` only if any inputs are not already on the AOI canonical grid.
4. Use `Extract by Mask` if AOI polygon clipping must be enforced at this step.
5. Use `Cell Statistics` with `MAXIMUM` to build each monthly composite from the scene rasters in the same month.
6. Use the ecozone raster as the zone dataset.
7. Use `Zonal Statistics as Table` or an equivalent zonal workflow to summarize each monthly composite by ecozone.
8. Export or join those zonal tables with the year classification table from [wet_dry_years.csv](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/config/wet_dry_years.csv).
9. Use `Summary Statistics` to compute the neutral baseline mean by:
   - AOI
   - index
   - ecozone
   - month
   - percentile/statistic
10. Join the neutral baseline table back onto the wet/dry monthly records.
11. Use `Add Field` and `Calculate Field` to compute:
    - `anomaly = value - baseline_mean`
12. Use `Summary Statistics` again to average anomalies across wet years and across dry years by:
    - AOI
    - ecozone
    - index
    - month
    - year class
13. Build the final line plot in ArcGIS Pro charts or export the final tables to another plotting environment.

## ArcGIS Pro Friction Point

The monthly max composite part is straightforward in ArcGIS Pro.

The difficult part is exact percentile-by-ecozone replication. The project plot depends on ecozone-level percentile summaries, and ArcGIS workflows are much cleaner for means than for repeated percentile extraction across many monthly rasters. If the exact percentile statistic is not directly available in the needed tool/version, that becomes the main bottleneck in a pure ArcGIS replication.

In short:

- monthly max composite replication is straightforward
- ecozone percentile extraction is the hard part
- table joins and anomaly calculations are manageable once the zonal percentile values exist

## Is This The Best Plot For The Comparison?

Not by itself. It is a valid plot, but it is not the clearest single comparison graphic.

Why it can be hard to interpret:

- it is anomaly relative to the neutral baseline, not relative to the all-years center
- it uses an upper-tail percentile, not an average ecozone response
- it is built from monthly maximum composites, which emphasize best observed conditions
- it overlays wet and dry trajectories on the same anomaly axis, which can make both lines above zero look counterintuitive

So the figure is useful as a supporting diagnostic, but it is not the easiest standalone plot for communicating how wet, neutral, and dry compare.

## Clearer Alternatives

### 1. Plot wet, neutral, and dry raw monthly curves directly

Instead of plotting anomalies relative to neutral, plot the actual monthly ecozone summary statistic for all three year classes:

- wet
- neutral
- dry

This is often easier to read because the viewer does not need to mentally decode the meaning of the zero line.

### 2. Plot wet-minus-neutral and dry-minus-neutral as side-by-side monthly summaries

This keeps the anomaly idea but makes the comparison more explicit. Month-by-month bars or paired markers are often easier to read than overlapping lines when the main goal is contrast.

### 3. Plot wet-minus-dry directly

If the main question is how strongly wet and dry diverge, this is often the clearest metric. A direct difference plot removes the extra interpretation step created by the neutral baseline.

### 4. Use a cumulative seasonal metric

For each AOI, ecozone, and index, summarize April through October into one number, such as:

- cumulative anomaly
- seasonal mean anomaly
- late-season mean anomaly

This is often better for summary tables and heatmaps.

### 5. Use fraction-below-baseline for stress interpretation

If the goal is to communicate suppression or stress extent, fraction-below-baseline is often more intuitive than a high-percentile anomaly trajectory.

### 6. Use mean or median instead of p99

If the intended message is general ecozone behavior, a central tendency metric is often clearer than `p99`. The `p99` statistic emphasizes the upper edge of the ecozone value distribution rather than typical conditions.

## Recommendation

The most defensible use of this figure is as a supporting plot, paired with a simpler comparison figure.

A clearer presentation set would be:

1. raw monthly class curves for `wet`, `neutral`, and `dry`
2. one compact seasonal summary metric such as cumulative anomaly
3. optionally, a `wet - dry` difference plot

In other words:

- good supporting figure
- not the best standalone figure if the goal is a clear wet vs neutral vs dry comparison
