# PRISM Annual Precipitation Classification

This workflow builds AOI-level wet / neutral / dry year labels from PRISM monthly precipitation.

Script:

```bash
python Analysis/DashboardPipeline/Climate/build_prism_growing_season_precip.py
```

Landsat-period run used for the current project tables:

```bash
python Analysis/DashboardPipeline/Climate/build_prism_growing_season_precip.py --start-year 1984 --end-year 2026 --download
```

## Data Source

- Source: PRISM Climate Group monthly precipitation.
- Variable: `ppt`.
- Preferred resolution: `4km`.
- Default download endpoint template:

```text
https://services.nacse.org/prism/data/get/us/{resolution}/{variable}/{yyyymm}
```

The script can either download missing monthly rasters with `--download` or read already-downloaded rasters from `--input-dir`.

## Precipitation Period

The current PRISM classification uses full calendar-year precipitation.

Monthly PRISM precipitation is summed for:

```text
January, February, March, April, May, June, July, August, September, October, November, December
```

This is documented in output columns as:

```text
Calendar-year total using all monthly PRISM ppt totals for Jan-Dec.
```

Earlier versions used a May-Sep approximation of the team-defined May 15-Sep 15 growing season. That behavior has been replaced for this PRISM table.

## AOI Extraction

The script reads the project AOI shapefile configured by `src.paths.project_path("tnc_aoi_shapefile")`.

AOI rows are mapped to:

- `north`: George Washington National Forest / Virginia AOI
- `south`: Smoky Mountains / Nantahala-region AOI

For each monthly PRISM raster, the workflow masks the raster to each AOI polygon and computes mean monthly precipitation in millimeters.

## Annual Aggregation

For each AOI and year:

```text
annual_precip_mm = sum(mean monthly PRISM ppt for Jan-Dec)
```

The script also writes a monthly extraction table for debugging missing or unusual inputs.

## Classification

Classification is AOI-relative and rank based:

- `wet`: top 20% of years within that AOI
- `dry`: bottom 20% of years within that AOI
- `neutral`: middle 60%

The exported metrics include:

- annual precipitation total
- anomaly from AOI mean
- AOI-relative z-score
- percentile/rank within AOI
- wet / neutral / dry classification

If an AOI-year has fewer than twelve monthly inputs, it is marked `incomplete` and reported at runtime.
Incomplete AOI-years are retained in the output tables but excluded from AOI mean, anomaly, z-score, percentile, and rank calculations.

For the May 2026 run, PRISM monthly data were available only through April 2026, so calendar-year 2026 is listed as `incomplete`.

## Outputs

Primary output:

```text
config/prism_growing_season_year_classes.csv
```

Supporting outputs:

```text
Results/tables/prism_growing_season_precip_summary.csv
Results/tables/prism_monthly_precip_extractions.csv
Results/tables/prism_growing_season_precip_summary.metadata.json
Results/figures/prism_growing_season_precip_classification.png
```

If static PNG export fails, the script attempts to write an HTML fallback next to the requested figure path.

## Comparison To Previous USDM Method

The previous project moisture context was `config/wet_dry_years.csv`, based on USDM / drought.gov-style county averages. That method used county/category summaries, including D1 and W1 percentages, and classified years with threshold logic on a net score.

The PRISM workflow is intended to replace or supplement that method because it is:

- raster based
- terrain aware at the PRISM grid scale
- extractable directly to the project AOI polygons
- continuous in millimeters rather than categorical county drought/wetness area

## Limitations

- Calendar-year precipitation is broader than the team-defined growing season and can include winter/spring/fall moisture effects that may not directly map to canopy growing-season response.
- AOI mean precipitation is an area summary; it does not capture within-AOI topographic gradients unless analyzed separately.
- Rank-based top/bottom 20% labels depend on the selected year range.
- PRISM monthly products and availability vary by date; recent years may be provisional or incomplete.
- The script uses `geopandas` to read AOI polygons when available. If `geopandas` is unavailable, it falls back to the GDAL `ogr2ogr` command.
