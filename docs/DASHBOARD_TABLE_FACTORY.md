# Dashboard Table Factory

The table factory builds the precomputed summary tables used by the Streamlit dashboard. It separates expensive raster processing from interactive dashboard use: the dashboard reads summary tables, manifests, and Parquet datasets rather than raw Sentinel-2 or Landsat rasters.

## Purpose

The dashboard table pipeline converts aligned satellite index caches and trait rasters into dashboard-ready products for:

- whole-AOI summaries
- thermal ecozone summaries
- forest-community-group summaries
- forest community summaries

The dashboard can then filter by AOI, sensor, vegetation index, year, cloud threshold, spatial percentile, temporal aggregation, thermal ecozone, forest-community group, and forest community without reading rasters at runtime.

## Inputs

Primary inputs:

- Sentinel-2 and Landsat index rasters stored in AOI-aligned caches
- cache manifests for scene date, cloud metadata, platform, path/row, and file provenance
- AOI-aligned thermal ecozone rasters
- AOI-aligned forest-community rasters and inventories
- project path configuration in `config/project_paths.yaml`

The table factory treats existing rasters and manifests as source material. It does not use legacy Excel outputs as source data.

## Output Location

Default output directory:

```text
SummaryTables/dashboard_data/
```

This directory is generated data and is ignored by git. It is required for dashboard runtime unless a dashboard package provides an equivalent data directory.

## Main Products

Base products:

- `scene_catalog.csv` / `scene_catalog.parquet`
- `scene_summary.csv` / `scene_summary.parquet`
- `temporal_summary.csv` / `temporal_summary.parquet`
- `data_dictionary.md`

Thermal ecozone products:

- `scene_summary_ecozone.csv` / `.parquet`
- `temporal_summary_ecozone.csv` / `.parquet`
- `scene_summary_ecozone_manifest.csv` / `.parquet`
- `temporal_summary_ecozone_manifest.csv` / `.parquet`
- `data_dictionary_ecozone.md`

Forest-community products:

- `scene_summary_forest_community.parquet`
- `temporal_summary_forest_community` partitioned Parquet dataset
- `scene_summary_forest_community_manifest.csv` / `.parquet`
- `temporal_summary_forest_community_manifest.csv` / `.parquet`
- `data_dictionary_forest_community.md`

Forest-community-group products:

- `scene_summary_forest_ecozone_group` partitioned Parquet dataset
- `temporal_summary_forest_ecozone_group` partitioned Parquet dataset
- `scene_summary_forest_ecozone_group_manifest.csv` / `.parquet`
- `temporal_summary_forest_ecozone_group_manifest.csv` / `.parquet`

Optimized runtime products:

- `optimized_parquet/`
- `partitioned_parquet/`

The dashboard prefers partitioned Parquet when present, then optimized Parquet, then root-level Parquet/CSV fallbacks.

## Processing Flow

1. Build a canonical scene catalog from Sentinel-2 and Landsat manifests.
2. Read aligned index rasters scene-by-scene.
3. Compute scene-level spatial percentiles for whole AOIs and trait classes.
4. Re-bin scene summaries into `scene`, `half_month`, and `month` temporal products.
5. Apply cloud-threshold filtering during summary/table construction rather than recomputing rasters.
6. Add `growing_season_day` so the dashboard can render normalized May 15-Sep 15 growing-season overlays.
7. Build manifests describing available layer combinations and year ranges.
8. Optimize large tables into sorted and partitioned Parquet datasets for dashboard filtering.

## Canonical Masks

Sentinel-2:

- `s2_scl4_veg_v1`
- accepted dashboard baseline uses `SCL = 4` vegetation pixels only
- accepted dashboard baseline was not harmonized for the early-2022 Sentinel-2 processing-baseline shift

Landsat:

- `ls_clear_terrestrial_v1`
- QA-based mask semantics are documented in `DATA_METHODS_SHORT.md` and `DATA_PROVENANCE_AND_CACHE_CHARACTERISTICS.md`

Mask identifiers are stored as metadata fields. They are fixed dataset definitions, not dashboard user controls.

Interpretation note: abrupt Sentinel-2 changes around 2022 may reflect the uncorrected Sentinel-2 processing-baseline shift in the accepted dashboard cache, not necessarily canopy response. Use Landsat overlap, PRISM context, and cross-year patterning when interpreting that period.

## Key Columns

Common dashboard columns include:

- `analysis_scope`
- `sensor`
- `aoi`
- `index`
- `date`, `year`, `doy`, `growing_season_day`
- `season_filter`
- `temporal_agg`
- `temporal_percentile`
- `spatial_percentile`
- `cloud_threshold`
- `pixel_mask_id`
- `n_pixels`
- `valid_pixel_fraction`
- `n_scenes`
- `value`

Thermal ecozone rows add:

- `ecozone_code`
- `ecozone_label`

Forest community rows add:

- `forest_community_code`
- `forest_community_display_code`
- `forest_community_label`
- `forest_community_source_dataset`
- `forest_community_source_value`
- `forest_community_source_key`

Forest-community-group rows use:

- `ecozone_group_code`
- `ecozone_group_label`
- `ecozone_group_raw`

The `ecozone_group_*` names are retained in the schema for compatibility with existing code, but these fields represent TNC forest-community groups rather than the dashboard's broad thermal ecozone tier.

## Common Commands

Run commands from the repository root.

Build base scene and temporal products:

```bash
python Analysis/DashboardPipeline/TableFactory/build_dashboard_tables.py
```

Build thermal ecozone products:

```bash
python Analysis/DashboardPipeline/TableFactory/build_dashboard_ecozone_tables.py all
```

Build forest community and forest-community-group products:

```bash
python Analysis/DashboardPipeline/TableFactory/build_dashboard_forest_community_tables.py all
```

Optimize large segment tables for dashboard runtime:

```bash
python Analysis/DashboardPipeline/TableFactory/optimize_dashboard_ecozone_parquet.py
```

Optimize selected stems only:

```bash
python Analysis/DashboardPipeline/TableFactory/optimize_dashboard_ecozone_parquet.py --stem scene_summary_forest_community --stem temporal_summary_forest_community
```

Optional development limiter:

```bash
python Analysis/DashboardPipeline/TableFactory/build_dashboard_tables.py --limit-scenes-per-group 2
```

Optional year window:

```bash
python Analysis/DashboardPipeline/TableFactory/build_dashboard_tables.py --start-year 1990 --end-year 1999 scene-summary
```

## Validation

Useful checks:

```bash
python Analysis/DashboardPipeline/TableFactory/check_dashboard_readiness.py
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -m unittest tests.test_dashboard_loader
```

The readiness check validates expected table availability. The unit test exercises dashboard loading/filtering behavior against sample data.

## Runtime Boundary

The Streamlit dashboard uses the generated tables and PRISM classification file at runtime. It does not:

- download Sentinel-2 or Landsat scenes
- read raw GeoTIFF rasters
- rebuild trait rasters
- recompute spatial or temporal percentiles

That boundary is what makes the dashboard package small enough for non-programmer handoff compared with the full rebuild workspace.
