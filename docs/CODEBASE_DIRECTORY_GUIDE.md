# Codebase Directory Guide

This guide maps the public repository and separates active dashboard/data-pipeline code from generated outputs, exploratory work, and archive material.

## Active Project Structure

| Path | Role | Active status |
| --- | --- | --- |
| `dashboard_app.py` | Streamlit dashboard entry point | Active runtime. |
| `src/` | Shared application and pipeline modules | Active core code. |
| `Cache/` | Sentinel-2 and Landsat cache builders/auditors | Active for full data rebuilds; not needed for dashboard-only users. |
| `Analysis/DashboardPipeline/TableFactory/` | Dashboard summary table builders and parquet optimization | Active preprocessing pipeline. |
| `Analysis/DashboardPipeline/Climate/` | PRISM annual precipitation classification workflow | Active supplemental climate/context workflow. |
| `Preprocessing/Traits/ForestCommunity/` | Forest community raster preparation | Active shared preprocessing. |
| `Preprocessing/Traits/Terrain/` | DEM, slope/aspect, and terrain trait preparation | Active shared preprocessing. |
| `config/` | Path and classification config files | Active configuration; local paths may need editing on other machines. |
| `docs/` | Curated project and pipeline documentation | Active documentation. Historical notes are not part of the public documentation set. |
| `tests/` | Dashboard loader/figure tests | Active validation. |
| `requirements-dashboard.txt` | Dashboard Python dependencies | Active handoff/install file. |

## `src/` Modules

| File | Role |
| --- | --- |
| `paths.py` | Resolves project paths from `config/project_paths.yaml`. |
| `aoi.py` | AOI configuration helpers and AOI shapefile access. |
| `cli.py` | Shared CLI argument helpers. |
| `labels.py` | Label dictionaries and small display-label corrections. |
| `landsat.py` | Landsat cache loading and aligned trait raster helpers. |
| `sentinel.py` | Sentinel cache loading and aligned trait raster helpers. Important for thermal-ecozone table rebuilds. |
| `forest_community.py` | Forest community inventory/raster loading and Landsat/Sentinel alignment helpers. |
| `table_factory.py` | Base scene catalog, scene summaries, temporal summaries. |
| `table_factory_ecozone.py` | Thermal-ecozone scene/temporal summary builders. |
| `table_factory_forest_community.py` | Forest community and forest-community-group summaries. |
| `dashboard_schema.py` | Canonical dashboard schema and config dataclass. |
| `dashboard_data.py` | Dashboard table loading, filtering, parquet/manifest handling. |
| `dashboard_figures.py` | Plotly figure construction, PRISM year bands, exports. |
| `ecozone_scenelevel.py` | Older scene-level thermal-ecozone analysis helper. The active table factory still reuses its shared thermal-ecozone constants. |

## Active Data Rebuild Flow

Run these commands from the repository root after activating a Python environment with the project dependencies installed. A local `.venv` is the recommended setup for reproducible dashboard use.

1. Acquire/cache satellite index rasters:
   - `Cache/build_sentinel_cache.py`
   - `Cache/build_landsat_cache.py`
2. Prepare forest-community raster inputs:
   - `Preprocessing/Traits/ForestCommunity/prep_forest_community.py`
   - `Preprocessing/Traits/Terrain/build_elevation_cache.py`
3. Download or ingest PRISM monthly precipitation and build AOI-level precipitation classifications:
   - `Analysis/DashboardPipeline/Climate/build_prism_growing_season_precip.py`
4. Build dashboard base summaries:
   - `Analysis/DashboardPipeline/TableFactory/build_dashboard_tables.py`
5. Build thermal-ecozone summaries:
   - `Analysis/DashboardPipeline/TableFactory/build_dashboard_ecozone_tables.py`
6. Build forest-community and forest-community-group summaries:
   - `Analysis/DashboardPipeline/TableFactory/build_dashboard_forest_community_tables.py`
7. Optimize/write dashboard parquet datasets:
   - `Analysis/DashboardPipeline/TableFactory/optimize_dashboard_ecozone_parquet.py`
8. Run dashboard:
   - `streamlit run dashboard_app.py`

## Dashboard-Only Runtime Assets

A non-programmer dashboard package generally needs only:

| Path | Role |
| --- | --- |
| `dashboard_app.py` | App entry point. |
| `src/` | Runtime modules used by dashboard. |
| `config/prism_growing_season_year_classes.csv` | PRISM year-band/classification table. |
| `SummaryTables/dashboard_data/partitioned_parquet/` | Main dashboard-ready summary data. |
| `SummaryTables/dashboard_data/*_manifest.csv` and `*_manifest.parquet` | Filter/year-range metadata for dashboard. |
| `requirements-dashboard.txt` | Install dependencies. |

Dashboard-only users do not need raw rasters, Planetary Computer access, AOI caches, or table factory scripts unless they are rebuilding data.

## Generated Output Directories

| Path | Role | Cleanup guidance |
| --- | --- | --- |
| `SummaryTables/dashboard_data/` | Current generated dashboard tables/manifests/parquet | Keep for dashboard use; reproducible from source caches but expensive to rebuild. |
| `SummaryTables/dashboard_data/partitioned_parquet/` | Preferred dashboard data store | Keep for handoff/runtime. |
| `SummaryTables/dashboard_data/optimized_parquet/` | Optimized intermediate parquet outputs | Useful but less central than `partitioned_parquet/`. |
| `SummaryTables/dashboard_data/archive/` | Previous generated table versions, if present | Optional rollback material; not needed for dashboard-only packages. |
| `SummaryTables/dashboard_data_test_*` | Test/dev generated table outputs, if present | Not needed for public release or dashboard-only packages. |
| `Results/rasters/` | Generated/intermediate rasters, including PRISM rasters | Needed for rebuild/provenance, not dashboard-only. |
| `Results/figures/` | Generated figures and exports | Keep selected report/deliverable figures; not required for dashboard runtime. |
| `Results/0-CacheBaseData/`, `1_Foundation/`, `2_Anomaly_Onset/`, etc. | Organized analysis deliverables, if present | Report/deliverable outputs, not dashboard runtime. |

## Exploratory Or Legacy Code

These directories contain useful analysis history but are not currently part of the core dashboard runtime:

| Path | Notes |
| --- | --- |
| `Analysis/SupportingAnalyses/EcozoneInvestigations/` | Many ecological investigation scripts. Some remain useful for report figures, but most are not required by dashboard runtime. |
| `Analysis/SupportingAnalyses/Terrain/` | Elevation/aspect/gradient supporting analyses. Not dashboard runtime. |
| `Analysis/SupportingAnalyses/SlopeAspect/Crosstab/` | Cross-tabulation analyses. Not dashboard runtime. |
| `Analysis/SupportingAnalyses/CompositesAndAnomalies/` | Annual/monthly composites and anomaly rasters. Useful for map products, not dashboard table runtime. |
| `Analysis/SupportingAnalyses/Diagnostics/` | Diagnostic/demo products. Not runtime. |
| `Analysis/SupportingAnalyses/ArcGISExports/` | ArcGIS Pro layer-package workflow. Optional delivery path, not dashboard runtime. |
| `Charts/` | Legacy/static chart image assets, if present. Active scripts were moved into dashboard or supporting-analysis folders. |
| `docs/Archived/` | Historical documentation archive, if present. Public docs are the curated files directly under `docs/`. |
| `Archived/LegacyTraits/` | Older duplicate trait scripts, if present. Current trait prep lives under `Preprocessing/Traits/`. |
| `Archived/` | Archived Python analyses, if present. |
| `ScriptSnapshots/` | Historical snapshots. |

## Large Data And Reproducibility Notes

- `SummaryTables/dashboard_data/` is the main dashboard-runtime data store.
- Satellite caches, `Results/rasters/`, `Results/0-CacheBaseData/`, and other `Results/` analysis outputs can dominate storage for rebuild/report workflows.
- Dashboard-only use is about 2-10 GB and 8-16 GB RAM.
- Dashboard table rebuild from existing caches is about 500-700+ GB and 16-32 GB RAM.
- Full raw-to-dashboard rebuild is about 600 GB-1 TB+ and 32 GB RAM.
- Dashboard-only runtime can be packaged much smaller by including only code, manifests, and partitioned parquet summary tables.
- Optional ArcGIS Pro export scripts, if used, live under `Analysis/SupportingAnalyses/ArcGISExports/` and are separate from the Streamlit dashboard runtime.

## Release Hygiene

Before a formal release, review `git status` from the repository root and keep active source/docs separate from generated outputs. Generated dashboard tables, figures, rasters, deliverable packages, Streamlit settings, archived notes, and chart image outputs are intentionally ignored by git.
