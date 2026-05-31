# Codebase Directory Guide

This guide maps the main project directories and separates active dashboard/data-pipeline code from generated outputs, exploratory work, and archive material.

## Project Root

| Path | Role | Notes |
| --- | --- | --- |
| `AOI/` | Large local geospatial source/cached AOI assets | Contains north/south AOI caches, TNC forest community source data, AOI layer-package material. Required for rebuilding some rasters/tables, not required for dashboard-only use. |
| `Python/` | Active Python project and dashboard repo | Main codebase for acquisition, preprocessing, dashboard table generation, Streamlit dashboard, docs, tests, and generated results. |
| `Screenshots/` | Manual screenshots | Used for UI review/debugging. Not part of runtime. |
| `Tasks/` | Project task notes | Planning/history. Not runtime. |
| `Archived/` | Older root-level project material | Archive/reference only unless explicitly revived. |
| `Notebooks/` | Notebook experiments | Exploratory. Not part of active dashboard pipeline. |
| `MangroveDocumentation/`, `practiceShapeNDVI/`, `tnc-forest-analysis/` | Older or adjacent workspaces | Treat as external/legacy context, not active dashboard flow. |
| `*.pdf`, `*.png`, `*.zip` at root | Reports, screenshots, packaged AOI artifacts | Static deliverables/reference. |

## Active Python Project

| Path | Role | Active status |
| --- | --- | --- |
| `Python/dashboard_app.py` | Streamlit dashboard entry point | Active runtime. |
| `Python/src/` | Shared application and pipeline modules | Active core code. |
| `Python/Cache/` | Sentinel-2 and Landsat cache builders/auditors | Active for full data rebuilds; not needed for dashboard-only users. |
| `Python/Analysis/TableFactory/` | Dashboard summary table builders and parquet optimization | Active preprocessing pipeline. |
| `Python/Analysis/Traits/Forest/` | Forest community raster preparation | Active when rebuilding forest-community inputs. |
| `Python/Analysis/Climate/` | PRISM and satellite-index year classification workflows | Active supplemental climate/context workflows. |
| `Python/config/` | Path and classification config files | Active configuration; local paths may need editing on other machines. |
| `Python/docs/` | Project and pipeline documentation | Active documentation. |
| `Python/tests/` | Dashboard loader/figure tests | Active validation. |
| `Python/requirements-dashboard.txt` | Dashboard Python dependencies | Active handoff/install file. |

## `Python/src/` Modules

| File | Role |
| --- | --- |
| `paths.py` | Resolves project paths from `config/project_paths.yaml`. |
| `aoi.py` | AOI configuration helpers and AOI shapefile access. |
| `cli.py` | Shared CLI argument helpers. |
| `labels.py` | Label dictionaries and small display-label corrections. |
| `landsat.py` | Landsat cache loading and aligned trait raster helpers. |
| `sentinel.py` | Sentinel cache loading and aligned trait raster helpers. Important for ecozone table rebuilds. |
| `forest_community.py` | Forest community inventory/raster loading and Landsat/Sentinel alignment helpers. |
| `table_factory.py` | Base scene catalog, scene summaries, temporal summaries. |
| `table_factory_ecozone.py` | Thermal-ecozone scene/temporal summary builders. |
| `table_factory_forest_community.py` | Forest community and forest-community-group summaries. |
| `dashboard_schema.py` | Canonical dashboard schema and config dataclass. |
| `dashboard_data.py` | Dashboard table loading, filtering, parquet/manifest handling. |
| `dashboard_figures.py` | Plotly figure construction, PRISM year bands, exports. |
| `ecozone_scenelevel.py` | Shared scene-level ecozone analysis helper for older/diagnostic scripts. |

## Active Data Rebuild Flow

1. Acquire/cache satellite index rasters:
   - `Python/Cache/build_sentinel_cache.py`
   - `Python/Cache/build_landsat_cache.py`
2. Prepare forest-community raster inputs:
   - `Python/Analysis/Traits/Forest/prep_forest_community.py`
3. Build dashboard base summaries:
   - `Python/Analysis/TableFactory/build_dashboard_tables.py`
4. Build thermal-ecozone summaries:
   - `Python/Analysis/TableFactory/build_dashboard_ecozone_tables.py`
5. Build forest-community and forest-community-group summaries:
   - `Python/Analysis/TableFactory/build_dashboard_forest_community_tables.py`
6. Optimize/write dashboard parquet datasets:
   - `Python/Analysis/TableFactory/optimize_dashboard_ecozone_parquet.py`
7. Run dashboard:
   - `streamlit run Python/dashboard_app.py` from the `Python/` directory, or `streamlit run dashboard_app.py` after changing into `Python/`.

## Dashboard-Only Runtime Assets

A non-programmer dashboard package generally needs only:

| Path | Role |
| --- | --- |
| `Python/dashboard_app.py` | App entry point. |
| `Python/src/` | Runtime modules used by dashboard. |
| `Python/config/prism_growing_season_year_classes.csv` | PRISM year-band/classification table. |
| `Python/Results/tables/dashboard_data/partitioned_parquet/` | Main dashboard-ready summary data. |
| `Python/Results/tables/dashboard_data/*_manifest.csv` and `*_manifest.parquet` | Filter/year-range metadata for dashboard. |
| `Python/requirements-dashboard.txt` | Install dependencies. |

Dashboard-only users do not need raw rasters, Planetary Computer access, AOI caches, or table factory scripts unless they are rebuilding data.

## Generated Output Directories

| Path | Role | Cleanup guidance |
| --- | --- | --- |
| `Python/Results/tables/dashboard_data/` | Current generated dashboard tables/manifests/parquet | Keep for dashboard use; reproducible from source caches but expensive to rebuild. |
| `Python/Results/tables/dashboard_data/partitioned_parquet/` | Preferred dashboard data store | Keep for handoff/runtime. |
| `Python/Results/tables/dashboard_data/optimized_parquet/` | Optimized intermediate parquet outputs | Useful but less central than `partitioned_parquet/`. |
| `Python/Results/tables/dashboard_data/archive/` | Previous generated table versions | Archive/delete only after confirming no rollback needed. |
| `Python/Results/tables/dashboard_data_test_*` | Test/dev generated table outputs | Safe archive/delete candidates. |
| `Python/Results/rasters/` | Generated/intermediate rasters, including PRISM rasters | Needed for rebuild/provenance, not dashboard-only. |
| `Python/Results/figures/` | Generated figures and exports | Keep selected deliverables; old exploratory figures can be archived. |
| `Python/Results/0-CacheBaseData/`, `1_Foundation/`, `2_Anomaly_Onset/`, etc. | Organized analysis deliverables | Report/deliverable outputs, not dashboard runtime. |

## Exploratory Or Legacy Code

These directories contain useful analysis history but are not currently part of the core dashboard runtime:

| Path | Notes |
| --- | --- |
| `Python/Analysis/Traits/Ecozone/` | Many ecological investigation scripts. Some remain useful for report figures, but most are not required by dashboard runtime. |
| `Python/Analysis/Traits/Elevation/` | Elevation/aspect/gradient analyses and prep. Rebuild support or exploratory analysis. |
| `Python/Analysis/Crosstab/` | Cross-tabulation analyses. Not dashboard runtime. |
| `Python/Analysis/Indices/` | Annual/monthly composites and anomaly rasters. Useful for map products, not dashboard table runtime. |
| `Python/Analysis/Diagnostics/` | Diagnostic/demo products. Not runtime. |
| `Python/Analysis/arcgis/` | ArcGIS Pro layer-package workflow. Optional delivery path, not dashboard runtime. |
| `Python/Charts/` | Static chart/export scripts. Some export support, mostly legacy/auxiliary. |
| `Python/Traits/` | Older duplicate trait scripts; `Python/Analysis/Traits/` appears more canonical now. |
| `Python/Archived/` | Archived Python analyses. |
| `Python/ScriptSnapshots/` | Historical snapshots. |

## Large Data And Reproducibility Notes

- `AOI/` and `Python/Results/` dominate storage.
- Raw/cached raster rebuild capability is hundreds of GB.
- Dashboard-only runtime can be packaged much smaller by including only code, manifests, and partitioned parquet summary tables.
- The dashboard does not use QGIS or DuckDB currently.
- ArcGIS Pro appears only in optional layer-package workflows under `Python/Analysis/arcgis/`.

## Known Current Housekeeping Flags

At the time this guide was created, the nested `Python/` git repo had unrelated dirty/generated files. Important examples:

- `Python/src/sentinel.py` and `Python/src/ecozone_scenelevel.py` were untracked but referenced by active or semi-active code.
- Several generated result files were untracked.
- Several old figure files were marked deleted.

Before a formal release, review `git status` and commit active source files separately from generated outputs.
