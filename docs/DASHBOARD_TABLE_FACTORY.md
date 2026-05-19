# Dashboard Table Factory

This project now separates heavy raster preprocessing from dashboard use.

## Purpose

The table factory regenerates clean dashboard-ready tables from:

- aligned Sentinel-2 index rasters
- aligned Landsat index rasters
- cache manifests

It does not rely on legacy Excel outputs as source data.

## Architectural stance

- Existing caches and manifests are treated as source material.
- Dashboard products are regenerated into a new canonical schema.
- The dashboard reads only precomputed tables.
- Scene masking is fixed by sensor-specific baseline definitions.

## Canonical masks

Sentinel-2:

- `s2_scl4_veg_v1`
- `SCL = 4` vegetation pixels only

Landsat:

- `ls_clear_terrestrial_v1`
- QA-based clear terrestrial mask excluding fill, dilated cloud, cirrus, cloud, cloud shadow, snow, and water

These are preserved as metadata fields, not exposed as user-facing toggles.

## Products

Default output directory:

- `Results/tables/dashboard_data/`

Products:

- `scene_catalog.csv`
- `scene_summary.csv`
- `temporal_summary.csv`
- optional parallel ecozone tables:
  - `scene_summary_ecozone.csv`
  - `temporal_summary_ecozone.csv`
  - `data_dictionary_ecozone.md`
- parquet versions when a parquet engine is installed
- `data_dictionary.md`

## Processing flow

1. Read cache manifests and build a canonical scene catalog.
2. Read aligned rasters scene-by-scene and compute spatial percentiles.
3. Store scene-level summaries in long form.
4. Re-bin scene summaries into `scene`, `half_month`, and `month` products.
5. Apply cloud-threshold filtering at summary time, not by recomputing rasters.
6. Use `growing_season_day` in `scene_summary` so the dashboard can render a normalized growing-season overlay without a separate table.

## Run

From the `Python/` directory:

```bash
python Analysis/TableFactory/build_dashboard_tables.py
```

Ecozone table prep:

```bash
python Analysis/TableFactory/build_dashboard_ecozone_tables.py scene-summary
python Analysis/TableFactory/build_dashboard_ecozone_tables.py temporal-summary
python Analysis/TableFactory/build_dashboard_ecozone_tables.py data-dictionary
```

Optional dev limiter:

```bash
python Analysis/TableFactory/build_dashboard_tables.py --limit-scenes-per-group 2
```

Optional year window:

```bash
python Analysis/TableFactory/build_dashboard_tables.py --start-year 1990 --end-year 1999 scene-summary
```

## Staged subcommands

You can build and test each artifact boundary independently:

```bash
python Analysis/TableFactory/build_dashboard_tables.py scene-catalog
python Analysis/TableFactory/build_dashboard_tables.py scene-summary --limit-scenes-per-group 5
python Analysis/TableFactory/build_dashboard_tables.py temporal-summary
python Analysis/TableFactory/build_dashboard_tables.py data-dictionary
```

Decade-at-a-time example:

```bash
python Analysis/TableFactory/build_dashboard_tables.py --start-year 1990 --end-year 1999 scene-catalog
python Analysis/TableFactory/build_dashboard_tables.py --start-year 1990 --end-year 1999 scene-summary
python Analysis/TableFactory/build_dashboard_tables.py --start-year 1990 --end-year 1999 temporal-summary
```

Notes:

- `scene-summary` reuses `scene_catalog` if it already exists, or builds it automatically.
- `temporal-summary` reuses `scene_summary` if it already exists, or builds it automatically.
- Running the CLI with no subcommand is equivalent to `all`.

## Stretch goals

- Add a reproducible growing-season animation export workflow that renders one highlighted year at a time and writes a StoryMap-friendly GIF or MP4.
- Update the Sentinel baseline used for dashboard-ready tables to a refreshed trusted version.
- Add an RNG view from the scene catalog to the dashboard that matches the selected layer.
- Final dashboard test/polish pass.
