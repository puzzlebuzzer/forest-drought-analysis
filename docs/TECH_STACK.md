# Tech Stack

This document summarizes the technology stack for the Appalachian Ecozone-Vegetation Analysis dashboard and rebuild pipeline.

## Product Components

- Streamlit dashboard application
- Plotly interactive figures with CSV and PNG export support
- Precomputed dashboard summary tables
- Sentinel-2 and Landsat raster cache builders
- PRISM precipitation classification workflow
- Forest-community and thermal-ecozone preprocessing workflows
- Supporting analysis scripts for report figures and exploratory investigation

## Runtime Stack

Dashboard-only use requires:

- Python
- Streamlit
- pandas
- Plotly
- PyArrow / Parquet support
- PyYAML
- rasterio
- openpyxl
- Kaleido for Plotly PNG export

The dashboard is run from the repository root:

```bash
streamlit run dashboard_app.py
```

## Rebuild Stack

Full or partial data rebuild workflows use:

- Microsoft Planetary Computer STAC access
- `pystac-client`
- `planetary-computer`
- numpy
- geopandas
- shapely
- rasterio / GDAL-backed geospatial IO
- matplotlib for generated figures
- PyArrow for optimized and partitioned Parquet products

The project rebuild and deliverable-packaging workflow was developed in a Bash/Linux-style shell environment, primarily WSL on Windows. Equivalent environments should work if the Python and geospatial dependencies install correctly.

## Remote Data And Source Platforms

- Microsoft Planetary Computer
- STAC API
- Sentinel-2 Level-2A
- Landsat Collection 2 Level 2
- PRISM Climate Group monthly precipitation
- TNC Appalachian forest-community/ecozone source data
- Copernicus DEM GLO-30

## Data Formats

Runtime/dashboard formats:

- Parquet
- CSV manifests
- JSON/YAML configuration
- CSV and PNG dashboard exports
- HTML Plotly exports where enabled

Rebuild/provenance formats:

- GeoTIFF
- Shapefile
- CSV
- JSON
- YAML
- Excel workbooks for some supporting analyses
- ArcGIS layer packages (`.lpkx`) for optional GIS delivery

## Storage And Memory

Estimated resource tiers:

| Use level | Storage | RAM |
| --- | ---: | ---: |
| Dashboard-only package | 2-10 GB | 8 GB minimum, 16 GB recommended |
| Dashboard table rebuild from existing caches | 500-700+ GB | 16 GB minimum, 32 GB recommended |
| Full raw-to-dashboard workspace | 600 GB-1 TB+ | 32 GB recommended |

The large storage tiers are driven mainly by AOI satellite caches, generated rasters, and summary-table products. Dashboard-only users do not need raw rasters or AOI caches.

## Development Tools

- Git / GitHub
- Markdown documentation
- Python virtual environments
- WSL/Linux shell for development
- ArcGIS Pro for optional layer-package workflows

## Scientific Vocabulary

- AOI: north and south Appalachian study areas
- NDVI, NDMI, EVI: vegetation indices used by the dashboard
- Thermal ecozone: broad cool/intermediate/hot tier
- Forest-community group: TNC group tier within forest-community data
- Forest community: fine ecological category used as the canonical detailed vegetation class
- PRISM wet/neutral/dry years: external precipitation context
- Spatial percentile: percentile across valid pixels within an AOI or selected trait class
- Temporal percentile: percentile across scenes within a temporal bin

## Short Reusable Summary

The project uses a Python, Streamlit, and Plotly dashboard backed by precomputed pandas/PyArrow Parquet summary tables. Raster rebuild workflows use Microsoft Planetary Computer STAC data, rasterio/geopandas/numpy/pandas, AOI-aligned GeoTIFF caches, PRISM precipitation rasters, TNC forest-community traits, and Copernicus DEM-derived terrain data. Dashboard-only handoff requires the app code, configuration, Python dependencies, and generated `SummaryTables/dashboard_data/` products, not the raw satellite cache.
