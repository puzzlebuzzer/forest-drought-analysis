# build_mosaic_lpkx.py — ArcGIS Layer Package Builder

Creates time-enabled mosaic datasets from the annual-max GeoTIFFs and packages
them as **`Results/TNC_Appalachian_Indices.lpkx`** for delivery to Jean.

---

## What it builds

| Mosaic name | Source folder | Years |
|---|---|---|
| S2_NDVI_north | Results/rasters/annual_max/ndvi_north/ | 2017–2026 |
| S2_NDVI_south | Results/rasters/annual_max/ndvi_south/ | 2017–2026 |
| S2_NDMI_north | Results/rasters/annual_max/ndmi_north/ | 2017–2026 |
| S2_NDMI_south | Results/rasters/annual_max/ndmi_south/ | 2017–2026 |
| S2_EVI_north  | Results/rasters/annual_max/evi_north/  | 2017–2026 |
| S2_EVI_south  | Results/rasters/annual_max/evi_south/  | 2017–2026 |
| LS_NDVI_north | Results/rasters/landsat_annual_max/ndvi_north_landsat/ | 1984–2025 |
| LS_NDVI_south | Results/rasters/landsat_annual_max/ndvi_south_landsat/ | 1984–2025 |
| LS_NDMI_north | … | 1984–2025 |
| LS_NDMI_south | … | 1984–2025 |
| LS_EVI_north  | … | 1984–2025 |
| LS_EVI_south  | … | 1984–2025 |

Landsat mosaics are skipped automatically if the cache hasn't been built yet.
Re-run the script once `build_landsat_annual_composites.py` finishes to add them.

Each mosaic is **time-enabled** with an `AcquiredDate` field set to July 1 of each
year (peak growing season proxy), so Jean can use the ArcGIS Pro **Time Slider**
to animate vegetation change across years.

---

## How to run

This script **requires the ArcGIS Pro Python environment** (not the base Anaconda env).

### Option A — from ArcGIS Pro Python Command Prompt
1. Open **ArcGIS Pro Python Command Prompt** (Start → ArcGIS → Python Command Prompt)
2. Navigate to the project root:
   ```
   cd /d "D:\path\to\your\project"
   ```
3. Run:
   ```
   python Analysis/arcgis/build_mosaic_lpkx.py
   ```

### Option B — from PowerShell / cmd using the full Python path
```powershell
& "C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe" `
    Analysis/arcgis/build_mosaic_lpkx.py
```

### Option C — from within ArcGIS Pro
1. Open the **Catalog pane** → right-click the script → **Run**
   (or use the Python window: `exec(open('Analysis/arcgis/build_mosaic_lpkx.py').read())`)

---

## Outputs

| File | Description |
|---|---|
| `Results/mosaic_scratch.gdb` | File geodatabase containing all mosaic datasets |
| `Results/TNC_Appalachian_Indices.lpkx` | Packaged layer file for delivery |

The `.lpkx` consolidates the mosaics and their source rasters into a single
portable file (~MB to GB depending on how many Landsat scenes are included).

---

## What Jean does with the .lpkx

1. **Open ArcGIS Pro** → drag the `.lpkx` onto the map, or
   **Catalog pane** → right-click → **Unpack** → add to map
2. Open the **Time Slider** (Map tab → Time group → Time Slider)
3. Set step size to **1 Year**
4. Press **Play** — the raster animates by year
5. Switch between NDVI / NDMI / EVI layers to compare indices
6. Sentinel-2 layers show 2017–2026; Landsat layers show 1984–2025

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `arcpy not available` | Run using ArcGIS Pro Python, not base conda |
| `No .tif files found` | Check that the annual-max scripts have been run |
| PackageLayer fails with size error | Reduce AOIs or indices; or use `convert_data="NO_CONVERT"` to reference rasters in-place |
| Time slider shows no data | Open the mosaic's Properties → Time → confirm Start/End time field |
