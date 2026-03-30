# Analysis Gap Review

Date: 2026-03-29

Purpose: compare current project goals against scripts and validated archived outputs to identify what appears complete, what is only implemented, and what is still missing or unverified.

## Scope Used For This Review

- Goals and priorities from `docs/PROJECT_OVERVIEW.md`
- Session evidence from `logs/2026-03-28.md`
- Implemented scripts under `Analysis/`
- Archived validated outputs visible in `Results_cache1/`

This review distinguishes three states:

- `Validated archive evidence`: output is present in `Results_cache1/`
- `Implemented but not evidenced`: a script exists, but matching validated archived output was not found in `Results_cache1/`
- `Gap`: neither a validated output nor enough documentation exists to treat the goal as complete

## Goal Coverage Summary

### 1. Ecozone productivity using NDVI patterns

Status: `Mostly covered with validated archive evidence`

Evidence:

- `Analysis/Traits/Ecozone/ecozone_peak_productivity.py`
- `Results_cache1/figures/ecozone_ndvi_p95_p99_p100.png`
- `Results_cache1/figures/ecozone_ecological_space.png`

Remaining gap:

- The expected summary table `ecozone_peak_summary.xlsx` is not visible in `Results_cache1/`

### 2. Moisture dynamics using NDMI trends

Status: `Partially covered`

Validated archive evidence:

- `Results_cache1/figures/ecozone_ndmi_p95_p99_p100.png`
- `Results_cache1/figures/ecozone_ndmi_seasonal.png`
- `Results_cache1/figures/ecozone_ndmi_drought_response.png`
- `Results_cache1/figures/ecozone_moisture_seasons.png`
- `Results_cache1/figures/ecozone_moisture_amplitude.png`
- `Results_cache1/figures/ecozone_ndmi_amplitude_timeseries.png`

Implemented but not evidenced:

- `Analysis/Traits/Ecozone/ecozone_longterm_trend.py`

Main gap:

- The overview explicitly mentions NDMI trends, but the visible validated archive supports seasonal, drought-class, and amplitude analysis more clearly than explicit long-term trend deliverables

### 3. NDVI vs NDMI relationships

Status: `Covered with validated archive evidence`

Evidence:

- `Results_cache1/figures/ecozone_ecological_space.png`
- `Results_cache1/figures/ecozone_drought_ecological_space.png`
- `Results_cache1/figures/ecozone_seasonal_sync.png`

Gap:

- No archived table or narrative summary ties these figures back to a single interpretation document

### 4. Seasonal peak timing by ecozone

Status: `Covered with validated archive evidence`

Evidence:

- `Analysis/Traits/Ecozone/ecozone_seasonal_curves.py`
- `Results_cache1/figures/ecozone_ndvi_seasonal.png`
- `Results_cache1/figures/ecozone_ndmi_seasonal.png`
- `Results_cache1/figures/ecozone_evi_seasonal.png`
- `Results_cache1/figures/ecozone_seasonal_sync.png`

Gap:

- The summary workbook `ecozone_seasonal_summary.xlsx` is not visible in the validated archive

### 5. Identification of wet vs dry years using annual NDMI context

Status: `Covered with validated archive evidence`

Evidence:

- `Analysis/Traits/Ecozone/plot_moisture_year_classification.py`
- `Results_cache1/figures/moisture_year_classification.png`
- `Analysis/Traits/Ecozone/ecozone_drought_response.py`

Gap:

- The classification figure is archived, but the exact classification source file and version should be cited in a collaborator-facing summary

### 6. Peak vegetation response in wet vs dry years

Status: `Covered with validated archive evidence`

Evidence:

- `Results_cache1/figures/ecozone_ndvi_drought_response.png`
- `Results_cache1/figures/ecozone_ndmi_drought_response.png`
- `Results_cache1/figures/ecozone_evi_drought_response.png`
- `Results_cache1/figures/ecozone_drought_ecological_space.png`

Gap:

- The expected workbook `ecozone_drought_response.xlsx` is not visible in `Results_cache1/`

### 7. Moisture resilience by ecozone

Status: `Partially covered`

Evidence supporting the goal indirectly:

- `Results_cache1/figures/ecozone_ndmi_drought_response.png`
- `Results_cache1/figures/ecozone_moisture_amplitude.png`
- `Results_cache1/figures/ecozone_ndmi_amplitude_timeseries.png`

Main gap:

- There is no explicit resilience metric or dedicated resilience deliverable visible in the validated archive
- The goal may currently be inferred from drought-response and amplitude analyses rather than directly measured

### 8. Growing season moisture stress

Status: `Covered with validated archive evidence`

Evidence:

- `Analysis/Traits/Ecozone/ecozone_moisture_stress.py`
- `Results_cache1/figures/ecozone_moisture_amplitude.png`
- `Results_cache1/figures/ecozone_moisture_seasons.png`

Gap:

- No archived workbook `ecozone_moisture_stress.xlsx` is visible

### 9. Elevation gradients and drought response

Status: `Partially covered`

Validated archive evidence:

- `Analysis/Traits/Elevation/ecozone_elevation_gradient.py`
- `Results_cache1/figures/elevation_ndvi_gradient.png`
- `Results_cache1/figures/elevation_ndmi_gradient.png`

Implemented but not evidenced:

- `Analysis/Traits/Elevation/ecozone_elevation_gradient_landsat.py`

Main gap:

- Elevation gradients appear covered, but elevation-specific drought-response outputs are not clearly separated or documented in the validated archive

## Additional Implemented Work Outside The Main Goal List

### Aspect-stratified index analysis

Status: `Validated archive evidence exists, but roadmap role is unclear`

Evidence:

- `Analysis/Crosstab/Index/crosstab_aspect_index.py`
- `Results_cache1/figures/aspect_index_summary.png`
- `Results_cache1/figures/north_ndvi_aspect_timeseries.png`
- `Results_cache1/figures/north_ndmi_aspect_timeseries.png`
- `Results_cache1/figures/north_evi_aspect_timeseries.png`
- `Results_cache1/figures/south_ndvi_aspect_timeseries.png`
- `Results_cache1/figures/south_ndmi_aspect_timeseries.png`
- `Results_cache1/figures/south_evi_aspect_timeseries.png`

Gap:

- These outputs are not explicitly referenced in `PROJECT_OVERVIEW.md`, so their priority relative to the main ecozone roadmap is ambiguous

### Inter-trait crosstabs

Status: `Implemented but not evidenced`

Scripts present:

- `Analysis/Crosstab/InterTraits/crosstab_aspect_ftype.py`
- `Analysis/Crosstab/InterTraits/crosstab_aspect_fgroup.py`
- `Analysis/Crosstab/InterTraits/crosstab_ecozone_ftype.py`

Gap:

- No matching validated archived outputs were found in `Results_cache1/`
- These may be exploratory support analyses, but that role is not documented

### Raster and package production

Status: `Partially evidenced`

Evidence:

- `Results_cache1/TNC_Appalachian_Indices.lpkx`
- `Results_cache1/TNC_Appalachian_Indices_Monthly.lpkx`
- `Results_cache1/LAYER_PACKAGES.txt`

Gap:

- The archived package names do not fully align with the current ArcGIS packaging documentation and script naming
- `LAYER_PACKAGES.txt` still references the now-abandoned `_3_24` rebuild as if it were still the intended next step

## Cross-Cutting Gaps

### Documentation gaps

- There is no single inventory linking each goal to a script, output file, and trust status
- There is no collaborator-facing interpretation summary for the archived figures
- There is no recorded run history for when the archived figures were generated

### Reproducibility gaps

- Current scripts write to `Results/`, while trusted historical outputs live in `Results_cache1/`
- Expected Excel summary tables are referenced in several scripts but are not visible in the validated archive
- Script existence does not yet equal validated reproducibility against the authoritative `_3_4` cache

### Trust-boundary gaps

- The overview notes that some scripts were modified during the `_3_24` era, but there is not yet a per-script trust inventory
- Landsat analyses exist in code, but the validated archive is dominated by Sentinel-era figures

## Recommended Next Moves

- Build a per-analysis inventory with columns for goal, script, expected outputs, archived outputs found, and trust level
- Decide which analyses count as formal deliverables versus exploratory side analyses
- Update archived-output documentation to reflect the current decision to abandon `_3_24`
- Preserve or regenerate missing summary tables for the validated analyses
- Pick one representative analysis and re-run it against `_3_4` into a clearly labeled verification location to test present reproducibility
