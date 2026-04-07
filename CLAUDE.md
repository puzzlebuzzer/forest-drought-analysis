# TNC Appalachian Terrain–Vegetation Project — Claude Context

This file is read at the start of every session. It captures decisions, conventions, and hard-won fixes so they don't have to be re-explained after context resets.

---

## What This Project Is

Capstone analysis for TNC comparing vegetation productivity and moisture (NDVI, NDMI, EVI) between north- and south-facing slopes across two Appalachian study areas, using Sentinel-2 (2017–present, 10 m) and Landsat C2 (1984–present, 30 m) imagery from Microsoft Planetary Computer. Terrain stratification uses DEM-derived aspect masks. Forest composition from FIA Forest Type Group rasters. Final deliverables are figures, rasters, and ArcGIS layer packages.

---

## Study Areas (AOIs)

| AOI   | Name                        | Cache root name  |
|-------|-----------------------------|------------------|
| north | George Washington NF (GWNF) | GWNF_cache       |
| south | Great Smoky Mountains       | Smoky_cache      |

Shapefile: `../AOI/TNC_AOI_LayerPkg/TNC_AOIs.shp`, filtered by `LscapeID`.

---

## Cache Versioning Convention

Caches now use unsuffixed active directories:

- `GWNF_cache/` and `Smoky_cache/` — the current cache roots for Sentinel, Landsat, and trait rasters

**Build scripts** use unsuffixed base paths from `project_paths.yaml`. A suffix can still be supplied explicitly, but the default is now no suffix. The yaml keys for this are `north_s2_build_base`, `south_s2_build_base`, `north_landsat_build_base`, `south_landsat_build_base`.

**Analysis scripts** point directly at the unsuffixed cache paths (`north_index_cache_root`, `north_landsat_index_root`, etc.).

---

## Cache Directory Structure

```
GWNF_cache/
├── s2/
│   └── indices/
│       ├── NDVI/       *.tif + cache_manifest.json
│       ├── NDMI/
│       ├── EVI/
│       └── SCL/
└── landsat/
    └── indices/
        ├── NDVI/
        ├── NDMI/
        ├── EVI/
        └── QA_PIXEL/
```

Traits (terrain, forest, ecozone) live under the unsuffixed cache roots and do not need to be rebuilt.

---

## Build Scripts

| Script                        | Data source      | Default suffix | Run command |
|-------------------------------|------------------|----------------|-------------|
| `Cache/build_sentinel_cache.py`  | Sentinel-2 L2A   | none           | `python Cache/build_sentinel_cache.py` |
| `Cache/build_landsat_cache.py`   | Landsat C2 L2    | none           | `python Cache/build_landsat_cache.py` |

Both write a `build_progress_{aoi}.txt` summary file into the cache directory, updated live.

To target one AOI: `--aoi north` or `--aoi south`
To target specific indices: `--indices NDVI NDMI`

---

## Planetary Computer — Rate Limit Behavior

PC issues 403s when the account/IP is rate-limited. Key facts:

- Signed URLs have ~1 hour TTL. If a run exceeds ~1 hour without re-signing, URLs expire and produce persistent 403s that are indistinguishable from rate limits.
- **Fix in place**: both scripts call `item = planetary_computer.sign(item)` at the top of the retry loop (before every fetch attempt), so URLs are always fresh.
- **Retry cap**: `MAX_RATE_LIMIT_RETRIES = 12` (1 hour of retries at 5 min each). After 12 consecutive 403s on one scene, the scene is skipped and marked failed. Re-running the script will retry failed scenes.
- `RATE_LIMIT_SLEEP = 300` (5 minutes between retries).
- Rate limits are account-wide — running S2 and Landsat simultaneously will share the quota.

---

## Sentinel-2 Harmonization (PB 04.00)

Scenes with `s2:processing_baseline >= 4.0` have a +0.1 reflectance offset applied by ESA. The build script corrects for this:

```python
if float(pb) >= 4.0:
    scaled = {k: np.clip(v - 0.1, 0.0, None) for k, v in scaled.items()}
```

This is applied after the standard scale/offset and before index computation. The manifest entry for each scene records `processing_baseline` and `harmonized: true/false`.

---

## Project Axes Framework (A–H, skipping I)

The analysis space is defined by 8 axes. "I" is intentionally skipped to avoid confusion with the number 1.

| Axis | Name | Values |
|------|------|--------|
| A | Temporal scope | A1=single scene, A2=monthly composite, A3=annual composite, A4=multi-year trend, A5=full archive |
| B | Index | B1=NDVI, B2=NDMI, B3=EVI |
| C | Primary stratification | C0=none, C1=north-facing, C2=south-facing, C3=east/west, C4=all aspects, C5=flat |
| D | Cross-stratification | D0=none, D1=forest type group, D2=ecozone |
| E | Aggregation | E1=p25, E2=p50, E3=p75, E4=mean, E5=max, E6=pixel-level |
| F | Deliverable | F1=time series plot, F2=aspect difference plot, F3=raster, F4=table, F5=map, F6=layer package, F7=anomaly raster, F8=seasonal curve |
| G | Index transformation | G1=raw index, G2=phenological amplitude (peak−baseline), G3=anomaly (z-score vs baseline) |
| H | Data source | H1=Sentinel-2 (current cache), H2=Sentinel-2 (alternate cache branch), H3=Landsat current cache, H4=Landsat alternate cache branch |

Current focus: H1/H3 on the active unsuffixed caches.

---

## Results Directory Structure

```
Results/              ← current outputs
Results_cache1/       ← historical validated outputs
```

Both directories have `figures/`, `figures/landsat/`, `rasters/`, `tables/`.

Layer packages (`.lpkx`) are documented in `LAYER_PACKAGES.txt` in each Results directory but excluded from git (too large). See `Results_cache1/LAYER_PACKAGES.txt` for the existing ones.

---

## Git / Repository Notes

- PNGs in `Results/figures/` and `Results/figures/landsat/` (and same for `Results_cache1/`) are tracked.
- `*.lpkx` files are gitignored.
- Cache directories (`*_cache*/`) are outside the repo root and never committed.
- `src/cli.py` contains shared CLI argument helpers including `add_cache_suffix_arg(parser, default)`.

---

## Obsidian Vault

A documentation vault exists at `../Project_Appalachia/tnc-forest-analysis/`. Currently contains `Project Axes.md` with the full axes framework. Intended to grow with decision logs, deliverable inventory, and code change notes.

---

## Key Decisions (Don't Re-litigate These)

- **Why p75?** Captures strong canopy signal, robust to residual cloud/shadow noise in individual scenes.
- **Why keep cloud shadow in index rasters?** Shadow has low NDVI and won't survive max compositing. QA_PIXEL is saved for optional post-hoc filtering.
- **Why subtract 0.1 for PB04+?** ESA added a radiometric offset in processing baseline 4.0 that inflates reflectance. Without correction, pre- and post-PB04 scenes are not comparable.
- **Why separate build vs analysis paths in yaml?** Build scripts need a clean unsuffixed base to append version suffixes. Analysis scripts need to point at a specific completed version.
- **S2 indices in the former `_3_4` branch were deleted** (~362 GB freed). The active unsuffixed caches now hold the retained Sentinel/Landsat/trait data.
