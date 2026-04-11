# Project Ontology

This file is a compact ontology of the Appalachian terrain-vegetation project: data sources, caches, trait layers, build steps, analysis families, outputs, and decision points.

## Flowchart

```mermaid
flowchart TD
    A[Project Purpose<br/>Compare vegetation and moisture dynamics across AOIs, ecozones, and year classes] --> B[Study Areas / AOIs]
    B --> B1[North / GWNF]
    B --> B2[South / Smoky]

    A --> C[Remote-Sensing Sources]
    C --> C1[Sentinel-2 L2A]
    C --> C2[Landsat C2 L2]

    A --> D[Static Spatial Traits]
    D --> D1[Ecozone]
    D --> D2[Elevation]
    D --> D3[Forest Type / Group]
    D --> D4[Terrain / Aspect]

    C1 --> E[Cache Builders]
    C2 --> E
    E --> E1[build_sentinel_cache.py]
    E --> E2[build_landsat_cache.py]
    E --> E3[run_cache_builder.sh]

    E1 --> F[AOI-Aligned Cache Roots]
    E2 --> F
    D --> F
    F --> F1[GWNF_cache]
    F --> F2[Smoky_cache]

    F1 --> G[Index Scene Caches]
    F2 --> G
    G --> G1[s2 / indices / NDVI NDMI EVI SCL]
    G --> G2[landsat / indices / NDVI NDMI EVI QA_PIXEL]
    G --> G3[s2 / traits / ecozone terrain forest]

    G --> H[Composite Builders]
    H --> H1[build_monthly_composites.py]
    H --> H2[build_annual_composites.py]
    H --> H3[build_landsat_monthly_composites.py]
    H --> H4[build_landsat_annual_composites.py]

    H --> I[Composite Outputs]
    I --> I1[Results/0-CacheBaseData/monthly_max]
    I --> I2[Results/0-CacheBaseData/annual_max]

    I --> J[Anomaly Raster Builders]
    J --> J1[build_anomaly_rasters.py]
    J --> J2[build_landsat_anomaly_rasters.py]

    J --> K[Analysis Inputs]
    G3 --> K
    K --> K1[Monthly anomaly summaries]
    K --> K2[Annual anomaly summaries]
    K --> K3[Year labels from wet_dry_years.csv]
    K --> K4[AOI/ecozone/trait masks]

    K --> L[Core Ecozone Investigations]
    L --> L1[Peak productivity]
    L --> L2[Seasonal curves]
    L --> L3[Drought response]
    L --> L4[Moisture stress]
    L --> L5[Long-term trend]

    K --> M[Sentinel Investigation Layer]
    M --> M1[Onset timing]
    M --> M2[Monthly trajectories]
    M --> M3[Percentile spread]
    M --> M4[Fraction below baseline]
    M --> M5[Distribution shape]
    M --> M6[Simple recovery]
    M --> M7[Spatial consistency]

    M --> N[Comparison Layer]
    N --> N1[Index role summary]
    N --> N2[Ecozone comparative dynamics]
    N --> N3[Wet / normal / dry comparison]
    N --> N4[Ecozone x year-type crossed comparison]

    I --> O[Presentation / Demo Layer]
    K3 --> O
    O --> O1[anomaly_from_normal_demo.py]
    O --> O2[Preview PNGs]
    O --> O3[GeoTIFF anomaly demos]

    L --> P[Results Families]
    M --> P
    N --> P
    O --> P
    P --> P1[Results/1_Foundation]
    P --> P2[Results/2_Anomaly_Onset]
    P --> P3[Results/3_Anomaly_Progression]
    P --> P4[Results/4_Anomaly_Recovery]
    P --> P5[Results/Other]
    P --> P6[Results_cache1 historical validated archive]

    A --> Q[Decision / Interpretation Layer]
    Q --> Q1[By AOI]
    Q --> Q2[By index: NDVI NDMI EVI]
    Q --> Q3[By ecozone]
    Q --> Q4[By year type: wet neutral dry]
    Q --> Q5[By response phase: onset progression recovery]

    Q1 --> R[North behaves more like classic stress-response]
    Q1 --> S[South behaves more asymmetrically]
    Q2 --> T[NDVI strongest overall signal]
    Q2 --> U[NDMI strongest ecozone / timing differentiation]
    Q2 --> V[EVI least stable]

    style A fill:#e8f1f8,stroke:#5c7c99,stroke-width:2px
    style F fill:#eef7ea,stroke:#5f8a55
    style I fill:#eef7ea,stroke:#5f8a55
    style P fill:#f8f1e8,stroke:#9b7a4f
    style Q fill:#f7eaf2,stroke:#94607f
```

## Entity Types

| Layer | Entity | Role |
|---|---|---|
| Domain | AOI | North/GWNF and South/Smoky study regions |
| Domain | Index | NDVI, NDMI, EVI |
| Domain | Trait | Ecozone, elevation, forest, terrain/aspect |
| Domain | Year class | Wet, neutral/normal, dry |
| Domain | Response phase | Onset, progression, recovery, spatial extent |
| Source | Scene cache | Per-scene AOI-aligned rasters plus manifests |
| Derived data | Monthly composite | Per-pixel monthly max composite |
| Derived data | Annual composite | Per-pixel annual max composite |
| Derived data | Anomaly raster | Composite minus baseline or comparable anomaly form |
| Analysis | Investigation | Script family that summarizes one ecological question |
| Analysis | Comparison layer | Script family that compares indices, ecozones, or year classes |
| Output | Summary CSV | Compact tables used for interpretation and follow-on plots |
| Output | Figure | PNG summaries, heatmaps, trajectories, demos |
| Output | Historical archive | `Results_cache1`, the validated legacy deliverable set |

## Main Processes

1. Acquire scene-level remote sensing data into AOI-specific caches.
2. Align all scenes to AOI-specific canonical grids.
3. Preserve trait rasters on the same aligned grid.
4. Build monthly and annual max composites from cached scenes.
5. Derive anomaly products relative to baselines or year-class references.
6. Summarize ecozone behavior from those anomalies.
7. Compare indices, ecozones, and year classes using derived CSV outputs.
8. Produce presentation artifacts such as quicklook PNGs, heatmaps, and demo GeoTIFFs.

## Key Decisions And Rules

| Topic | Current rule |
|---|---|
| Active cache layout | Unsuffixed `GWNF_cache` and `Smoky_cache` |
| Sentinel cache retry policy | Exit immediately on any `403`; outer runner restarts |
| Landsat cache retry policy | Exit immediately on any `403`; outer runner restarts |
| Monthly composite definition | Per-pixel monthly maximum across valid scenes |
| Annual composite definition | Per-pixel annual maximum across valid scenes |
| Trusted output archive | `Results_cache1` |
| Current working output tree | `Results/` |
| Comparative focus | Relative differences across ecozones/AOIs/year classes, more than exact polygon-total claims |

## Script Families

| Family | Main location | Purpose |
|---|---|---|
| Cache acquisition | `Python/Cache/` | Build Sentinel and Landsat per-scene caches |
| Composite generation | `Python/Analysis/Indices/` | Build monthly/annual composites and anomaly rasters |
| Trait preparation | `Python/Analysis/Traits/` | Prepare and verify ecozone/elevation/forest/terrain masks |
| Ecozone core analyses | `Python/Analysis/Traits/Ecozone/` | Seasonal, drought, stress, trend, and ecozone comparisons |
| Crosstabs | `Python/Analysis/Crosstab/` | Inter-trait and aspect cross-tabulation |
| Diagnostics | `Python/Analysis/Diagnostics/` | Demo and troubleshooting products |
| ArcGIS packaging | `Python/Analysis/arcgis/` | Layer package / mosaic support |

## Practical Reading Order

1. [PROJECT_OVERVIEW.md](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/docs/PROJECT_OVERVIEW.md)
2. [DATA_METHODS_SHORT.md](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/docs/DATA_METHODS_SHORT.md)
3. [project_paths.yaml](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/config/project_paths.yaml)
4. [build_sentinel_cache.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Cache/build_sentinel_cache.py)
5. [build_landsat_cache.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Cache/build_landsat_cache.py)
6. [investigation_common.py](/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python/Analysis/Traits/Ecozone/investigation_common.py)
7. The `Results/2_Anomaly_Onset`, `Results/3_Anomaly_Progression`, `Results/4_Anomaly_Recovery`, and `Results/Other` trees

## Short Summary

The project is a layered ecological analysis system:

- remote-sensing scene caches are the base
- AOI-aligned trait rasters provide ecological stratification
- monthly and annual composites turn scene caches into analyzable time slices
- anomaly investigations summarize response timing, shape, magnitude, and recovery
- comparison scripts convert those summaries into explicit cross-index, cross-ecozone, and cross-year-type interpretations
- presentation scripts produce compact demo rasters and figures for communication
