# Dashboard Value Derivation Examples

This document traces two dashboard hover values from source acquisition through cached rasters, summary tables, and final dashboard display. The goal is to make explicit that dashboard points are precomputed multi-stage summaries, not raw pixels or single-scene values.

For the broader data-methods context, see:

- `DATA_METHODS_SHORT.md`
- `DATA_PROVENANCE_AND_CACHE_CHARACTERISTICS.md`
- `DASHBOARD_TABLE_FACTORY.md`

## General Pattern

Most dashboard time-series points are built through this chain:

1. Query satellite scenes from Microsoft Planetary Computer.
2. Reproject source bands to a project AOI-aligned grid.
3. Scale reflectance and calculate NDVI, NDMI, or EVI.
4. Apply the cache's pixel mask and write one index GeoTIFF per scene.
5. Build scene-level summaries by calculating a spatial percentile over valid pixels.
6. Optionally subset pixels by thermal ecozone, forest-community group, or forest community.
7. Group scene summaries into a temporal bin such as `scene`, `half_month`, or `month`.
8. Calculate a temporal percentile across the scene-level values in that bin.
9. Load the precomputed summary row in Streamlit and display it through Plotly.

The dashboard's z-score exclusion filters can remove displayed points, but they do not recalculate the stored summary values.

Current dashboard selection model: each layer has base settings such as AOI, sensor, index, cloud threshold, spatial percentile, temporal aggregation, and temporal percentile. Spatial subsets are selected below the layer in the layer-list checklist. The checklist can show the layer's `Overall Combined` series, thermal ecozones, forest-community groups, group-level combined lines, and individual forest communities. The older separate Scope dropdown has been collapsed into those layer checklist selections. Internally, the summary tables still store an `analysis_scope` field such as `overall`, `ecozone`, `forest_ecozone_group`, or `forest_community`.

## Example 1: Landsat NDVI, Cove Forest Group

Dashboard hover value:

| Field | Value |
| --- | --- |
| Date | `1999-07-16` |
| Displayed value | `0.9456` |
| Exact stored value | `0.945563` |
| Sensor | `ls` |
| AOI | `north` |
| Index | `ndvi` |
| Layer checklist selection | Cove forest group, combined |
| Underlying table scope | forest-community group |
| Temporal aggregation | `half_month` |
| Spatial percentile | `p99` |
| Temporal percentile | `p99` |
| Mask id | `ls_clear_terrestrial_v1` |

Scientific interpretation:

> This point is the 99th percentile across scene-level p99 NDVI values for valid Landsat pixels in the north AOI's Cove forest group during the second half of July 1999.

### Derivation

The Landsat cache builder queries Microsoft Planetary Computer's `landsat-c2-l2` collection for Landsat 5, 7, 8, and 9 scenes intersecting the AOI polygon. For NDVI, the relevant bands are red, near infrared, and `qa_pixel`.

Landsat Collection 2 surface reflectance is scaled as:

```text
reflectance = raw * 0.0000275 - 0.2
```

NDVI is then calculated as:

```text
NDVI = (NIR - Red) / (NIR + Red)
```

The Landsat bands are reprojected to the project Landsat grid in `EPSG:32617` at 30 m resolution. Pixels outside the AOI polygon are set to `NaN`. The cache masks invalid reflectance plus dilated cloud, cirrus, cloud, and snow before writing the NDVI GeoTIFF. The Landsat mask metadata is documented in `DATA_METHODS_SHORT.md` and `DATA_PROVENANCE_AND_CACHE_CHARACTERISTICS.md`.

For forest-community-group summaries, the TNC forest-community raster is aligned to the Landsat grid with nearest-neighbor resampling. "Cove forest, combined" means all detailed forest communities assigned to the Cove forest group are pooled before the statistic is calculated.

For each qualifying Landsat scene, the table factory reads the cached NDVI raster, keeps finite pixels in the Cove forest group, and calculates the spatial p99 NDVI. That produces one scene-level p99 value per scene.

The dashboard date `1999-07-16` is the half-month bin start, not necessarily an acquisition date. This bin covers July 16 through July 31, 1999.

The six scene-level inputs for this bin were:

| Scene date | Platform | Path/row | Cloud cover | Scene p99 NDVI |
| --- | --- | --- | ---: | ---: |
| `1999-07-18` | LT05 | p017r033 | 6% | `0.918222` |
| `1999-07-18` | LT05 | p017r034 | 5% | `0.896812` |
| `1999-07-19` | LE07 | p016r033 | 37% | `0.938699` |
| `1999-07-19` | LE07 | p016r034 | 7% | `0.945924` |
| `1999-07-26` | LE07 | p017r033 | 17% | `0.921151` |
| `1999-07-26` | LE07 | p017r034 | 7% | `0.904538` |

The temporal p99 of those six scene-level p99 values is:

```text
0.945563
```

The dashboard hover rounds this to four decimals:

```text
0.9456
```

## Example 2: Sentinel-2 NDMI, Whole South AOI

Dashboard hover value:

| Field | Value |
| --- | --- |
| Date | `2019-06-01` |
| Displayed value | `0.4526` |
| Exact stored value | `0.452625` |
| Sensor | `s2` |
| AOI | `south` |
| Index | `ndmi` |
| Layer checklist selection | Overall Combined |
| Underlying table scope | overall |
| Temporal aggregation | `month` |
| Spatial percentile | `p75` |
| Temporal percentile | `p95` |
| Mask id | `s2_scl4_veg_v1` |

Scientific interpretation:

> This point is the 95th percentile across June 2019 scene-level p75 NDMI values, where each scene-level value is the 75th percentile NDMI across valid Sentinel-2 pixels in the whole south AOI.

### Derivation

The accepted Sentinel-2 cache used for dashboard summaries was built from Microsoft Planetary Computer's `sentinel-2-l2a` collection. For NDMI, the relevant bands are:

```text
B08 = near infrared
B11 = shortwave infrared
```

NDMI is calculated as:

```text
NDMI = (B08 - B11) / (B08 + B11)
```

Sentinel-2 reflectance bands in the accepted cache were scaled as:

```text
reflectance = raw / 10000
```

The accepted Sentinel cache used a vegetation-only `SCL == 4` mask plus numeric validity screening, on a canonical `EPSG:32617` grid at 10 m resolution. It was not harmonized for the early-2022 Sentinel-2 processing-baseline shift; this example is from 2019, so that specific shift is not relevant to this point.

Because this point uses the layer checklist's `Overall Combined` selection, no thermal-ecozone, forest-community-group, or forest-community subset is applied. Each qualifying June 2019 Sentinel scene contributes one scene-level value: the spatial p75 NDMI across valid south-AOI pixels.

The dashboard date `2019-06-01` is the month bucket start, covering June 1 through June 30, 2019.

The 17 scene-level p75 NDMI inputs were:

```text
0.392525
0.398669
0.410554
0.411686
0.422832
0.450442
0.449143
0.461355
0.371355
0.376321
0.389031
0.373382
0.380899
0.397850
0.412093
0.407524
0.413632
```

The temporal p95 of those scene-level p75 values is:

```text
0.452625
```

The dashboard hover rounds this to:

```text
0.4526
```

## Contrast Between The Examples

The two values differ in nearly every dashboard setting:

| Dimension | Landsat Cove forest example | Sentinel-2 south AOI example |
| --- | --- | --- |
| Sensor | Landsat | Sentinel-2 |
| AOI | north | south |
| Index | NDVI | NDMI |
| Spatial scope | Cove forest group, combined | whole AOI |
| Temporal bin | half-month | month |
| Spatial statistic | p99 | p75 |
| Temporal statistic | p99 | p95 |
| Interpretation | high-end canopy greenness in a specific forest group | upper-quartile canopy moisture signal across an AOI |

Together, the examples show why dashboard values should be read as configurable summaries of precomputed raster products. A point's meaning depends on the sensor, AOI, vegetation index, spatial subset, spatial percentile, temporal bin, temporal percentile, cloud threshold, and mask definition.
