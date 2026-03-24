#!/usr/bin/env python3
"""
ecozone_elevation_gradient_landsat.py
--------------------------------------
Landsat C2 parallel of elevation gradient analysis

Elevation moisture gradient: p95 NDVI and NDMI by elevation band.
(Axis C3 — elevation band stratification)

Bins all pixels in each AOI into fixed 200m elevation bands, then
computes scene-level p95 within each band and means across all scenes.

Elevation data is reprojected from terrain_dir/elevation.tif (10m) to
the Landsat 30m canonical grid (9x the area per pixel).

Ecozone VALUE field: 1=Cool  2=Intermediate  3=Hot  (0 excluded)

Outputs:
  Results/figures/
    landsat_elevation_ndvi_gradient.png
    landsat_elevation_ndmi_gradient.png
    landsat_elevation_evi_gradient.png
  Results/tables/
    landsat_elevation_gradient_summary.xlsx

Run from project root:
  python Analysis/Traits/Elevation/ecozone_elevation_gradient_landsat.py
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject as warp_reproject
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.aoi import get_aoi_config
from src.landsat import load_landsat_scenes, load_landsat_ecozone
from src.paths import project_path

# ── Configuration ──────────────────────────────────────────────────────────────

AOIS                = ["north", "south"]
VALID_ECOZONE_CODES = [1, 2, 3]
ECOZONE_LABELS      = {1: "Cool", 2: "Intermediate", 3: "Hot"}
ECOZONE_COLORS      = {1: "#4E90C8", 2: "#72B063", 3: "#D9534F"}
AOI_DISPLAY         = {"north": "GW National Forest", "south": "Great Smoky Mtns"}
AOI_COLORS          = {"north": "#5A7FA8", "south": "#7A9E5A"}
INDICES             = ["NDVI", "NDMI", "EVI"]
MIN_PIXELS          = 200   # minimum valid pixels per band per scene to include

BAND_WIDTH_M = 200          # elevation band width in metres
MIN_BAND_PX  = 500          # minimum pixels in a band to show in output (reduced for 30m pixels)

FIGURES_DIR = project_path("results_figures_landsat_dir")
TABLES_DIR  = project_path("results_tables_landsat_dir")

for d in (FIGURES_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_elevation_bands(
    elevation: np.ndarray,
    band_width: int,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], np.ndarray]]:
    """
    Build fixed-width elevation bands.
    Returns list of (lo, hi) tuples and a dict of band → boolean mask.
    Only bands with >= MIN_BAND_PX valid elevation pixels are included.
    """
    elev_valid = elevation[np.isfinite(elevation)]
    lo_global  = int(np.floor(elev_valid.min() / band_width) * band_width)
    hi_global  = int(np.ceil(elev_valid.max()  / band_width) * band_width)

    bands  = []
    masks  = {}
    counts = {}
    for lo in range(lo_global, hi_global, band_width):
        hi   = lo + band_width
        mask = (elevation >= lo) & (elevation < hi) & np.isfinite(elevation)
        n    = int(mask.sum())
        if n >= MIN_BAND_PX:
            key = (lo, hi)
            bands.append(key)
            masks[key]  = mask
            counts[key] = n

    return bands, masks, counts


def scene_p95_by_band(
    scenes: list[dict],
    band_masks: dict[tuple[int, int], np.ndarray],
) -> dict[tuple[int, int], list[float]]:
    """
    Per scene, compute p95 of valid (non-NaN) pixels within each elevation band.
    Returns {band: [per_scene_p95]}.
    """
    results = {band: [] for band in band_masks}

    for i, scene in enumerate(scenes):
        if i % 50 == 0:
            print(f"      {i:>4}/{len(scenes)} scenes...", flush=True)
        with rasterio.open(scene["filepath"]) as src:
            data = src.read(1)
        valid = np.isfinite(data)

        for band, bmask in band_masks.items():
            combined = bmask & valid
            if combined.sum() >= MIN_PIXELS:
                results[band].append(float(np.percentile(data[combined], 95)))

    return results


# ── Main loop ──────────────────────────────────────────────────────────────────

# gradient[aoi][index][band] = mean p95 across scenes
gradient: dict = {}
band_info: dict = {}   # [aoi] = (bands list, counts dict)

for aoi in AOIS:
    cfg = get_aoi_config(aoi)
    print(f"\n{'='*62}")
    print(f"  {aoi.upper()} AOI — {AOI_DISPLAY[aoi]}")
    print(f"{'='*62}")

    # Load Landsat ecozone (to get canonical Landsat grid)
    try:
        ecozone_arr, dst_height, dst_width, dst_transform = load_landsat_ecozone(aoi)
    except FileNotFoundError as e:
        print(f"\n  Error: {e}")
        print("  Skipping this AOI.")
        continue

    # Reproject elevation to Landsat 30m grid
    elev_src_path = cfg.terrain_dir / "elevation.tif"
    elev_arr = np.zeros((dst_height, dst_width), dtype=np.float32)
    with rasterio.open(elev_src_path) as src:
        warp_reproject(
            source=rasterio.band(src, 1),
            destination=elev_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:32617",
            resampling=Resampling.bilinear,
        )

    print(f"\n  Elevation range: {np.nanmin(elev_arr):.0f}m – {np.nanmax(elev_arr):.0f}m")

    bands, band_masks, counts = make_elevation_bands(elev_arr, BAND_WIDTH_M)
    band_info[aoi] = (bands, counts)

    print(f"  Elevation bands ({BAND_WIDTH_M}m, ≥{MIN_BAND_PX:,} px):")
    for (lo, hi) in bands:
        print(f"    {lo:>5}–{hi:<5}m  {counts[(lo,hi)]:>10,} px")

    gradient[aoi] = {}

    for index_name in INDICES:
        print(f"\n  [{index_name}] Loading scenes...")
        scenes = load_landsat_scenes(aoi, index_name)

        if not scenes:
            print(f"    No scenes available — skipping this index.")
            continue

        print(f"    {len(scenes)} scenes  — computing p95 per elevation band...")
        per_scene = scene_p95_by_band(scenes, band_masks)

        gradient[aoi][index_name] = {}
        for band in bands:
            vals = per_scene[band]
            gradient[aoi][index_name][band] = (
                float(np.nanmean(vals)) if vals else np.nan
            )

        # Console summary
        print(f"    {'Band (m)':<14} {'p95':>8}  {'n_scenes':>10}")
        for band in bands:
            val = gradient[aoi][index_name][band]
            n   = len(per_scene[band])
            lo, hi = band
            print(f"    {lo}-{hi:<10}  {val:>8.4f}  {n:>10}")


# ── Figure helper ──────────────────────────────────────────────────────────────

def gradient_figure(index_name: str, title: str, xlabel: str, outfile) -> None:
    """
    Horizontal line chart: x = mean p95 index, y = elevation band midpoint.
    One line per AOI.  Bands with no data are omitted.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    for aoi in AOIS:
        if aoi not in gradient or index_name not in gradient[aoi]:
            continue

        bands, counts = band_info[aoi]
        color = AOI_COLORS[aoi]

        midpoints = []
        values    = []
        for band in bands:
            val = gradient[aoi][index_name][band]
            if not np.isnan(val):
                midpoints.append((band[0] + band[1]) / 2)
                values.append(val)

        ax.plot(
            values, midpoints,
            color=color, linewidth=2.2, marker="o", markersize=5,
            label=AOI_DISPLAY[aoi], zorder=3,
        )

        # Pixel count annotations on the right
        for mp, val, band in zip(midpoints, values, [b for b in bands if not np.isnan(gradient[aoi][index_name][b])]):
            ax.annotate(
                f"{counts[band]/1000:.0f}k px",
                xy=(val, mp), xytext=(5, 0),
                textcoords="offset points",
                fontsize=7, color=color, va="center", alpha=0.7,
            )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Elevation band midpoint (m)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {outfile}")


# ── Figures ────────────────────────────────────────────────────────────────────

gradient_figure(
    index_name="NDVI",
    title="Elevation Gradient — Mean p95 NDVI by Elevation Band — Landsat C2\n(all scenes pooled)",
    xlabel="Mean p95 NDVI",
    outfile=FIGURES_DIR / "landsat_elevation_ndvi_gradient.png",
)

gradient_figure(
    index_name="NDMI",
    title="Elevation Gradient — Mean p95 NDMI by Elevation Band — Landsat C2\n(all scenes pooled)",
    xlabel="Mean p95 NDMI",
    outfile=FIGURES_DIR / "landsat_elevation_ndmi_gradient.png",
)

gradient_figure(
    index_name="EVI",
    title="Elevation Gradient — Mean p95 EVI by Elevation Band — Landsat C2\n(all scenes pooled)",
    xlabel="Mean p95 EVI",
    outfile=FIGURES_DIR / "landsat_elevation_evi_gradient.png",
)


# ── Spreadsheet ────────────────────────────────────────────────────────────────

rows = []
for aoi in AOIS:
    if aoi not in band_info:
        continue
    bands, counts = band_info[aoi]
    for band in bands:
        lo, hi = band
        row = {
            "AOI":            AOI_DISPLAY[aoi],
            "Elev Band Lo m": lo,
            "Elev Band Hi m": hi,
            "Midpoint m":     (lo + hi) / 2,
            "Pixel Count":    counts[band],
        }
        for index_name in INDICES:
            if aoi in gradient and index_name in gradient[aoi]:
                row[f"{index_name} p95"] = round(
                    gradient[aoi][index_name][band], 6
                )
            else:
                row[f"{index_name} p95"] = np.nan
        rows.append(row)

df = pd.DataFrame(rows)
table_path = TABLES_DIR / "landsat_elevation_gradient_summary.xlsx"
df.to_excel(table_path, index=False, sheet_name="Elevation Gradient")
print(f"\nSaved: {table_path}")

print("\nLandsat C2 elevation gradient analysis complete.")
