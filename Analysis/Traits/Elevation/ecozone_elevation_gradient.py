#!/usr/bin/env python3
"""
ecozone_elevation_gradient.py
------------------------------
Elevation moisture gradient: p95 NDVI and NDMI by elevation band.
(Axis C3 — elevation band stratification)

Bins all pixels in each AOI into fixed 200m elevation bands, then
computes scene-level p95 within each band and means across all scenes.
Optionally cross-stratified by ecozone (optional — see CROSS_ECOZONE flag).

Elevation data: terrain_dir/elevation.tif  (10m, UTM 17N, from Copernicus DEM)

Ecozone VALUE field: 1=Cool  2=Intermediate  3=Hot  (0 excluded)

Outputs:
  Results/figures/
    elevation_ndvi_gradient.png
    elevation_ndmi_gradient.png
  Results/tables/
    elevation_gradient_summary.xlsx

Run from project root:
  python Analysis/Traits/Elevation/ecozone_elevation_gradient.py
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import rasterio

from src.aoi import get_aoi_config
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
MIN_BAND_PX  = 5000         # minimum pixels in a band to show in output

FIGURES_DIR = project_path("results_figures_dir")
TABLES_DIR  = project_path("results_tables_dir")

for d in (FIGURES_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_scenes(index_dir: Path, manifest_path: Path) -> list[dict]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    scenes = []
    for _, meta in manifest.items():
        fp = index_dir / meta["filename"]
        if fp.exists():
            scenes.append({
                "date":     datetime.fromisoformat(meta["date"]),
                "filepath": fp,
            })
    return sorted(scenes, key=lambda s: s["date"])


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

    # Load elevation
    elev_path = cfg.terrain_dir / "elevation.tif"
    with rasterio.open(elev_path) as src:
        elevation = src.read(1).astype(np.float32)

    print(f"\n  Elevation range: {np.nanmin(elevation):.0f}m – {np.nanmax(elevation):.0f}m")

    bands, band_masks, counts = make_elevation_bands(elevation, BAND_WIDTH_M)
    band_info[aoi] = (bands, counts)

    print(f"  Elevation bands ({BAND_WIDTH_M}m, ≥{MIN_BAND_PX:,} px):")
    for (lo, hi) in bands:
        print(f"    {lo:>5}–{hi:<5}m  {counts[(lo,hi)]:>10,} px")

    gradient[aoi] = {}

    for index_name in INDICES:
        index_dir     = cfg.index_cache_root / index_name
        manifest_path = index_dir / "cache_manifest.json"

        print(f"\n  [{index_name}] Loading scenes...")
        scenes = load_scenes(index_dir, manifest_path)
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

def gradient_figure(index_name: str, title: str, xlabel: str, outfile: Path) -> None:
    """
    Horizontal line chart: x = mean p95 index, y = elevation band midpoint.
    One line per AOI.  Bands with no data are omitted.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    for aoi in AOIS:
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
    title="Elevation Gradient — Mean p95 NDVI by Elevation Band\n(all scenes pooled)",
    xlabel="Mean p95 NDVI",
    outfile=FIGURES_DIR / "elevation_ndvi_gradient.png",
)

gradient_figure(
    index_name="NDMI",
    title="Elevation Gradient — Mean p95 NDMI by Elevation Band\n(all scenes pooled)",
    xlabel="Mean p95 NDMI",
    outfile=FIGURES_DIR / "elevation_ndmi_gradient.png",
)


# ── Spreadsheet ────────────────────────────────────────────────────────────────

rows = []
for aoi in AOIS:
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
            row[f"{index_name} p95"] = round(
                gradient[aoi][index_name][band], 6
            )
        rows.append(row)

df = pd.DataFrame(rows)
table_path = TABLES_DIR / "elevation_gradient_summary.xlsx"
df.to_excel(table_path, index=False, sheet_name="Elevation Gradient")
print(f"\nSaved: {table_path}")

print("\nElevation gradient analysis complete.")
