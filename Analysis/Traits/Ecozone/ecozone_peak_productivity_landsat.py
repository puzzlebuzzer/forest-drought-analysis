#!/usr/bin/env python3
"""
ecozone_peak_productivity_landsat.py
------------------------------------
Landsat C2 parallel of Analyses 1, 2, and 3 — TNC Appalachian Terrain–Vegetation Roadmap

Analysis 1 — Peak vegetation productivity: p95/p99/p100 NDVI by ecozone
Analysis 2 — Peak canopy moisture:         p95/p99/p100 NDMI by ecozone
Analysis 3 — Ecological space:             NDVI vs NDMI scatter by ecozone (p95)
Analysis 1b — Enhanced vegetation index:   p95/p99/p100 EVI by ecozone (if cached)

For each AOI, iterates all cached scenes for NDVI and NDMI.
Per scene, computes the 95th, 99th, and 100th percentile of valid index
pixels within each ecozone mask. Summary values are the mean of all
scene-level percentile values across all scenes.

Ecozone VALUE field:
  1 = Cool
  2 = Intermediate
  3 = Hot
  0 = excluded (nodata / boundary artifact)

Outputs:
  Results/figures/landsat/
    landsat_ecozone_ndvi_p95_p99_p100.png
    landsat_ecozone_ndmi_p95_p99_p100.png
    landsat_ecozone_evi_p95_p99_p100.png   (only if EVI cache exists)
    landsat_ecozone_ecological_space.png
  Results/tables/landsat/
    landsat_ecozone_peak_summary.xlsx

Run from project root:
  python Analysis/Traits/Ecozone/ecozone_peak_productivity_landsat.py
"""

from datetime import datetime
from pathlib import Path

import matplotlib.colors as mc
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from src.landsat import get_landsat_index_root, load_landsat_scenes, load_landsat_ecozone
from src.paths import project_path

# ── Configuration ──────────────────────────────────────────────────────────────

AOIS               = ["north", "south"]
VALID_ECOZONE_CODES = [1, 2, 3]
ECOZONE_LABELS     = {1: "Cool", 2: "Intermediate", 3: "Hot"}
ECOZONE_COLORS     = {1: "#4E90C8", 2: "#72B063", 3: "#D9534F"}
AOI_DISPLAY        = {"north": "GW National Forest", "south": "Great Smoky Mtns"}
AOI_MARKERS        = {"north": "o", "south": "s"}
INDICES            = ["NDVI", "NDMI", "EVI"]
PERCENTILES        = [95, 99, 100]
PCT_LABELS         = {95: "p95", 99: "p99", 100: "p100 (max)"}
MIN_PIXELS         = 100   # minimum valid pixels to compute a percentile

FIGURES_DIR = project_path("results_figures_landsat_dir")
TABLES_DIR  = project_path("results_tables_landsat_dir")

for d in (FIGURES_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ── Color helpers ──────────────────────────────────────────────────────────────

def lighten(color_hex: str, factor: float) -> tuple:
    """Blend a hex color toward white. factor=0 → original, factor=1 → white."""
    r, g, b = mc.to_rgb(color_hex)
    return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)


# Lighten factors per percentile: p95 = full color, p99 = lighter, p100 = lightest
_LIGHTEN = {95: 0.0, 99: 0.4, 100: 0.65}

BAR_COLORS = {
    (code, pct): lighten(ECOZONE_COLORS[code], _LIGHTEN[pct])
    for code in VALID_ECOZONE_CODES
    for pct in PERCENTILES
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def scene_percentiles_by_ecozone(
    scenes: list[dict],
    eco_masks: dict[int, np.ndarray],
    percentiles: list[int],
) -> dict[int, dict[int, list[float]]]:
    """
    For each scene, compute requested percentiles within each ecozone mask.
    All percentiles are computed from the same pixel extraction — one raster
    read per scene.
    Returns: {ecozone_code: {percentile: [per_scene_values]}}
    """
    results = {
        code: {pct: [] for pct in percentiles}
        for code in VALID_ECOZONE_CODES
    }

    for i, scene in enumerate(scenes):
        if i % 50 == 0:
            print(f"      {i:>4}/{len(scenes)} scenes...", flush=True)
        with rasterio.open(scene["filepath"]) as src:
            data = src.read(1)
        valid = ~np.isnan(data)
        for code in VALID_ECOZONE_CODES:
            combined = eco_masks[code] & valid
            if combined.sum() >= MIN_PIXELS:
                pixel_vals = data[combined]
                for pct in percentiles:
                    results[code][pct].append(float(np.percentile(pixel_vals, pct)))

    return results


# ── Main loop: both AOIs, both indices ────────────────────────────────────────

# summary[aoi][index_name][code][pct] = mean across all scenes
summary: dict = {}
eco_pixel_counts: dict = {}   # [aoi][code] = pixel count

for aoi in AOIS:
    print(f"\n{'='*62}")
    print(f"  {aoi.upper()} AOI — {AOI_DISPLAY[aoi]}")
    print(f"{'='*62}")

    ecozone_arr, _, _, _ = load_landsat_ecozone(aoi)
    print(f"\n  Ecozone raster: Landsat 30m reprojected")

    eco_masks = {code: (ecozone_arr == code) for code in VALID_ECOZONE_CODES}
    eco_pixel_counts[aoi] = {
        code: int(eco_masks[code].sum()) for code in VALID_ECOZONE_CODES
    }

    for code in VALID_ECOZONE_CODES:
        print(f"    {ECOZONE_LABELS[code]:>12}: {eco_pixel_counts[aoi][code]:>10,} px")
    excluded = int((ecozone_arr == 0).sum())
    if excluded:
        print(f"    {'Excluded (0)':>12}: {excluded:>10,} px")

    summary[aoi] = {}

    for index_name in INDICES:
        print(f"\n  [{index_name}] Loading scenes from manifest...")
        scenes = load_landsat_scenes(aoi, index_name)

        if not scenes:
            print(f"    0 scenes on disk")
            print(f"  [{index_name}] Skipping — no Landsat cache found.")
            print(f"             Run: python Cache/build_landsat_cache.py --aoi {aoi}")
            summary[aoi][index_name] = {
                code: {pct: np.nan for pct in PERCENTILES}
                for code in VALID_ECOZONE_CODES
            }
            continue

        print(f"    {len(scenes)} scenes on disk")

        print(f"  [{index_name}] Computing p95/p99/p100 per ecozone per scene...")
        per_scene = scene_percentiles_by_ecozone(scenes, eco_masks, PERCENTILES)

        summary[aoi][index_name] = {}
        for code in VALID_ECOZONE_CODES:
            summary[aoi][index_name][code] = {}
            for pct in PERCENTILES:
                vals = per_scene[code][pct]
                summary[aoi][index_name][code][pct] = (
                    float(np.nanmean(vals)) if vals else np.nan
                )
            d = summary[aoi][index_name][code]
            print(
                f"    {ECOZONE_LABELS[code]:>12}:  "
                f"p95={d[95]:.4f}  p99={d[99]:.4f}  p100={d[100]:.4f}"
                f"  (n={len(per_scene[code][95])} scenes)"
            )


# ── Console summary table ──────────────────────────────────────────────────────

print(f"\n{'='*62}")
print("  SUMMARY — Mean pXX by AOI / Index / Ecozone")
print(f"{'='*62}")
for aoi in AOIS:
    print(f"\n  {AOI_DISPLAY[aoi]}")
    for index_name in INDICES:
        print(f"    {index_name}  {'Ecozone':<14} {'p95':>8}  {'p99':>8}  {'p100':>8}")
        print(f"          {'-'*44}")
        for code in VALID_ECOZONE_CODES:
            d = summary[aoi][index_name][code]
            print(
                f"          {ECOZONE_LABELS[code]:<14}"
                f" {d[95]:>8.4f}  {d[99]:>8.4f}  {d[100]:>8.4f}"
            )


# ── Figure helper: grouped bar chart ──────────────────────────────────────────

def bar_figure(index_name: str, title: str, ylabel: str, outfile: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    bar_width = 0.22
    n_pcts    = len(PERCENTILES)
    x         = np.arange(len(VALID_ECOZONE_CODES))

    for ax, aoi in zip(axes, AOIS):
        for i, pct in enumerate(PERCENTILES):
            offset = (i - (n_pcts - 1) / 2) * bar_width
            vals   = [summary[aoi][index_name][code][pct] for code in VALID_ECOZONE_CODES]
            colors = [BAR_COLORS[(code, pct)] for code in VALID_ECOZONE_CODES]
            hatch  = "////" if pct == 100 else None

            bars = ax.bar(
                x + offset, vals,
                width=bar_width,
                color=colors,
                edgecolor="#444444" if hatch else "white",
                linewidth=0.7,
                hatch=hatch,
                label=PCT_LABELS[pct],
            )

            # Value labels on p95 bars only — p99 and p100 readable in spreadsheet
            if pct == 95:
                for bar, val in zip(bars, vals):
                    if not np.isnan(val):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.003,
                            f"{val:.3f}",
                            ha="center", va="bottom", fontsize=8, fontweight="bold",
                        )

        ax.set_xticks(x)
        ax.set_xticklabels([ECOZONE_LABELS[c] for c in VALID_ECOZONE_CODES])
        ax.set_title(AOI_DISPLAY[aoi], fontsize=12)
        ax.set_xlabel("Ecozone")
        ax.set_ylabel(ylabel if aoi == "north" else "")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        all_vals = [
            summary[aoi][index_name][code][pct]
            for code in VALID_ECOZONE_CODES
            for pct in PERCENTILES
        ]
        valid_vals = [v for v in all_vals if not np.isnan(v)]
        if valid_vals:
            ymin = max(0, min(valid_vals) - 0.05) if index_name in ("NDVI", "EVI") else min(valid_vals) - 0.05
            ymax = max(valid_vals) + 0.10
            ax.set_ylim(ymin, ymax)

    # Legend: ecozone colors + percentile shading key
    eco_patches = [
        mpatches.Patch(color=ECOZONE_COLORS[c], label=ECOZONE_LABELS[c])
        for c in VALID_ECOZONE_CODES
    ]
    pct_patches = [
        mpatches.Patch(
            facecolor="lightgray",
            edgecolor="#444" if pct == 100 else "white",
            hatch="////" if pct == 100 else None,
            label=PCT_LABELS[pct],
        )
        for pct in PERCENTILES
    ]
    fig.legend(
        handles=eco_patches + pct_patches,
        loc="lower center",
        ncol=len(eco_patches) + len(pct_patches),
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.06),
    )

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {outfile}")


# ── Figure 1: p95/p99/p100 NDVI ───────────────────────────────────────────────

bar_figure(
    index_name="NDVI",
    title="Peak Vegetation Productivity — p95 / p99 / p100 NDVI by Ecozone — Landsat C2",
    ylabel="Mean NDVI (scene-level percentile averaged across all scenes)",
    outfile=FIGURES_DIR / "landsat_ecozone_ndvi_p95_p99_p100.png",
)


# ── Figure 2: p95/p99/p100 NDMI ───────────────────────────────────────────────

bar_figure(
    index_name="NDMI",
    title="Peak Canopy Moisture — p95 / p99 / p100 NDMI by Ecozone — Landsat C2",
    ylabel="Mean NDMI (scene-level percentile averaged across all scenes)",
    outfile=FIGURES_DIR / "landsat_ecozone_ndmi_p95_p99_p100.png",
)


# ── Figure 1b: p95/p99/p100 EVI (only if cache exists) ───────────────────────

if any(
    not np.isnan(summary[aoi]["EVI"][code][95])
    for aoi in AOIS for code in VALID_ECOZONE_CODES
):
    bar_figure(
        index_name="EVI",
        title="Peak EVI by Ecozone — p95 / p99 / p100 — Landsat C2",
        ylabel="Mean EVI (scene-level percentile averaged across all scenes)",
        outfile=FIGURES_DIR / "landsat_ecozone_evi_p95_p99_p100.png",
    )
else:
    print("\n[EVI] Bar figure skipped — no EVI cache found.")
    print("      Run: python Cache/build_landsat_cache.py --aoi north")
    print("      Then: python Cache/build_landsat_cache.py --aoi south")


# ── Figure 3: Ecological space scatter (p95) ──────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 7))

for aoi in AOIS:
    for code in VALID_ECOZONE_CODES:
        x = summary[aoi]["NDVI"][code][95]
        y = summary[aoi]["NDMI"][code][95]
        ax.scatter(
            x, y,
            color=ECOZONE_COLORS[code],
            marker=AOI_MARKERS[aoi],
            s=160,
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )
        short_aoi = AOI_DISPLAY[aoi].split()[0]
        ax.annotate(
            f"{ECOZONE_LABELS[code]} ({short_aoi})",
            xy=(x, y),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=8.5,
        )

ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

quadrant_style = dict(fontsize=8, color="#888888", style="italic", ha="center", va="center")
ax.text(0.80, 0.78, "Productive\n& Moist", transform=ax.transAxes, **quadrant_style)
ax.text(0.80, 0.12, "Productive\n& Dry",   transform=ax.transAxes, **quadrant_style)
ax.text(0.14, 0.12, "Low Prod.\n& Dry",    transform=ax.transAxes, **quadrant_style)
ax.text(0.14, 0.78, "Low Prod.\n& Moist",  transform=ax.transAxes, **quadrant_style)

eco_patches = [
    mpatches.Patch(color=ECOZONE_COLORS[c], label=ECOZONE_LABELS[c])
    for c in VALID_ECOZONE_CODES
]
aoi_handles = [
    plt.Line2D(
        [0], [0],
        marker=AOI_MARKERS[aoi], color="w",
        markerfacecolor="#888888", markeredgecolor="black",
        markersize=9, label=AOI_DISPLAY[aoi],
    )
    for aoi in AOIS
]
ax.legend(handles=eco_patches + aoi_handles, loc="lower right", fontsize=9, framealpha=0.9)

ax.set_xlabel("Mean p95 NDVI  (Vegetation Productivity)", fontsize=11)
ax.set_ylabel("Mean p95 NDMI  (Canopy Moisture)", fontsize=11)
ax.set_title(
    "Ecological Space — Peak Productivity vs Canopy Moisture\nby Ecozone and AOI  (p95) — Landsat C2",
    fontsize=12, fontweight="bold",
)
ax.grid(True, alpha=0.2, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out3 = FIGURES_DIR / "landsat_ecozone_ecological_space.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out3}")


# ── Spreadsheet ───────────────────────────────────────────────────────────────

rows = []
for aoi in AOIS:
    for code in VALID_ECOZONE_CODES:
        row = {
            "AOI":          AOI_DISPLAY[aoi],
            "Ecozone":      ECOZONE_LABELS[code],
            "Ecozone Code": code,
            "Pixel Count":  eco_pixel_counts[aoi][code],
        }
        for index_name in INDICES:
            for pct in PERCENTILES:
                row[f"{index_name} {PCT_LABELS[pct]}"] = round(
                    summary[aoi][index_name][code][pct], 6
                )
        rows.append(row)

df = pd.DataFrame(rows)
table_path = TABLES_DIR / "landsat_ecozone_peak_summary.xlsx"
df.to_excel(table_path, index=False, sheet_name="Ecozone Summary")
print(f"\nSaved: {table_path}")


print("\nAnalyses 1–3 (Landsat C2) complete.")
