#!/usr/bin/env python3
"""
ecozone_drought_response.py
---------------------------
Analyses 5, 6, and 7 — TNC Appalachian Terrain–Vegetation Roadmap

Analysis 5 — Wet vs. dry year peak NDVI by ecozone
Analysis 6 — Wet vs. dry year peak NDMI by ecozone
Analysis 7 — Ecological space shift: NDVI vs NDMI scatter by moisture year class
Analysis 5b — Wet vs. dry year peak EVI by ecozone  (if EVI cache exists)

Classification source: config/wet_dry_years.csv (USDM-derived, AOI-specific)
  wet     — growing-season W1 significantly exceeds D1
  neutral — balanced or near-normal growing season
  dry     — growing-season D1 significantly exceeds W1

Aggregation: all scenes from years belonging to a classification are pooled.
Scene-level p95/p99/p100 is computed within each ecozone mask, then the mean
across all contributing scenes is the summary value.

Ecozone VALUE field: 1=Cool  2=Intermediate  3=Hot  (0 excluded)

Outputs:
  Results/figures/
    ecozone_ndvi_drought_response.png
    ecozone_ndmi_drought_response.png
    ecozone_evi_drought_response.png   (only if EVI cache exists)
    ecozone_drought_ecological_space.png
  Results/tables/
    ecozone_drought_response.xlsx

Run from project root:
  python Analysis/Traits/Ecozone/ecozone_drought_response.py
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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
AOI_MARKERS         = {"north": "o", "south": "s"}
INDICES             = ["NDVI", "NDMI", "EVI"]
PERCENTILES         = [95, 99, 100]
PCT_LABELS          = {95: "p95", 99: "p99", 100: "p100 (max)"}
MIN_PIXELS          = 100

CLASSIFICATIONS = ["wet", "neutral", "dry"]
CLASS_COLORS    = {"wet": "#3A7FC1", "neutral": "#8C8C8C", "dry": "#C97834"}
CLASS_LABELS    = {"wet": "Wet years", "neutral": "Neutral years", "dry": "Dry years"}

FIGURES_DIR = project_path("results_figures_dir")
TABLES_DIR  = project_path("results_tables_dir")

for d in (FIGURES_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ── Load wet/dry year classification ──────────────────────────────────────────

wdy_path = project_path("config_dir") / "wet_dry_years.csv"
wdy_df   = pd.read_csv(wdy_path)

# {(aoi, year): classification}
year_class = {
    (row.aoi, int(row.year)): row.classification
    for _, row in wdy_df.iterrows()
}

print("Wet/dry year classification loaded:")
for aoi in AOIS:
    by_cls = {}
    for (a, yr), cls in year_class.items():
        if a == aoi:
            by_cls.setdefault(cls, []).append(yr)
    for cls in CLASSIFICATIONS:
        years = sorted(by_cls.get(cls, []))
        print(f"  {aoi:>5} / {cls:>8}: {years}")


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


def scene_percentiles_by_ecozone(
    scenes: list[dict],
    eco_masks: dict[int, np.ndarray],
    percentiles: list[int],
) -> dict[int, dict[int, list[float]]]:
    """Per-scene percentile extraction within each ecozone mask."""
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
                px = data[combined]
                for pct in percentiles:
                    results[code][pct].append(float(np.percentile(px, pct)))
    return results


# ── Main loop: both AOIs, both indices, all three classifications ──────────────

# summary[aoi][index][classification][code][pct] = mean across classified scenes
summary: dict = {}
# scene_counts[aoi][index][classification] = number of scenes
scene_counts: dict = {}

for aoi in AOIS:
    cfg = get_aoi_config(aoi)
    print(f"\n{'='*66}")
    print(f"  {aoi.upper()} AOI — {AOI_DISPLAY[aoi]}")
    print(f"{'='*66}")

    ecozone_path = cfg.ecozone_dir / "tnc_ecozone_simplified_snapped.tif"
    with rasterio.open(ecozone_path) as src:
        ecozone = src.read(1)

    eco_masks = {code: (ecozone == code) for code in VALID_ECOZONE_CODES}

    # Map year → classification for this AOI
    aoi_year_cls = {yr: cls for (a, yr), cls in year_class.items() if a == aoi}

    summary[aoi]      = {}
    scene_counts[aoi] = {}

    for index_name in INDICES:
        index_dir     = cfg.index_cache_root / index_name
        manifest_path = index_dir / "cache_manifest.json"

        if not manifest_path.exists():
            print(f"\n  [{index_name}] Skipping — cache manifest not found.")
            print(f"             Run: python Cache/build_sentinel_cache.py --aoi {aoi} --indices {index_name}")
            summary[aoi][index_name] = {
                cls: {code: {pct: np.nan for pct in PERCENTILES} for code in VALID_ECOZONE_CODES}
                for cls in CLASSIFICATIONS
            }
            scene_counts[aoi][index_name] = {cls: 0 for cls in CLASSIFICATIONS}
            continue

        print(f"\n  [{index_name}] Loading scenes...")
        all_scenes = load_scenes(index_dir, manifest_path)
        print(f"    {len(all_scenes)} total scenes on disk")

        # Group scenes by classification; skip years not in the CSV
        classified_scenes: dict[str, list[dict]] = {c: [] for c in CLASSIFICATIONS}
        unclassified = 0
        for scene in all_scenes:
            cls = aoi_year_cls.get(scene["date"].year)
            if cls in CLASSIFICATIONS:
                classified_scenes[cls].append(scene)
            else:
                unclassified += 1

        if unclassified:
            print(f"    {unclassified} scenes from years not in wet_dry_years.csv (skipped)")

        for cls in CLASSIFICATIONS:
            n     = len(classified_scenes[cls])
            years = sorted({s["date"].year for s in classified_scenes[cls]})
            print(f"    {cls:>8}: {n:>4} scenes  (years: {', '.join(str(y) for y in years)})")

        summary[aoi][index_name]      = {}
        scene_counts[aoi][index_name] = {}

        for cls in CLASSIFICATIONS:
            scenes = classified_scenes[cls]
            scene_counts[aoi][index_name][cls] = len(scenes)

            if not scenes:
                summary[aoi][index_name][cls] = {
                    code: {pct: np.nan for pct in PERCENTILES}
                    for code in VALID_ECOZONE_CODES
                }
                continue

            print(f"\n  [{index_name} / {cls}] Computing p95/p99/p100 per ecozone...")
            per_scene = scene_percentiles_by_ecozone(scenes, eco_masks, PERCENTILES)

            summary[aoi][index_name][cls] = {}
            for code in VALID_ECOZONE_CODES:
                summary[aoi][index_name][cls][code] = {}
                for pct in PERCENTILES:
                    vals = per_scene[code][pct]
                    summary[aoi][index_name][cls][code][pct] = (
                        float(np.nanmean(vals)) if vals else np.nan
                    )
            print(f"    {'Ecozone':<14} {'p95':>8}  {'p99':>8}  {'p100':>8}")
            for code in VALID_ECOZONE_CODES:
                d = summary[aoi][index_name][cls][code]
                print(
                    f"    {ECOZONE_LABELS[code]:<14}"
                    f" {d[95]:>8.4f}  {d[99]:>8.4f}  {d[100]:>8.4f}"
                )


# ── Figure helper: drought response bar chart ──────────────────────────────────

def drought_bar_figure(
    index_name: str,
    title: str,
    ylabel: str,
    outfile: Path,
) -> None:
    """
    Grouped bar chart: x = ecozone, bar groups = wet / neutral / dry (left→right).
    Bars show p95. P100 shown as a small diamond marker on top of each bar.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    bar_width = 0.22
    n_cls     = len(CLASSIFICATIONS)
    x         = np.arange(len(VALID_ECOZONE_CODES))

    for ax, aoi in zip(axes, AOIS):
        for i, cls in enumerate(CLASSIFICATIONS):
            offset    = (i - (n_cls - 1) / 2) * bar_width
            p95_vals  = [summary[aoi][index_name][cls][code][95]  for code in VALID_ECOZONE_CODES]
            p100_vals = [summary[aoi][index_name][cls][code][100] for code in VALID_ECOZONE_CODES]
            color     = CLASS_COLORS[cls]
            n_scenes  = scene_counts[aoi][index_name][cls]

            bars = ax.bar(
                x + offset, p95_vals,
                width=bar_width,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                label=f"{CLASS_LABELS[cls]}  (n={n_scenes})",
                zorder=3,
            )

            # p100 as a small diamond marker on each bar
            for bar, p100 in zip(bars, p100_vals):
                if not np.isnan(p100):
                    ax.plot(
                        bar.get_x() + bar.get_width() / 2,
                        p100,
                        marker="D",
                        markersize=4.5,
                        color=color,
                        markeredgecolor="#333333",
                        markeredgewidth=0.6,
                        zorder=5,
                    )

            # p95 value labels
            for bar, val in zip(bars, p95_vals):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{val:.3f}",
                        ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold",
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
            summary[aoi][index_name][cls][code][pct]
            for cls in CLASSIFICATIONS
            for code in VALID_ECOZONE_CODES
            for pct in PERCENTILES
        ]
        valid_vals = [v for v in all_vals if not np.isnan(v)]
        if valid_vals:
            ymin = max(0, min(valid_vals) - 0.04) if index_name in ("NDVI", "EVI") else min(valid_vals) - 0.04
            ymax = max(valid_vals) + 0.09
            ax.set_ylim(ymin, ymax)

    cls_patches = [
        mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_LABELS[c])
        for c in CLASSIFICATIONS
    ]
    p100_handle = plt.Line2D(
        [0], [0], marker="D", color="w",
        markerfacecolor="#666666", markeredgecolor="#333333",
        markeredgewidth=0.6, markersize=6,
        label="p100 (max) marker",
    )
    fig.legend(
        handles=cls_patches + [p100_handle],
        loc="lower center", ncol=4,
        fontsize=9, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.07),
    )

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {outfile}")


# ── Figures 1 and 2: NDVI and NDMI drought response ───────────────────────────

drought_bar_figure(
    index_name="NDVI",
    title="Peak Vegetation Productivity by Moisture Year — p95 NDVI by Ecozone\n"
          "(diamonds = p100 max;  bars = mean of all scenes in classified years)",
    ylabel="Mean p95 NDVI  (scene-level percentile averaged across classified scenes)",
    outfile=FIGURES_DIR / "ecozone_ndvi_drought_response.png",
)

drought_bar_figure(
    index_name="NDMI",
    title="Peak Canopy Moisture by Moisture Year — p95 NDMI by Ecozone\n"
          "(diamonds = p100 max;  bars = mean of all scenes in classified years)",
    ylabel="Mean p95 NDMI  (scene-level percentile averaged across classified scenes)",
    outfile=FIGURES_DIR / "ecozone_ndmi_drought_response.png",
)


# ── Figure 2b: EVI drought response (only if cache exists) ───────────────────

if any(
    not np.isnan(summary[aoi]["EVI"][cls][code][95])
    for aoi in AOIS for cls in CLASSIFICATIONS for code in VALID_ECOZONE_CODES
):
    drought_bar_figure(
        index_name="EVI",
        title="Peak EVI by Moisture Year — p95 by Ecozone\n"
              "(diamonds = p100 max;  bars = mean of all scenes in classified years)",
        ylabel="Mean p95 EVI  (scene-level percentile averaged across classified scenes)",
        outfile=FIGURES_DIR / "ecozone_evi_drought_response.png",
    )
else:
    print("\n[EVI] Drought bar figure skipped — no EVI cache found.")
    print("      Run: python Cache/build_sentinel_cache.py --aoi north --indices EVI")
    print("      Then: python Cache/build_sentinel_cache.py --aoi south --indices EVI")


# ── Figure 3: Ecological space shift ──────────────────────────────────────────
# x = p95 NDVI, y = p95 NDMI
# Fill color = moisture classification  |  Marker shape = AOI
# Ecozone shown via outline color
# Thin dashed lines connect wet → neutral → dry for each ecozone × AOI

fig, ax = plt.subplots(figsize=(9, 8))

for aoi in AOIS:
    for code in VALID_ECOZONE_CODES:
        eco_color = ECOZONE_COLORS[code]

        # Collect valid (x, y) points per classification for this ecozone × AOI
        pts = {}
        for cls in CLASSIFICATIONS:
            xv = summary[aoi]["NDVI"][cls][code][95]
            yv = summary[aoi]["NDMI"][cls][code][95]
            if not (np.isnan(xv) or np.isnan(yv)):
                pts[cls] = (xv, yv)

        # Connect points wet → neutral → dry with a thin dashed line
        ordered = [pts[c] for c in CLASSIFICATIONS if c in pts]
        if len(ordered) > 1:
            xs, ys = zip(*ordered)
            ax.plot(
                xs, ys,
                color=eco_color, linewidth=1.0,
                linestyle="--", alpha=0.55, zorder=2,
            )

        # Draw each classification point
        for cls, (xv, yv) in pts.items():
            ax.scatter(
                xv, yv,
                color=CLASS_COLORS[cls],
                marker=AOI_MARKERS[aoi],
                s=150,
                edgecolors=eco_color,
                linewidths=1.8,
                zorder=4,
            )

        # Label near the wet point (or whichever is leftmost)
        if pts:
            label_cls = "wet" if "wet" in pts else list(pts.keys())[0]
            ax.annotate(
                f"{ECOZONE_LABELS[code]} ({AOI_DISPLAY[aoi].split()[0]})",
                xy=pts[label_cls],
                xytext=(7, 4),
                textcoords="offset points",
                fontsize=7.5,
                color=eco_color,
            )

ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.7, alpha=0.45)

# Legend: classification fills + AOI shapes + ecozone outlines
cls_handles = [
    plt.Line2D(
        [0], [0], marker="o", color="w",
        markerfacecolor=CLASS_COLORS[c],
        markeredgecolor="gray",
        markersize=9, label=CLASS_LABELS[c],
    )
    for c in CLASSIFICATIONS
]
aoi_handles = [
    plt.Line2D(
        [0], [0], marker=AOI_MARKERS[aoi], color="w",
        markerfacecolor="#888888", markeredgecolor="black",
        markersize=9, label=AOI_DISPLAY[aoi],
    )
    for aoi in AOIS
]
eco_handles = [
    mpatches.Patch(
        facecolor="none", edgecolor=ECOZONE_COLORS[c],
        linewidth=2, label=f"{ECOZONE_LABELS[c]} ecozone",
    )
    for c in VALID_ECOZONE_CODES
]
ax.legend(
    handles=cls_handles + aoi_handles + eco_handles,
    loc="lower right", fontsize=8.5, framealpha=0.9, ncol=2,
)

ax.set_xlabel("Mean p95 NDVI  (Vegetation Productivity)", fontsize=11)
ax.set_ylabel("Mean p95 NDMI  (Canopy Moisture)", fontsize=11)
ax.set_title(
    "Ecological Space Shift by Moisture Year\n"
    "NDVI vs NDMI by Ecozone  (p95)  —  dashed lines connect wet → neutral → dry",
    fontsize=12, fontweight="bold",
)
ax.grid(True, alpha=0.2, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out_scatter = FIGURES_DIR / "ecozone_drought_ecological_space.png"
plt.savefig(out_scatter, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_scatter}")


# ── Spreadsheet ────────────────────────────────────────────────────────────────

rows = []
for aoi in AOIS:
    for cls in CLASSIFICATIONS:
        for code in VALID_ECOZONE_CODES:
            row = {
                "AOI":            AOI_DISPLAY[aoi],
                "Classification": cls.capitalize(),
                "Ecozone":        ECOZONE_LABELS[code],
                "Ecozone Code":   code,
            }
            for index_name in INDICES:
                n = scene_counts[aoi][index_name][cls]
                row[f"{index_name} Scene Count"] = n
                for pct in PERCENTILES:
                    row[f"{index_name} {PCT_LABELS[pct]}"] = round(
                        summary[aoi][index_name][cls][code][pct], 6
                    )
            rows.append(row)

df = pd.DataFrame(rows)
table_path = TABLES_DIR / "ecozone_drought_response.xlsx"
df.to_excel(table_path, index=False, sheet_name="Drought Response")
print(f"\nSaved: {table_path}")

print("\nAnalyses 5–7 complete.")
