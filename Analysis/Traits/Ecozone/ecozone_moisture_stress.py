#!/usr/bin/env python3
"""
ecozone_moisture_stress.py
--------------------------
Analysis 8 — TNC Appalachian Terrain–Vegetation Roadmap

Phenological moisture amplitude: growing-season p95 NDMI minus dormant-season
p95 NDMI, by ecozone.  (Axis G2 — phenological amplitude transformation)

Motivation: raw NDMI values include a persistent background signal from
evergreen understory (e.g. rhododendron) that stays moisture-active year-round.
Subtracting the dormant baseline isolates the canopy moisture response and
corrects for the north/south slope paradox flagged by TNC.

Season definitions:
  Growing : April–September   (months 4–9)   — green-up through peak canopy
  Dormant : November–March    (months 11–3)  — fully deciduous dormancy
  Excluded: October                          — senescence transition

Aggregation: all scenes in each season are pooled across all years.
Per scene, p95 NDMI is computed within each ecozone mask.
Season summary = mean of those scene-level p95 values.
Amplitude = growing_p95 - dormant_p95.

Ecozone VALUE field: 1=Cool  2=Intermediate  3=Hot  (0 excluded)

Outputs:
  Results/figures/
    ecozone_moisture_amplitude.png     amplitude bar chart by ecozone + AOI
    ecozone_moisture_seasons.png       growing vs dormant p95 side-by-side
  Results/tables/
    ecozone_moisture_stress.xlsx

Run from project root:
  python Analysis/Traits/Ecozone/ecozone_moisture_stress.py
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
MIN_PIXELS          = 100

GROWING_MONTHS = {4, 5, 6, 7, 8, 9}     # April–September
DORMANT_MONTHS = {11, 12, 1, 2, 3}      # November–March
# Month 10 (October) intentionally excluded — senescence transition

SEASONS        = ["growing", "dormant"]
SEASON_LABELS  = {"growing": "Growing (Apr–Sep)", "dormant": "Dormant (Nov–Mar)"}
SEASON_MONTHS  = {"growing": GROWING_MONTHS, "dormant": DORMANT_MONTHS}
SEASON_COLORS  = {"growing": "#5A9E6F", "dormant": "#7B9FBE"}

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


def seasonal_p95_by_ecozone(
    scenes: list[dict],
    eco_masks: dict[int, np.ndarray],
) -> dict[int, list[float]]:
    """
    For each scene, compute p95 NDMI within each ecozone mask.
    Returns {ecozone_code: [per_scene_p95_values]}.
    """
    results = {code: [] for code in VALID_ECOZONE_CODES}
    for i, scene in enumerate(scenes):
        if i % 50 == 0:
            print(f"      {i:>4}/{len(scenes)} scenes...", flush=True)
        with rasterio.open(scene["filepath"]) as src:
            data = src.read(1)
        valid = ~np.isnan(data)
        for code in VALID_ECOZONE_CODES:
            combined = eco_masks[code] & valid
            if combined.sum() >= MIN_PIXELS:
                results[code].append(float(np.percentile(data[combined], 95)))
    return results


# ── Main loop ──────────────────────────────────────────────────────────────────

# season_summary[aoi][season][code] = mean p95 NDMI across scenes in that season
season_summary: dict = {}
# scene_counts[aoi][season] = number of contributing scenes
scene_counts: dict = {}

for aoi in AOIS:
    cfg = get_aoi_config(aoi)
    print(f"\n{'='*62}")
    print(f"  {aoi.upper()} AOI — {AOI_DISPLAY[aoi]}")
    print(f"{'='*62}")

    ecozone_path = cfg.ecozone_dir / "tnc_ecozone_simplified_snapped.tif"
    with rasterio.open(ecozone_path) as src:
        ecozone = src.read(1)

    eco_masks = {code: (ecozone == code) for code in VALID_ECOZONE_CODES}

    index_dir     = cfg.index_cache_root / "NDMI"
    manifest_path = index_dir / "cache_manifest.json"

    print(f"\n  [NDMI] Loading scenes...")
    all_scenes = load_scenes(index_dir, manifest_path)
    print(f"    {len(all_scenes)} total scenes on disk")

    # Partition scenes by season
    season_scenes: dict[str, list[dict]] = {s: [] for s in SEASONS}
    excluded = 0
    for scene in all_scenes:
        m = scene["date"].month
        if m in GROWING_MONTHS:
            season_scenes["growing"].append(scene)
        elif m in DORMANT_MONTHS:
            season_scenes["dormant"].append(scene)
        else:
            excluded += 1   # October

    print(f"    Growing (Apr–Sep): {len(season_scenes['growing']):>4} scenes")
    print(f"    Dormant (Nov–Mar): {len(season_scenes['dormant']):>4} scenes")
    print(f"    Excluded (Oct):    {excluded:>4} scenes")

    season_summary[aoi] = {}
    scene_counts[aoi]   = {}

    for season in SEASONS:
        scenes = season_scenes[season]
        scene_counts[aoi][season] = len(scenes)

        print(f"\n  [{season}] Computing p95 NDMI per ecozone ({len(scenes)} scenes)...")
        per_scene = seasonal_p95_by_ecozone(scenes, eco_masks)

        season_summary[aoi][season] = {}
        for code in VALID_ECOZONE_CODES:
            vals = per_scene[code]
            season_summary[aoi][season][code] = (
                float(np.nanmean(vals)) if vals else np.nan
            )

    # Console: season values and amplitude
    print(f"\n  {'Ecozone':<14} {'Growing p95':>12}  {'Dormant p95':>12}  {'Amplitude':>10}")
    print(f"  {'-'*52}")
    for code in VALID_ECOZONE_CODES:
        g = season_summary[aoi]["growing"][code]
        d = season_summary[aoi]["dormant"][code]
        amp = g - d if not (np.isnan(g) or np.isnan(d)) else np.nan
        print(
            f"  {ECOZONE_LABELS[code]:<14}"
            f" {g:>12.4f}  {d:>12.4f}  {amp:>10.4f}"
        )


# ── Figure 1: Amplitude bar chart ─────────────────────────────────────────────
# x = ecozone, bars = amplitude (growing p95 - dormant p95), one panel per AOI

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
fig.suptitle(
    "Phenological Moisture Amplitude by Ecozone\n"
    "Growing-season p95 NDMI  minus  Dormant-season p95 NDMI  (Apr–Sep  vs  Nov–Mar)",
    fontsize=13, fontweight="bold", y=1.02,
)

x = np.arange(len(VALID_ECOZONE_CODES))

for ax, aoi in zip(axes, AOIS):
    amplitudes = []
    for code in VALID_ECOZONE_CODES:
        g   = season_summary[aoi]["growing"][code]
        d   = season_summary[aoi]["dormant"][code]
        amp = g - d if not (np.isnan(g) or np.isnan(d)) else np.nan
        amplitudes.append(amp)

    colors = [ECOZONE_COLORS[c] for c in VALID_ECOZONE_CODES]
    bars   = ax.bar(x, amplitudes, width=0.5, color=colors, edgecolor="white",
                    linewidth=0.7, zorder=3)

    for bar, val in zip(bars, amplitudes):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    ax.axhline(y=0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([ECOZONE_LABELS[c] for c in VALID_ECOZONE_CODES], fontsize=11)
    ax.set_title(AOI_DISPLAY[aoi], fontsize=12)
    ax.set_xlabel("Ecozone")
    ax.set_ylabel("NDMI Amplitude  (growing − dormant)" if aoi == "north" else "")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    valid = [v for v in amplitudes if not np.isnan(v)]
    if valid:
        ax.set_ylim(0, max(valid) + 0.06)

eco_patches = [
    mpatches.Patch(color=ECOZONE_COLORS[c], label=ECOZONE_LABELS[c])
    for c in VALID_ECOZONE_CODES
]
fig.legend(
    handles=eco_patches, loc="lower center", ncol=3,
    fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.06),
)

plt.tight_layout()
out1 = FIGURES_DIR / "ecozone_moisture_amplitude.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out1}")


# ── Figure 2: Growing vs dormant p95 side-by-side ─────────────────────────────
# Shows the raw season values alongside each other so amplitude context is clear

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
fig.suptitle(
    "NDMI by Season and Ecozone — Growing (Apr–Sep) vs Dormant (Nov–Mar)\n"
    "p95  |  all years pooled",
    fontsize=13, fontweight="bold", y=1.02,
)

bar_width = 0.3
x         = np.arange(len(VALID_ECOZONE_CODES))

for ax, aoi in zip(axes, AOIS):
    for i, season in enumerate(SEASONS):
        offset = (i - 0.5) * bar_width
        vals   = [season_summary[aoi][season][code] for code in VALID_ECOZONE_CODES]
        color  = SEASON_COLORS[season]
        n      = scene_counts[aoi][season]

        bars = ax.bar(
            x + offset, vals,
            width=bar_width,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=f"{SEASON_LABELS[season]}  (n={n})",
            zorder=3,
        )

        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([ECOZONE_LABELS[c] for c in VALID_ECOZONE_CODES], fontsize=11)
    ax.set_title(AOI_DISPLAY[aoi], fontsize=12)
    ax.set_xlabel("Ecozone")
    ax.set_ylabel("Mean p95 NDMI" if aoi == "north" else "")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    all_vals = [
        season_summary[aoi][s][code]
        for s in SEASONS for code in VALID_ECOZONE_CODES
    ]
    valid = [v for v in all_vals if not np.isnan(v)]
    if valid:
        ax.set_ylim(min(valid) - 0.04, max(valid) + 0.08)

fig.legend(
    loc="lower center", ncol=2,
    fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.07),
)

plt.tight_layout()
out2 = FIGURES_DIR / "ecozone_moisture_seasons.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")


# ── Spreadsheet ────────────────────────────────────────────────────────────────

rows = []
for aoi in AOIS:
    for code in VALID_ECOZONE_CODES:
        g   = season_summary[aoi]["growing"][code]
        d   = season_summary[aoi]["dormant"][code]
        amp = round(g - d, 6) if not (np.isnan(g) or np.isnan(d)) else np.nan
        rows.append({
            "AOI":                  AOI_DISPLAY[aoi],
            "Ecozone":              ECOZONE_LABELS[code],
            "Ecozone Code":         code,
            "Growing Scene Count":  scene_counts[aoi]["growing"],
            "Dormant Scene Count":  scene_counts[aoi]["dormant"],
            "Growing p95 NDMI":     round(g, 6),
            "Dormant p95 NDMI":     round(d, 6),
            "Amplitude (G - D)":    amp,
        })

df = pd.DataFrame(rows)
table_path = TABLES_DIR / "ecozone_moisture_stress.xlsx"
df.to_excel(table_path, index=False, sheet_name="Moisture Amplitude")
print(f"\nSaved: {table_path}")

print("\nAnalysis 8 complete.")
