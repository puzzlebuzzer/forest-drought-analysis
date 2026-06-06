#!/usr/bin/env python3
"""
ecozone_seasonal_curves.py
--------------------------
Analysis 4 — TNC Appalachian Ecozone–Vegetation Roadmap

Seasonal peak timing: monthly p95 and p100 NDVI, NDMI, and EVI by ecozone.

For each AOI, all cached scenes are grouped by calendar month (pooled
across all years). Per scene, p95 and p100 are computed within each
ecozone mask. Per month, the mean of those scene-level values is the
summary statistic.

Questions addressed:
  - Do cooler ecozones green up later / lose greenness earlier?
  - Do hot ecozones show earlier summer moisture decline?
  - Are NDMI moisture curves synchronized with NDVI greenness curves?

Ecozone VALUE field: 1=Cool  2=Intermediate  3=Hot  (0 excluded)

Outputs:
  Results/figures/
    ecozone_ndvi_seasonal.png        NDVI curves by ecozone, both AOIs
    ecozone_ndmi_seasonal.png        NDMI curves by ecozone, both AOIs
    ecozone_evi_seasonal.png         EVI curves by ecozone  (only if EVI cache exists)
    ecozone_seasonal_sync.png        NDVI + NDMI overlaid (synchronization)
  Results/tables/
    ecozone_seasonal_summary.xlsx    Monthly p95 / p100 per ecozone

Run from project root:
  python Analysis/SupportingAnalyses/EcozoneInvestigations/ecozone_seasonal_curves.py
  python Analysis/SupportingAnalyses/EcozoneInvestigations/ecozone_seasonal_curves.py --scene-level
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
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
INDICES             = ["NDVI", "NDMI", "EVI"]
MONTHS              = list(range(1, 13))
MONTH_NAMES         = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MIN_PIXELS          = 100

FIGURES_DIR = project_path("results_figures_dir")
TABLES_DIR  = project_path("results_tables_dir")

for d in (FIGURES_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)


parser = argparse.ArgumentParser(description="Ecozone seasonal curves")
parser.add_argument(
    "--scene-level",
    action="store_true",
    help=(
        "Skip monthly aggregation in the figure outputs and plot one point per scene "
        "using the scene acquisition month on the x-axis. Writes alternate scene-level "
        "figure and table names."
    ),
)
parser.add_argument(
    "--points-only",
    action="store_true",
    help=(
        "When used with --scene-level, plot only scene-level points and omit the "
        "monthly median overlay line."
    ),
)
args = parser.parse_args()
PLOT_SCENE_LEVEL = args.scene_level
POINTS_ONLY = args.points_only

if POINTS_ONLY and not PLOT_SCENE_LEVEL:
    parser.error("--points-only requires --scene-level")


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_scenes(index_dir: Path, manifest_path: Path) -> list[dict]:
    """Return scenes present in the manifest and on disk, sorted by date."""
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


def monthly_percentiles_by_ecozone(
    scenes: list[dict],
    eco_masks: dict[int, np.ndarray],
    collect_monthly: bool = True,
) -> tuple[dict[int, dict[int, dict[int, list[float]]]], list[dict]]:
    """
    For each scene, compute p95 and p100 within each ecozone mask.
    Group results by calendar month. Single raster read per scene.

    Returns:
      - {month: {ecozone_code: {95: [scene_vals], 100: [scene_vals]}}}
      - scene-level records for optional point plotting / export
    """
    results = (
        {
            m: {code: {95: [], 100: []} for code in VALID_ECOZONE_CODES}
            for m in MONTHS
        }
        if collect_monthly else {}
    )
    scene_records: list[dict] = []

    for i, scene in enumerate(scenes):
        if i % 50 == 0:
            print(f"      {i:>4}/{len(scenes)} scenes...", flush=True)
        month = scene["date"].month
        with rasterio.open(scene["filepath"]) as src:
            data = src.read(1)
        valid = ~np.isnan(data)
        for code in VALID_ECOZONE_CODES:
            combined = eco_masks[code] & valid
            if combined.sum() >= MIN_PIXELS:
                px = data[combined]
                p95 = float(np.percentile(px, 95))
                p100 = float(np.percentile(px, 100))
                if collect_monthly:
                    results[month][code][95].append(p95)
                    results[month][code][100].append(p100)
                scene_records.append({
                    "scene_date": scene["date"],
                    "scene_month": month,
                    "ecozone_code": code,
                    "p95": p95,
                    "p100": p100,
                })

    return results, scene_records


# ── Main loop: both AOIs, both indices ────────────────────────────────────────

# monthly_summary[aoi][index][month][code][pct] = mean across scenes
monthly_summary: dict = {}
# scene_counts[aoi][index][month] = number of contributing scenes
scene_counts: dict = {}
scene_level_records: list[dict] = []

for aoi in AOIS:
    cfg = get_aoi_config(aoi)
    print(f"\n{'='*62}")
    print(f"  {aoi.upper()} AOI — {AOI_DISPLAY[aoi]}")
    print(f"{'='*62}")

    ecozone_path = cfg.ecozone_dir / "tnc_ecozone_simplified_snapped.tif"
    with rasterio.open(ecozone_path) as src:
        ecozone = src.read(1)

    eco_masks = {code: (ecozone == code) for code in VALID_ECOZONE_CODES}
    monthly_summary[aoi] = {}
    scene_counts[aoi]    = {}

    for index_name in INDICES:
        index_dir     = cfg.index_cache_root / index_name
        manifest_path = index_dir / "cache_manifest.json"

        if not manifest_path.exists():
            print(f"\n  [{index_name}] Skipping — cache manifest not found.")
            print(f"             Run: python Cache/build_sentinel_cache.py --aoi {aoi} --indices {index_name}")
            monthly_summary[aoi][index_name] = {
                m: {code: {95: np.nan, 100: np.nan} for code in VALID_ECOZONE_CODES}
                for m in MONTHS
            }
            scene_counts[aoi][index_name] = {m: 0 for m in MONTHS}
            continue

        print(f"\n  [{index_name}] Loading scenes...")
        scenes = load_scenes(index_dir, manifest_path)
        print(f"    {len(scenes)} scenes on disk")

        if PLOT_SCENE_LEVEL and POINTS_ONLY:
            print(f"  [{index_name}] Computing scene-level p95/p100 per ecozone...")
        else:
            print(f"  [{index_name}] Computing monthly p95/p100 per ecozone...")
        raw, per_scene_records = monthly_percentiles_by_ecozone(
            scenes,
            eco_masks,
            collect_monthly=not (PLOT_SCENE_LEVEL and POINTS_ONLY),
        )
        for record in per_scene_records:
            scene_level_records.append({
                "AOI": AOI_DISPLAY[aoi],
                "AOI Key": aoi,
                "Index": index_name,
                "Scene Date": record["scene_date"].date().isoformat(),
                "Month": record["scene_month"],
                "Month Name": MONTH_NAMES[record["scene_month"] - 1],
                "Ecozone": ECOZONE_LABELS[record["ecozone_code"]],
                "Ecozone Code": record["ecozone_code"],
                "p95": round(record["p95"], 6),
                "p100 (max)": round(record["p100"], 6),
            })

        if PLOT_SCENE_LEVEL and POINTS_ONLY:
            continue

        monthly_summary[aoi][index_name] = {}
        scene_counts[aoi][index_name]    = {}

        for m in MONTHS:
            monthly_summary[aoi][index_name][m] = {}
            # Scene count = max contributing scenes across ecozones for this month
            scene_counts[aoi][index_name][m] = max(
                len(raw[m][code][95]) for code in VALID_ECOZONE_CODES
            )
            for code in VALID_ECOZONE_CODES:
                monthly_summary[aoi][index_name][m][code] = {
                    95:  float(np.nanmean(raw[m][code][95]))  if raw[m][code][95]  else np.nan,
                    100: float(np.nanmean(raw[m][code][100])) if raw[m][code][100] else np.nan,
                }

        # Console summary
        print(f"\n    {'':14}", end="")
        for name in MONTH_NAMES:
            print(f" {name:>6}", end="")
        print()
        for code in VALID_ECOZONE_CODES:
            print(f"    {ECOZONE_LABELS[code]:<14}", end="")
            for m in MONTHS:
                val = monthly_summary[aoi][index_name][m][code][95]
                print(f" {val:>6.3f}" if not np.isnan(val) else f" {'---':>6}", end="")
            print()
        print(f"    {'Scene count':<14}", end="")
        for m in MONTHS:
            print(f" {scene_counts[aoi][index_name][m]:>6}", end="")
        print()


# ── Figure helper: seasonal line chart ────────────────────────────────────────

def seasonal_figure(
    index_name: str,
    title: str,
    ylabel: str,
    outfile: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    for ax, aoi in zip(axes, AOIS):
        for code in VALID_ECOZONE_CODES:
            color     = ECOZONE_COLORS[code]
            p95_vals  = [monthly_summary[aoi][index_name][m][code][95]  for m in MONTHS]
            p100_vals = [monthly_summary[aoi][index_name][m][code][100] for m in MONTHS]

            # p95 main line
            ax.plot(
                MONTHS, p95_vals,
                color=color, linewidth=2.2, marker="o", markersize=4,
                label=ECOZONE_LABELS[code], zorder=3,
            )

            # p95–p100 shaded band
            pairs = [(m, lo, hi) for m, lo, hi in zip(MONTHS, p95_vals, p100_vals)
                     if not (np.isnan(lo) or np.isnan(hi))]
            if pairs:
                xs, lows, highs = zip(*pairs)
                ax.fill_between(xs, lows, highs, alpha=0.12, color=color, linewidth=0, zorder=2)

        # Scene count as subtle background bars on a secondary axis
        counts = [scene_counts[aoi][index_name][m] for m in MONTHS]
        ax_cnt = ax.twinx()
        ax_cnt.bar(MONTHS, counts, alpha=0.07, color="gray", width=0.8, zorder=0)
        ax_cnt.set_ylabel("Scene count", fontsize=8, color="#aaaaaa")
        ax_cnt.tick_params(axis="y", labelsize=7, labelcolor="#aaaaaa")
        ax_cnt.set_ylim(0, max(counts) * 6)   # keeps bars unobtrusive
        ax_cnt.spines["top"].set_visible(False)

        ax.set_title(AOI_DISPLAY[aoi], fontsize=12)
        ax.set_xlabel("Month")
        ax.set_ylabel(ylabel if aoi == "north" else "")
        ax.set_xticks(MONTHS)
        ax.set_xticklabels(MONTH_NAMES, fontsize=9)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.set_zorder(ax_cnt.get_zorder() + 1)
        ax.patch.set_visible(False)

    # Legend (ecozone colors) + p95/p100 band note — placed outside panels
    eco_handles = [
        mpatches.Patch(color=ECOZONE_COLORS[c], label=ECOZONE_LABELS[c])
        for c in VALID_ECOZONE_CODES
    ]
    band_handle = mpatches.Patch(
        facecolor="gray", alpha=0.3, label="p95–p100 range"
    )
    fig.legend(
        handles=eco_handles + [band_handle],
        loc="lower center", ncol=4, fontsize=9,
        framealpha=0.9, bbox_to_anchor=(0.5, -0.06),
    )

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {outfile}")


def seasonal_scene_figure(
    index_name: str,
    title: str,
    ylabel: str,
    outfile: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    scene_df = pd.DataFrame(scene_level_records)
    if not scene_df.empty:
        scene_df["Scene Date"] = pd.to_datetime(scene_df["Scene Date"])

    for ax, aoi in zip(axes, AOIS):
        for code in VALID_ECOZONE_CODES:
            color = ECOZONE_COLORS[code]
            subset = scene_df[
                (scene_df["AOI Key"] == aoi)
                & (scene_df["Index"] == index_name)
                & (scene_df["Ecozone Code"] == code)
            ].sort_values("Scene Date").copy()
            if subset.empty:
                continue

            ax.plot(
                subset["Scene Date"],
                subset["p95"],
                color=color,
                alpha=0.8,
                linewidth=1.4,
                marker="o",
                markersize=2.8,
                label=ECOZONE_LABELS[code],
                zorder=3,
            )

            if not POINTS_ONLY:
                valid_band = subset[["Scene Date", "p95", "p100 (max)"]].dropna()
                if not valid_band.empty:
                    ax.fill_between(
                        valid_band["Scene Date"],
                        valid_band["p95"],
                        valid_band["p100 (max)"],
                        alpha=0.1,
                        color=color,
                        linewidth=0,
                        zorder=2,
                    )

        ax.set_title(AOI_DISPLAY[aoi], fontsize=12)
        ax.set_xlabel("Scene date")
        ax.set_ylabel(ylabel if aoi == "north" else "")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    eco_handles = [
        mpatches.Patch(color=ECOZONE_COLORS[c], label=ECOZONE_LABELS[c])
        for c in VALID_ECOZONE_CODES
    ]
    line_handle = plt.Line2D(
        [0], [0], color="#666666", linewidth=1.5, marker="o",
        alpha=0.8, markersize=4, label="Scene-level p95"
    )
    handles = eco_handles + [line_handle]
    if not POINTS_ONLY:
        band_handle = mpatches.Patch(
            facecolor="gray", alpha=0.3, label="p95–p100 range"
        )
        handles.append(band_handle)
    fig.legend(
        handles=handles,
        loc="lower center", ncol=5 if not POINTS_ONLY else 4, fontsize=9,
        framealpha=0.9, bbox_to_anchor=(0.5, -0.06),
    )

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {outfile}")


# ── Figure 1: NDVI seasonal ───────────────────────────────────────────────────

if PLOT_SCENE_LEVEL:
    seasonal_scene_figure(
        index_name="NDVI",
        title=(
            "Seasonal Peak Vegetation — Scene-level p95 NDVI by Ecozone\n"
            + ("(scene-date line plot)" if POINTS_ONLY else "(scene-date line plot, shaded band = p95 to p100)")
        ),
        ylabel="Scene-level p95 NDVI",
        outfile=FIGURES_DIR / "ecozone_ndvi_seasonal_scenelevel.png",
    )
else:
    seasonal_figure(
        index_name="NDVI",
        title="Seasonal Peak Vegetation — Monthly p95 NDVI by Ecozone\n(shaded band = p95 to p100)",
        ylabel="Mean monthly p95 NDVI",
        outfile=FIGURES_DIR / "ecozone_ndvi_seasonal.png",
    )


# ── Figure 2: NDMI seasonal ───────────────────────────────────────────────────

if PLOT_SCENE_LEVEL:
    seasonal_scene_figure(
        index_name="NDMI",
        title=(
            "Seasonal Canopy Moisture — Scene-level p95 NDMI by Ecozone\n"
            + ("(scene-date line plot)" if POINTS_ONLY else "(scene-date line plot, shaded band = p95 to p100)")
        ),
        ylabel="Scene-level p95 NDMI",
        outfile=FIGURES_DIR / "ecozone_ndmi_seasonal_scenelevel.png",
    )
else:
    seasonal_figure(
        index_name="NDMI",
        title="Seasonal Canopy Moisture — Monthly p95 NDMI by Ecozone\n(shaded band = p95 to p100)",
        ylabel="Mean monthly p95 NDMI",
        outfile=FIGURES_DIR / "ecozone_ndmi_seasonal.png",
    )


# ── Figure 2b: EVI seasonal (only if cache exists) ───────────────────────────

evi_available = False
if PLOT_SCENE_LEVEL and POINTS_ONLY:
    evi_available = any(
        record["Index"] == "EVI"
        for record in scene_level_records
    )
else:
    evi_available = any(
        not np.isnan(monthly_summary[aoi]["EVI"][m][code][95])
        for aoi in AOIS for m in MONTHS for code in VALID_ECOZONE_CODES
    )

if evi_available:
    if PLOT_SCENE_LEVEL:
        seasonal_scene_figure(
            index_name="EVI",
            title=(
                "Seasonal EVI — Scene-level p95 by Ecozone\n"
                + ("(scene-date line plot)" if POINTS_ONLY else "(scene-date line plot, shaded band = p95 to p100)")
            ),
            ylabel="Scene-level p95 EVI",
            outfile=FIGURES_DIR / "ecozone_evi_seasonal_scenelevel.png",
        )
    else:
        seasonal_figure(
            index_name="EVI",
            title="Seasonal EVI — Monthly p95 by Ecozone\n(shaded band = p95 to p100)",
            ylabel="Mean monthly p95 EVI",
            outfile=FIGURES_DIR / "ecozone_evi_seasonal.png",
        )
else:
    print("\n[EVI] Seasonal figure skipped — no EVI cache found.")
    print("      Run: python Cache/build_sentinel_cache.py --aoi north --indices EVI")
    print("      Then: python Cache/build_sentinel_cache.py --aoi south --indices EVI")


# ── Figure 3: NDVI + NDMI synchronization ────────────────────────────────────
# Solid lines = NDVI (left y-axis)  |  Dashed lines = NDMI (right y-axis)
# Ecozone color is consistent across both indices.

if not PLOT_SCENE_LEVEL:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Seasonal Synchronization — NDVI Greenness vs NDMI Moisture by Ecozone\n"
        "(solid = NDVI, left axis  |  dashed = NDMI, right axis)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, aoi in zip(axes, AOIS):
        ax2 = ax.twinx()

        for code in VALID_ECOZONE_CODES:
            color     = ECOZONE_COLORS[code]
            ndvi_vals = [monthly_summary[aoi]["NDVI"][m][code][95] for m in MONTHS]
            ndmi_vals = [monthly_summary[aoi]["NDMI"][m][code][95] for m in MONTHS]

            ax.plot(
                MONTHS, ndvi_vals,
                color=color, linewidth=2.2, linestyle="-",
                marker="o", markersize=3.5, zorder=3,
            )
            ax2.plot(
                MONTHS, ndmi_vals,
                color=color, linewidth=2.2, linestyle="--",
                marker="s", markersize=3.5, zorder=3,
            )

        ax.set_title(AOI_DISPLAY[aoi], fontsize=12)
        ax.set_xlabel("Month")
        ax.set_ylabel("p95 NDVI  (solid)", fontsize=10)
        ax2.set_ylabel("p95 NDMI  (dashed)", fontsize=10)
        ax.set_xticks(MONTHS)
        ax.set_xticklabels(MONTH_NAMES, fontsize=9)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    # Legend: ecozone colors + line style key
    eco_handles = [
        mpatches.Patch(color=ECOZONE_COLORS[c], label=ECOZONE_LABELS[c])
        for c in VALID_ECOZONE_CODES
    ]
    style_handles = [
        plt.Line2D([0], [0], color="#555", linewidth=2, linestyle="-",  label="NDVI  (solid)"),
        plt.Line2D([0], [0], color="#555", linewidth=2, linestyle="--", label="NDMI  (dashed)"),
    ]
    fig.legend(
        handles=eco_handles + style_handles,
        loc="lower center", ncol=5, fontsize=9,
        framealpha=0.9, bbox_to_anchor=(0.5, -0.07),
    )

    plt.tight_layout()
    sync_out = FIGURES_DIR / "ecozone_seasonal_sync.png"
    plt.savefig(sync_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {sync_out}")
else:
    print("Skipping monthly synchronization figure in --scene-level mode.")


# ── Spreadsheet ───────────────────────────────────────────────────────────────

rows = []
table_path = TABLES_DIR / "ecozone_seasonal_summary.xlsx"
if PLOT_SCENE_LEVEL:
    table_path = TABLES_DIR / "ecozone_seasonal_scenelevel_summary.xlsx"
    df = pd.DataFrame(scene_level_records)
else:
    for aoi in AOIS:
        for index_name in INDICES:
            for m in MONTHS:
                for code in VALID_ECOZONE_CODES:
                    d = monthly_summary[aoi][index_name][m][code]
                    rows.append({
                        "AOI":          AOI_DISPLAY[aoi],
                        "Index":        index_name,
                        "Month":        m,
                        "Month Name":   MONTH_NAMES[m - 1],
                        "Ecozone":      ECOZONE_LABELS[code],
                        "Ecozone Code": code,
                        "Scene Count":  scene_counts[aoi][index_name][m],
                        "p95":          round(d[95],  6),
                        "p100 (max)":   round(d[100], 6),
                    })
    df = pd.DataFrame(rows)
df.to_excel(table_path, index=False, sheet_name="Monthly by Ecozone")
print(f"\nSaved: {table_path}")

print("\nAnalysis 4 complete.")
