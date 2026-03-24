#!/usr/bin/env python3
"""
ecozone_longterm_trend.py
-------------------------
Long-term annual vegetation trends from Landsat C2 (1984–present).

This analysis is UNIQUE TO LANDSAT. There is no S2 parallel.

For each year from 1984–present, computes the annual mean of per-scene p95
within each ecozone. Produces an annual time series of p95 by ecozone for
each spectral index.

Outputs:
  Results/figures_landsat/
    landsat_ndvi_longterm_trend.png
    landsat_ndmi_longterm_trend.png
    landsat_evi_longterm_trend.png   (only if EVI cache exists)
  Results/tables_landsat/
    landsat_longterm_trend.xlsx

Each figure shows:
  - 1×2 subplots (north AOI | south AOI)
  - Lines for each ecozone (colored by ECOZONE_COLORS)
  - Solid thin line (1.5 px) for annual p95 values with circle markers
  - Dashed thicker line (2.5 px) for 5-year rolling mean (min_periods=3)
  - Shaded background bands for wet/dry years (from wet_dry_years.csv)
    * Wet years: light blue (alpha=0.12)
    * Dry years: light orange (alpha=0.12)
    * Only years in 2017–2025 range

Run from project root:
  python Analysis/Traits/Ecozone/ecozone_longterm_trend.py
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

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
INDICES            = ["NDVI", "NDMI", "EVI"]
MIN_PIXELS         = 100

FIGURES_DIR = project_path("results_figures_landsat_dir")
TABLES_DIR  = project_path("results_tables_landsat_dir")

for d in (FIGURES_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ── Load wet/dry year classifications ──────────────────────────────────────────

def load_wet_dry_years() -> dict[str, dict[int, str]]:
    """
    Load wet/dry year classifications from config/wet_dry_years.csv.
    Returns: {aoi: {year: classification}}
    """
    wet_dry_path = project_path("config_dir") / "wet_dry_years.csv"
    wd = defaultdict(dict)

    if wet_dry_path.exists():
        df = pd.read_csv(wet_dry_path)
        for _, row in df.iterrows():
            aoi = row["aoi"]
            year = int(row["year"])
            classification = row["classification"]
            wd[aoi][year] = classification

    return dict(wd)


WET_DRY_YEARS = load_wet_dry_years()


# ── Main loop: both AOIs, all indices ──────────────────────────────────────────

# trend[aoi][index_name][year][code] = mean p95 across scenes in that year
trend: dict = {}
scene_counts: dict = {}  # trend_counts[aoi][index_name][year][code] = scene count

for aoi in AOIS:
    print(f"\n{'='*62}")
    print(f"  {aoi.upper()} AOI — {AOI_DISPLAY[aoi]}")
    print(f"{'='*62}")

    # Load ecozone array and create masks
    try:
        ecozone_arr, _, _, _ = load_landsat_ecozone(aoi)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        continue

    eco_masks = {code: (ecozone_arr == code) for code in VALID_ECOZONE_CODES}

    for code in VALID_ECOZONE_CODES:
        pixel_count = int(eco_masks[code].sum())
        print(f"    {ECOZONE_LABELS[code]:>12}: {pixel_count:>10,} px")

    trend[aoi] = {}
    scene_counts[aoi] = {}

    for index_name in INDICES:
        print(f"\n  [{index_name}] Loading Landsat scenes...")

        scenes = load_landsat_scenes(aoi, index_name)

        if not scenes:
            print(f"      Skipping {index_name} — no scenes available.")
            trend[aoi][index_name] = {code: {} for code in VALID_ECOZONE_CODES}
            scene_counts[aoi][index_name] = {code: {} for code in VALID_ECOZONE_CODES}
            continue

        print(f"    {len(scenes)} scenes loaded")

        # Group scenes by year
        year_scenes = defaultdict(list)
        for scene in scenes:
            year = scene["date"].year
            year_scenes[year].append(scene)

        print(f"    Year range: {min(year_scenes.keys())}–{max(year_scenes.keys())}")

        # For each year, compute annual mean p95 by ecozone
        trend[aoi][index_name] = {code: {} for code in VALID_ECOZONE_CODES}
        scene_counts[aoi][index_name] = {code: {} for code in VALID_ECOZONE_CODES}

        for year in sorted(year_scenes.keys()):
            year_list = year_scenes[year]

            # Per-ecozone: collect all p95 values across scenes in this year
            eco_p95_lists = {code: [] for code in VALID_ECOZONE_CODES}

            for scene in year_list:
                with rasterio.open(scene["filepath"]) as src:
                    data = src.read(1)

                valid = ~np.isnan(data)

                for code in VALID_ECOZONE_CODES:
                    combined = eco_masks[code] & valid
                    if combined.sum() >= MIN_PIXELS:
                        pixel_vals = data[combined]
                        p95_val = float(np.percentile(pixel_vals, 95))
                        eco_p95_lists[code].append(p95_val)

            # Compute mean p95 per ecozone for this year
            for code in VALID_ECOZONE_CODES:
                if eco_p95_lists[code]:
                    mean_p95 = float(np.nanmean(eco_p95_lists[code]))
                    trend[aoi][index_name][code][year] = mean_p95
                    scene_counts[aoi][index_name][code][year] = len(eco_p95_lists[code])
                else:
                    trend[aoi][index_name][code][year] = np.nan
                    scene_counts[aoi][index_name][code][year] = 0

        # Summary
        print(f"\n  [{index_name}] Annual trend summary:")
        for code in VALID_ECOZONE_CODES:
            years_with_data = [
                year for year in trend[aoi][index_name][code]
                if not np.isnan(trend[aoi][index_name][code][year])
            ]
            if years_with_data:
                vals = [
                    trend[aoi][index_name][code][year]
                    for year in years_with_data
                ]
                print(
                    f"    {ECOZONE_LABELS[code]:>12}: "
                    f"{len(years_with_data)} years (range: {min(vals):.4f}–{max(vals):.4f})"
                )


# ── Figure generation ──────────────────────────────────────────────────────────

def plot_longterm_trend(index_name: str) -> None:
    """
    Plot long-term annual p95 trends by ecozone for one index.
    Shows 1×2 subplots (north | south) with wet/dry year shading.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"Long-term {index_name} Trend by Ecozone — Landsat C2 (1984–2025)",
        fontsize=14, fontweight="bold", y=0.98
    )

    for ax, aoi in zip(axes, AOIS):
        if index_name not in trend[aoi]:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        # Collect all years across all ecozones
        all_years = set()
        for code in VALID_ECOZONE_CODES:
            all_years.update(trend[aoi][index_name][code].keys())

        if not all_years:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        all_years = sorted(all_years)
        year_min, year_max = min(all_years), max(all_years)

        # Shade wet/dry years (only if in wet_dry_years.csv)
        aoi_wd = WET_DRY_YEARS.get(aoi, {})
        for year in all_years:
            if year in aoi_wd:
                classification = aoi_wd[year]
                if classification == "wet":
                    ax.axvspan(year - 0.5, year + 0.5, color="blue", alpha=0.12, zorder=0)
                elif classification == "dry":
                    ax.axvspan(year - 0.5, year + 0.5, color="orange", alpha=0.12, zorder=0)

        # Plot per-ecozone trends
        for code in VALID_ECOZONE_CODES:
            yearly_data = trend[aoi][index_name][code]

            # Collect years and values in order
            years = sorted([y for y in all_years if y in yearly_data])
            values = [yearly_data[y] for y in years]

            # Filter out NaN values for plotting
            valid_idx = [i for i, v in enumerate(values) if not np.isnan(v)]
            plot_years = [years[i] for i in valid_idx]
            plot_vals = [values[i] for i in valid_idx]

            if not plot_vals:
                continue

            # Annual solid line with markers
            ax.plot(
                plot_years, plot_vals,
                color=ECOZONE_COLORS[code],
                linewidth=1.5,
                marker="o",
                markersize=4,
                label=ECOZONE_LABELS[code],
                zorder=2,
            )

            # 5-year rolling mean (dashed, thicker)
            years_array = np.array(plot_years)
            vals_array = np.array(plot_vals)

            # Use pandas rolling for easier handling
            s = pd.Series(vals_array, index=years_array)
            rolling = s.rolling(window=5, min_periods=3, center=True).mean()

            # Only plot rolling mean where it exists (min_periods=3 respected by pandas)
            rolling_years = rolling.index[rolling.notna()].tolist()
            rolling_vals = rolling[rolling.notna()].values.tolist()

            if rolling_vals:
                ax.plot(
                    rolling_years, rolling_vals,
                    color=ECOZONE_COLORS[code],
                    linewidth=2.5,
                    linestyle="--",
                    zorder=2.5,
                )

        ax.set_xlim(year_min - 1, year_max + 1)
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel(f"Mean Annual p95 {index_name}", fontsize=11)
        ax.set_title(AOI_DISPLAY[aoi], fontsize=12)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend
    eco_patches = [
        mpatches.Patch(color=ECOZONE_COLORS[c], label=ECOZONE_LABELS[c])
        for c in VALID_ECOZONE_CODES
    ]
    trend_lines = [
        plt.Line2D([0], [0], color="gray", linewidth=1.5, marker="o", markersize=4,
                   label="Annual values"),
        plt.Line2D([0], [0], color="gray", linewidth=2.5, linestyle="--",
                   label="5-year rolling mean"),
    ]
    wet_dry_patches = [
        mpatches.Patch(facecolor="blue", alpha=0.12, label="Wet years"),
        mpatches.Patch(facecolor="orange", alpha=0.12, label="Dry years"),
    ]

    fig.legend(
        handles=eco_patches + trend_lines + wet_dry_patches,
        loc="lower center",
        ncol=len(eco_patches) + len(trend_lines) + len(wet_dry_patches),
        fontsize=9,
        framealpha=0.95,
        bbox_to_anchor=(0.5, -0.08),
    )

    plt.tight_layout()
    outfile = FIGURES_DIR / f"landsat_{index_name.lower()}_longterm_trend.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {outfile}")


# Check if any indices have data before plotting
for index_name in INDICES:
    has_data = False
    for aoi in AOIS:
        if index_name in trend[aoi]:
            for code in VALID_ECOZONE_CODES:
                yearly = trend[aoi][index_name][code]
                if any(not np.isnan(v) for v in yearly.values()):
                    has_data = True
                    break
        if has_data:
            break

    if has_data:
        print(f"\n[{index_name}] Generating long-term trend figure...")
        plot_longterm_trend(index_name)
    else:
        print(f"\n[{index_name}] Skipped — no data available.")


# ── Spreadsheet ────────────────────────────────────────────────────────────────

rows = []
for aoi in AOIS:
    for index_name in INDICES:
        if index_name not in trend[aoi]:
            continue

        for code in VALID_ECOZONE_CODES:
            yearly_data = trend[aoi][index_name][code]
            year_counts = scene_counts[aoi][index_name][code]

            for year in sorted(yearly_data.keys()):
                row = {
                    "AOI":          AOI_DISPLAY[aoi],
                    "Index":        index_name,
                    "Year":         year,
                    "Ecozone":      ECOZONE_LABELS[code],
                    "Ecozone Code": code,
                    "p95":          round(yearly_data[year], 6),
                    "Scene Count":  year_counts.get(year, 0),
                }
                rows.append(row)

if rows:
    df = pd.DataFrame(rows)
    table_path = TABLES_DIR / "landsat_longterm_trend.xlsx"
    df.to_excel(table_path, index=False, sheet_name="Long-term Trend")
    print(f"\nSaved: {table_path}")
else:
    print("\nNo data to write to spreadsheet.")

print("\nLong-term trend analysis complete.")
