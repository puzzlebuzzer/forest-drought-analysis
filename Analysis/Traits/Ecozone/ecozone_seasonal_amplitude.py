#!/usr/bin/env python3
"""
ecozone_seasonal_amplitude.py
------------------------------
Analysis 9 — TNC Appalachian Terrain–Vegetation Roadmap

Annual seasonal amplitude: summer peak minus winter baseline, per ecozone,
per year (2017–present).  Computed from the monthly max composites, so
this script is fast (one GeoTIFF per month rather than hundreds of scenes).

Why amplitude?
  Seasonal amplitude = summer peak − winter baseline.
  If the 2022 Sentinel-2 processing baseline (PB04.00) introduced a purely
  additive reflectance offset, the summer and winter values would both
  increase by the same amount and the difference would be stable across the
  2022 transition.  A step in amplitude would indicate the offset is
  non-uniform across the growing/dormant season.

Season definitions:
  Summer  : June–August    (months 6–8)   — peak canopy
  Winter  : December–February             — dormant / leaf-off
  (same as the "summer" and "winter" concept the user asked about)

Per-year per-ecozone summary:
  1. For each month in the season window, read the monthly max composite
     and extract median pixel value within the ecozone mask.
  2. Season value = mean of those monthly medians (only months that exist).
  3. Amplitude = summer_median - winter_median.

Important: winter spans the calendar boundary (Dec of year Y, Jan+Feb of Y+1).
  We pair the Dec of year Y-1 with the summer of year Y (so winter precedes
  the growing season it bookends), matching the convention used for dormancy.

Outputs:
  Results/figures/
    ecozone_ndvi_amplitude_timeseries.png   — NDVI amplitude by year, both AOIs
    ecozone_ndmi_amplitude_timeseries.png   — NDMI amplitude by year
    ecozone_evi_amplitude_timeseries.png    — EVI amplitude by year
    ecozone_amplitude_all_indices.png       — NDVI / NDMI / EVI grid
  Results/tables/
    ecozone_seasonal_amplitude.xlsx

Run from project root:
  python Analysis/Traits/Ecozone/ecozone_seasonal_amplitude.py
"""

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
INDICES             = ["NDVI", "NDMI", "EVI"]
VALID_ECOZONE_CODES = [1, 2, 3]
ECOZONE_LABELS      = {1: "Cool", 2: "Intermediate", 3: "Hot"}
ECOZONE_COLORS      = {1: "#4E90C8", 2: "#72B063", 3: "#D9534F"}
AOI_DISPLAY         = {"north": "GW National Forest", "south": "Great Smoky Mtns"}
MIN_PIXELS          = 100    # minimum valid pixels to accept a monthly composite
YEAR_FLOOR          = 2017

# Summer: Jun–Aug.  Winter: Dec (prior year) + Jan–Feb (current year).
SUMMER_MONTHS = [6, 7, 8]
WINTER_MONTHS_PREV_DEC = [12]      # taken from year - 1
WINTER_MONTHS_CURR     = [1, 2]    # taken from current year

MONTHLY_MAX_DIR = project_path("monthly_max_dir")
FIGURES_DIR     = project_path("results_figures_dir")
TABLES_DIR      = project_path("results_tables_dir")

for d in (FIGURES_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_monthly_composite(monthly_dir: Path, year: int, month: int) -> np.ndarray | None:
    """Return float32 array for the given year-month, or None if not on disk."""
    path = monthly_dir / f"{year}_{month:02d}.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def ecozone_median(arr: np.ndarray, eco_masks: dict[int, np.ndarray]) -> dict[int, float]:
    """Compute median within each ecozone mask; NaN if too few valid pixels."""
    out = {}
    for code, mask in eco_masks.items():
        valid = mask & np.isfinite(arr)
        if valid.sum() >= MIN_PIXELS:
            out[code] = float(np.nanmedian(arr[valid]))
        else:
            out[code] = np.nan
    return out


def season_mean(
    monthly_dir: Path,
    year: int,
    months: list[int],
    eco_masks: dict[int, np.ndarray],
    alt_year: int | None = None,
) -> dict[int, float]:
    """
    Average the per-ecozone medians across `months` for `year`.
    If `alt_year` is given those months are drawn from alt_year instead.
    Returns NaN for an ecozone if no months had enough data.
    """
    yr = alt_year if alt_year is not None else year
    stacks: dict[int, list[float]] = {c: [] for c in VALID_ECOZONE_CODES}

    for m in months:
        arr = load_monthly_composite(monthly_dir, yr, m)
        if arr is None:
            continue
        meds = ecozone_median(arr, eco_masks)
        for code in VALID_ECOZONE_CODES:
            if not np.isnan(meds[code]):
                stacks[code].append(meds[code])

    return {
        code: float(np.mean(stacks[code])) if stacks[code] else np.nan
        for code in VALID_ECOZONE_CODES
    }


# ── Main computation ───────────────────────────────────────────────────────────

# amplitude[aoi][index][year][ecozone_code] = float
amplitude: dict = {}
# raw season values for the spreadsheet
records: list[dict] = []

for aoi in AOIS:
    cfg = get_aoi_config(aoi)
    amplitude[aoi] = {}

    ecozone_path = cfg.ecozone_dir / "tnc_ecozone_simplified_snapped.tif"
    with rasterio.open(ecozone_path) as src:
        ecozone_arr = src.read(1)

    eco_masks = {code: (ecozone_arr == code) for code in VALID_ECOZONE_CODES}

    print(f"\n{'='*62}")
    print(f"  {aoi.upper()} AOI — {AOI_DISPLAY[aoi]}")
    print(f"{'='*62}")

    for index_name in INDICES:
        monthly_dir = MONTHLY_MAX_DIR / f"{index_name.lower()}_{aoi}"
        if not monthly_dir.exists():
            print(f"  [{index_name}] monthly_max dir not found — skipping.")
            amplitude[aoi][index_name] = {}
            continue

        # Discover available years
        tif_years = sorted({
            int(p.stem.split("_")[0])
            for p in monthly_dir.glob("*.tif")
            if p.stem.split("_")[0].isdigit()
            and int(p.stem.split("_")[0]) >= YEAR_FLOOR
        })

        amplitude[aoi][index_name] = {}
        print(f"\n  [{index_name}] Years: {tif_years}")
        print(f"  {'Year':<8}", end="")
        for code in VALID_ECOZONE_CODES:
            print(f"  {ECOZONE_LABELS[code]:>14}", end="")
        print()

        for year in tif_years:
            # Summer of this year
            summer = season_mean(monthly_dir, year, SUMMER_MONTHS, eco_masks)

            # Winter = Dec of year-1 + Jan,Feb of year
            winter_dec  = season_mean(
                monthly_dir, year, WINTER_MONTHS_PREV_DEC, eco_masks,
                alt_year=year - 1,
            )
            winter_jf   = season_mean(monthly_dir, year, WINTER_MONTHS_CURR, eco_masks)

            # Combine Dec(Y-1) + Jan(Y) + Feb(Y) into single mean
            winter: dict[int, float] = {}
            for code in VALID_ECOZONE_CODES:
                vals = [v for v in [winter_dec[code], winter_jf[code]] if not np.isnan(v)]
                winter[code] = float(np.mean(vals)) if vals else np.nan

            amp: dict[int, float] = {}
            for code in VALID_ECOZONE_CODES:
                if not np.isnan(summer[code]) and not np.isnan(winter[code]):
                    amp[code] = summer[code] - winter[code]
                else:
                    amp[code] = np.nan

            amplitude[aoi][index_name][year] = amp

            print(f"  {year:<8}", end="")
            for code in VALID_ECOZONE_CODES:
                val = amp[code]
                print(f"  {val:>14.4f}" if not np.isnan(val) else f"  {'---':>14}", end="")
            print()

            for code in VALID_ECOZONE_CODES:
                records.append({
                    "AOI":            AOI_DISPLAY[aoi],
                    "Index":          index_name,
                    "Year":           year,
                    "Ecozone":        ECOZONE_LABELS[code],
                    "Ecozone Code":   code,
                    "Summer (Jun-Aug) median": round(summer[code], 6)  if not np.isnan(summer[code]) else None,
                    "Winter (Dec-Feb) median": round(winter[code], 6)  if not np.isnan(winter[code]) else None,
                    "Amplitude (Summer-Winter)": round(amp[code], 6)   if not np.isnan(amp[code])    else None,
                })


# ── Spreadsheet ───────────────────────────────────────────────────────────────

df = pd.DataFrame(records)
table_path = TABLES_DIR / "ecozone_seasonal_amplitude.xlsx"
df.to_excel(table_path, index=False, sheet_name="Annual Amplitude by Ecozone")
print(f"\nSaved table: {table_path}")


# ── Figure helper ─────────────────────────────────────────────────────────────

def amplitude_figure(
    index_name: str,
    title: str,
    ylabel: str,
    outfile: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    # 2022 baseline shift annotation
    SHIFT_YEAR = 2022

    for ax, aoi in zip(axes, AOIS):
        amp_data = amplitude.get(aoi, {}).get(index_name, {})
        years = sorted(amp_data.keys())
        if not years:
            ax.set_title(f"{AOI_DISPLAY[aoi]} — no data")
            continue

        # Shaded annotation for 2022 shift
        ax.axvspan(SHIFT_YEAR - 0.4, SHIFT_YEAR + 0.4, color="#FFA500",
                   alpha=0.15, label="S2 baseline shift (PB04.00)")
        ax.axvline(SHIFT_YEAR, color="#FFA500", linewidth=1.0,
                   linestyle="--", alpha=0.7)

        for code in VALID_ECOZONE_CODES:
            color = ECOZONE_COLORS[code]
            vals  = [amp_data[yr].get(code, np.nan) for yr in years]
            ax.plot(
                years, vals,
                color=color, linewidth=2.0, marker="o", markersize=5,
                label=ECOZONE_LABELS[code], zorder=3,
            )
            # Trend line (where data exists)
            valid_pairs = [(yr, v) for yr, v in zip(years, vals) if not np.isnan(v)]
            if len(valid_pairs) >= 3:
                xs, ys = zip(*valid_pairs)
                z = np.polyfit(xs, ys, 1)
                p = np.poly1d(z)
                ax.plot(xs, p(np.array(xs)), color=color,
                        linewidth=1.0, linestyle=":", alpha=0.6, zorder=2)

        ax.set_title(AOI_DISPLAY[aoi], fontsize=12)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel if aoi == "north" else "")
        ax.set_xticks(years)
        ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right", fontsize=8)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend
    eco_handles = [
        mpatches.Patch(color=ECOZONE_COLORS[c], label=ECOZONE_LABELS[c])
        for c in VALID_ECOZONE_CODES
    ]
    shift_handle = mpatches.Patch(
        facecolor="#FFA500", alpha=0.3, label="S2 PB04.00 baseline shift"
    )
    trend_handle = plt.Line2D(
        [0], [0], color="#888", linewidth=1.0, linestyle=":",
        label="Linear trend (dotted)"
    )
    fig.legend(
        handles=eco_handles + [shift_handle, trend_handle],
        loc="lower center", ncol=5, fontsize=9,
        framealpha=0.9, bbox_to_anchor=(0.5, -0.08),
    )

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")


# ── Per-index figures ─────────────────────────────────────────────────────────

for idx_name, ylabel_unit in [
    ("NDVI", "Amplitude  (NDVI units)"),
    ("NDMI", "Amplitude  (NDMI units)"),
    ("EVI",  "Amplitude  (EVI units)"),
]:
    has_data = any(
        amplitude.get(aoi, {}).get(idx_name, {})
        for aoi in AOIS
    )
    if has_data:
        amplitude_figure(
            index_name=idx_name,
            title=(
                f"Annual Seasonal Amplitude — {idx_name}\n"
                f"(Summer Jun–Aug median  −  Winter Dec–Feb median)  |  by ecozone"
            ),
            ylabel=ylabel_unit,
            outfile=FIGURES_DIR / f"ecozone_{idx_name.lower()}_amplitude_timeseries.png",
        )
    else:
        print(f"\n[{idx_name}] No amplitude data — skipping figure.")


# ── Multi-index summary grid ───────────────────────────────────────────────────
# 3-row × 2-col grid: rows = NDVI / NDMI / EVI, cols = north / south

valid_indices = [
    idx for idx in INDICES
    if any(amplitude.get(aoi, {}).get(idx) for aoi in AOIS)
]

if valid_indices:
    n_rows = len(valid_indices)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4.5 * n_rows), sharex=False)
    if n_rows == 1:
        axes = [axes]   # ensure 2-D indexing works

    fig.suptitle(
        "Annual Seasonal Amplitude — All Indices\n"
        "(Summer Jun–Aug  minus  Winter Dec–Feb, median by ecozone)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for row_i, idx_name in enumerate(valid_indices):
        for col_j, aoi in enumerate(AOIS):
            ax = axes[row_i][col_j]
            amp_data = amplitude.get(aoi, {}).get(idx_name, {})
            years = sorted(amp_data.keys())

            if not years:
                ax.set_title(f"{idx_name} – {AOI_DISPLAY[aoi]} — no data", fontsize=10)
                continue

            ax.axvspan(2022 - 0.4, 2022 + 0.4, color="#FFA500",
                       alpha=0.12, zorder=0)
            ax.axvline(2022, color="#FFA500", linewidth=0.9,
                       linestyle="--", alpha=0.6, zorder=1)

            for code in VALID_ECOZONE_CODES:
                color = ECOZONE_COLORS[code]
                vals  = [amp_data[yr].get(code, np.nan) for yr in years]
                ax.plot(
                    years, vals, color=color,
                    linewidth=1.8, marker="o", markersize=4,
                    label=ECOZONE_LABELS[code], zorder=3,
                )

            ax.set_title(f"{idx_name} — {AOI_DISPLAY[aoi]}", fontsize=10)
            ax.set_xlabel("Year" if row_i == n_rows - 1 else "")
            ax.set_ylabel(f"Amplitude ({idx_name})" if col_j == 0 else "")
            ax.set_xticks(years)
            ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right", fontsize=7)
            ax.grid(True, alpha=0.18, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    eco_handles = [
        mpatches.Patch(color=ECOZONE_COLORS[c], label=ECOZONE_LABELS[c])
        for c in VALID_ECOZONE_CODES
    ]
    shift_handle = mpatches.Patch(
        facecolor="#FFA500", alpha=0.3, label="S2 PB04.00 shift"
    )
    fig.legend(
        handles=eco_handles + [shift_handle],
        loc="lower center", ncol=4, fontsize=9,
        framealpha=0.9, bbox_to_anchor=(0.5, -0.04),
    )

    plt.tight_layout()
    grid_out = FIGURES_DIR / "ecozone_amplitude_all_indices.png"
    plt.savefig(grid_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved summary grid: {grid_out}")

print("\nAnalysis 9 complete.")
