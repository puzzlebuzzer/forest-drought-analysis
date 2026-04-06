#!/usr/bin/env python3
"""
Lightweight Sentinel spatial consistency diagnostic for anomaly maps.

Purpose:
  Provide a quick spatial check on whether anomaly signals are widespread or
  patchy for selected AOIs, indices, months, and year selections.

Diagnostic logic:
  - Build a monthly neutral baseline raster per AOI / index / month using the
    median across available neutral-year monthly composites.
  - Compute anomaly rasters as:

      anomaly = monthly_composite - neutral_baseline

  - Save simple spatial PNG maps plus a small coverage-summary CSV.

Coverage summary:
  - negative_fraction : fraction of valid AOI pixels with anomaly < -NEAR_ZERO
  - near_zero_fraction: fraction of valid AOI pixels with |anomaly| <= NEAR_ZERO
  - positive_fraction : fraction of valid AOI pixels with anomaly >  NEAR_ZERO

Interpretation:
  - High negative_fraction suggests anomaly is spatially widespread and below
    baseline over much of the AOI.
  - High positive_fraction suggests widespread above-baseline conditions.
  - High near_zero_fraction suggests weak or spatially mixed anomalies.

Parameterization:
  Edit the user settings below to target one or more months, indices, AOIs,
  and either year-groups or explicit years.

Outputs:
  Results/Other/spatial_consistency_check/
    anomaly_<index>_<aoi>_<year>_<month>.png
    spatial_consistency_summary.csv
    baseline_availability.csv
"""

from pathlib import Path
import gc
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from investigation_common import (
    AOIS,
    AOI_DISPLAY,
    INDICES,
    MONTH_NAMES,
    PROJECT_ROOT,
    load_monthly_array,
    load_wet_dry_years,
)

# ── User settings ─────────────────────────────────────────────────────────────

SELECTED_AOIS = ["north", "south"]
SELECTED_INDICES = ["NDMI", "NDVI"]
SELECTED_MONTHS = [4, 7, 10]

# Use one of the following approaches:
# 1. Keep SELECTED_YEARS empty and use year-groups from wet_dry_years.csv
# 2. Provide explicit years in SELECTED_YEARS
SELECTED_YEAR_GROUPS = ["dry", "wet"]
SELECTED_YEARS: list[int] = []

INCLUDE_EVI_IF_EASY = False

# Small threshold used for the positive / near-zero / negative coverage split.
NEAR_ZERO = 0.02

CMAP = "RdBu_r"

# ── Paths / outputs ───────────────────────────────────────────────────────────

OUT_DIR = PROJECT_ROOT / "Results" / "Other" / "spatial_consistency_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

wet_dry = load_wet_dry_years()


def resolve_indices() -> list[str]:
    indices = list(SELECTED_INDICES)
    if INCLUDE_EVI_IF_EASY and "EVI" not in indices:
        indices.append("EVI")
    return indices


def build_monthly_neutral_baseline(aoi: str, index_name: str, month: int) -> tuple[np.ndarray | None, int]:
    neutral_years = sorted(
        wet_dry[
            (wet_dry["aoi"] == aoi)
            & (wet_dry["classification"] == "neutral")
        ]["year"].unique()
    )

    arrays = []
    for year in neutral_years:
        arr = load_monthly_array(index_name, aoi, int(year), month)
        if arr is not None:
            arrays.append(arr)

    if not arrays:
        return None, 0

    stack = np.stack(arrays, axis=0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        baseline = np.nanmedian(stack, axis=0)
    return baseline.astype(np.float32), len(arrays)


def years_to_process(aoi: str) -> list[tuple[int, str]]:
    if SELECTED_YEARS:
        rows = wet_dry[wet_dry["aoi"] == aoi]
        year_class = {
            int(r["year"]): r["classification"]
            for _, r in rows.iterrows()
        }
        return [(year, year_class.get(year, "unclassified")) for year in SELECTED_YEARS]

    rows = wet_dry[
        (wet_dry["aoi"] == aoi)
        & (wet_dry["classification"].isin(SELECTED_YEAR_GROUPS))
    ].sort_values("year")
    return [(int(r["year"]), r["classification"]) for _, r in rows.iterrows()]


def anomaly_limits(percentile_samples: list[np.ndarray]) -> tuple[float, float]:
    if not percentile_samples:
        return -1.0, 1.0
    vals = np.concatenate(percentile_samples)
    if vals.size == 0:
        return -1.0, 1.0
    lim = float(np.nanpercentile(np.abs(vals), 98))
    lim = max(lim, 0.05)
    return -lim, lim


def anomaly_sample(anomaly: np.ndarray, sample_size: int = 250_000) -> np.ndarray:
    finite = np.isfinite(anomaly)
    valid_count = int(finite.sum())
    if valid_count == 0:
        return np.array([], dtype=np.float32)

    # For large rasters, avoid materializing every finite pixel just to derive a
    # plotting range. A coarse grid sample is sufficient for percentile-based
    # color scaling and keeps memory bounded.
    if valid_count <= sample_size:
        return anomaly[finite].astype(np.float32, copy=False)

    stride = max(int(np.ceil(np.sqrt(valid_count / sample_size))), 1)
    sampled = anomaly[::stride, ::stride]
    sampled = sampled[np.isfinite(sampled)]
    if sampled.size == 0:
        sampled = anomaly[finite]
    if sampled.size > sample_size:
        step = max(sampled.size // sample_size, 1)
        sampled = sampled[::step][:sample_size]
    return sampled.astype(np.float32, copy=False)


def compute_anomaly(arr: np.ndarray, baseline_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(arr) & np.isfinite(baseline_arr)
    anomaly = np.full(arr.shape, np.nan, dtype=np.float32)
    np.subtract(arr, baseline_arr, out=anomaly, where=finite)
    return anomaly, finite


baseline_records: list[dict] = []
summary_records: list[dict] = []
percentile_samples: list[np.ndarray] = []

for aoi in SELECTED_AOIS:
    for index_name in resolve_indices():
        for month in SELECTED_MONTHS:
            baseline_arr, n_neutral = build_monthly_neutral_baseline(aoi, index_name, month)
            baseline_records.append({
                "aoi": aoi,
                "aoi_label": AOI_DISPLAY.get(aoi, aoi),
                "index": index_name,
                "month": month,
                "month_name": MONTH_NAMES[month],
                "neutral_rasters_used": n_neutral,
                "baseline_available": baseline_arr is not None,
            })
            if baseline_arr is None:
                continue

            for year, classification in years_to_process(aoi):
                arr = load_monthly_array(index_name, aoi, year, month)
                if arr is None:
                    continue

                anomaly, finite = compute_anomaly(arr, baseline_arr)
                valid_count = int(finite.sum())
                if valid_count == 0:
                    negative_fraction = np.nan
                    near_zero_fraction = np.nan
                    positive_fraction = np.nan
                else:
                    negative_fraction = float((anomaly[finite] < -NEAR_ZERO).sum() / valid_count)
                    near_zero_fraction = float((np.abs(anomaly[finite]) <= NEAR_ZERO).sum() / valid_count)
                    positive_fraction = float((anomaly[finite] > NEAR_ZERO).sum() / valid_count)

                summary_records.append({
                    "aoi": aoi,
                    "aoi_label": AOI_DISPLAY.get(aoi, aoi),
                    "index": index_name,
                    "year": year,
                    "classification": classification,
                    "month": month,
                    "month_name": MONTH_NAMES[month],
                    "valid_pixel_count": valid_count,
                    "negative_fraction": negative_fraction,
                    "near_zero_fraction": near_zero_fraction,
                    "positive_fraction": positive_fraction,
                })
                percentile_samples.append(anomaly_sample(anomaly))

                del arr, anomaly, finite

            del baseline_arr
            gc.collect()


baseline_df = pd.DataFrame(baseline_records).sort_values(["index", "aoi", "month"])
summary_df = pd.DataFrame(summary_records).sort_values(["index", "aoi", "year", "month"])

baseline_df.to_csv(OUT_DIR / "baseline_availability.csv", index=False)
summary_df.to_csv(OUT_DIR / "spatial_consistency_summary.csv", index=False)

vmin, vmax = anomaly_limits(percentile_samples)

for aoi in SELECTED_AOIS:
    for index_name in resolve_indices():
        for month in SELECTED_MONTHS:
            baseline_arr, _ = build_monthly_neutral_baseline(aoi, index_name, month)
            if baseline_arr is None:
                continue

            for year, classification in years_to_process(aoi):
                arr = load_monthly_array(index_name, aoi, year, month)
                if arr is None:
                    continue

                anomaly, _ = compute_anomaly(arr, baseline_arr)

                fig, ax = plt.subplots(figsize=(7, 7))
                im = ax.imshow(anomaly, cmap=CMAP, vmin=vmin, vmax=vmax)
                ax.set_title(
                    f"{index_name} | {AOI_DISPLAY.get(aoi, aoi)}\n"
                    f"{classification} {year} | {MONTH_NAMES[month]} anomaly vs neutral baseline",
                    fontsize=11,
                )
                ax.set_xticks([])
                ax.set_yticks([])

                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Anomaly (monthly composite - neutral baseline)")

                out_path = OUT_DIR / f"anomaly_{index_name.lower()}_{aoi}_{year}_{month:02d}.png"
                plt.tight_layout()
                plt.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

                del arr, anomaly

            del baseline_arr
            gc.collect()

print(f"Prepared spatial consistency outputs in: {OUT_DIR}")
