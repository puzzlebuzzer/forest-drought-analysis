#!/usr/bin/env python3
"""
Prototype Sentinel investigation: fraction of ecozone pixels below baseline.

Question:
  For each month, what fraction of ecozone pixels fall below a neutral-year
  baseline in wet and dry years?

Conservative implementation:
  - Build a monthly neutral baseline raster per AOI / index / month using the
    median across all available neutral-year monthly composites.
  - For each wet-year and dry-year monthly composite, compute the fraction of
    finite ecozone pixels that fall below the corresponding neutral baseline
    raster value at the same pixel location.

What "below baseline" means here:
  A pixel is counted as below baseline when:

      monthly_value < neutral_baseline_value

  using the monthly composite raster and the neutral-year monthly median raster
  for the same AOI, index, and calendar month.

Why this implementation:
  - It is directly interpretable.
  - It uses the current monthly Sentinel data structure without adding heavier
    statistical modeling.
  - It preserves pixel-level spatial context, unlike a pure ecozone-summary
    threshold.

Limitations:
  - The baseline is a neutral-year monthly median raster, not a long-term
    climatology or a per-pixel trend model.
  - If neutral-year monthly coverage is sparse for a given month, the baseline
    may be less stable.

Outputs:
  Results/2_Anomaly_Onset/fraction_below_baseline/
    fraction_below_baseline_ndvi.png
    fraction_below_baseline_ndmi.png
    fraction_below_baseline_evi.png
    fraction_below_baseline.csv
    neutral_baseline_availability.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from investigation_common import (
    AOIS,
    AOI_DISPLAY,
    CLASS_COLORS,
    ECOZONE_COLORS,
    ECOZONE_LABELS,
    GROWING_MONTHS,
    INDICES,
    MONTH_NAMES,
    PROJECT_ROOT,
    VALID_ECOZONE_CODES,
    load_ecozone_masks,
    load_monthly_array,
    load_wet_dry_years,
)

OUT_DIR = PROJECT_ROOT / "Results" / "2_Anomaly_Onset" / "fraction_below_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

wet_dry = load_wet_dry_years()


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
    baseline = np.nanmedian(stack, axis=0)
    return baseline.astype(np.float32), len(arrays)


records: list[dict] = []
baseline_records: list[dict] = []

for aoi in AOIS:
    eco_masks = load_ecozone_masks(aoi)
    for index_name in INDICES:
        baseline_by_month: dict[int, np.ndarray | None] = {}
        for month in GROWING_MONTHS:
            baseline_arr, n_neutral = build_monthly_neutral_baseline(aoi, index_name, month)
            baseline_by_month[month] = baseline_arr
            baseline_records.append({
                "aoi": aoi,
                "aoi_label": AOI_DISPLAY[aoi],
                "index": index_name,
                "month": month,
                "month_name": MONTH_NAMES[month],
                "neutral_rasters_used": n_neutral,
                "baseline_available": baseline_arr is not None,
            })

        aoi_years = wet_dry[wet_dry["aoi"] == aoi].copy()
        for _, row in aoi_years.iterrows():
            year = int(row["year"])
            classification = row["classification"]
            for month in GROWING_MONTHS:
                baseline_arr = baseline_by_month[month]
                if baseline_arr is None:
                    continue
                arr = load_monthly_array(index_name, aoi, year, month)
                if arr is None:
                    continue

                finite = np.isfinite(arr) & np.isfinite(baseline_arr)
                for code in VALID_ECOZONE_CODES:
                    mask = finite & eco_masks[code]
                    valid_count = int(mask.sum())
                    if valid_count == 0:
                        frac = np.nan
                    else:
                        below = int((arr[mask] < baseline_arr[mask]).sum())
                        frac = below / valid_count

                    records.append({
                        "aoi": aoi,
                        "aoi_label": AOI_DISPLAY[aoi],
                        "index": index_name,
                        "year": year,
                        "month": month,
                        "month_name": MONTH_NAMES[month],
                        "classification": classification,
                        "ecozone_code": code,
                        "ecozone_label": ECOZONE_LABELS[code],
                        "fraction_below_baseline": frac,
                        "valid_pixel_count": valid_count,
                    })

results = pd.DataFrame(records).sort_values(
    ["index", "aoi", "classification", "year", "month", "ecozone_code"]
)
baseline_df = pd.DataFrame(baseline_records).sort_values(["index", "aoi", "month"])

results.to_csv(OUT_DIR / "fraction_below_baseline.csv", index=False)
baseline_df.to_csv(OUT_DIR / "neutral_baseline_availability.csv", index=False)


summary = (
    results.groupby(
        ["aoi", "aoi_label", "index", "classification", "ecozone_code", "ecozone_label", "month", "month_name"],
        dropna=False,
    )["fraction_below_baseline"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={
        "mean": "group_mean_fraction_below_baseline",
        "std": "group_sd_fraction_below_baseline",
        "count": "group_n",
    })
)

for index_name in INDICES:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(
        f"Fraction of Ecozone Pixels Below Neutral Baseline | {index_name}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    for ax, aoi in zip(axes, AOIS):
        subset = summary[
            (summary["aoi"] == aoi)
            & (summary["index"] == index_name)
            & (summary["classification"].isin(["wet", "dry"]))
        ].sort_values(["classification", "ecozone_code", "month"])

        for code in VALID_ECOZONE_CODES:
            color = ECOZONE_COLORS[code]
            for classification, linestyle in [("wet", "-"), ("dry", "--")]:
                curve = subset[
                    (subset["ecozone_code"] == code)
                    & (subset["classification"] == classification)
                ].sort_values("month")
                if curve.empty:
                    continue

                ax.plot(
                    curve["month"],
                    curve["group_mean_fraction_below_baseline"],
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.1,
                    marker="o",
                    markersize=4,
                    alpha=0.95,
                    label=f"{ECOZONE_LABELS[code]} ({classification})",
                )

        ax.axhline(0.5, color="#777777", linewidth=1.0, linestyle=":", alpha=0.7)
        ax.set_title(AOI_DISPLAY[aoi], fontsize=12)
        ax.set_xticks(GROWING_MONTHS)
        ax.set_xticklabels([MONTH_NAMES[m] for m in GROWING_MONTHS])
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Month")
        ax.set_ylabel("Fraction below neutral baseline" if aoi == "north" else "")
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.08),
        framealpha=0.9,
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(
        OUT_DIR / f"fraction_below_baseline_{index_name.lower()}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

print(f"Prepared fraction-below-baseline outputs in: {OUT_DIR}")
