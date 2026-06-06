#!/usr/bin/env python3
"""
Prototype Sentinel investigation: ecozone monthly distribution shape.

Question:
  How does the full monthly distribution shape differ across wet, dry, and
  neutral years for each AOI / ecozone / index combination?

Method:
  - Read existing Sentinel monthly composites from Results/0-CacheBaseData/monthly_max.
  - Reuse the existing AOI wet/dry labels from config/wet_dry_years.csv.
  - Reuse the existing AOI ecozone masks.
  - For each monthly raster, compute ecozone percentiles directly from the
    pixel distribution.
  - Summarize those percentile values by year-group (wet, neutral, dry),
    month, AOI, ecozone, and index.

Assumptions:
  - This is descriptive analysis only.
  - Percentile curves summarize the shape of each ecozone's monthly value
    distribution, not the trajectory of a single pixel or scene.
  - Months with insufficient finite pixels inside an ecozone return NaN for
    all requested percentiles.

Outputs:
  Results/3_Anomaly_Progression/distribution_shape/
    distribution_shape_ndvi.png
    distribution_shape_ndmi.png
    distribution_shape_evi.png
    distribution_shape_monthly_percentiles.csv
    distribution_shape_group_summary.csv
"""

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
    build_monthly_ecozone_dataframe,
)

SUMMARY_PERCENTILES = [25, 50, 75, 90, 99]

OUT_DIR = PROJECT_ROOT / "Results" / "3_Anomaly_Progression" / "distribution_shape"
OUT_DIR.mkdir(parents=True, exist_ok=True)


monthly = build_monthly_ecozone_dataframe(
    indices=INDICES,
    summary_percentiles=SUMMARY_PERCENTILES,
)

monthly = monthly[monthly["month"].isin(GROWING_MONTHS)].copy()

group_summary = (
    monthly.groupby(
        [
            "aoi",
            "aoi_label",
            "index",
            "classification",
            "ecozone_code",
            "ecozone_label",
            "month",
            "month_name",
            "summary_percentile",
        ],
        dropna=False,
    )["value"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={
        "mean": "group_mean_value",
        "std": "group_sd_value",
        "count": "group_n",
    })
)

monthly.to_csv(OUT_DIR / "distribution_shape_monthly_percentiles.csv", index=False)
group_summary.to_csv(OUT_DIR / "distribution_shape_group_summary.csv", index=False)


def percentile_line_style(percentile: int) -> dict:
    if percentile == 25:
        return {"linestyle": ":", "linewidth": 1.3, "alpha": 0.65}
    if percentile == 50:
        return {"linestyle": "-", "linewidth": 2.4, "alpha": 1.0}
    if percentile == 75:
        return {"linestyle": "--", "linewidth": 1.7, "alpha": 0.9}
    if percentile == 90:
        return {"linestyle": "-.", "linewidth": 1.5, "alpha": 0.8}
    return {"linestyle": "-", "linewidth": 1.2, "alpha": 0.55}


for index_name in INDICES:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=False)
    fig.suptitle(
        f"Monthly Distribution Shape by Ecozone and Year Group | {index_name}",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    for row_idx, aoi in enumerate(AOIS):
        for col_idx, code in enumerate(VALID_ECOZONE_CODES):
            ax = axes[row_idx, col_idx]
            subset = group_summary[
                (group_summary["aoi"] == aoi)
                & (group_summary["index"] == index_name)
                & (group_summary["ecozone_code"] == code)
            ].sort_values(["classification", "summary_percentile", "month"])

            for classification in ["wet", "neutral", "dry"]:
                class_subset = subset[subset["classification"] == classification]
                if class_subset.empty:
                    continue

                # Plot all requested percentile curves for the same class color.
                for percentile in SUMMARY_PERCENTILES:
                    curve = class_subset[
                        class_subset["summary_percentile"] == percentile
                    ].sort_values("month")
                    if curve.empty:
                        continue

                    style = percentile_line_style(percentile)
                    ax.plot(
                        curve["month"],
                        curve["group_mean_value"],
                        color=CLASS_COLORS[classification],
                        label=f"{classification} p{percentile}",
                        **style,
                    )

                # Light band between p25 and p75 to make the core shape easier to read.
                p25 = class_subset[class_subset["summary_percentile"] == 25].sort_values("month")
                p75 = class_subset[class_subset["summary_percentile"] == 75].sort_values("month")
                if len(p25) == len(GROWING_MONTHS) and len(p75) == len(GROWING_MONTHS):
                    ax.fill_between(
                        p25["month"],
                        p25["group_mean_value"],
                        p75["group_mean_value"],
                        color=CLASS_COLORS[classification],
                        alpha=0.08,
                        linewidth=0,
                    )

            ax.set_title(f"{AOI_DISPLAY[aoi]} | {ECOZONE_LABELS[code]}", fontsize=11)
            ax.set_xticks(GROWING_MONTHS)
            ax.set_xticklabels([MONTH_NAMES[m] for m in GROWING_MONTHS])
            ax.grid(True, alpha=0.2, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if col_idx == 0:
                ax.set_ylabel(f"{index_name} value")
            if row_idx == len(AOIS) - 1:
                ax.set_xlabel("Month")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.03),
        framealpha=0.9,
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(
        OUT_DIR / f"distribution_shape_{index_name.lower()}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

print(f"Prepared distribution-shape outputs in: {OUT_DIR}")
