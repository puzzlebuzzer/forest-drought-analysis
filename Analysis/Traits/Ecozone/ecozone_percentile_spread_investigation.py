#!/usr/bin/env python3
"""
Prototype Sentinel investigation: percentile spread / compression by ecozone.

Question:
  Do wet, dry, and neutral years differ in how spread out the ecozone value
  distribution is through the growing season?

Method:
  - Read existing Sentinel monthly composites from Results/0-CacheBaseData/monthly_max.
  - Reuse the existing AOI wet/dry labels from config/wet_dry_years.csv.
  - Reuse the existing AOI ecozone masks.
  - For each monthly raster, compute ecozone percentiles directly from the
    pixel distribution.
  - Derive descriptive spread metrics:
      p99 - p50   : upper-tail spread / upper-end heterogeneity
      p75 - p25   : core interquartile spread / middle-distribution heterogeneity
  - Summarize those metrics by AOI, ecozone, index, month, and year-group.

Ecological interpretation notes:
  - p99 - p50 increases when the highest-value portion of the ecozone pulls
    away from the central tendency, suggesting stronger upper-tail heterogeneity.
  - p75 - p25 increases when the middle of the ecozone distribution broadens,
    suggesting greater within-ecozone spread across the bulk of pixels.
  - These are descriptive spread metrics, not variance models or significance tests.

Outputs:
  Results/3_Anomaly_Progression/percentile_spread/
    percentile_spread_ndvi.png
    percentile_spread_ndmi.png
    percentile_spread_evi.png
    percentile_spread_monthly_percentiles.csv
    percentile_spread_metrics.csv
    percentile_spread_group_summary.csv
"""

import matplotlib.pyplot as plt
import pandas as pd

from investigation_common import (
    AOIS,
    AOI_DISPLAY,
    CLASS_COLORS,
    ECOZONE_LABELS,
    GROWING_MONTHS,
    INDICES,
    MONTH_NAMES,
    PROJECT_ROOT,
    VALID_ECOZONE_CODES,
    build_monthly_ecozone_dataframe,
)

SUMMARY_PERCENTILES = [25, 50, 75, 99]
SPREAD_METRICS = [
    ("p99_minus_p50", 99, 50, "Upper-tail spread (p99 - p50)"),
    ("p75_minus_p25", 75, 25, "Core spread (p75 - p25)"),
]

OUT_DIR = PROJECT_ROOT / "Results" / "3_Anomaly_Progression" / "percentile_spread"
OUT_DIR.mkdir(parents=True, exist_ok=True)


monthly = build_monthly_ecozone_dataframe(
    indices=INDICES,
    summary_percentiles=SUMMARY_PERCENTILES,
)
monthly = monthly[monthly["month"].isin(GROWING_MONTHS)].copy()

pivot = (
    monthly.pivot_table(
        index=[
            "aoi",
            "aoi_label",
            "index",
            "year",
            "month",
            "month_name",
            "classification",
            "ecozone_code",
            "ecozone_label",
        ],
        columns="summary_percentile",
        values="value",
        aggfunc="first",
    )
    .reset_index()
)

metric_frames: list[pd.DataFrame] = []
for metric_name, upper_pct, lower_pct, metric_label in SPREAD_METRICS:
    frame = pivot[
        [
            "aoi",
            "aoi_label",
            "index",
            "year",
            "month",
            "month_name",
            "classification",
            "ecozone_code",
            "ecozone_label",
            upper_pct,
            lower_pct,
        ]
    ].copy()
    frame["spread_metric"] = metric_name
    frame["spread_label"] = metric_label
    frame["upper_percentile"] = upper_pct
    frame["lower_percentile"] = lower_pct
    frame["spread_value"] = frame[upper_pct] - frame[lower_pct]
    metric_frames.append(frame)

spread_metrics = pd.concat(metric_frames, ignore_index=True)

group_summary = (
    spread_metrics.groupby(
        [
            "aoi",
            "aoi_label",
            "index",
            "classification",
            "ecozone_code",
            "ecozone_label",
            "month",
            "month_name",
            "spread_metric",
            "spread_label",
            "upper_percentile",
            "lower_percentile",
        ],
        dropna=False,
    )["spread_value"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={
        "mean": "group_mean_spread",
        "std": "group_sd_spread",
        "count": "group_n",
    })
)

monthly.to_csv(OUT_DIR / "percentile_spread_monthly_percentiles.csv", index=False)
spread_metrics.to_csv(OUT_DIR / "percentile_spread_metrics.csv", index=False)
group_summary.to_csv(OUT_DIR / "percentile_spread_group_summary.csv", index=False)


for index_name in INDICES:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=False)
    fig.suptitle(
        f"Percentile Spread by Ecozone and Year Group | {index_name}",
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
            ].sort_values(["spread_metric", "classification", "month"])

            for classification in ["wet", "neutral", "dry"]:
                class_subset = subset[subset["classification"] == classification]
                if class_subset.empty:
                    continue

                for metric_name, _, _, metric_label in SPREAD_METRICS:
                    curve = class_subset[
                        class_subset["spread_metric"] == metric_name
                    ].sort_values("month")
                    if curve.empty:
                        continue

                    linestyle = "-" if metric_name == "p99_minus_p50" else "--"
                    linewidth = 2.2 if metric_name == "p99_minus_p50" else 1.8

                    ax.plot(
                        curve["month"],
                        curve["group_mean_spread"],
                        color=CLASS_COLORS[classification],
                        linestyle=linestyle,
                        linewidth=linewidth,
                        marker="o",
                        markersize=4,
                        alpha=0.95,
                        label=f"{classification} | {metric_name}",
                    )

            ax.set_title(f"{AOI_DISPLAY[aoi]} | {ECOZONE_LABELS[code]}", fontsize=11)
            ax.set_xticks(GROWING_MONTHS)
            ax.set_xticklabels([MONTH_NAMES[m] for m in GROWING_MONTHS])
            ax.grid(True, alpha=0.2, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if col_idx == 0:
                ax.set_ylabel(f"{index_name} spread")
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
        ncol=3,
        bbox_to_anchor=(0.5, -0.03),
        framealpha=0.9,
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(
        OUT_DIR / f"percentile_spread_{index_name.lower()}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

print(f"Prepared percentile-spread outputs in: {OUT_DIR}")
