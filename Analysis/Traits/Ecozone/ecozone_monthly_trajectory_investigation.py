#!/usr/bin/env python3
"""
Prototype Sentinel investigation: monthly anomaly trajectories by ecozone.

Question:
  After divergence begins, how do wet and dry year trajectories unfold
  through the season?

Method:
  - Read existing Sentinel monthly composites.
  - Extract monthly ecozone percentile values for NDVI, NDMI, and EVI.
  - Define the baseline seasonal curve as the neutral-year monthly percentile.
  - Compute wet-year and dry-year anomalies relative to that baseline.

Outputs:
  Results/3_Anomaly_Progression/monthly_trajectories/
    trajectories_<index>_p50.png, trajectories_<index>_p75.png
    trajectory_monthly_anomalies_p50.csv, trajectory_monthly_anomalies_p75.csv
    trajectory_summary_p50.csv, trajectory_summary_p75.csv
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
    SUMMARY_PERCENTILES,
    VALID_ECOZONE_CODES,
    baseline_monthly_stats,
    build_monthly_ecozone_dataframe,
)

OUT_DIR = PROJECT_ROOT / "Results" / "3_Anomaly_Progression" / "monthly_trajectories"
OUT_DIR.mkdir(parents=True, exist_ok=True)

monthly = build_monthly_ecozone_dataframe(indices=INDICES, summary_percentiles=SUMMARY_PERCENTILES)
baseline = baseline_monthly_stats(monthly, baseline_class="neutral")
print("Computing monthly trajectory anomalies...")

plot_df = monthly.merge(
    baseline,
    on=["summary_percentile", "aoi", "index", "ecozone_code", "month"],
    how="left",
)
plot_df["anomaly"] = plot_df["value"] - plot_df["baseline_mean"]
plot_df = plot_df[
    plot_df["classification"].isin(["wet", "dry"])
    & plot_df["month"].isin(GROWING_MONTHS)
].copy()

monthly_anomalies = (
    plot_df.groupby(
        [
            "summary_percentile",
            "aoi",
            "aoi_label",
            "index",
            "ecozone_code",
            "ecozone_label",
            "classification",
            "month",
            "month_name",
        ],
        dropna=False,
    )["anomaly"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={"mean": "mean_anomaly", "std": "sd_anomaly", "count": "n_year_months"})
)

summary_rows: list[dict] = []
for _, group in monthly_anomalies.groupby(["aoi", "index", "ecozone_code", "classification"], dropna=False):
    ordered = group.sort_values("month")
    peak_idx = ordered["mean_anomaly"].idxmin() if ordered["classification"].iloc[0] == "dry" else ordered["mean_anomaly"].idxmax()
    peak = monthly_anomalies.loc[peak_idx]
    seasonal_mean = float(ordered["mean_anomaly"].mean())
    late_mean = float(ordered[ordered["month"].isin([8, 9, 10])]["mean_anomaly"].mean())
    summary_rows.append({
        "summary_percentile": int(peak["summary_percentile"]),
        "aoi": peak["aoi"],
        "aoi_label": peak["aoi_label"],
        "index": peak["index"],
        "ecozone_code": int(peak["ecozone_code"]),
        "ecozone_label": peak["ecozone_label"],
        "classification": peak["classification"],
        "seasonal_mean_anomaly": seasonal_mean,
        "peak_month": int(peak["month"]),
        "peak_month_name": peak["month_name"],
        "peak_anomaly": float(peak["mean_anomaly"]),
        "late_season_mean_anomaly": late_mean,
    })

summary_df = pd.DataFrame(summary_rows).sort_values(
    ["summary_percentile", "index", "aoi", "classification", "ecozone_code"]
)

print("Writing trajectory CSV outputs...")
monthly_anomalies.to_csv(OUT_DIR / "trajectory_monthly_anomalies_all_percentiles.csv", index=False)
summary_df.to_csv(OUT_DIR / "trajectory_summary_all_percentiles.csv", index=False)

for summary_percentile in SUMMARY_PERCENTILES:
    monthly_pct = monthly_anomalies[monthly_anomalies["summary_percentile"] == summary_percentile]
    summary_pct = summary_df[summary_df["summary_percentile"] == summary_percentile]
    monthly_pct.to_csv(OUT_DIR / f"trajectory_monthly_anomalies_p{summary_percentile}.csv", index=False)
    summary_pct.to_csv(OUT_DIR / f"trajectory_summary_p{summary_percentile}.csv", index=False)

    for index_name in INDICES:
        print(f"Rendering trajectory plot for p{summary_percentile} {index_name}...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        fig.suptitle(
            f"Monthly p{summary_percentile} Anomaly Trajectories vs Neutral Baseline | {index_name}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )

        for ax, aoi in zip(axes, AOIS):
            for code in VALID_ECOZONE_CODES:
                color = ECOZONE_COLORS[code]
                for classification, linestyle in [("wet", "-"), ("dry", "--")]:
                    series = monthly_pct[
                        (monthly_pct["index"] == index_name)
                        & (monthly_pct["aoi"] == aoi)
                        & (monthly_pct["ecozone_code"] == code)
                        & (monthly_pct["classification"] == classification)
                    ].sort_values("month")
                    if series.empty:
                        continue
                    ax.plot(
                        series["month"],
                        series["mean_anomaly"],
                        color=color,
                        linestyle=linestyle,
                        linewidth=2.1,
                        marker="o",
                        markersize=4,
                        alpha=0.95,
                        label=f"{ECOZONE_LABELS[code]} ({classification})",
                    )

            ax.axhline(0, color="#666666", linewidth=1.0, alpha=0.8)
            ax.set_title(AOI_DISPLAY[aoi], fontsize=12)
            ax.set_xticks(GROWING_MONTHS)
            ax.set_xticklabels([MONTH_NAMES[m] for m in GROWING_MONTHS])
            ax.set_xlabel("Month")
            ax.set_ylabel(
            f"Monthly p{summary_percentile} anomaly vs neutral baseline" if aoi == "north" else ""
        )
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
            bbox_to_anchor=(0.5, -0.1),
            framealpha=0.9,
            fontsize=9,
        )
        plt.tight_layout()
        plt.savefig(
            OUT_DIR / f"trajectories_{index_name.lower()}_p{summary_percentile}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

print(f"Saved trajectory outputs to: {OUT_DIR}")
