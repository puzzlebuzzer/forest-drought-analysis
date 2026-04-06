#!/usr/bin/env python3
"""
Prototype Sentinel investigation: simple recovery by ecozone.

Question:
  After dry-year stress, do ecozones appear to recover quickly, partially,
  or slowly?

Conservative recovery logic:
  1. Compute monthly ecozone percentile values from the existing Sentinel monthly
     composites for NDVI, NDMI, and EVI.
  2. Define baseline monthly values from neutral years only.
  3. For each dry year, compute:
       - late-season anomaly: mean anomaly in Aug-Oct of the dry year
       - following-spring anomaly: mean anomaly in Apr-Jun of the next year
  4. Record wet-year anomaly trajectories as a reference.
  5. Classify dry-year recovery:
       - quick: following spring is near baseline (abs anomaly <= 0.02)
       - partial: following spring improved materially relative to late season
       - slow: still clearly depressed or not materially improved

Limitation:
  This is a descriptive next-spring check, not a full resilience model.
  It does not attempt to infer multi-year legacy effects beyond the immediate
  following spring.

Outputs:
  Results/4_Anomaly_Recovery/simple_recovery/
    recovery_<index>_p50.png, recovery_<index>_p75.png
    simple_recovery_by_year_p50.csv, simple_recovery_by_year_p75.csv
    simple_recovery_summary_p50.csv, simple_recovery_summary_p75.csv
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from investigation_common import (
    AOIS,
    AOI_DISPLAY,
    ECOZONE_COLORS,
    ECOZONE_LABELS,
    FOLLOWING_SPRING_MONTHS,
    INDICES,
    LATE_SEASON_MONTHS,
    RECOVERY_NEAR_BASELINE_TOLERANCE,
    PROJECT_ROOT,
    SUMMARY_PERCENTILES,
    VALID_ECOZONE_CODES,
    baseline_monthly_stats,
    build_monthly_ecozone_dataframe,
    classify_recovery,
)

OUT_DIR = PROJECT_ROOT / "Results" / "4_Anomaly_Recovery" / "simple_recovery"
OUT_DIR.mkdir(parents=True, exist_ok=True)

monthly = build_monthly_ecozone_dataframe(indices=INDICES, summary_percentiles=SUMMARY_PERCENTILES)
baseline = baseline_monthly_stats(monthly, baseline_class="neutral")
print("Computing simple recovery summaries...")
full_df = monthly.merge(
    baseline,
    on=["summary_percentile", "aoi", "index", "ecozone_code", "month"],
    how="left",
)
full_df["anomaly"] = full_df["value"] - full_df["baseline_mean"]

year_rows: list[dict] = []

for summary_percentile in SUMMARY_PERCENTILES:
    print(f"Percentile p{summary_percentile}...")
    for index_name in INDICES:
        print(f"  Processing recovery metrics for {index_name}...")
        for aoi in AOIS:
            print(f"    AOI: {aoi}")
            wet_years = sorted(
                full_df[
                    (full_df["summary_percentile"] == summary_percentile)
                    & (full_df["aoi"] == aoi)
                    & (full_df["index"] == index_name)
                    & (full_df["classification"] == "wet")
                ]["year"].unique()
            )
            dry_years = sorted(
                full_df[
                    (full_df["summary_percentile"] == summary_percentile)
                    & (full_df["aoi"] == aoi)
                    & (full_df["index"] == index_name)
                    & (full_df["classification"] == "dry")
                ]["year"].unique()
            )
            for dry_year in dry_years:
                next_year = int(dry_year) + 1
                for code in VALID_ECOZONE_CODES:
                    late = full_df[
                        (full_df["summary_percentile"] == summary_percentile)
                        & (full_df["aoi"] == aoi)
                        & (full_df["index"] == index_name)
                        & (full_df["ecozone_code"] == code)
                        & (full_df["year"] == dry_year)
                        & (full_df["month"].isin(LATE_SEASON_MONTHS))
                    ]["anomaly"]
                    spring = full_df[
                        (full_df["summary_percentile"] == summary_percentile)
                        & (full_df["aoi"] == aoi)
                        & (full_df["index"] == index_name)
                        & (full_df["ecozone_code"] == code)
                        & (full_df["year"] == next_year)
                        & (full_df["month"].isin(FOLLOWING_SPRING_MONTHS))
                    ]["anomaly"]

                    late_mean = float(late.mean()) if not late.empty else np.nan
                    spring_mean = float(spring.mean()) if not spring.empty else np.nan
                    recovery_status = classify_recovery(late_mean, spring_mean)

                    year_rows.append({
                        "summary_percentile": summary_percentile,
                        "aoi": aoi,
                        "aoi_label": AOI_DISPLAY[aoi],
                        "index": index_name,
                        "ecozone_code": code,
                        "ecozone_label": ECOZONE_LABELS[code],
                        "dry_year": int(dry_year),
                        "followup_year": next_year,
                        "late_season_anomaly": late_mean,
                        "following_spring_anomaly": spring_mean,
                        "net_change": spring_mean - late_mean if np.isfinite(late_mean) and np.isfinite(spring_mean) else np.nan,
                        "recovery_status": recovery_status,
                    })

            for wet_year in wet_years:
                next_year = int(wet_year) + 1
                for code in VALID_ECOZONE_CODES:
                    late = full_df[
                        (full_df["summary_percentile"] == summary_percentile)
                        & (full_df["aoi"] == aoi)
                        & (full_df["index"] == index_name)
                        & (full_df["ecozone_code"] == code)
                        & (full_df["year"] == wet_year)
                        & (full_df["month"].isin(LATE_SEASON_MONTHS))
                    ]["anomaly"]
                    spring = full_df[
                        (full_df["summary_percentile"] == summary_percentile)
                        & (full_df["aoi"] == aoi)
                        & (full_df["index"] == index_name)
                        & (full_df["ecozone_code"] == code)
                        & (full_df["year"] == next_year)
                        & (full_df["month"].isin(FOLLOWING_SPRING_MONTHS))
                    ]["anomaly"]

                    late_mean = float(late.mean()) if not late.empty else np.nan
                    spring_mean = float(spring.mean()) if not spring.empty else np.nan
                    year_rows.append({
                        "summary_percentile": summary_percentile,
                        "aoi": aoi,
                        "aoi_label": AOI_DISPLAY[aoi],
                        "index": index_name,
                        "ecozone_code": code,
                        "ecozone_label": ECOZONE_LABELS[code],
                        "dry_year": int(wet_year),
                        "followup_year": next_year,
                        "late_season_anomaly": late_mean,
                        "following_spring_anomaly": spring_mean,
                        "net_change": spring_mean - late_mean if np.isfinite(late_mean) and np.isfinite(spring_mean) else np.nan,
                        "recovery_status": "wet_reference",
                    })

year_df = pd.DataFrame(year_rows).sort_values(
    ["summary_percentile", "index", "aoi", "dry_year", "ecozone_code"]
)
summary_df = (
    year_df.groupby(
        ["summary_percentile", "aoi", "aoi_label", "index", "ecozone_code", "ecozone_label", "recovery_status"],
        dropna=False,
    )
    .agg(
        mean_late_season_anomaly=("late_season_anomaly", "mean"),
        mean_following_spring_anomaly=("following_spring_anomaly", "mean"),
        mean_net_change=("net_change", "mean"),
        recovery_cases=("recovery_status", "count"),
    )
    .reset_index()
    .sort_values(["index", "aoi", "ecozone_code", "recovery_status"])
)

dry_counts = (
    year_df[year_df["recovery_status"].isin(["quick", "partial", "slow", "insufficient_data"])]
    .groupby(["summary_percentile", "aoi", "index", "ecozone_code", "recovery_status"])
    .size()
    .unstack(fill_value=0)
)

print("Writing recovery CSV outputs...")
year_df.to_csv(OUT_DIR / "simple_recovery_by_year_all_percentiles.csv", index=False)
summary_df.to_csv(OUT_DIR / "simple_recovery_summary_all_percentiles.csv", index=False)

for summary_percentile in SUMMARY_PERCENTILES:
    year_pct = year_df[year_df["summary_percentile"] == summary_percentile]
    summary_pct = summary_df[summary_df["summary_percentile"] == summary_percentile]
    year_pct.to_csv(OUT_DIR / f"simple_recovery_by_year_p{summary_percentile}.csv", index=False)
    summary_pct.to_csv(OUT_DIR / f"simple_recovery_summary_p{summary_percentile}.csv", index=False)

    for index_name in INDICES:
        print(f"Rendering recovery plot for p{summary_percentile} {index_name}...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        fig.suptitle(
            f"Simple Recovery Check After Dry-Year Stress | p{summary_percentile} {index_name}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )

        for ax, aoi in zip(axes, AOIS):
            subset = summary_pct[
                (summary_pct["index"] == index_name)
                & (summary_pct["aoi"] == aoi)
            ].sort_values(["ecozone_code", "recovery_status"])

            for code in VALID_ECOZONE_CODES:
                color = ECOZONE_COLORS[code]
                dry_subset = subset[
                    (subset["ecozone_code"] == code)
                    & (subset["recovery_status"].isin(["quick", "partial", "slow", "insufficient_data"]))
                ]
                wet_subset = subset[
                    (subset["ecozone_code"] == code)
                    & (subset["recovery_status"] == "wet_reference")
                ]

                if not dry_subset.empty:
                    dry_late = float(dry_subset["mean_late_season_anomaly"].mean())
                    dry_spring = float(dry_subset["mean_following_spring_anomaly"].mean())
                    ax.plot(
                        [0, 1],
                        [dry_late, dry_spring],
                        color=color,
                        marker="o",
                        linewidth=2.4,
                        linestyle="--",
                        label=f"{ECOZONE_LABELS[code]} dry",
                    )
                else:
                    dry_spring = np.nan

                if not wet_subset.empty:
                    wet_late = float(wet_subset["mean_late_season_anomaly"].mean())
                    wet_spring = float(wet_subset["mean_following_spring_anomaly"].mean())
                    ax.plot(
                        [0, 1],
                        [wet_late, wet_spring],
                        color=color,
                        marker="o",
                        linewidth=2.0,
                        linestyle="-",
                        alpha=0.65,
                        label=f"{ECOZONE_LABELS[code]} wet",
                    )
                    ref_y = wet_spring if np.isnan(dry_spring) else dry_spring
                else:
                    ref_y = dry_spring

                if (summary_percentile, aoi, index_name, code) in dry_counts.index:
                    counts = dry_counts.loc[(summary_percentile, aoi, index_name, code)]
                    quick = int(counts.get("quick", 0))
                    partial = int(counts.get("partial", 0))
                    slow = int(counts.get("slow", 0))
                    insufficient = int(counts.get("insufficient_data", 0))
                    if np.isfinite(ref_y):
                        ax.text(
                            1.03,
                            ref_y,
                            f"{quick}/{partial}/{slow}/{insufficient}",
                            color=color,
                            fontsize=9,
                            va="center",
                        )

            ax.axhline(0, color="#666666", linewidth=1.0, alpha=0.8)
            ax.axhspan(
                -RECOVERY_NEAR_BASELINE_TOLERANCE,
                RECOVERY_NEAR_BASELINE_TOLERANCE,
                color="#cccccc",
                alpha=0.18,
            )
            ax.set_title(AOI_DISPLAY[aoi], fontsize=12)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Dry late season", "Following spring"])
            ax.set_ylabel(
                f"Mean p{summary_percentile} anomaly vs neutral baseline" if aoi == "north" else ""
            )
            ax.grid(True, alpha=0.2, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05), framealpha=0.9)
        plt.tight_layout()
        plt.savefig(
            OUT_DIR / f"recovery_{index_name.lower()}_p{summary_percentile}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

print(f"Saved recovery outputs to: {OUT_DIR}")
