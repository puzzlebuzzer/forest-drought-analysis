#!/usr/bin/env python3
"""
Prototype Sentinel investigation: onset timing by ecozone.

Question:
  For each ecozone and index, when do dry and wet years first diverge from
  baseline seasonal behavior?

Inputs:
  - Existing Sentinel monthly composites at Results/0-CacheBaseData/monthly_max
  - Existing AOI wet/dry labels in config/wet_dry_years.csv
  - Existing AOI ecozone masks under cfg.ecozone_dir

Baseline definition:
  Baseline is the neutral-year monthly percentile summary by ecozone.

Divergence logic:
  For each growing-season month, compare the wet-year or dry-year anomaly
  (class monthly percentile minus neutral baseline percentile) against a transparent
  descriptive threshold:

    abs(class anomaly) >= max(0.03, 1.0 * baseline monthly std)

  The first month meeting that rule is the onset month for that class.

Outputs:
  Results/2_Anomaly_Onset/onset_timing/
    onset_<index>_p50.png, onset_<index>_p75.png
    onset_timing_summary_p50.csv, onset_timing_summary_p75.csv
    onset_timing_monthly_details_p50.csv, onset_timing_monthly_details_p75.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from investigation_common import (
    AOIS,
    AOI_DISPLAY,
    DIVERGENCE_MIN_ABS_ANOMALY,
    DIVERGENCE_STD_MULTIPLIER,
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

OUT_DIR = PROJECT_ROOT / "Results" / "2_Anomaly_Onset" / "onset_timing"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def month_series(
    df: pd.DataFrame,
    summary_percentile: int,
    aoi: str,
    index_name: str,
    ecozone_code: int,
) -> pd.DataFrame:
    subset = df[
        (df["summary_percentile"] == summary_percentile)
        & (df["aoi"] == aoi)
        & (df["index"] == index_name)
        & (df["ecozone_code"] == ecozone_code)
        & (df["month"].isin(GROWING_MONTHS))
    ].copy()
    return subset.sort_values("month").reset_index(drop=True)


monthly = build_monthly_ecozone_dataframe(indices=INDICES, summary_percentiles=SUMMARY_PERCENTILES)
baseline = baseline_monthly_stats(monthly, baseline_class="neutral")
print("Computing onset timing summaries...")

summary_rows: list[dict] = []
detail_rows: list[dict] = []

for summary_percentile in SUMMARY_PERCENTILES:
    print(f"Percentile p{summary_percentile}...")
    for index_name in INDICES:
        print(f"  Processing onset timing for {index_name}...")
        for aoi in AOIS:
            print(f"    AOI: {aoi}")
            for code in VALID_ECOZONE_CODES:
                subset = month_series(monthly, summary_percentile, aoi, index_name, code)
                grouped = (
                    subset.groupby(["classification", "month"])["value"]
                    .agg(["mean", "count"])
                    .reset_index()
                    .rename(columns={"mean": "class_value", "count": "class_count"})
                )
                merged = grouped.merge(
                    baseline[
                        (baseline["summary_percentile"] == summary_percentile)
                        & (baseline["aoi"] == aoi)
                        & (baseline["index"] == index_name)
                        & (baseline["ecozone_code"] == code)
                    ][["month", "baseline_mean", "baseline_std", "baseline_count"]],
                    on="month",
                    how="left",
                )
                merged["anomaly"] = merged["class_value"] - merged["baseline_mean"]
                merged["threshold"] = np.maximum(
                    DIVERGENCE_MIN_ABS_ANOMALY,
                    DIVERGENCE_STD_MULTIPLIER * merged["baseline_std"].fillna(0.0),
                )
                merged["diverges"] = (
                    merged["classification"].isin(["wet", "dry"])
                    & merged["anomaly"].abs().ge(merged["threshold"])
                )

                for anomaly_class in ["wet", "dry"]:
                    class_rows = merged[merged["classification"] == anomaly_class].sort_values("month")
                    onset = class_rows[class_rows["diverges"]].head(1)
                    if onset.empty:
                        onset_month = np.nan
                        onset_name = None
                        onset_anomaly = np.nan
                        onset_threshold = np.nan
                    else:
                        onset_month = int(onset.iloc[0]["month"])
                        onset_name = MONTH_NAMES[onset_month]
                        onset_anomaly = float(onset.iloc[0]["anomaly"])
                        onset_threshold = float(onset.iloc[0]["threshold"])

                    summary_rows.append({
                        "summary_percentile": summary_percentile,
                        "aoi": aoi,
                        "aoi_label": AOI_DISPLAY[aoi],
                        "index": index_name,
                        "ecozone_code": code,
                        "ecozone_label": ECOZONE_LABELS[code],
                        "classification": anomaly_class,
                        "onset_month": onset_month,
                        "onset_month_name": onset_name,
                        "onset_anomaly": onset_anomaly,
                        "onset_threshold": onset_threshold,
                        "class_months_available": int((class_rows["class_count"] > 0).sum()),
                        "neutral_months_available": int((class_rows["baseline_count"].fillna(0) > 0).sum()),
                    })

                for _, row in merged.iterrows():
                    detail_rows.append({
                        "summary_percentile": summary_percentile,
                        "aoi": aoi,
                        "aoi_label": AOI_DISPLAY[aoi],
                        "index": index_name,
                        "ecozone_code": code,
                        "ecozone_label": ECOZONE_LABELS[code],
                        "classification": row["classification"],
                        "month": int(row["month"]),
                        "month_name": MONTH_NAMES[int(row["month"])],
                        "class_value": row["class_value"],
                        "class_count": int(row["class_count"]),
                        "baseline_value": row["baseline_mean"],
                        "baseline_std": row["baseline_std"],
                        "baseline_count": row["baseline_count"],
                        "anomaly": row["anomaly"],
                        "threshold": row["threshold"],
                        "diverges": bool(row["diverges"]),
                    })


summary_df = pd.DataFrame(summary_rows).sort_values(
    ["summary_percentile", "index", "aoi", "classification", "ecozone_code"]
)
details_df = pd.DataFrame(detail_rows).sort_values(
    ["summary_percentile", "index", "aoi", "ecozone_code", "classification", "month"]
)

print("Writing onset CSV outputs...")
summary_df.to_csv(OUT_DIR / "onset_timing_summary_all_percentiles.csv", index=False)
details_df.to_csv(OUT_DIR / "onset_timing_monthly_details_all_percentiles.csv", index=False)

for summary_percentile in SUMMARY_PERCENTILES:
    summary_pct = summary_df[summary_df["summary_percentile"] == summary_percentile]
    details_pct = details_df[details_df["summary_percentile"] == summary_percentile]
    summary_df_file = OUT_DIR / f"onset_timing_summary_p{summary_percentile}.csv"
    details_df_file = OUT_DIR / f"onset_timing_monthly_details_p{summary_percentile}.csv"
    summary_pct.to_csv(summary_df_file, index=False)
    details_pct.to_csv(details_df_file, index=False)

    for index_name in INDICES:
        print(f"Rendering onset plot for p{summary_percentile} {index_name}...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        fig.suptitle(
            f"Wet/Dry Onset Timing vs Neutral Baseline | p{summary_percentile} {index_name}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )

        for ax, aoi in zip(axes, AOIS):
            for code in VALID_ECOZONE_CODES:
                color = ECOZONE_COLORS[code]
                for anomaly_class, linestyle in [("wet", "-"), ("dry", "--")]:
                    class_series = details_pct[
                        (details_pct["index"] == index_name)
                        & (details_pct["aoi"] == aoi)
                        & (details_pct["ecozone_code"] == code)
                        & (details_pct["classification"] == anomaly_class)
                    ].sort_values("month")
                    if class_series.empty:
                        continue

                    x = class_series["month"].to_numpy()
                    y = class_series["anomaly"].to_numpy(dtype=float)
                    th = class_series["threshold"].to_numpy(dtype=float)

                    ax.plot(
                        x,
                        y,
                        color=color,
                        marker="o",
                        linewidth=2.2,
                        linestyle=linestyle,
                        label=f"{ECOZONE_LABELS[code]} ({anomaly_class})",
                    )
                    ax.plot(x, th, color=color, linestyle=":", linewidth=1.0, alpha=0.35)
                    ax.plot(x, -th, color=color, linestyle=":", linewidth=1.0, alpha=0.35)

                    onset = summary_pct[
                        (summary_pct["index"] == index_name)
                        & (summary_pct["aoi"] == aoi)
                        & (summary_pct["ecozone_code"] == code)
                        & (summary_pct["classification"] == anomaly_class)
                    ].iloc[0]
                    if pd.notna(onset["onset_month"]):
                        ax.scatter(
                            [int(onset["onset_month"])],
                            [float(onset["onset_anomaly"])],
                            color=color,
                            edgecolor="black",
                            s=70,
                            zorder=4,
                        )

            ax.axhline(0, color="#666666", linewidth=1.0, alpha=0.8)
            ax.set_title(AOI_DISPLAY[aoi], fontsize=12)
            ax.set_xticks(GROWING_MONTHS)
            ax.set_xticklabels([MONTH_NAMES[m] for m in GROWING_MONTHS])
            ax.set_xlabel("Month")
            ax.set_ylabel(
                f"Wet/dry p{summary_percentile} anomaly vs neutral baseline" if aoi == "north" else ""
            )
            ax.grid(True, alpha=0.2, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05), framealpha=0.9)
        plt.tight_layout()
        plt.savefig(
            OUT_DIR / f"onset_{index_name.lower()}_p{summary_percentile}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

print(f"Saved onset outputs to: {OUT_DIR}")
