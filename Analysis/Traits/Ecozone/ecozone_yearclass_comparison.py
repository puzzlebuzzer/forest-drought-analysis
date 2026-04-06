#!/usr/bin/env python3
"""
Wet / normal / dry year comparison using existing ecozone-level outputs only.

Purpose:
  Build a compact comparison layer across year classes for onset, magnitude,
  trajectory, and spatial extent. This script reads existing summary CSVs and
  writes unified comparison tables plus a few simple plots.

Interpretation guardrails:
  - "Normal" is the neutral baseline.
  - For anomaly-based metrics, normal is represented as 0.0 because the source
    anomaly outputs are already defined relative to the neutral baseline.
  - For onset timing, neutral has no divergence month in the source outputs, so
    normal onset timing remains NA.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "Results" / "Other" / "ecozone_yearclass_comparison"

ONSET_SUMMARY_FILE = (
    PROJECT_ROOT / "Results" / "2_Anomaly_Onset" / "onset_timing" / "onset_timing_summary_all_percentiles.csv"
)
ONSET_DETAILS_FILE = (
    PROJECT_ROOT / "Results" / "2_Anomaly_Onset" / "onset_timing" / "onset_timing_monthly_details_all_percentiles.csv"
)
TRAJECTORY_MONTHLY_FILE = (
    PROJECT_ROOT / "Results" / "3_Anomaly_Progression" / "monthly_trajectories" / "trajectory_monthly_anomalies_all_percentiles.csv"
)
FRACTION_BELOW_FILE = (
    PROJECT_ROOT / "Results" / "2_Anomaly_Onset" / "fraction_below_baseline" / "fraction_below_baseline.csv"
)
ECOZONE_REFERENCE_FILE = (
    PROJECT_ROOT / "Results" / "Other" / "ecozone_comparative_dynamics" / "ecozone_reference_metrics.csv"
)

BASE_INDICES = ["NDVI", "NDMI"]
INCLUDE_EVI_IF_EASY = False
GROWING_MONTHS = [4, 5, 6, 7, 8, 9, 10]
EARLY_MONTHS = [4, 5, 6]
LATE_MONTHS = [8, 9, 10]
SPATIAL_MONTHS = [4, 6, 8, 10]


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input CSV not found: {path}")
    return path


def resolve_indices() -> list[str]:
    indices = list(BASE_INDICES)
    if INCLUDE_EVI_IF_EASY and "EVI" not in indices:
        indices.append("EVI")
    return indices


def month_name(month: float) -> str | None:
    if pd.isna(month):
        return None
    return {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }.get(int(round(month)))


def onset_metrics(onset_summary: pd.DataFrame) -> pd.DataFrame:
    onset = (
        onset_summary.groupby(
            ["aoi", "aoi_label", "index", "ecozone_code", "ecozone_label", "classification"],
            dropna=False,
        )
        .agg(
            value=("onset_month", "mean"),
            onset_anomaly=("onset_anomaly", "mean"),
        )
        .reset_index()
    )
    onset["metric_type"] = "onset_month"

    onset_mag = onset.copy()
    onset_mag["value"] = onset_mag["onset_anomaly"]
    onset_mag["metric_type"] = "onset_anomaly"
    return pd.concat(
        [
            onset[["aoi", "aoi_label", "index", "ecozone_code", "ecozone_label", "classification", "metric_type", "value"]],
            onset_mag[["aoi", "aoi_label", "index", "ecozone_code", "ecozone_label", "classification", "metric_type", "value"]],
        ],
        ignore_index=True,
    )


def trajectory_metrics(trajectory_monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["aoi", "aoi_label", "index", "ecozone_code", "ecozone_label", "classification"]

    for group_keys, group in trajectory_monthly.groupby(keys, dropna=False):
        aoi, aoi_label, index_name, ecozone_code, ecozone_label, classification = group_keys
        month_lookup = group.groupby("month")["mean_anomaly"].mean()
        cumulative = float(month_lookup.reindex(GROWING_MONTHS).fillna(0.0).sum())
        early = float(month_lookup.reindex(EARLY_MONTHS).mean())
        late = float(month_lookup.reindex(LATE_MONTHS).mean())
        late_minus_early = late - early

        rows.extend([
            {
                "aoi": aoi,
                "aoi_label": aoi_label,
                "index": index_name,
                "ecozone_code": int(ecozone_code),
                "ecozone_label": ecozone_label,
                "classification": classification,
                "metric_type": "cumulative_anomaly",
                "value": cumulative,
            },
            {
                "aoi": aoi,
                "aoi_label": aoi_label,
                "index": index_name,
                "ecozone_code": int(ecozone_code),
                "ecozone_label": ecozone_label,
                "classification": classification,
                "metric_type": "late_minus_early_anomaly",
                "value": late_minus_early,
            },
            {
                "aoi": aoi,
                "aoi_label": aoi_label,
                "index": index_name,
                "ecozone_code": int(ecozone_code),
                "ecozone_label": ecozone_label,
                "classification": classification,
                "metric_type": "early_season_anomaly",
                "value": early,
            },
            {
                "aoi": aoi,
                "aoi_label": aoi_label,
                "index": index_name,
                "ecozone_code": int(ecozone_code),
                "ecozone_label": ecozone_label,
                "classification": classification,
                "metric_type": "late_season_anomaly",
                "value": late,
            },
        ])
    return pd.DataFrame(rows)


def spatial_metrics(fraction_below: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = (
        fraction_below.groupby(
            ["aoi", "aoi_label", "index", "ecozone_code", "ecozone_label", "classification", "month"],
            dropna=False,
        )["fraction_below_baseline"]
        .mean()
        .reset_index()
    )
    for _, row in grouped.iterrows():
        rows.append({
            "aoi": row["aoi"],
            "aoi_label": row["aoi_label"],
            "index": row["index"],
            "ecozone_code": int(row["ecozone_code"]),
            "ecozone_label": row["ecozone_label"],
            "classification": row["classification"],
            "metric_type": f"fraction_below_{int(row['month']):02d}",
            "value": float(row["fraction_below_baseline"]),
        })
    return pd.DataFrame(rows)


def add_normal_reference(metrics_long: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    group_cols = ["aoi", "aoi_label", "index", "ecozone_code", "ecozone_label", "metric_type"]

    for group_keys, group in metrics_long.groupby(group_cols, dropna=False):
        aoi, aoi_label, index_name, ecozone_code, ecozone_label, metric_type = group_keys
        if metric_type.startswith("fraction_below_"):
            neutral = group[group["classification"] == "neutral"]["value"]
            if neutral.empty:
                normal_value = np.nan
            else:
                normal_value = float(neutral.mean())
        elif metric_type == "onset_month":
            normal_value = np.nan
        else:
            normal_value = 0.0

        records.append({
            "aoi": aoi,
            "aoi_label": aoi_label,
            "index": index_name,
            "ecozone_code": int(ecozone_code),
            "ecozone_label": ecozone_label,
            "classification": "normal",
            "metric_type": metric_type,
            "value": normal_value,
        })

    return pd.concat([metrics_long, pd.DataFrame(records)], ignore_index=True)


def build_unified_comparison(metrics_long: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        metrics_long.groupby(
            ["aoi", "aoi_label", "index", "ecozone_code", "ecozone_label", "metric_type", "classification"],
            dropna=False,
        )["value"]
        .mean()
        .reset_index()
    )

    wide = (
        grouped.pivot_table(
            index=["aoi", "aoi_label", "index", "ecozone_code", "ecozone_label", "metric_type"],
            columns="classification",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    for col in ["wet", "normal", "dry"]:
        if col not in wide.columns:
            wide[col] = np.nan

    wide = wide.rename(columns={
        "aoi": "AOI",
        "index": "index",
        "ecozone_label": "ecozone",
        "wet": "value_wet",
        "normal": "value_normal",
        "dry": "value_dry",
    })
    wide["delta_wet_normal"] = wide["value_wet"] - wide["value_normal"]
    wide["delta_dry_normal"] = wide["value_dry"] - wide["value_normal"]

    # Keep requested columns first.
    cols = [
        "AOI",
        "index",
        "ecozone",
        "metric_type",
        "value_wet",
        "value_normal",
        "value_dry",
        "delta_wet_normal",
        "delta_dry_normal",
        "aoi_label",
        "ecozone_code",
    ]
    return wide[cols].sort_values(["AOI", "index", "ecozone_code", "metric_type"])


def build_summary_tables(unified: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_aoi_index = (
        unified.groupby(["AOI", "index", "metric_type"], dropna=False)
        .agg(
            mean_value_wet=("value_wet", "mean"),
            mean_value_normal=("value_normal", "mean"),
            mean_value_dry=("value_dry", "mean"),
            mean_delta_wet_normal=("delta_wet_normal", "mean"),
            mean_delta_dry_normal=("delta_dry_normal", "mean"),
            ecozone_count=("ecozone", "count"),
        )
        .reset_index()
        .sort_values(["AOI", "index", "metric_type"])
    )

    summary_by_ecozone = (
        unified.groupby(["AOI", "index", "ecozone", "metric_type"], dropna=False)
        .agg(
            value_wet=("value_wet", "first"),
            value_normal=("value_normal", "first"),
            value_dry=("value_dry", "first"),
            delta_wet_normal=("delta_wet_normal", "first"),
            delta_dry_normal=("delta_dry_normal", "first"),
        )
        .reset_index()
        .sort_values(["AOI", "index", "ecozone", "metric_type"])
    )
    return summary_aoi_index, summary_by_ecozone


def heatmap_frame(unified: pd.DataFrame, metric_type: str, delta_col: str) -> pd.DataFrame:
    frame = unified[unified["metric_type"] == metric_type].copy()
    frame["row_label"] = frame["AOI"].astype(str) + " | " + frame["index"].astype(str)
    heat = frame.pivot_table(index="row_label", columns="ecozone", values=delta_col, aggfunc="first")
    return heat.reindex(columns=sorted(heat.columns))


def write_heatmap_figure(unified: pd.DataFrame) -> None:
    panels = [
        ("cumulative_anomaly", "delta_wet_normal", "Wet - Normal Cumulative"),
        ("cumulative_anomaly", "delta_dry_normal", "Dry - Normal Cumulative"),
        ("fraction_below_10", "delta_wet_normal", "Wet - Normal Oct Extent"),
        ("fraction_below_10", "delta_dry_normal", "Dry - Normal Oct Extent"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (metric_type, delta_col, title) in zip(axes.flat, panels):
        heat = heatmap_frame(unified, metric_type, delta_col)
        values = heat.to_numpy(dtype=float)
        masked = np.ma.masked_invalid(values)
        vmax = np.nanmax(np.abs(values))
        if np.isnan(vmax) or vmax == 0:
            vmax = 1.0
        im = ax.imshow(masked, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(heat.columns)), heat.columns)
        ax.set_yticks(range(len(heat.index)), heat.index)
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                value = heat.iloc[i, j]
                text = "NA" if pd.isna(value) else f"{value:.2f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("Delta relative to normal")
    fig.suptitle("Wet / Normal / Dry Comparison Summary", fontsize=13)
    fig.savefig(OUT_DIR / "yearclass_comparison_heatmaps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_onset_plot(unified: pd.DataFrame) -> None:
    onset = unified[unified["metric_type"] == "onset_month"].copy()
    onset["row_label"] = onset["AOI"].astype(str) + " | " + onset["index"].astype(str)
    row_order = onset[["row_label"]].drop_duplicates()["row_label"].tolist()
    ecozones = sorted(onset["ecozone"].dropna().unique().tolist())
    positions = np.arange(len(row_order))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True)
    for ax, class_col, title in [
        (axes[0], "value_wet", "Wet Onset Month"),
        (axes[1], "value_dry", "Dry Onset Month"),
    ]:
        for offset, ecozone in enumerate(ecozones):
            subset = onset[onset["ecozone"] == ecozone].set_index("row_label").reindex(row_order)
            values = subset[class_col].to_numpy(dtype=float)
            ax.bar(positions + (offset - 1) * width, values, width=width, label=ecozone)
        ax.set_title(title)
        ax.set_xticks(positions, row_order, rotation=20, ha="right")
        ax.set_ylabel("Month")
        ax.set_ylim(0, 12)
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(OUT_DIR / "yearclass_onset_months.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    indices = resolve_indices()

    onset_summary = pd.read_csv(require_file(ONSET_SUMMARY_FILE))
    onset_details = pd.read_csv(require_file(ONSET_DETAILS_FILE))
    trajectory_monthly = pd.read_csv(require_file(TRAJECTORY_MONTHLY_FILE))
    fraction_below = pd.read_csv(require_file(FRACTION_BELOW_FILE))
    require_file(ECOZONE_REFERENCE_FILE)

    onset_summary = onset_summary[onset_summary["index"].isin(indices)].copy()
    onset_details = onset_details[onset_details["index"].isin(indices)].copy()
    trajectory_monthly = trajectory_monthly[trajectory_monthly["index"].isin(indices)].copy()
    fraction_below = fraction_below[fraction_below["index"].isin(indices)].copy()

    metrics_long = pd.concat(
        [
            onset_metrics(onset_summary),
            trajectory_metrics(trajectory_monthly),
            spatial_metrics(fraction_below),
        ],
        ignore_index=True,
    )
    metrics_long = add_normal_reference(metrics_long)

    unified = build_unified_comparison(metrics_long)
    summary_aoi_index, summary_by_ecozone = build_summary_tables(unified)

    unified.to_csv(OUT_DIR / "yearclass_unified_comparison.csv", index=False)
    summary_aoi_index.to_csv(OUT_DIR / "yearclass_summary_by_aoi_index.csv", index=False)
    summary_by_ecozone.to_csv(OUT_DIR / "yearclass_summary_by_ecozone.csv", index=False)

    write_heatmap_figure(unified)
    write_onset_plot(unified)

    print(f"Saved year-class comparison outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
