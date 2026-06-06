#!/usr/bin/env python3
"""
Compact cross-index summary built from existing ecozone investigation CSVs.

Purpose:
  Compare NDVI, NDMI, and EVI by AOI and ecozone using already-generated onset,
  progression, and recovery summaries. This script does not recompute raster
  products. It reads existing CSV outputs and writes compact comparative tables
  and figures for interpretation.

Outputs:
  Results/Other/index_role_summary/
    index_role_comparison_by_class.csv
    index_role_comparison_overall.csv
    index_role_overall_ranking.csv
    index_role_metric_heatmaps.png
    index_role_metric_bars.png

Interpretation guardrails:
  - Scores are descriptive heuristics, not significance tests.
  - Missing values are preserved when source outputs do not support a metric.
  - Simple rules are used so thresholds remain easy to review and adjust.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "Results" / "Other" / "index_role_summary"

ONSET_DIR = PROJECT_ROOT / "Results" / "2_Anomaly_Onset" / "onset_timing"
PROGRESSION_DIR = PROJECT_ROOT / "Results" / "3_Anomaly_Progression"
RECOVERY_DIR = PROJECT_ROOT / "Results" / "4_Anomaly_Recovery" / "simple_recovery"

ONSET_SUMMARY_FILE = ONSET_DIR / "onset_timing_summary_all_percentiles.csv"
TRAJECTORY_SUMMARY_FILE = (
    PROGRESSION_DIR / "monthly_trajectories" / "trajectory_summary_all_percentiles.csv"
)
TRAJECTORY_MONTHLY_FILE = (
    PROGRESSION_DIR / "monthly_trajectories" / "trajectory_monthly_anomalies_all_percentiles.csv"
)
SPREAD_GROUP_FILE = (
    PROGRESSION_DIR / "percentile_spread" / "percentile_spread_group_summary.csv"
)
RECOVERY_YEAR_FILE = RECOVERY_DIR / "simple_recovery_by_year_all_percentiles.csv"

INDICES = ["NDVI", "NDMI", "EVI"]
CLASSIFICATIONS = ["dry", "wet"]
RECOVERY_LABELS = ["quick", "partial", "slow", "insufficient_data"]

MEANINGFUL_ANOMALY = 0.02
ONSET_MONTH_SPAN_BAD = 3.0
PEAK_MONTH_SPAN_BAD = 3.0
GOOD_CORE_SPREAD = 0.05
BAD_CORE_SPREAD = 0.20

CMAP = "YlGnBu"


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input CSV not found: {path}")
    return path


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def inverse_score(value: float, good: float, bad: float) -> float:
    if pd.isna(value):
        return np.nan
    if value <= good:
        return 1.0
    if value >= bad:
        return 0.0
    return clamp01((bad - value) / (bad - good))


def month_name(month: float) -> str | None:
    if pd.isna(month):
        return None
    month_int = int(round(month))
    return {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }.get(month_int)


def dominant_label(series: pd.Series) -> str | None:
    counts = series.value_counts()
    if counts.empty:
        return None
    top = counts.index.tolist()
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        return "tie"
    return str(top[0])


def onset_metrics(onset_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = ["aoi", "aoi_label", "index", "ecozone_code", "ecozone_label", "classification"]

    for keys, group in onset_summary.groupby(group_cols, dropna=False):
        aoi, aoi_label, index_name, ecozone_code, ecozone_label, classification = keys
        valid = group.dropna(subset=["onset_month"]).copy()
        months = valid["onset_month"].astype(float)
        anomalies = valid["onset_anomaly"].astype(float)
        available_records = int(len(valid))
        onset_month_mean = float(months.mean()) if available_records else np.nan
        onset_month_span = float(months.max() - months.min()) if available_records else np.nan
        onset_month_stability = inverse_score(onset_month_span, good=0.0, bad=ONSET_MONTH_SPAN_BAD)

        sign_values = np.sign(anomalies[np.abs(anomalies) >= MEANINGFUL_ANOMALY])
        if len(sign_values) == 0:
            onset_direction_label = "mixed_or_small"
            onset_direction_consistency = np.nan
        else:
            dominant_sign = 1 if sign_values.sum() > 0 else -1 if sign_values.sum() < 0 else 0
            if dominant_sign == 0:
                onset_direction_label = "mixed"
                onset_direction_consistency = 0.5
            else:
                onset_direction_label = "positive" if dominant_sign > 0 else "negative"
                onset_direction_consistency = float((sign_values == dominant_sign).mean())

        rows.append({
            "aoi": aoi,
            "aoi_label": aoi_label,
            "index": index_name,
            "ecozone_code": int(ecozone_code),
            "ecozone_label": ecozone_label,
            "classification": classification,
            "onset_records": available_records,
            "onset_month_mean": onset_month_mean,
            "onset_month_name_mean": month_name(onset_month_mean),
            "onset_month_span": onset_month_span,
            "onset_month_stability": onset_month_stability,
            "onset_direction_label": onset_direction_label,
            "onset_direction_consistency": onset_direction_consistency,
        })

    return pd.DataFrame(rows)


def progression_metrics(
    trajectory_summary: pd.DataFrame,
    trajectory_monthly: pd.DataFrame,
    spread_group: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    core_spread = spread_group[spread_group["spread_metric"] == "p75_minus_p25"].copy()
    spread_lookup = (
        core_spread.groupby(
            ["aoi", "index", "ecozone_code", "classification"], dropna=False
        )["group_mean_spread"]
        .mean()
        .rename("mean_core_spread")
        .reset_index()
    )

    peak_lookup = (
        trajectory_summary.groupby(
            ["aoi", "aoi_label", "index", "ecozone_code", "ecozone_label", "classification"],
            dropna=False,
        )
        .agg(
            seasonal_mean_anomaly=("seasonal_mean_anomaly", "mean"),
            late_season_mean_anomaly=("late_season_mean_anomaly", "mean"),
            peak_month_mean=("peak_month", "mean"),
            peak_month_span=("peak_month", lambda s: float(np.nanmax(s) - np.nanmin(s))),
            peak_anomaly_mean=("peak_anomaly", "mean"),
        )
        .reset_index()
    )

    group_cols = ["aoi", "aoi_label", "index", "ecozone_code", "ecozone_label", "classification"]
    for keys, group in trajectory_monthly.groupby(group_cols, dropna=False):
        aoi, aoi_label, index_name, ecozone_code, ecozone_label, classification = keys
        active = group.loc[np.abs(group["mean_anomaly"]) >= MEANINGFUL_ANOMALY, "mean_anomaly"].astype(float)

        if active.empty:
            direction_label = "mixed_or_small"
            direction_sign_consistency = np.nan
        else:
            sign_sum = float(np.sign(active).sum())
            dominant_sign = 1 if sign_sum > 0 else -1 if sign_sum < 0 else 0
            if dominant_sign == 0:
                direction_label = "mixed"
                direction_sign_consistency = 0.5
            else:
                direction_label = "positive" if dominant_sign > 0 else "negative"
                direction_sign_consistency = float((np.sign(active) == dominant_sign).mean())

        row = {
            "aoi": aoi,
            "aoi_label": aoi_label,
            "index": index_name,
            "ecozone_code": int(ecozone_code),
            "ecozone_label": ecozone_label,
            "classification": classification,
            "direction_label": direction_label,
            "direction_sign_consistency": direction_sign_consistency,
            "active_month_count": int(active.shape[0]),
        }
        rows.append(row)

    progression = pd.DataFrame(rows)
    progression = progression.merge(
        peak_lookup,
        on=["aoi", "aoi_label", "index", "ecozone_code", "ecozone_label", "classification"],
        how="left",
    ).merge(
        spread_lookup,
        on=["aoi", "index", "ecozone_code", "classification"],
        how="left",
    )

    progression["peak_month_name_mean"] = progression["peak_month_mean"].map(month_name)
    progression["peak_month_stability"] = progression["peak_month_span"].map(
        lambda x: inverse_score(x, good=0.0, bad=PEAK_MONTH_SPAN_BAD)
    )
    progression["spread_reliability"] = progression["mean_core_spread"].map(
        lambda x: inverse_score(x, good=GOOD_CORE_SPREAD, bad=BAD_CORE_SPREAD)
    )
    progression["trajectory_clarity"] = progression[
        ["direction_sign_consistency", "peak_month_stability", "spread_reliability"]
    ].mean(axis=1, skipna=True)
    return progression


def recovery_metrics(recovery_year: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = ["aoi", "aoi_label", "index", "ecozone_code", "ecozone_label"]

    for keys, group in recovery_year.groupby(group_cols, dropna=False):
        aoi, aoi_label, index_name, ecozone_code, ecozone_label = keys
        evaluable = group[group["recovery_status"].isin(RECOVERY_LABELS)].copy()
        wet_reference_cases = int((group["recovery_status"] == "wet_reference").sum())

        if evaluable.empty:
            dominant_recovery = None
            recovery_label_consistency = np.nan
            recovery_cases = 0
        else:
            dominant_recovery = dominant_label(evaluable["recovery_status"])
            counts = evaluable["recovery_status"].value_counts()
            recovery_label_consistency = float(counts.iloc[0] / counts.sum())
            recovery_cases = int(counts.sum())

        row = {
            "aoi": aoi,
            "aoi_label": aoi_label,
            "index": index_name,
            "ecozone_code": int(ecozone_code),
            "ecozone_label": ecozone_label,
            "dominant_recovery_label": dominant_recovery,
            "recovery_label_consistency": recovery_label_consistency,
            "recovery_cases": recovery_cases,
            "wet_reference_cases": wet_reference_cases,
        }

        for label in RECOVERY_LABELS:
            if recovery_cases == 0:
                row[f"{label}_fraction"] = np.nan
            else:
                row[f"{label}_fraction"] = float((evaluable["recovery_status"] == label).mean())

        rows.append(row)

    return pd.DataFrame(rows)


def overall_metrics(by_class: pd.DataFrame, recovery: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["aoi", "aoi_label", "index", "ecozone_code", "ecozone_label"]

    overall = (
        by_class.groupby(group_cols, dropna=False)
        .agg(
            onset_month_stability=("onset_month_stability", "mean"),
            onset_direction_consistency=("onset_direction_consistency", "mean"),
            direction_sign_consistency=("direction_sign_consistency", "mean"),
            mean_core_spread=("mean_core_spread", "mean"),
            spread_reliability=("spread_reliability", "mean"),
            trajectory_clarity=("trajectory_clarity", "mean"),
            onset_month_span_mean=("onset_month_span", "mean"),
            peak_month_span_mean=("peak_month_span", "mean"),
        )
        .reset_index()
    )

    dry = (
        by_class[by_class["classification"] == "dry"][
            ["aoi", "index", "ecozone_code", "onset_month_mean", "direction_label", "seasonal_mean_anomaly"]
        ]
        .rename(
            columns={
                "onset_month_mean": "dry_onset_month_mean",
                "direction_label": "dry_direction_label",
                "seasonal_mean_anomaly": "dry_seasonal_mean_anomaly",
            }
        )
    )
    wet = (
        by_class[by_class["classification"] == "wet"][
            ["aoi", "index", "ecozone_code", "onset_month_mean", "direction_label", "seasonal_mean_anomaly"]
        ]
        .rename(
            columns={
                "onset_month_mean": "wet_onset_month_mean",
                "direction_label": "wet_direction_label",
                "seasonal_mean_anomaly": "wet_seasonal_mean_anomaly",
            }
        )
    )

    overall = overall.merge(dry, on=["aoi", "index", "ecozone_code"], how="left")
    overall = overall.merge(wet, on=["aoi", "index", "ecozone_code"], how="left")
    overall = overall.merge(
        recovery,
        on=["aoi", "aoi_label", "index", "ecozone_code", "ecozone_label"],
        how="left",
    )

    overall["composite_reliability_score"] = overall[
        [
            "onset_month_stability",
            "direction_sign_consistency",
            "trajectory_clarity",
            "recovery_label_consistency",
        ]
    ].mean(axis=1, skipna=True)

    overall["role_note"] = np.where(
        overall["wet_seasonal_mean_anomaly"].abs() > overall["dry_seasonal_mean_anomaly"].abs(),
        "stronger wet-year signal",
        "stronger dry-year signal",
    )
    overall.loc[
        overall["dry_direction_label"].isna() & overall["wet_direction_label"].isna(),
        "role_note",
    ] = "insufficient classification detail"

    overall["reliability_rank_within_group"] = (
        overall.groupby(["aoi", "ecozone_code"])["composite_reliability_score"]
        .rank(ascending=False, method="min")
    )

    return overall.sort_values(["aoi", "ecozone_code", "reliability_rank_within_group", "index"])


def add_class_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["group_label"] = out["aoi"].astype(str) + " | " + out["ecozone_label"].astype(str)
    return out


def write_heatmaps(overall: pd.DataFrame) -> None:
    metrics = [
        ("onset_month_stability", "Onset Month Stability"),
        ("direction_sign_consistency", "Direction Consistency"),
        ("trajectory_clarity", "Trajectory Clarity"),
        ("recovery_label_consistency", "Recovery Label Consistency"),
    ]

    plot_df = add_class_labels(overall)
    row_order = (
        plot_df[["aoi", "ecozone_code", "group_label"]]
        .drop_duplicates()
        .sort_values(["aoi", "ecozone_code"])
    )
    rows = row_order["group_label"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, metrics):
        matrix = (
            plot_df.pivot_table(index="group_label", columns="index", values=metric)
            .reindex(index=rows, columns=INDICES)
        )
        masked = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
        im = ax.imshow(masked, cmap=CMAP, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(INDICES)), INDICES)
        ax.set_yticks(range(len(rows)), rows)
        for i in range(masked.shape[0]):
            for j in range(masked.shape[1]):
                value = matrix.iloc[i, j]
                text = "NA" if pd.isna(value) else f"{value:.2f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("Heuristic score (0 to 1)")
    fig.suptitle("Cross-Index Reliability Comparison by AOI and Ecozone", fontsize=13)
    fig.savefig(OUT_DIR / "index_role_metric_heatmaps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_bar_figure(overall: pd.DataFrame) -> None:
    metrics = [
        ("onset_month_stability", "Onset"),
        ("direction_sign_consistency", "Direction"),
        ("trajectory_clarity", "Trajectory"),
        ("recovery_label_consistency", "Recovery"),
        ("composite_reliability_score", "Composite"),
    ]

    agg = (
        overall.groupby("index", dropna=False)[[metric for metric, _ in metrics]]
        .mean()
        .reindex(INDICES)
    )

    x = np.arange(len(INDICES))
    width = 0.14
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for offset, (metric, label) in enumerate(metrics):
        positions = x + (offset - 2) * width
        ax.bar(positions, agg[metric].to_numpy(dtype=float), width=width, label=label)

    ax.set_xticks(x, INDICES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean heuristic score")
    ax.set_title("Mean Cross-Index Reliability Scores Across AOIs and Ecozones")
    ax.legend(ncols=5, fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "index_role_metric_bars.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    onset_summary = pd.read_csv(require_file(ONSET_SUMMARY_FILE))
    trajectory_summary = pd.read_csv(require_file(TRAJECTORY_SUMMARY_FILE))
    trajectory_monthly = pd.read_csv(require_file(TRAJECTORY_MONTHLY_FILE))
    spread_group = pd.read_csv(require_file(SPREAD_GROUP_FILE))
    recovery_year = pd.read_csv(require_file(RECOVERY_YEAR_FILE))

    onset_summary = onset_summary[onset_summary["index"].isin(INDICES)].copy()
    trajectory_summary = trajectory_summary[trajectory_summary["index"].isin(INDICES)].copy()
    trajectory_monthly = trajectory_monthly[trajectory_monthly["index"].isin(INDICES)].copy()
    spread_group = spread_group[spread_group["index"].isin(INDICES)].copy()
    recovery_year = recovery_year[recovery_year["index"].isin(INDICES)].copy()

    onset = onset_metrics(onset_summary)
    progression = progression_metrics(trajectory_summary, trajectory_monthly, spread_group)
    by_class = onset.merge(
        progression,
        on=["aoi", "aoi_label", "index", "ecozone_code", "ecozone_label", "classification"],
        how="outer",
    ).sort_values(["aoi", "ecozone_code", "index", "classification"])

    recovery = recovery_metrics(recovery_year)
    overall = overall_metrics(by_class, recovery)
    ranking = (
        overall[
            [
                "aoi",
                "aoi_label",
                "ecozone_code",
                "ecozone_label",
                "index",
                "composite_reliability_score",
                "reliability_rank_within_group",
                "role_note",
            ]
        ]
        .sort_values(["aoi", "ecozone_code", "reliability_rank_within_group", "index"])
    )

    by_class.to_csv(OUT_DIR / "index_role_comparison_by_class.csv", index=False)
    overall.to_csv(OUT_DIR / "index_role_comparison_overall.csv", index=False)
    ranking.to_csv(OUT_DIR / "index_role_overall_ranking.csv", index=False)

    write_heatmaps(overall)
    write_bar_figure(overall)

    print(f"Saved index role summary outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
