#!/usr/bin/env python3
"""
Explicit ecozone-vs-ecozone comparison layer for Sentinel ecozone summaries.

Purpose:
  Read existing ecozone-level summary CSV outputs and produce direct between-
  ecozone comparisons within each AOI, index, and year-group classification.
  The intent is descriptive: make it easy to say one ecozone diverges earlier,
  has stronger suppression, or shows a more sustained response than another.

Outputs:
  Results/Other/ecozone_comparative_dynamics/
    ecozone_reference_metrics.csv
    ecozone_pairwise_comparisons.csv
    ecozone_pairwise_summary.csv
    ecozone_onset_heatmaps.png
    ecozone_magnitude_heatmaps.png
    ecozone_trajectory_heatmaps.png
    ecozone_recovery_heatmaps.png
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "Results" / "Other" / "ecozone_comparative_dynamics"

ONSET_SUMMARY_FILE = (
    PROJECT_ROOT / "Results" / "2_Anomaly_Onset" / "onset_timing" / "onset_timing_summary_all_percentiles.csv"
)
ONSET_DETAILS_FILE = (
    PROJECT_ROOT / "Results" / "2_Anomaly_Onset" / "onset_timing" / "onset_timing_monthly_details_all_percentiles.csv"
)
FRACTION_BELOW_FILE = (
    PROJECT_ROOT / "Results" / "2_Anomaly_Onset" / "fraction_below_baseline" / "fraction_below_baseline.csv"
)
TRAJECTORY_SUMMARY_FILE = (
    PROJECT_ROOT / "Results" / "3_Anomaly_Progression" / "monthly_trajectories" / "trajectory_summary_all_percentiles.csv"
)
TRAJECTORY_MONTHLY_FILE = (
    PROJECT_ROOT / "Results" / "3_Anomaly_Progression" / "monthly_trajectories" / "trajectory_monthly_anomalies_all_percentiles.csv"
)
RECOVERY_YEAR_FILE = (
    PROJECT_ROOT / "Results" / "4_Anomaly_Recovery" / "simple_recovery" / "simple_recovery_by_year_all_percentiles.csv"
)

INCLUDE_EVI_IF_EASY = False
BASE_INDICES = ["NDVI", "NDMI"]
ECOZONE_ORDER = [1, 2, 3]
ECOZONE_NAME = {1: "Cool", 2: "Intermediate", 3: "Hot"}
KEY_MONTHS = [4, 6, 8, 10]
GROWING_MONTHS = [4, 5, 6, 7, 8, 9, 10]
EARLY_MONTHS = [4, 5, 6]
LATE_MONTHS = [8, 9, 10]
RECOVERY_LABELS = ["quick", "partial", "slow", "insufficient_data"]


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


def interpretation_note(metric_name: str, difference: float, ecozone_a: str, ecozone_b: str) -> str:
    if pd.isna(difference):
        return "comparison unavailable from source summaries"

    if metric_name == "onset_month":
        if abs(difference) < 0.25:
            return f"{ecozone_a} and {ecozone_b} have similar mean onset timing"
        earlier = ecozone_a if difference < 0 else ecozone_b
        later = ecozone_b if difference < 0 else ecozone_a
        return f"{earlier} diverges earlier than {later} by about {abs(difference):.1f} months"

    if metric_name in {"onset_anomaly", "peak_anomaly", "cumulative_anomaly", "late_minus_early_anomaly", "mean_net_change"}:
        if abs(difference) < 0.01:
            return f"{ecozone_a} and {ecozone_b} are very similar on {metric_name}"
        stronger = ecozone_a if abs(difference) == abs(difference) and difference > 0 else ecozone_b
        weaker = ecozone_b if stronger == ecozone_a else ecozone_a
        if metric_name == "late_minus_early_anomaly":
            return f"{stronger} shows a more positive late-season shift than {weaker}"
        if metric_name == "mean_net_change":
            return f"{stronger} shows a larger spring recovery shift than {weaker}"
        return f"{stronger} shows the larger {metric_name.replace('_', ' ')} magnitude"

    if metric_name.startswith("fraction_below_"):
        if abs(difference) < 0.02:
            return f"{ecozone_a} and {ecozone_b} have similar below-baseline area in that month"
        higher = ecozone_a if difference > 0 else ecozone_b
        lower = ecozone_b if higher == ecozone_a else ecozone_a
        return f"{higher} has more area below baseline than {lower}"

    if metric_name == "peak_month":
        if abs(difference) < 0.25:
            return f"{ecozone_a} and {ecozone_b} reach peak anomaly at a similar time"
        earlier = ecozone_a if difference < 0 else ecozone_b
        later = ecozone_b if difference < 0 else ecozone_a
        return f"{earlier} reaches peak anomaly earlier than {later}"

    if metric_name == "dominant_recovery_label":
        if difference == 0:
            return f"{ecozone_a} and {ecozone_b} share the same dominant recovery label"
        return f"{ecozone_a} and {ecozone_b} differ in dominant recovery label"

    return "descriptive ecozone comparison"


def onset_reference(onset_summary: pd.DataFrame) -> pd.DataFrame:
    onset = (
        onset_summary.groupby(
            ["aoi", "aoi_label", "index", "classification", "ecozone_code", "ecozone_label"],
            dropna=False,
        )
        .agg(
            onset_month=("onset_month", "mean"),
            onset_anomaly=("onset_anomaly", "mean"),
            onset_records=("onset_month", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )
    onset["peak_suppression_month"] = np.nan
    return onset


def onset_monthly_reference(onset_details: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        onset_details.groupby(
            ["aoi", "aoi_label", "index", "classification", "ecozone_code", "ecozone_label", "month", "month_name"],
            dropna=False,
        )
        .agg(mean_anomaly=("anomaly", "mean"))
        .reset_index()
    )
    return grouped


def fraction_below_reference(fraction_below: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        fraction_below.groupby(
            ["aoi", "aoi_label", "index", "classification", "ecozone_code", "ecozone_label", "month", "month_name"],
            dropna=False,
        )
        .agg(
            fraction_below_baseline=("fraction_below_baseline", "mean"),
            valid_pixel_count=("valid_pixel_count", "mean"),
        )
        .reset_index()
    )
    return grouped


def trajectory_reference(
    trajectory_summary: pd.DataFrame,
    trajectory_monthly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        trajectory_summary.groupby(
            ["aoi", "aoi_label", "index", "classification", "ecozone_code", "ecozone_label"],
            dropna=False,
        )
        .agg(
            seasonal_mean_anomaly=("seasonal_mean_anomaly", "mean"),
            peak_month=("peak_month", "mean"),
            peak_anomaly=("peak_anomaly", "mean"),
            late_season_mean_anomaly=("late_season_mean_anomaly", "mean"),
        )
        .reset_index()
    )

    monthly = (
        trajectory_monthly.groupby(
            ["aoi", "aoi_label", "index", "classification", "ecozone_code", "ecozone_label", "month", "month_name"],
            dropna=False,
        )
        .agg(mean_anomaly=("mean_anomaly", "mean"))
        .reset_index()
    )

    rows: list[dict] = []
    keys = ["aoi", "aoi_label", "index", "classification", "ecozone_code", "ecozone_label"]
    for group_keys, group in monthly.groupby(keys, dropna=False):
        aoi, aoi_label, index_name, classification, ecozone_code, ecozone_label = group_keys
        month_lookup = group.set_index("month")["mean_anomaly"]
        cumulative = float(month_lookup.reindex(GROWING_MONTHS).fillna(0.0).sum())
        early = float(month_lookup.reindex(EARLY_MONTHS).mean())
        late = float(month_lookup.reindex(LATE_MONTHS).mean())
        late_minus_early = late - early
        rows.append({
            "aoi": aoi,
            "aoi_label": aoi_label,
            "index": index_name,
            "classification": classification,
            "ecozone_code": int(ecozone_code),
            "ecozone_label": ecozone_label,
            "cumulative_anomaly": cumulative,
            "early_season_mean_anomaly": early,
            "late_season_mean_anomaly_from_monthly": late,
            "late_minus_early_anomaly": late_minus_early,
        })

    shape = pd.DataFrame(rows)
    summary = summary.merge(
        shape,
        on=["aoi", "aoi_label", "index", "classification", "ecozone_code", "ecozone_label"],
        how="left",
    )
    return summary, monthly


def recovery_reference(recovery_year: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["aoi", "aoi_label", "index", "ecozone_code", "ecozone_label"]

    for group_keys, group in recovery_year.groupby(keys, dropna=False):
        aoi, aoi_label, index_name, ecozone_code, ecozone_label = group_keys
        evaluable = group[group["recovery_status"].isin(RECOVERY_LABELS)].copy()
        if evaluable.empty:
            dominant_label = None
            label_consistency = np.nan
            mean_late = np.nan
            mean_spring = np.nan
            mean_net = np.nan
            evaluable_cases = 0
        else:
            counts = evaluable["recovery_status"].value_counts()
            if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
                dominant_label = "tie"
            else:
                dominant_label = str(counts.index[0])
            label_consistency = float(counts.iloc[0] / counts.sum())
            mean_late = float(evaluable["late_season_anomaly"].mean())
            mean_spring = float(evaluable["following_spring_anomaly"].mean())
            mean_net = float(evaluable["net_change"].mean())
            evaluable_cases = int(len(evaluable))

        rows.append({
            "aoi": aoi,
            "aoi_label": aoi_label,
            "index": index_name,
            "classification": "dry",
            "ecozone_code": int(ecozone_code),
            "ecozone_label": ecozone_label,
            "dominant_recovery_label": dominant_label,
            "recovery_label_consistency": label_consistency,
            "mean_late_season_anomaly": mean_late,
            "mean_following_spring_anomaly": mean_spring,
            "mean_net_change": mean_net,
            "recovery_cases": evaluable_cases,
        })
    return pd.DataFrame(rows)


def build_reference_metrics(
    onset_ref: pd.DataFrame,
    fraction_ref: pd.DataFrame,
    trajectory_ref: pd.DataFrame,
    recovery_ref: pd.DataFrame,
) -> pd.DataFrame:
    reference = onset_ref.merge(
        trajectory_ref,
        on=["aoi", "aoi_label", "index", "classification", "ecozone_code", "ecozone_label"],
        how="outer",
    ).merge(
        recovery_ref,
        on=["aoi", "aoi_label", "index", "classification", "ecozone_code", "ecozone_label"],
        how="left",
    )

    for month in KEY_MONTHS:
        month_slice = fraction_ref[fraction_ref["month"] == month][
            ["aoi", "index", "classification", "ecozone_code", "fraction_below_baseline"]
        ].rename(columns={"fraction_below_baseline": f"fraction_below_{month:02d}"})
        reference = reference.merge(
            month_slice,
            on=["aoi", "index", "classification", "ecozone_code"],
            how="left",
        )

    return reference.sort_values(["aoi", "index", "classification", "ecozone_code"])


def pairwise_rows_from_reference(reference: pd.DataFrame) -> list[dict]:
    metric_specs = [
        ("onset_timing", "onset_month"),
        ("onset_magnitude", "onset_anomaly"),
        ("magnitude", "fraction_below_04"),
        ("magnitude", "fraction_below_06"),
        ("magnitude", "fraction_below_08"),
        ("magnitude", "fraction_below_10"),
        ("trajectory", "peak_month"),
        ("trajectory", "peak_anomaly"),
        ("trajectory", "cumulative_anomaly"),
        ("trajectory", "late_minus_early_anomaly"),
        ("recovery", "mean_late_season_anomaly"),
        ("recovery", "mean_following_spring_anomaly"),
        ("recovery", "mean_net_change"),
    ]

    rows: list[dict] = []
    keys = ["aoi", "aoi_label", "index", "classification"]
    for group_keys, group in reference.groupby(keys, dropna=False):
        aoi, aoi_label, index_name, classification = group_keys
        group = group.sort_values("ecozone_code")
        for left_idx, right_idx in combinations(group.index.tolist(), 2):
            left = group.loc[left_idx]
            right = group.loc[right_idx]
            ecozone_a = str(left["ecozone_label"])
            ecozone_b = str(right["ecozone_label"])

            for comparison_type, metric_name in metric_specs:
                value_a = left.get(metric_name, np.nan)
                value_b = right.get(metric_name, np.nan)
                difference = np.nan if pd.isna(value_a) or pd.isna(value_b) else float(value_a - value_b)
                rows.append({
                    "aoi": aoi,
                    "aoi_label": aoi_label,
                    "index": index_name,
                    "year_group": classification,
                    "ecozone_a": ecozone_a,
                    "ecozone_b": ecozone_b,
                    "comparison_type": comparison_type,
                    "metric_name": metric_name,
                    "value_a": value_a,
                    "value_b": value_b,
                    "difference": difference,
                    "interpretation_note": interpretation_note(metric_name, difference, ecozone_a, ecozone_b),
                })

            label_a = left.get("dominant_recovery_label")
            label_b = right.get("dominant_recovery_label")
            label_diff = np.nan if pd.isna(label_a) or pd.isna(label_b) else float(label_a != label_b)
            rows.append({
                "aoi": aoi,
                "aoi_label": aoi_label,
                "index": index_name,
                "year_group": classification,
                "ecozone_a": ecozone_a,
                "ecozone_b": ecozone_b,
                "comparison_type": "recovery",
                "metric_name": "dominant_recovery_label",
                "value_a": label_a,
                "value_b": label_b,
                "difference": label_diff,
                "interpretation_note": interpretation_note("dominant_recovery_label", label_diff, ecozone_a, ecozone_b),
            })

    return rows


def build_pairwise_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    numeric = pairwise[pd.to_numeric(pairwise["difference"], errors="coerce").notna()].copy()
    numeric["abs_difference"] = numeric["difference"].abs()
    summary = (
        numeric.groupby(["aoi", "index", "year_group", "comparison_type", "metric_name"], dropna=False)
        .agg(
            mean_abs_difference=("abs_difference", "mean"),
            max_abs_difference=("abs_difference", "max"),
            pair_count=("abs_difference", "count"),
        )
        .reset_index()
        .sort_values(["aoi", "index", "year_group", "comparison_type", "metric_name"])
    )
    return summary


def heatmap_matrix(pairwise: pd.DataFrame, aoi: str, index_name: str, year_group: str, metric_name: str) -> pd.DataFrame:
    labels = [ECOZONE_NAME[code] for code in ECOZONE_ORDER]
    values = np.full((len(labels), len(labels)), np.nan, dtype=float)
    np.fill_diagonal(values, 0.0)
    matrix = pd.DataFrame(values, index=labels, columns=labels)

    subset = pairwise[
        (pairwise["aoi"] == aoi)
        & (pairwise["index"] == index_name)
        & (pairwise["year_group"] == year_group)
        & (pairwise["metric_name"] == metric_name)
    ]

    for _, row in subset.iterrows():
        a = row["ecozone_a"]
        b = row["ecozone_b"]
        diff = row["difference"]
        if a in matrix.index and b in matrix.columns:
            matrix.loc[a, b] = diff
        if b in matrix.index and a in matrix.columns and pd.notna(diff):
            matrix.loc[b, a] = -diff

    return matrix


def write_metric_heatmaps(pairwise: pd.DataFrame, metric_name: str, title: str, out_name: str, cmap: str = "coolwarm") -> None:
    indices = sorted(pairwise["index"].dropna().unique().tolist())
    year_groups = sorted(pairwise["year_group"].dropna().unique().tolist())
    aois = sorted(pairwise["aoi"].dropna().unique().tolist())

    nrows = len(aois)
    ncols = max(1, len(indices) * len(year_groups))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), constrained_layout=True)
    axes = np.atleast_2d(axes)

    vmax = pairwise.loc[pairwise["metric_name"] == metric_name, "difference"].abs().max()
    if pd.isna(vmax) or vmax == 0:
        vmax = 1.0

    for row_idx, aoi in enumerate(aois):
        col_idx = 0
        for index_name in indices:
            for year_group in year_groups:
                ax = axes[row_idx, col_idx]
                matrix = heatmap_matrix(pairwise, aoi, index_name, year_group, metric_name)
                masked = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
                im = ax.imshow(masked, cmap=cmap, vmin=-vmax, vmax=vmax)
                ax.set_title(f"{aoi} | {index_name} | {year_group}", fontsize=10)
                ax.set_xticks(range(len(matrix.columns)), matrix.columns)
                ax.set_yticks(range(len(matrix.index)), matrix.index)
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        value = matrix.iloc[i, j]
                        text = "NA" if pd.isna(value) else f"{value:.2f}"
                        ax.text(j, i, text, ha="center", va="center", fontsize=8)
                col_idx += 1

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("Ecozone A - Ecozone B difference")
    fig.suptitle(title, fontsize=13)
    fig.savefig(OUT_DIR / out_name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    indices = resolve_indices()

    onset_summary = pd.read_csv(require_file(ONSET_SUMMARY_FILE))
    onset_details = pd.read_csv(require_file(ONSET_DETAILS_FILE))
    fraction_below = pd.read_csv(require_file(FRACTION_BELOW_FILE))
    trajectory_summary = pd.read_csv(require_file(TRAJECTORY_SUMMARY_FILE))
    trajectory_monthly = pd.read_csv(require_file(TRAJECTORY_MONTHLY_FILE))
    recovery_year = pd.read_csv(require_file(RECOVERY_YEAR_FILE))

    onset_summary = onset_summary[onset_summary["index"].isin(indices)].copy()
    onset_details = onset_details[onset_details["index"].isin(indices)].copy()
    fraction_below = fraction_below[fraction_below["index"].isin(indices)].copy()
    trajectory_summary = trajectory_summary[trajectory_summary["index"].isin(indices)].copy()
    trajectory_monthly = trajectory_monthly[trajectory_monthly["index"].isin(indices)].copy()
    recovery_year = recovery_year[recovery_year["index"].isin(indices)].copy()

    onset_ref = onset_reference(onset_summary)
    onset_monthly_reference(onset_details)
    fraction_ref = fraction_below_reference(fraction_below)
    trajectory_ref, _ = trajectory_reference(trajectory_summary, trajectory_monthly)
    recovery_ref = recovery_reference(recovery_year)

    reference = build_reference_metrics(onset_ref, fraction_ref, trajectory_ref, recovery_ref)
    pairwise = pd.DataFrame(pairwise_rows_from_reference(reference))
    summary = build_pairwise_summary(pairwise)

    reference.to_csv(OUT_DIR / "ecozone_reference_metrics.csv", index=False)
    pairwise.to_csv(OUT_DIR / "ecozone_pairwise_comparisons.csv", index=False)
    summary.to_csv(OUT_DIR / "ecozone_pairwise_summary.csv", index=False)

    write_metric_heatmaps(
        pairwise,
        metric_name="onset_month",
        title="Ecozone Onset Timing Differences",
        out_name="ecozone_onset_heatmaps.png",
    )
    write_metric_heatmaps(
        pairwise,
        metric_name="fraction_below_06",
        title="Ecozone June Below-Baseline Fraction Differences",
        out_name="ecozone_magnitude_heatmaps.png",
    )
    write_metric_heatmaps(
        pairwise,
        metric_name="cumulative_anomaly",
        title="Ecozone Cumulative Apr-Oct Anomaly Differences",
        out_name="ecozone_trajectory_heatmaps.png",
    )
    write_metric_heatmaps(
        pairwise,
        metric_name="mean_net_change",
        title="Ecozone Recovery Net-Change Differences",
        out_name="ecozone_recovery_heatmaps.png",
    )

    print(f"Saved ecozone comparative dynamics outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
