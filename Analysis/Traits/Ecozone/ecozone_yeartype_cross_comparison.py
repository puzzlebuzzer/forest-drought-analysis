#!/usr/bin/env python3
"""
Explicit ecozone x year-type comparison layer from existing summary outputs.

Purpose:
  Produce compact ecozone/year-type tables and pairwise ecozone comparisons
  using the existing wet/normal/dry comparison outputs. This is descriptive and
  deterministic, intended to support direct statements about how ecozones differ
  within wet and dry conditions and how wet-vs-dry contrasts vary by ecozone.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "Results" / "Other" / "ecozone_yeartype_cross_comparison"

YEARCLASS_SUMMARY_FILE = (
    PROJECT_ROOT / "Results" / "Other" / "ecozone_yearclass_comparison" / "yearclass_summary_by_ecozone.csv"
)
REFERENCE_FILE = (
    PROJECT_ROOT / "Results" / "Other" / "ecozone_comparative_dynamics" / "ecozone_reference_metrics.csv"
)

BASE_INDICES = ["NDVI", "NDMI"]
INCLUDE_EVI_IF_EASY = False
YEAR_TYPES = ["wet", "normal", "dry"]

METRIC_MAP = {
    "onset": "onset_month",
    "magnitude": "cumulative_anomaly",
    "trajectory": "late_minus_early_anomaly",
    "spatial_extent": "fraction_below_10",
}


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input CSV not found: {path}")
    return path


def resolve_indices() -> list[str]:
    indices = list(BASE_INDICES)
    if INCLUDE_EVI_IF_EASY and "EVI" not in indices:
        indices.append("EVI")
    return indices


def load_yeartype_table(summary_by_ecozone: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (aoi, index_name, ecozone), group in summary_by_ecozone.groupby(["AOI", "index", "ecozone"], dropna=False):
        group = group.set_index("metric_type")
        for year_type in YEAR_TYPES:
            row = {
                "AOI": aoi,
                "index": index_name,
                "ecozone": ecozone,
                "year_type": year_type,
            }
            for output_name, metric_type in METRIC_MAP.items():
                col = f"value_{year_type}"
                if metric_type in group.index and col in group.columns:
                    row[output_name] = group.loc[metric_type, col]
                else:
                    row[output_name] = np.nan
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["AOI", "index", "ecozone", "year_type"])


def build_pairwise(yeartype_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    metrics = ["onset", "magnitude", "trajectory", "spatial_extent"]

    for (aoi, index_name, year_type), group in yeartype_table.groupby(["AOI", "index", "year_type"], dropna=False):
        group = group.sort_values("ecozone")
        for left_idx, right_idx in combinations(group.index.tolist(), 2):
            left = group.loc[left_idx]
            right = group.loc[right_idx]
            for metric in metrics:
                left_value = left[metric]
                right_value = right[metric]
                difference = np.nan if pd.isna(left_value) or pd.isna(right_value) else float(left_value - right_value)
                rows.append({
                    "AOI": aoi,
                    "index": index_name,
                    "ecozone_a": left["ecozone"],
                    "ecozone_b": right["ecozone"],
                    "year_type": year_type,
                    "metric": metric,
                    "difference": difference,
                })
    return pd.DataFrame(rows).sort_values(["AOI", "index", "year_type", "metric", "ecozone_a", "ecozone_b"])


def build_crossed_differences(yeartype_table: pd.DataFrame) -> pd.DataFrame:
    wide = (
        yeartype_table.pivot_table(
            index=["AOI", "index", "ecozone"],
            columns="year_type",
            values=["onset", "magnitude", "trajectory", "spatial_extent"],
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns = [
        "_".join([str(part) for part in col if str(part) != ""]).strip("_")
        if isinstance(col, tuple) else str(col)
        for col in wide.columns
    ]

    rows: list[dict] = []
    for _, row in wide.iterrows():
        rows.append({
            "AOI": row["AOI"],
            "index": row["index"],
            "ecozone": row["ecozone"],
            "delta_onset_wet_dry": row.get("onset_wet", np.nan) - row.get("onset_dry", np.nan),
            "delta_magnitude_wet_dry": row.get("magnitude_wet", np.nan) - row.get("magnitude_dry", np.nan),
            "delta_trajectory_wet_dry": row.get("trajectory_wet", np.nan) - row.get("trajectory_dry", np.nan),
            "delta_spatial_extent_wet_dry": row.get("spatial_extent_wet", np.nan) - row.get("spatial_extent_dry", np.nan),
        })
    return pd.DataFrame(rows).sort_values(["AOI", "index", "ecozone"])


def build_crossed_pairwise(crossed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    metrics = [
        "delta_onset_wet_dry",
        "delta_magnitude_wet_dry",
        "delta_trajectory_wet_dry",
        "delta_spatial_extent_wet_dry",
    ]

    for (aoi, index_name), group in crossed.groupby(["AOI", "index"], dropna=False):
        group = group.sort_values("ecozone")
        for left_idx, right_idx in combinations(group.index.tolist(), 2):
            left = group.loc[left_idx]
            right = group.loc[right_idx]
            for metric in metrics:
                lv = left[metric]
                rv = right[metric]
                difference = np.nan if pd.isna(lv) or pd.isna(rv) else float(lv - rv)
                rows.append({
                    "AOI": aoi,
                    "index": index_name,
                    "ecozone_a": left["ecozone"],
                    "ecozone_b": right["ecozone"],
                    "year_type": "wet_vs_dry_cross",
                    "metric": metric,
                    "difference": difference,
                })
    return pd.DataFrame(rows).sort_values(["AOI", "index", "metric", "ecozone_a", "ecozone_b"])


def statement_for_metric(metric: str, left_label: str, right_label: str, difference: float, year_type: str) -> str:
    if pd.isna(difference):
        return f"{left_label} vs {right_label} under {year_type}: {metric} unavailable"

    if metric == "onset":
        if abs(difference) < 0.25:
            return f"{left_label} and {right_label} have similar onset under {year_type}"
        earlier = left_label if difference < 0 else right_label
        later = right_label if difference < 0 else left_label
        return f"Under {year_type}, {earlier} diverges earlier than {later}"

    if metric == "spatial_extent":
        if abs(difference) < 0.02:
            return f"{left_label} and {right_label} have similar late-season spatial extent under {year_type}"
        higher = left_label if difference > 0 else right_label
        lower = right_label if difference > 0 else left_label
        return f"Under {year_type}, {higher} has greater below-baseline extent than {lower}"

    if metric in {"magnitude", "trajectory"}:
        if abs(difference) < 0.02:
            return f"{left_label} and {right_label} are similar on {metric} under {year_type}"
        stronger = left_label if abs(left_value := difference) == abs(difference) and difference > 0 else right_label
        weaker = right_label if stronger == left_label else left_label
        return f"Under {year_type}, {stronger} shows the larger {metric} signal than {weaker}"

    if metric.startswith("delta_"):
        pretty = metric.replace("delta_", "").replace("_wet_dry", "")
        if abs(difference) < 0.02:
            return f"{left_label} and {right_label} have similar wet-vs-dry change in {pretty}"
        larger = left_label if difference > 0 else right_label
        smaller = right_label if difference > 0 else left_label
        return f"{larger} shows the larger wet-vs-dry shift in {pretty} than {smaller}"

    return f"{left_label} vs {right_label} under {year_type}: {metric} difference {difference:.2f}"


def build_statements(pairwise: pd.DataFrame, crossed_pairwise: pd.DataFrame) -> pd.DataFrame:
    statements: list[dict] = []
    for _, row in pd.concat([pairwise, crossed_pairwise], ignore_index=True).iterrows():
        statements.append({
            "AOI": row["AOI"],
            "index": row["index"],
            "ecozone_a": row["ecozone_a"],
            "ecozone_b": row["ecozone_b"],
            "year_type": row["year_type"],
            "metric": row["metric"],
            "statement": statement_for_metric(
                row["metric"],
                row["ecozone_a"],
                row["ecozone_b"],
                row["difference"],
                row["year_type"],
            ),
        })
    return pd.DataFrame(statements)


def write_statement_text(statements: pd.DataFrame) -> None:
    lines: list[str] = []
    for (aoi, index_name), group in statements.groupby(["AOI", "index"], dropna=False):
        lines.append(f"{aoi} | {index_name}")
        lines.append("")
        for _, row in group.iterrows():
            lines.append(f"- {row['statement']}")
        lines.append("")

    (OUT_DIR / "ecozone_yeartype_statements.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    indices = resolve_indices()

    summary_by_ecozone = pd.read_csv(require_file(YEARCLASS_SUMMARY_FILE))
    reference = pd.read_csv(require_file(REFERENCE_FILE))

    summary_by_ecozone = summary_by_ecozone[summary_by_ecozone["index"].isin(indices)].copy()
    reference = reference[reference["index"].isin(indices)].copy()

    yeartype_table = load_yeartype_table(summary_by_ecozone)
    pairwise = build_pairwise(yeartype_table)
    crossed = build_crossed_differences(yeartype_table)
    crossed_pairwise = build_crossed_pairwise(crossed)
    statements = build_statements(pairwise, crossed_pairwise)

    yeartype_table.to_csv(OUT_DIR / "ecozone_yeartype_table.csv", index=False)
    pairwise.to_csv(OUT_DIR / "ecozone_yeartype_pairwise_comparisons.csv", index=False)
    crossed.to_csv(OUT_DIR / "ecozone_yeartype_crossed_differences.csv", index=False)
    crossed_pairwise.to_csv(OUT_DIR / "ecozone_yeartype_crossed_pairwise.csv", index=False)
    statements.to_csv(OUT_DIR / "ecozone_yeartype_statements.csv", index=False)
    write_statement_text(statements)

    print(f"Saved ecozone year-type cross-comparison outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
