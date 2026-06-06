#!/usr/bin/env python3
"""
Shared helpers for new Sentinel ecozone investigation scripts.

This module is intentionally read-only with respect to the existing project
state: it loads monthly Sentinel composites, AOI ecozone masks, and the
existing AOI wet/dry year classification table without modifying any current
analysis logic or results.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from src.aoi import get_aoi_config
from src.paths import project_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

AOIS = ["north", "south"]
INDICES = ["NDVI", "NDMI", "EVI"]
VALID_ECOZONE_CODES = [1, 2, 3]
ECOZONE_LABELS = {1: "Cool", 2: "Intermediate", 3: "Hot"}
ECOZONE_COLORS = {1: "#4E90C8", 2: "#72B063", 3: "#D9534F"}
AOI_DISPLAY = {"north": "GW National Forest", "south": "Great Smoky Mtns"}
CLASS_COLORS = {"wet": "#3A7FC1", "neutral": "#8C8C8C", "dry": "#C97834"}
SUMMARY_PERCENTILES = [50, 75]

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

GROWING_MONTHS = [4, 5, 6, 7, 8, 9, 10]
LATE_SEASON_MONTHS = [8, 9, 10]
FOLLOWING_SPRING_MONTHS = [4, 5, 6]

MIN_PIXELS = 100

# Conservative, explicit thresholds. These are descriptive heuristics, not
# significance tests. Scripts expose them in comments and outputs.
DIVERGENCE_STD_MULTIPLIER = 1.0
DIVERGENCE_MIN_ABS_ANOMALY = 0.03
RECOVERY_NEAR_BASELINE_TOLERANCE = 0.02
RECOVERY_IMPROVEMENT_FRACTION = 0.5

# Current populated monthly composite location on disk.
MONTHLY_BASE_DIR = PROJECT_ROOT / "Results" / "0-CacheBaseData" / "monthly_max"


def ensure_monthly_base_dir() -> Path:
    if not MONTHLY_BASE_DIR.exists():
        raise FileNotFoundError(
            f"Monthly Sentinel composites not found at {MONTHLY_BASE_DIR}"
        )
    return MONTHLY_BASE_DIR


def load_wet_dry_years() -> pd.DataFrame:
    path = project_path("config_dir") / "wet_dry_years.csv"
    df = pd.read_csv(path)
    df["year"] = df["year"].astype(int)
    return df.sort_values(["aoi", "year"]).reset_index(drop=True)


def load_ecozone_masks(aoi: str) -> dict[int, np.ndarray]:
    cfg = get_aoi_config(aoi)
    ecozone_path = cfg.ecozone_dir / "tnc_ecozone_simplified_snapped.tif"
    with rasterio.open(ecozone_path) as src:
        ecozone = src.read(1)
    return {code: (ecozone == code) for code in VALID_ECOZONE_CODES}


def monthly_file(index_name: str, aoi: str, year: int, month: int) -> Path:
    return ensure_monthly_base_dir() / f"{index_name.lower()}_{aoi}" / f"{year}_{month:02d}.tif"


def load_monthly_array(index_name: str, aoi: str, year: int, month: int) -> np.ndarray | None:
    path = monthly_file(index_name, aoi, year, month)
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def ecozone_percentiles(
    arr: np.ndarray,
    eco_masks: dict[int, np.ndarray],
    summary_percentiles: list[int],
) -> dict[int, dict[int, float]]:
    values = {}
    finite = np.isfinite(arr)
    for code, mask in eco_masks.items():
        values[code] = {}
        valid = finite & mask
        if valid.sum() >= MIN_PIXELS:
            px = arr[valid]
            for percentile in summary_percentiles:
                values[code][percentile] = float(np.nanpercentile(px, percentile))
        else:
            for percentile in summary_percentiles:
                values[code][percentile] = np.nan
    return values


def build_monthly_ecozone_dataframe(
    indices: list[str] | None = None,
    aois: list[str] | None = None,
    summary_percentiles: list[int] | None = None,
) -> pd.DataFrame:
    use_indices = indices or INDICES
    use_aois = aois or AOIS
    use_percentiles = summary_percentiles or SUMMARY_PERCENTILES
    wet_dry = load_wet_dry_years()
    records: list[dict] = []

    print("Loading monthly Sentinel composites for investigation layer...")

    for aoi in use_aois:
        print(f"  AOI: {aoi}")
        eco_masks = load_ecozone_masks(aoi)
        aoi_years = sorted(wet_dry.loc[wet_dry["aoi"] == aoi, "year"].unique())

        for index_name in use_indices:
            print(f"    Index: {index_name}")
            loaded_months = 0
            for year in aoi_years:
                year_loaded = 0
                classification = wet_dry.loc[
                    (wet_dry["aoi"] == aoi) & (wet_dry["year"] == year),
                    "classification",
                ].iloc[0]
                for month in range(1, 13):
                    arr = load_monthly_array(index_name, aoi, int(year), month)
                    if arr is None:
                        continue
                    percentile_values = ecozone_percentiles(arr, eco_masks, use_percentiles)
                    loaded_months += 1
                    year_loaded += 1
                    for code in VALID_ECOZONE_CODES:
                        for percentile in use_percentiles:
                            value = percentile_values[code][percentile]
                            records.append({
                                "aoi": aoi,
                                "aoi_label": AOI_DISPLAY[aoi],
                                "index": index_name,
                                "year": int(year),
                                "month": month,
                                "month_name": MONTH_NAMES[month],
                                "classification": classification,
                                "ecozone_code": code,
                                "ecozone_label": ECOZONE_LABELS[code],
                                "summary_percentile": percentile,
                                "value": value,
                                "has_value": bool(np.isfinite(value)),
                            })
                print(
                    f"      {year} ({classification}): {year_loaded:>2} monthly rasters loaded"
                )
            print(f"    Completed {aoi} / {index_name}: {loaded_months} monthly rasters")

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No monthly Sentinel ecozone records were loaded.")
    print(f"Loaded {len(df):,} ecozone-month records total.")
    return df.sort_values(
        ["summary_percentile", "aoi", "index", "year", "month", "ecozone_code"]
    ).reset_index(drop=True)


def baseline_monthly_stats(df: pd.DataFrame, baseline_class: str = "neutral") -> pd.DataFrame:
    baseline = df[df["classification"] == baseline_class].copy()
    grouped = (
        baseline.groupby(
            ["summary_percentile", "aoi", "index", "ecozone_code", "month"], dropna=False
        )["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "baseline_mean",
            "std": "baseline_std",
            "count": "baseline_count",
        })
    )
    grouped["baseline_std"] = grouped["baseline_std"].fillna(0.0)
    return grouped


def classify_recovery(late_anomaly: float, spring_anomaly: float) -> str:
    if np.isnan(late_anomaly) or np.isnan(spring_anomaly):
        return "insufficient_data"
    if abs(spring_anomaly) <= RECOVERY_NEAR_BASELINE_TOLERANCE:
        return "quick"
    if late_anomaly < 0 and spring_anomaly > late_anomaly:
        improvement = spring_anomaly - late_anomaly
        needed = abs(late_anomaly) * RECOVERY_IMPROVEMENT_FRACTION
        if improvement >= needed:
            return "partial"
    return "slow"
