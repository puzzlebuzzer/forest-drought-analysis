#!/usr/bin/env python3
"""
Create a small, presentation-oriented set of anomaly-from-normal demo rasters.

This script reads existing Sentinel monthly composites and existing wet/dry/
neutral year labels, then writes a compact set of baseline rasters, anomaly
rasters, preview PNGs, and a manifest CSV for a fixed demo selection.

Anomaly definition:
    anomaly = target monthly composite - monthly neutral-year baseline

Baseline definition:
    per-pixel median across available neutral-year monthly composites for the
    same AOI / index / month

This script is intentionally narrow and presentation-oriented. It does not
attempt to build a full archive of anomalies.
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "Results" / "Other" / "anomaly_from_normal_demo"

ECOZONE_HELPER_DIR = PROJECT_ROOT / "Analysis" / "Traits" / "Ecozone"
if str(ECOZONE_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(ECOZONE_HELPER_DIR))

from investigation_common import AOI_DISPLAY, MONTH_NAMES, load_wet_dry_years, monthly_file  # noqa: E402


INDICES = ["NDVI", "NDMI"]
BASELINE_CACHE: dict[tuple[str, str, int], tuple[np.ndarray, dict]] = {}
DEMO_SPECS = [
    {
        "slug": "north_ndvi_dry_2023_10",
        "kind": "single",
        "aoi": "north",
        "index": "NDVI",
        "year_type": "dry",
        "year": 2023,
        "month": 10,
        "months": [10],
        "note": "Chosen as the strongest classic north dry late-season NDVI example.",
    },
    {
        "slug": "north_ndmi_dry_2023_10",
        "kind": "single",
        "aoi": "north",
        "index": "NDMI",
        "year_type": "dry",
        "year": 2023,
        "month": 10,
        "months": [10],
        "note": "Chosen as the strongest north dry late-season NDMI example.",
    },
    {
        "slug": "south_ndvi_wet_2020_07",
        "kind": "single",
        "aoi": "south",
        "index": "NDVI",
        "year_type": "wet",
        "year": 2020,
        "month": 7,
        "months": [7],
        "note": "Chosen as a strong south wet NDVI example with widespread positive anomaly.",
    },
    {
        "slug": "south_ndmi_wet_2020_07",
        "kind": "single",
        "aoi": "south",
        "index": "NDMI",
        "year_type": "wet",
        "year": 2020,
        "month": 7,
        "months": [7],
        "note": "Chosen as a strong south wet NDMI example with clear positive response.",
    },
    {
        "slug": "north_ndvi_dry_apr_oct_2023_mean",
        "kind": "composite",
        "aoi": "north",
        "index": "NDVI",
        "year_type": "dry",
        "years": [2023],
        "months": [4, 5, 6, 7, 8, 9, 10],
        "note": "Chosen as a compact north dry-season NDVI mean anomaly composite.",
    },
    {
        "slug": "south_ndmi_wet_apr_oct_2018_2021_mean",
        "kind": "composite",
        "aoi": "south",
        "index": "NDMI",
        "year_type": "wet",
        "years": [2018, 2019, 2020, 2021],
        "months": [4, 5, 6, 7, 8, 9, 10],
        "note": "Chosen as a compact south wet-season NDMI mean anomaly composite.",
    },
]


def ensure_out_dir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def build_monthly_neutral_baseline(aoi: str, index_name: str, month: int, wet_dry: pd.DataFrame) -> tuple[np.ndarray, dict]:
    cache_key = (aoi, index_name, month)
    if cache_key in BASELINE_CACHE:
        return BASELINE_CACHE[cache_key]

    neutral_years = sorted(
        wet_dry[
            (wet_dry["aoi"] == aoi)
            & (wet_dry["classification"] == "neutral")
        ]["year"].astype(int).tolist()
    )
    arrays: list[np.ndarray] = []
    source_paths: list[str] = []
    profile = None

    for year in neutral_years:
        path = monthly_file(index_name, aoi, year, month)
        if not path.exists():
            continue
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            arrays.append(arr)
            source_paths.append(str(path))
            if profile is None:
                profile = src.profile.copy()

    if not arrays or profile is None:
        raise FileNotFoundError(
            f"No neutral-year monthly composites found for {aoi} / {index_name} / month {month}"
        )

    stack = np.stack(arrays, axis=0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        baseline = np.nanmedian(stack, axis=0).astype(np.float32)

    meta = {
        "neutral_years": neutral_years,
        "neutral_source_paths": source_paths,
        "profile": profile,
    }
    BASELINE_CACHE[cache_key] = (baseline, meta)
    return baseline, meta


def load_target_raster(index_name: str, aoi: str, year: int, month: int) -> tuple[np.ndarray, dict, str]:
    path = monthly_file(index_name, aoi, year, month)
    if not path.exists():
        raise FileNotFoundError(f"Target monthly composite not found: {path}")
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    return arr, profile, str(path)


def compute_anomaly(target: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    finite = np.isfinite(target) & np.isfinite(baseline)
    anomaly = np.full(target.shape, np.nan, dtype=np.float32)
    np.subtract(target, baseline, out=anomaly, where=finite)
    return anomaly


def write_raster(path: Path, arr: np.ndarray, profile: dict) -> None:
    out_profile = profile.copy()
    out_profile.update(
        dtype="float32",
        count=1,
        compress="deflate",
        predictor=2,
        tiled=False,
        nodata=np.nan,
    )
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def preview_limits(arr: np.ndarray, minimum: float = 0.05) -> tuple[float, float]:
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return -1.0, 1.0
    lim = float(np.nanpercentile(np.abs(valid), 98))
    lim = max(lim, minimum)
    return -lim, lim


def write_preview(path: Path, arr: np.ndarray, title: str) -> None:
    vmin, vmax = preview_limits(arr)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(arr, cmap="RdBu", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Anomaly vs neutral baseline")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def selected_demo_specs() -> list[dict]:
    return DEMO_SPECS


def month_label(months: list[int]) -> str:
    if len(months) == 1:
        return MONTH_NAMES[months[0]]
    return f"{MONTH_NAMES[months[0]]}-{MONTH_NAMES[months[-1]]}"


def single_demo(spec: dict, wet_dry: pd.DataFrame) -> dict:
    aoi = spec["aoi"]
    index_name = spec["index"]
    year = spec["year"]
    month = spec["month"]

    print(f"Preparing single demo: {spec['slug']}")
    baseline, baseline_meta = build_monthly_neutral_baseline(aoi, index_name, month, wet_dry)
    target, profile, source_path = load_target_raster(index_name, aoi, year, month)
    anomaly = compute_anomaly(target, baseline)

    baseline_path = OUT_DIR / f"baseline_{spec['slug']}.tif"
    anomaly_path = OUT_DIR / f"anomaly_{spec['slug']}.tif"
    preview_path = OUT_DIR / f"preview_{spec['slug']}.png"

    write_raster(baseline_path, baseline, baseline_meta["profile"])
    write_raster(anomaly_path, anomaly, profile)
    write_preview(
        preview_path,
        anomaly,
        title=(
            f"{index_name} | {AOI_DISPLAY.get(aoi, aoi)}\n"
            f"{spec['year_type']} {year} {MONTH_NAMES[month]} anomaly vs neutral baseline"
        ),
    )

    return {
        "slug": spec["slug"],
        "kind": "single",
        "AOI": aoi,
        "index": index_name,
        "year": year,
        "month": month,
        "month_label": MONTH_NAMES[month],
        "year_type": spec["year_type"],
        "baseline_months_used": str(month),
        "target_years_used": str(year),
        "source_raster_paths": source_path,
        "neutral_source_paths": " | ".join(baseline_meta["neutral_source_paths"]),
        "baseline_raster_path": str(baseline_path),
        "anomaly_raster_path": str(anomaly_path),
        "preview_png_path": str(preview_path),
        "note": spec["note"],
    }


def composite_demo(spec: dict, wet_dry: pd.DataFrame) -> dict:
    aoi = spec["aoi"]
    index_name = spec["index"]
    months = spec["months"]
    years = spec["years"]

    print(f"Preparing composite demo: {spec['slug']}")

    source_paths: list[str] = []
    neutral_paths: list[str] = []
    profile = None
    anomaly_sum = None
    anomaly_count = None
    baseline_sum = None
    baseline_count = None

    for year in years:
        for month in months:
            print(f"  Loading {index_name} {aoi} {year}-{month:02d}")
            baseline, baseline_meta = build_monthly_neutral_baseline(aoi, index_name, month, wet_dry)
            target, profile, source_path = load_target_raster(index_name, aoi, year, month)
            anomaly = compute_anomaly(target, baseline)

            if anomaly_sum is None:
                anomaly_sum = np.zeros(anomaly.shape, dtype=np.float32)
                anomaly_count = np.zeros(anomaly.shape, dtype=np.uint16)
                baseline_sum = np.zeros(baseline.shape, dtype=np.float32)
                baseline_count = np.zeros(baseline.shape, dtype=np.uint16)

            anomaly_valid = np.isfinite(anomaly)
            baseline_valid = np.isfinite(baseline)
            anomaly_sum[anomaly_valid] += anomaly[anomaly_valid]
            anomaly_count[anomaly_valid] += 1
            baseline_sum[baseline_valid] += baseline[baseline_valid]
            baseline_count[baseline_valid] += 1

            source_paths.append(source_path)
            neutral_paths.extend(baseline_meta["neutral_source_paths"])
            del target, baseline, anomaly, anomaly_valid, baseline_valid

    if anomaly_sum is None or anomaly_count is None or baseline_sum is None or baseline_count is None or profile is None:
        raise RuntimeError(f"No inputs available for composite demo {spec['slug']}")

    anomaly_mean = np.full(anomaly_sum.shape, np.nan, dtype=np.float32)
    baseline_mean = np.full(baseline_sum.shape, np.nan, dtype=np.float32)
    np.divide(anomaly_sum, anomaly_count, out=anomaly_mean, where=anomaly_count > 0)
    np.divide(baseline_sum, baseline_count, out=baseline_mean, where=baseline_count > 0)

    baseline_path = OUT_DIR / f"baseline_{spec['slug']}.tif"
    anomaly_path = OUT_DIR / f"anomaly_{spec['slug']}.tif"
    preview_path = OUT_DIR / f"preview_{spec['slug']}.png"

    write_raster(baseline_path, baseline_mean, profile)
    write_raster(anomaly_path, anomaly_mean, profile)
    write_preview(
        preview_path,
        anomaly_mean,
        title=(
            f"{index_name} | {AOI_DISPLAY.get(aoi, aoi)}\n"
            f"{spec['year_type']} composite {month_label(months)} anomaly vs neutral baseline"
        ),
    )

    return {
        "slug": spec["slug"],
        "kind": "composite",
        "AOI": aoi,
        "index": index_name,
        "year": "",
        "month": "",
        "month_label": month_label(months),
        "year_type": spec["year_type"],
        "baseline_months_used": ",".join(str(m) for m in months),
        "target_years_used": ",".join(str(y) for y in years),
        "source_raster_paths": " | ".join(source_paths),
        "neutral_source_paths": " | ".join(sorted(set(neutral_paths))),
        "baseline_raster_path": str(baseline_path),
        "anomaly_raster_path": str(anomaly_path),
        "preview_png_path": str(preview_path),
        "note": spec["note"],
    }


def write_readme() -> None:
    text = """# Anomaly From Normal Demo

This folder is intended for a small presentation-ready set of Sentinel anomaly
rasters generated from existing monthly composites.

The script to generate these outputs is:

`Python/Analysis/Diagnostics/anomaly_from_normal_demo.py`

Run it manually from the project `Python/` directory:

```bash
python ./Analysis/Diagnostics/anomaly_from_normal_demo.py
```

Planned demo set:

- north NDVI dry October 2023
- north NDMI dry October 2023
- south NDVI wet July 2020
- south NDMI wet July 2020
- north NDVI dry Apr-Oct 2023 mean anomaly composite
- south NDMI wet Apr-Oct 2018-2021 mean anomaly composite

Outputs written by the script:

- baseline GeoTIFFs for each selected example
- anomaly GeoTIFFs for each selected example
- quicklook PNG previews
- `demo_manifest.csv`

Baseline rule:

- per-pixel median across available neutral-year monthly Sentinel composites
- computed separately by AOI, index, and month

The intent is demonstration only. This is not a full anomaly production archive.
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_out_dir()
    write_readme()
    wet_dry = load_wet_dry_years()

    manifest_rows: list[dict] = []
    for spec in selected_demo_specs():
        if spec["kind"] == "single":
            manifest_rows.append(single_demo(spec, wet_dry))
        elif spec["kind"] == "composite":
            manifest_rows.append(composite_demo(spec, wet_dry))
        else:
            raise ValueError(f"Unknown demo kind: {spec['kind']}")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT_DIR / "demo_manifest.csv", index=False)
    print(f"Saved anomaly-from-normal demo outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
