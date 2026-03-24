#!/usr/bin/env python3
"""
write_peak_summary_excel.py
---------------------------
Standalone Excel writer for Analysis 1+2 summary data.

Re-runs only the data computation (no figures). Use this if the main
ecozone_peak_productivity.py script completed its figures but failed
on the Excel write (e.g. missing openpyxl).

Run from project root:
  python Analysis/Traits/Ecozone/write_peak_summary_excel.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from src.aoi import get_aoi_config
from src.paths import project_path

# ── Configuration (must match ecozone_peak_productivity.py) ───────────────────

AOIS                = ["north", "south"]
VALID_ECOZONE_CODES = [1, 2, 3]
ECOZONE_LABELS      = {1: "Cool", 2: "Intermediate", 3: "Hot"}
AOI_DISPLAY         = {"north": "GW National Forest", "south": "Great Smoky Mtns"}
INDICES             = ["NDVI", "NDMI"]
PERCENTILES         = [95, 99, 100]
PCT_LABELS          = {95: "p95", 99: "p99", 100: "p100 (max)"}
MIN_PIXELS          = 100

TABLES_DIR = project_path("results_tables_dir")
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_scenes(index_dir: Path, manifest_path: Path) -> list[dict]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    scenes = []
    for _, meta in manifest.items():
        fp = index_dir / meta["filename"]
        if fp.exists():
            scenes.append({"date": datetime.fromisoformat(meta["date"]), "filepath": fp})
    return sorted(scenes, key=lambda s: s["date"])


def scene_percentiles_by_ecozone(scenes, eco_masks, percentiles):
    results = {code: {pct: [] for pct in percentiles} for code in VALID_ECOZONE_CODES}
    for i, scene in enumerate(scenes):
        if i % 50 == 0:
            print(f"      {i:>4}/{len(scenes)} scenes...", flush=True)
        with rasterio.open(scene["filepath"]) as src:
            data = src.read(1)
        valid = ~np.isnan(data)
        for code in VALID_ECOZONE_CODES:
            combined = eco_masks[code] & valid
            if combined.sum() >= MIN_PIXELS:
                px = data[combined]
                for pct in percentiles:
                    results[code][pct].append(float(np.percentile(px, pct)))
    return results


# ── Compute ────────────────────────────────────────────────────────────────────

summary: dict = {}
eco_pixel_counts: dict = {}

for aoi in AOIS:
    cfg = get_aoi_config(aoi)
    print(f"\n{'='*62}")
    print(f"  {aoi.upper()} AOI — {AOI_DISPLAY[aoi]}")
    print(f"{'='*62}")

    ecozone_path = cfg.ecozone_dir / "tnc_ecozone_simplified_snapped.tif"
    with rasterio.open(ecozone_path) as src:
        ecozone = src.read(1)

    eco_masks = {code: (ecozone == code) for code in VALID_ECOZONE_CODES}
    eco_pixel_counts[aoi] = {code: int(eco_masks[code].sum()) for code in VALID_ECOZONE_CODES}

    summary[aoi] = {}

    for index_name in INDICES:
        index_dir     = cfg.index_cache_root / index_name
        manifest_path = index_dir / "cache_manifest.json"

        print(f"\n  [{index_name}] Loading scenes...")
        scenes = load_scenes(index_dir, manifest_path)
        print(f"    {len(scenes)} scenes  — computing p95/p99/p100 per ecozone...")

        per_scene = scene_percentiles_by_ecozone(scenes, eco_masks, PERCENTILES)

        summary[aoi][index_name] = {}
        for code in VALID_ECOZONE_CODES:
            summary[aoi][index_name][code] = {}
            for pct in PERCENTILES:
                vals = per_scene[code][pct]
                summary[aoi][index_name][code][pct] = (
                    float(np.nanmean(vals)) if vals else np.nan
                )
            d = summary[aoi][index_name][code]
            print(
                f"    {ECOZONE_LABELS[code]:>12}:  "
                f"p95={d[95]:.4f}  p99={d[99]:.4f}  p100={d[100]:.4f}"
            )


# ── Write Excel ────────────────────────────────────────────────────────────────

rows = []
for aoi in AOIS:
    for code in VALID_ECOZONE_CODES:
        row = {
            "AOI":          AOI_DISPLAY[aoi],
            "Ecozone":      ECOZONE_LABELS[code],
            "Ecozone Code": code,
            "Pixel Count":  eco_pixel_counts[aoi][code],
        }
        for index_name in INDICES:
            for pct in PERCENTILES:
                row[f"{index_name} {PCT_LABELS[pct]}"] = round(
                    summary[aoi][index_name][code][pct], 6
                )
        rows.append(row)

df = pd.DataFrame(rows)
table_path = TABLES_DIR / "ecozone_peak_summary.xlsx"
df.to_excel(table_path, index=False, sheet_name="Ecozone Summary")
print(f"\nSaved: {table_path}")
print("\nDone.")
