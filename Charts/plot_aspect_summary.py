#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from pathlib import Path
import rasterio

from src.aoi import get_aoi_config

INDICES = ["NDVI", "NDMI", "EVI"]
AOIS = ["north", "south"]

def compute_diff_series(aoi, index):

    cfg = get_aoi_config(aoi)

    TERRAIN_DIR = cfg.terrain_dir
    INDEX_DIR = cfg.index_cache_root / index
    MANIFEST_PATH = INDEX_DIR / "cache_manifest.json"

    with rasterio.open(TERRAIN_DIR / "mask_south_facing.tif") as src:
        south_mask = src.read(1) == 1

    with rasterio.open(TERRAIN_DIR / "mask_north_facing.tif") as src:
        north_mask = src.read(1) == 1

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    scenes = []
    for _, meta in manifest.items():
        fp = INDEX_DIR / meta["filename"]
        if fp.exists():
            scenes.append({
                "date": datetime.fromisoformat(meta["date"]),
                "filepath": fp
            })

    scenes = sorted(scenes, key=lambda x: x["date"])

    dates = []
    diff = []

    for scene in scenes:

        with rasterio.open(scene["filepath"]) as src:
            data = src.read(1)

        valid = ~np.isnan(data)

        sv = valid & south_mask
        nv = valid & north_mask

        if sv.sum() < 100 or nv.sum() < 100:
            diff.append(np.nan)
        else:
            s = np.percentile(data[sv], 75)
            n = np.percentile(data[nv], 75)
            diff.append(s - n)

        dates.append(scene["date"])

    return np.array(dates), np.array(diff)


fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

for r, index in enumerate(INDICES):
    for c, aoi in enumerate(AOIS):

        dates, diff = compute_diff_series(aoi, index)

        ax = axes[r, c]
        ax.plot(dates, diff)
        ax.axhline(0, linestyle="--")

        ax.set_title(f"{index} ({aoi})")
        ax.set_ylabel("South − North")

plt.tight_layout()

output = Path("Results/figures/aspect_index_summary.png")
output.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output, dpi=150)

print(f"Saved {output}")
