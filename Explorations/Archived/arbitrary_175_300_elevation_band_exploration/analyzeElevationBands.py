# terrain_stratification_analysis.py

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

# ============= CONFIGURATION =============
INDEX = "NDVI"  # Options: "NDVI", "NDMI", "EVI"
PERCENTILES = [50, 75, 90]  # List of percentiles to track (e.g., [25, 50, 75, 99])
FORCE_RERUN = False  # True: reprocess all scenes even if cache exists

# Paths (auto-configured based on INDEX)
CACHE_DIR = f"/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/AOI/GWNF_cache/{INDEX}"
TERRAIN_DIR = "/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/AOI/GWNF_cache/terrain"
MANIFEST_PATH = f"/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/AOI/GWNF_cache/cache_manifest{'_' + INDEX.lower() if INDEX != 'NDVI' else ''}.json"
OUTPUT_DIR = "/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/Python"

# Index-specific settings
INDEX_RANGES = {
    "NDVI": (0, 1),
    "NDMI": (-1, 1),
    "EVI": (-1, 1)
}
INDEX_LABELS = {
    "NDVI": "Greenness",
    "NDMI": "Moisture Content",
    "EVI": "Enhanced Vegetation"
}
# ========================================

# Derived paths for this run's results cache
run_label = f"{INDEX}_p{'_'.join(map(str, PERCENTILES))}"
results_cache_path = Path(OUTPUT_DIR) / f"results_cache_{run_label}.npz"
results_log_path   = Path(OUTPUT_DIR) / f"results_cache_{run_label}.json"

print(f"Analyzing {INDEX} with percentiles: {PERCENTILES}")
print(f"Cache: {CACHE_DIR}")
print(f"Manifest: {MANIFEST_PATH}")

# ---- Try loading from results cache ----
loaded_from_cache = False
if not FORCE_RERUN and results_cache_path.exists() and results_log_path.exists():
    print(f"\nResults cache found: {results_cache_path.name}")
    with open(results_log_path) as f:
        log = json.load(f)
    print(f"  Run timestamp : {log['run_timestamp']}")
    print(f"  INDEX         : {log['index']}")
    print(f"  PERCENTILES   : {log['percentiles']}")
    print(f"  Scenes        : {log['num_scenes']}")
    print(f"  Date range    : {log['date_range'][0]} → {log['date_range'][1]}")
    print("Loading cached results (set FORCE_RERUN=True to reprocess)...")

    npz = np.load(results_cache_path, allow_pickle=True)
    dates      = [datetime.fromisoformat(d) for d in npz['dates'].tolist()]
    whole_aoi  = {p: npz[f'whole_aoi_p{p}'].tolist() for p in PERCENTILES}
    low        = {p: npz[f'low_p{p}'].tolist()       for p in PERCENTILES}
    mid        = {p: npz[f'mid_p{p}'].tolist()        for p in PERCENTILES}
    high       = {p: npz[f'high_p{p}'].tolist()       for p in PERCENTILES}
    loaded_from_cache = True
else:
    if FORCE_RERUN:
        print("\nFORCE_RERUN=True — reprocessing all scenes.")
    else:
        print("\nNo results cache found — processing scenes.")

    # ---- Load terrain masks ----
    print("\nLoading elevation band masks...")
    with rasterio.open(f"{TERRAIN_DIR}/mask_elev_low.tif") as src:
        low_mask = src.read(1).astype(bool)
    with rasterio.open(f"{TERRAIN_DIR}/mask_elev_mid.tif") as src:
        mid_mask = src.read(1).astype(bool)
    with rasterio.open(f"{TERRAIN_DIR}/mask_elev_high.tif") as src:
        high_mask = src.read(1).astype(bool)

    print(f"Low-elevation pixels:  {low_mask.sum():,}")
    print(f"Mid-elevation pixels:  {mid_mask.sum():,}")
    print(f"High-elevation pixels: {high_mask.sum():,}")

    # ---- Load manifest ----
    print("\nLoading scene manifest...")
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)

    scenes = []
    for scene_id, metadata in manifest.items():
        date = datetime.fromisoformat(metadata['date'])
        filename = metadata['filename']
        filepath = Path(CACHE_DIR) / filename
        if filepath.exists():
            scenes.append({'date': date, 'filepath': filepath})

    scenes = sorted(scenes, key=lambda x: x['date'])
    print(f"Found {len(scenes)} {INDEX} scenes")

    # ---- Process scenes ----
    dates     = []
    whole_aoi = {p: [] for p in PERCENTILES}
    low       = {p: [] for p in PERCENTILES}
    mid       = {p: [] for p in PERCENTILES}
    high      = {p: [] for p in PERCENTILES}

    print(f"\nComputing stratified percentiles...")
    for i, scene in enumerate(scenes):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(scenes)}...")

        with rasterio.open(scene['filepath']) as src:
            data = src.read(1)

        valid = ~np.isnan(data)

        for percentile in PERCENTILES:
            whole_aoi[percentile].append(
                np.percentile(data[valid], percentile) if valid.sum() > 0 else np.nan
            )
            for band_dict, band_mask in [(low, low_mask), (mid, mid_mask), (high, high_mask)]:
                band_valid = valid & band_mask
                band_dict[percentile].append(
                    np.percentile(data[band_valid], percentile) if band_valid.sum() > 100 else np.nan
                )

        dates.append(scene['date'])

    print(f"Computed percentiles for {len(dates)} scenes")

    # ---- Save results cache ----
    save_arrays = {'dates': np.array([d.isoformat() for d in dates])}
    for p in PERCENTILES:
        save_arrays[f'whole_aoi_p{p}'] = np.array(whole_aoi[p])
        save_arrays[f'low_p{p}']       = np.array(low[p])
        save_arrays[f'mid_p{p}']       = np.array(mid[p])
        save_arrays[f'high_p{p}']      = np.array(high[p])
    np.savez(results_cache_path, **save_arrays)

    log = {
        'run_timestamp' : datetime.now().isoformat(),
        'index'         : INDEX,
        'percentiles'   : PERCENTILES,
        'num_scenes'    : len(dates),
        'date_range'    : [dates[0].isoformat(), dates[-1].isoformat()],
        'results_file'  : results_cache_path.name,
        'force_rerun'   : FORCE_RERUN
    }
    with open(results_log_path, 'w') as f:
        json.dump(log, f, indent=2)

    print(f"Results cached to {results_cache_path.name}")
    print(f"Run log saved to  {results_log_path.name}")

# ---- Plot ----
num_percentiles = len(PERCENTILES)
fig, axes = plt.subplots(num_percentiles, 2, figsize=(16, 4 * num_percentiles))

if num_percentiles == 1:
    axes = axes.reshape(1, -1)

y_min, y_max = INDEX_RANGES[INDEX]

for idx, percentile in enumerate(PERCENTILES):
    ax_ts   = axes[idx, 0]
    ax_diff = axes[idx, 1]

    ax_ts.plot(dates, whole_aoi[percentile], 'o-', color='gray',   markersize=2, alpha=0.6, label='Whole AOI')
    ax_ts.plot(dates, low[percentile],       'o-', color='green',  markersize=2, alpha=0.6, label='Low elevation')
    ax_ts.plot(dates, mid[percentile],       'o-', color='orange', markersize=2, alpha=0.6, label='Mid elevation')
    ax_ts.plot(dates, high[percentile],      'o-', color='blue',   markersize=2, alpha=0.6, label='High elevation')

    ax_ts.set_ylabel(f'{INDEX} (p{percentile})')
    ax_ts.set_title(f'{INDEX} Time Series - p{percentile} ({INDEX_LABELS[INDEX]})')
    ax_ts.legend()
    ax_ts.grid(True, alpha=0.3)
    ax_ts.set_ylim(y_min, y_max)

    if idx == num_percentiles - 1:
        ax_ts.set_xlabel('Date')

    diff = np.array(high[percentile]) - np.array(low[percentile])
    ax_diff.plot(dates, diff, 'o-', color='purple', markersize=2, alpha=0.6)
    ax_diff.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax_diff.set_ylabel(f'{INDEX} Difference (High - Low)')
    ax_diff.set_title(f'High vs Low Elevation - p{percentile} Difference')
    ax_diff.grid(True, alpha=0.3)

    if idx == num_percentiles - 1:
        ax_diff.set_xlabel('Date')

    print(f"\n=== p{percentile} Statistics ===")
    print(f"Whole AOI: mean={np.nanmean(whole_aoi[percentile]):.3f}, std={np.nanstd(whole_aoi[percentile]):.3f}")
    print(f"Low:       mean={np.nanmean(low[percentile]):.3f}, std={np.nanstd(low[percentile]):.3f}")
    print(f"Mid:       mean={np.nanmean(mid[percentile]):.3f}, std={np.nanstd(mid[percentile]):.3f}")
    print(f"High:      mean={np.nanmean(high[percentile]):.3f}, std={np.nanstd(high[percentile]):.3f}")
    print(f"Difference (High-Low): mean={np.nanmean(diff):.3f}, std={np.nanstd(diff):.3f}")

plt.tight_layout()
manager = plt.get_current_fig_manager()
manager.window.attributes('-fullscreen', True)
plt.show()