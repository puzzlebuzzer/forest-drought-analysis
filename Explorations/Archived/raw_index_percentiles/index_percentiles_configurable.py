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

print(f"Analyzing {INDEX} with percentiles: {PERCENTILES}")
print(f"Cache: {CACHE_DIR}")
print(f"Manifest: {MANIFEST_PATH}")

print("\nLoading terrain masks...")
with rasterio.open(f"{TERRAIN_DIR}/mask_south_facing.tif") as src:
    south_mask = src.read(1).astype(bool)

with rasterio.open(f"{TERRAIN_DIR}/mask_north_facing.tif") as src:
    north_mask = src.read(1).astype(bool)

print(f"South-facing pixels: {south_mask.sum():,}")
print(f"North-facing pixels: {north_mask.sum():,}")

# Load manifest
print("\nLoading scene manifest...")
with open(MANIFEST_PATH, 'r') as f:
    manifest = json.load(f)

# Sort scenes by date
scenes = []
for scene_id, metadata in manifest.items():
    date = datetime.fromisoformat(metadata['date'])
    filename = metadata['filename']
    filepath = Path(CACHE_DIR) / filename
    if filepath.exists():
        scenes.append({
            'date': date,
            'filepath': filepath
        })

scenes = sorted(scenes, key=lambda x: x['date'])
print(f"Found {len(scenes)} {INDEX} scenes")

# Initialize storage for each percentile
dates = []
whole_aoi = {p: [] for p in PERCENTILES}
south = {p: [] for p in PERCENTILES}
north = {p: [] for p in PERCENTILES}

print(f"\nComputing stratified percentiles...")
for i, scene in enumerate(scenes):
    if i % 50 == 0:
        print(f"  Processing {i}/{len(scenes)}...")
    
    with rasterio.open(scene['filepath']) as src:
        data = src.read(1)
    
    # Get valid pixels (non-NaN)
    valid = ~np.isnan(data)
    
    # Compute percentiles for each stratum
    for percentile in PERCENTILES:
        # Whole AOI
        if valid.sum() > 0:
            whole_aoi[percentile].append(np.percentile(data[valid], percentile))
        else:
            whole_aoi[percentile].append(np.nan)
        
        # South-facing slopes
        south_valid = valid & south_mask
        if south_valid.sum() > 100:
            south[percentile].append(np.percentile(data[south_valid], percentile))
        else:
            south[percentile].append(np.nan)
        
        # North-facing slopes
        north_valid = valid & north_mask
        if north_valid.sum() > 100:
            north[percentile].append(np.percentile(data[north_valid], percentile))
        else:
            north[percentile].append(np.nan)
    
    dates.append(scene['date'])

print(f"Computed percentiles for {len(dates)} scenes")

# Create plots - one subplot per percentile
num_percentiles = len(PERCENTILES)
fig, axes = plt.subplots(num_percentiles, 2, figsize=(16, 5 * num_percentiles))

# Handle case of single percentile
if num_percentiles == 1:
    axes = axes.reshape(1, -1)

y_min, y_max = INDEX_RANGES[INDEX]

for idx, percentile in enumerate(PERCENTILES):
    ax_ts = axes[idx, 0]  # Time series
    ax_diff = axes[idx, 1]  # Difference
    
    # Plot time series
    ax_ts.plot(dates, whole_aoi[percentile], 'o-', color='gray', markersize=2, alpha=0.6, label='Whole AOI')
    ax_ts.plot(dates, south[percentile], 'o-', color='red', markersize=2, alpha=0.6, label='South-facing')
    ax_ts.plot(dates, north[percentile], 'o-', color='blue', markersize=2, alpha=0.6, label='North-facing')
    
    ax_ts.set_ylabel(f'{INDEX} (p{percentile})')
    ax_ts.set_title(f'{INDEX} Time Series - p{percentile} ({INDEX_LABELS[INDEX]})')
    ax_ts.legend()
    ax_ts.grid(True, alpha=0.3)
    ax_ts.set_ylim(y_min, y_max)
    
    if idx == num_percentiles - 1:
        ax_ts.set_xlabel('Date')
    
    # Plot difference
    diff = np.array(south[percentile]) - np.array(north[percentile])
    ax_diff.plot(dates, diff, 'o-', color='purple', markersize=2, alpha=0.6)
    ax_diff.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax_diff.set_ylabel(f'{INDEX} Difference (S - N)')
    ax_diff.set_title(f'South vs North - p{percentile} Difference')
    ax_diff.grid(True, alpha=0.3)
    
    if idx == num_percentiles - 1:
        ax_diff.set_xlabel('Date')
    
    # Print stats for this percentile
    print(f"\n=== p{percentile} Statistics ===")
    print(f"Whole AOI: mean={np.nanmean(whole_aoi[percentile]):.3f}, std={np.nanstd(whole_aoi[percentile]):.3f}")
    print(f"South: mean={np.nanmean(south[percentile]):.3f}, std={np.nanstd(south[percentile]):.3f}")
    print(f"North: mean={np.nanmean(north[percentile]):.3f}, std={np.nanstd(north[percentile]):.3f}")
    print(f"Difference (S-N): mean={np.nanmean(diff):.3f}, std={np.nanstd(diff):.3f}")

plt.tight_layout()
output_path = f"{OUTPUT_DIR}/terrain_stratification_{INDEX}_p{'_'.join(map(str, PERCENTILES))}.png"
plt.savefig(output_path, dpi=150)
print(f"\nPlot saved to {output_path}")
plt.show()