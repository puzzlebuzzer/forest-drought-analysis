#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from src.aoi import get_aoi_config
from src.cli import make_parser, add_aoi_arg
from src.labels import FOREST_TYPE_LABELS, label_forest_group


parser = make_parser("Generate charts for aspect analysis")
add_aoi_arg(parser)
args = parser.parse_args()

AOI = args.aoi
cfg = get_aoi_config(AOI)

FOREST_DIR = cfg.forest_type_dir
GROUP_DIR = cfg.forest_group_dir
TERRAIN_DIR = cfg.terrain_dir
CHART_DIR = Path("Charts")
CHART_DIR.mkdir(exist_ok=True)

print("Loading rasters...")

with rasterio.open(TERRAIN_DIR / "mask_south_facing.tif") as src:
    south = src.read(1) == 1

with rasterio.open(TERRAIN_DIR / "mask_north_facing.tif") as src:
    north = src.read(1) == 1

with rasterio.open(GROUP_DIR / "forest_type_group.tif") as src:
    group = src.read(1)

with rasterio.open(FOREST_DIR / "forest_type_type.tif") as src:
    species = src.read(1)


def pct(mask, condition):
    return 100 * np.sum(mask & condition) / np.sum(mask)


# -------------------------------------------------
# Forest group chart
# -------------------------------------------------

codes = [int(c) for c in np.unique(group) if c > 0]

labels = []
south_vals = []
north_vals = []

for code in codes:
    labels.append(label_forest_group(code))
    south_vals.append(pct(south, group == code))
    north_vals.append(pct(north, group == code))

x = np.arange(len(labels))
w = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - w / 2, south_vals, w, label="South-facing")
plt.bar(x + w / 2, north_vals, w, label="North-facing")

plt.xticks(x, labels, rotation=45, ha="right")
plt.ylabel("% of pixels")
plt.title(f"Forest Groups by Aspect ({AOI})")
plt.legend()
plt.tight_layout()

plt.savefig(CHART_DIR / f"{AOI}_forest_groups.png")
plt.close()


# -------------------------------------------------
# Species chart (top 8 forest types, excluding non-forest)
# -------------------------------------------------

codes = [int(c) for c in np.unique(species) if c != 0]

totals = []
for c in codes:
    totals.append((c, np.sum(species == c)))

totals.sort(key=lambda x: -x[1])
top = [c for c, _ in totals[:8]]

labels = []
south_vals = []
north_vals = []

for code in top:
    labels.append(FOREST_TYPE_LABELS.get(code, f"Code {code}"))
    south_vals.append(pct(south, species == code))
    north_vals.append(pct(north, species == code))

x = np.arange(len(labels))

plt.figure(figsize=(10, 5))
plt.bar(x - w / 2, south_vals, w, label="South-facing")
plt.bar(x + w / 2, north_vals, w, label="North-facing")

plt.xticks(x, labels, rotation=45, ha="right")
plt.ylabel("% of pixels")
plt.title(f"Major Forest Types by Aspect ({AOI})")
plt.legend()
plt.tight_layout()

plt.savefig(CHART_DIR / f"{AOI}_species.png")
plt.close()


# -------------------------------------------------
# Oak contrast chart
# -------------------------------------------------
# Chestnut oak = 502
# Northern red oak = 505

chestnut_south = pct(south, species == 502)
chestnut_north = pct(north, species == 502)

red_south = pct(south, species == 505)
red_north = pct(north, species == 505)

labels = ["Chestnut Oak", "Northern Red Oak"]
south_vals = [chestnut_south, red_south]
north_vals = [chestnut_north, red_north]

x = np.arange(len(labels))

plt.figure(figsize=(6, 4))
plt.bar(x - w / 2, south_vals, w, label="South-facing")
plt.bar(x + w / 2, north_vals, w, label="North-facing")

plt.xticks(x, labels)
plt.ylabel("% of pixels")
plt.title(f"Oak Species Contrast ({AOI})")
plt.legend()
plt.tight_layout()

plt.savefig(CHART_DIR / f"{AOI}_oak_contrast.png")
plt.close()

print(f"Charts written to {CHART_DIR}/")