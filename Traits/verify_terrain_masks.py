import rasterio
import numpy as np
from pathlib import Path

CACHE_BASE  = Path("/mnt/c/Users/rowan/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/AOI/GWNF_cache")
TRAITS_DIR  = CACHE_BASE / "traits"
TERRAIN_DIR = TRAITS_DIR / "terrain"
FOREST_DIR  = TRAITS_DIR / "forest" / "forest_type_group"
ECOZONE_DIR = TRAITS_DIR / "ecozone"

CANONICAL_SHAPE  = (9812, 6451)
CANONICAL_BOUNDS = (619300.153, 4203684.597, 683810.825, 4301812.355)

FOREST_TYPE_LABELS = {
    0: "Non-forest", 100: "White/red/jack pine", 120: "Spruce/fir",
    160: "Loblolly/shortleaf pine", 400: "Oak/pine", 500: "Oak/hickory",
    700: "Elm/ash/cottonwood", 800: "Maple/beech/birch",
}
ECOZONE_LABELS = {0: "Cool", 1: "Intermediate", 2: "Hot"}

def check_alignment(src):
    issues = []
    crs_epsg = src.crs.to_epsg() if src.crs else None
    res = (abs(src.res[0]), abs(src.res[1]))
    b = src.bounds
    if crs_epsg != 32617:
        issues.append(f"  ⚠  CRS: {src.crs}  (expected EPSG:32617)")
    else:
        print(f"  ✓  CRS: EPSG:32617")
    if abs(res[0] - 10.0) > 0.01 or abs(res[1] - 10.0) > 0.01:
        issues.append(f"  ⚠  Resolution: {res}  (expected 10m x 10m)")
    else:
        print(f"  ✓  Resolution: 10m x 10m")
    shape_ok = abs(src.height - CANONICAL_SHAPE[0]) <= 2 and abs(src.width - CANONICAL_SHAPE[1]) <= 2
    if not shape_ok:
        issues.append(f"  ⚠  Shape: {src.height} x {src.width}  (expected ~{CANONICAL_SHAPE[0]} x {CANONICAL_SHAPE[1]})")
    else:
        print(f"  ✓  Shape: {src.height} rows x {src.width} cols")
    bounds_ok = all(abs(getattr(b, k) - v) < 50
                    for k, v in zip(['left','bottom','right','top'], CANONICAL_BOUNDS))
    if not bounds_ok:
        issues.append(f"  ⚠  Bounds: {b.left:.0f} {b.bottom:.0f} {b.right:.0f} {b.top:.0f}  (expected {[int(x) for x in CANONICAL_BOUNDS]})")
    else:
        print(f"  ✓  Bounds: aligned")
    return issues

def print_issues(issues):
    if issues:
        for i in issues: print(i)
    else:
        print("  ✓  No alignment issues")

# ── Elevation ─────────────────────────────────────────────────
print(f"\n{'='*60}\nELEVATION\n{'='*60}")
with rasterio.open(TERRAIN_DIR / "elevation.tif") as src:
    issues = check_alignment(src)
    data = src.read(1, masked=True)
    valid = data.compressed()
    print(f"  ✓  Valid pixels: {len(valid):,}  ({len(valid)/data.size*100:.1f}%)")
    print(f"  ✓  Range: {valid.min():.0f}m – {valid.max():.0f}m")
    print(f"  ✓  P25={np.percentile(valid,25):.0f}m  P50={np.percentile(valid,50):.0f}m  P75={np.percentile(valid,75):.0f}m  Mean={valid.mean():.0f}m")
    if valid.min() < 0:
        issues.append(f"  ⚠  Negative elevation values (min={valid.min():.0f}m)")
    print_issues(issues)

# ── Terrain masks ─────────────────────────────────────────────
print(f"\n{'='*60}\nTERRAIN MASKS\n{'='*60}")
for fname, label in [
    ("mask_south_facing.tif",  "South-facing slopes   "),
    ("mask_north_facing.tif",  "North-facing slopes   "),
    ("mask_steep_slopes.tif",  "Steep slopes          "),
    ("mask_elev_low.tif",      "Elevation low  <175m  "),
    ("mask_elev_mid.tif",      "Elevation mid  175-300m"),
    ("mask_elev_high.tif",     "Elevation high >300m  "),
]:
    fpath = TERRAIN_DIR / fname
    if not fpath.exists():
        print(f"  ✗  MISSING: {fname}")
        continue
    with rasterio.open(fpath) as src:
        data = src.read(1)
        count = (data == 1).sum()
        print(f"  ✓  {label}  {count:>10,} px  ({count*100/1e6:6.1f} km²)")

# ── FIA Forest type group raster ──────────────────────────────
print(f"\n{'='*60}\nFIA FOREST TYPE GROUP RASTER\n{'='*60}")
with rasterio.open(FOREST_DIR / "forest_type_group.tif") as src:
    issues = check_alignment(src)
    data = src.read(1, masked=True)
    codes, counts = np.unique(data.compressed(), return_counts=True)
    total = counts.sum()
    for code, count in zip(codes, counts):
        label = FOREST_TYPE_LABELS.get(int(code), "Unknown")
        area  = count * 100 / 1e6
        pct   = count / total * 100
        mk    = "✓" if int(code) in FOREST_TYPE_LABELS else "⚠"
        print(f"  {mk}  Code {int(code):4d}  {area:7.1f} km²  ({pct:5.1f}%)  {label}")
    print_issues(issues)

# ── Forest type masks ─────────────────────────────────────────
print(f"\n{'='*60}\nFOREST TYPE MASKS  (traits/forest/forest_type_group/)\n{'='*60}")
for fname in ["mask_forest_400.tif", "mask_forest_500.tif", "mask_forest_800.tif"]:
    code  = int(fname.replace("mask_forest_","").replace(".tif",""))
    label = FOREST_TYPE_LABELS.get(code, "Unknown")
    fpath = FOREST_DIR / fname
    if not fpath.exists():
        print(f"  ✗  MISSING: {fname}")
        continue
    with rasterio.open(fpath) as src:
        data  = src.read(1)
        count = (data == 1).sum()
        print(f"  ✓  {fname}  ({label})  {count:,} px  ({count*100/1e6:.1f} km²)")

# ── Flag duplicate masks in terrain/ ─────────────────────────
dupes = sorted(TERRAIN_DIR.glob("mask_forest_*.tif"))
if dupes:
    print(f"\n  Note: forest type masks also found in traits/terrain/ —")
    print(f"  confirm these are identical to the ones in traits/forest/")
    for f in dupes:
        print(f"    {f.name}")

# ── TNC Ecozone ───────────────────────────────────────────────
print(f"\n{'='*60}\nTNC ECOZONE RASTER\n{'='*60}")
for label, path in [
    ("Snapped (use this)", ECOZONE_DIR / "tnc_ecozone_simplified_snapped.tif"),
    ("Raw pre-snap",       ECOZONE_DIR / "tnc_ecozone_simplified" / "fortype_aoiN.tif"),
]:
    print(f"\n  [{label}]")
    if not path.exists():
        print(f"  ✗  NOT FOUND: {path}")
        continue
    with rasterio.open(path) as src:
        issues = check_alignment(src)
        data   = src.read(1, masked=True)
        codes, counts = np.unique(data.compressed(), return_counts=True)
        total  = counts.sum()
        for code, count in zip(codes, counts):
            lbl  = ECOZONE_LABELS.get(int(code), f"Unknown ({int(code)})")
            area = count * 100 / 1e6
            pct  = count / total * 100
            print(f"  ✓  Code {int(code)}  {area:7.1f} km²  ({pct:5.1f}%)  {lbl}")
        print_issues(issues)

# ── Ecozone masks ─────────────────────────────────────────────
print(f"\n{'='*60}\nECOZONE MASKS\n{'='*60}")
for fname in ["mask_ecozone_0.tif", "mask_ecozone_1.tif", "mask_ecozone_2.tif"]:
    code  = int(fname.replace("mask_ecozone_","").replace(".tif",""))
    label = ECOZONE_LABELS.get(code, "Unknown")
    fpath = ECOZONE_DIR / fname
    if not fpath.exists():
        print(f"  ✗  MISSING: {fname}")
        continue
    with rasterio.open(fpath) as src:
        data  = src.read(1)
        count = (data == 1).sum()
        print(f"  ✓  {fname}  ({label:<15})  {count:,} px  ({count*100/1e6:.1f} km²)")

print(f"\n{'='*60}\nVERIFICATION COMPLETE\n{'='*60}")