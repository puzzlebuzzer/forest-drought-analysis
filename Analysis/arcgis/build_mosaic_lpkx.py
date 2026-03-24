"""
Analysis/arcgis/build_mosaic_lpkx.py
--------------------------------------
Creates time-enabled mosaic datasets from annual-max and monthly-max GeoTIFFs
(Sentinel-2 and, when available, Landsat) and packages them as ArcGIS layer
packages (.lpkx) for delivery.

Outputs:
    Results/TNC_Appalachian_Indices_Annual.lpkx   — one composite per year
    Results/TNC_Appalachian_Indices_Monthly.lpkx  — one composite per month
                                                    (skipped if not yet built)

Mosaic naming:
    Annual  S2:      S2_NDVI_north,         S2_NDVI_south,      …
    Annual  Landsat: LS_NDVI_north,         LS_NDVI_south,      …
    Monthly S2:      S2_NDVI_north_monthly, S2_NDVI_south_monthly, …
    Monthly Landsat: LS_NDVI_north_monthly, LS_NDVI_south_monthly, …

Run from the ArcGIS Pro Python environment (NOT the base conda env):
    cd <project root>
    python Analysis/arcgis/build_mosaic_lpkx.py
"""

import shutil
import sys
from pathlib import Path

try:
    import arcpy
except ImportError:
    print(
        "ERROR: arcpy not available.\n"
        "Run this script using the ArcGIS Pro Python environment, e.g.:\n"
        '  "C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\envs\\arcgispro-py3\\python.exe" '
        "Analysis/arcgis/build_mosaic_lpkx.py"
    )
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

S2_ANNUAL_MAX_DIR  = PROJECT_ROOT / "Results" / "rasters" / "annual_max"
LS_ANNUAL_MAX_DIR  = PROJECT_ROOT / "Results" / "rasters" / "landsat_annual_max"
S2_MONTHLY_MAX_DIR = PROJECT_ROOT / "Results" / "rasters" / "monthly_max"
LS_MONTHLY_MAX_DIR = PROJECT_ROOT / "Results" / "rasters" / "landsat_monthly_max"
RESULTS_DIR        = PROJECT_ROOT / "Results"
SCRATCH_GDB        = RESULTS_DIR / "mosaic_scratch.gdb"
LYRX_SCRATCH_DIR   = RESULTS_DIR / "_tmp_lyrx"

OUTPUT_ANNUAL  = RESULTS_DIR / "TNC_Appalachian_Indices_Annual.lpkx"
OUTPUT_MONTHLY = RESULTS_DIR / "TNC_Appalachian_Indices_Monthly.lpkx"

# ── Run flags — set False to skip rebuilding that package ──────────────────────
BUILD_ANNUAL  = False
BUILD_MONTHLY = True

# Indices and AOIs to process
S2_INDICES = ["ndvi", "ndmi", "evi"]
LS_INDICES = ["ndvi", "ndmi", "evi"]
AOIS       = ["north", "south"]

# AOI display names for layer labels
AOI_LABELS = {"north": "GWNF North", "south": "Smoky South"}

# Spatial reference: NAD83 / UTM Zone 17N — covers both AOIs
WKID = 32617


# ── Helpers ────────────────────────────────────────────────────────────────────

def ensure_gdb(gdb_path: Path) -> str:
    """Create file GDB if it does not exist; return its string path."""
    p = str(gdb_path)
    if not arcpy.Exists(p):
        arcpy.management.CreateFileGDB(str(gdb_path.parent), gdb_path.name)
        print(f"  Created GDB: {gdb_path.name}")
    else:
        print(f"  Using existing GDB: {gdb_path.name}")
    return p


def build_mosaic(
    gdb: str,
    mosaic_name: str,
    raster_dir: Path,
    display_label: str,
    granularity: str = "annual",   # "annual" | "monthly"
) -> str | None:
    """
    Create (or replace) a mosaic dataset in `gdb`, add all *.tif files from
    `raster_dir`, then time-enable on a derived AcquiredDate field.

    Annual  filenames: "2017.tif"     → AcquiredDate = July 1 of that year
    Monthly filenames: "2017_07.tif"  → AcquiredDate = 15th of that month

    Returns the full mosaic dataset path, or None if no .tif files were found.
    """
    tifs = sorted(raster_dir.glob("*.tif"))
    if not tifs:
        print(f"  [{display_label}] No .tif files in {raster_dir.name} — skipping.")
        return None

    mosaic_path = f"{gdb}/{mosaic_name}"
    sr = arcpy.SpatialReference(WKID)

    # Idempotent: delete and recreate
    if arcpy.Exists(mosaic_path):
        arcpy.management.Delete(mosaic_path)

    arcpy.management.CreateMosaicDataset(
        in_workspace=gdb,
        in_mosaicdataset_name=mosaic_name,
        coordinate_system=sr,
        num_bands=1,
        pixel_type="32_BIT_FLOAT",
    )

    arcpy.management.AddRastersToMosaicDataset(
        in_mosaic_dataset=mosaic_path,
        raster_type="Raster Dataset",
        input_path=str(raster_dir),
        filter="*.tif",
        update_cellsize_ranges="UPDATE_CELL_SIZES",
        update_boundary="UPDATE_BOUNDARY",
        maximum_pyramid_levels=-1,
        maximum_cell_size=0,
        minimum_dimension=1500,
        spatial_reference=sr,
        duplicate_items_action="OVERWRITE_DUPLICATES",
    )

    count = int(arcpy.management.GetCount(mosaic_path).getOutput(0))
    print(f"  [{display_label}] {count} scene(s) added.")

    # ── AcquiredDate field ────────────────────────────────────────────────────
    existing = {f.name for f in arcpy.ListFields(mosaic_path)}
    if "AcquiredDate" in existing:
        arcpy.management.DeleteField(mosaic_path, "AcquiredDate")
    arcpy.management.AddField(mosaic_path, "AcquiredDate", "DATE")

    if granularity == "monthly":
        # Name is e.g. "2017_07" — mid-month proxy: 15th
        arcpy.management.CalculateField(
            in_table=mosaic_path,
            field="AcquiredDate",
            expression="make_date(!Name!)",
            expression_type="PYTHON3",
            code_block=(
                "import datetime\n"
                "def make_date(name):\n"
                "    try:\n"
                "        stem = name.replace('.tif','').strip()\n"
                "        y, m = stem.split('_')\n"
                "        return datetime.datetime(int(y), int(m), 15)\n"
                "    except Exception:\n"
                "        return None\n"
            ),
        )
        time_msg = "Time-enabled on AcquiredDate (15th of each month)."
    else:
        # Name is e.g. "2017" — July 1 peak-season proxy
        arcpy.management.CalculateField(
            in_table=mosaic_path,
            field="AcquiredDate",
            expression="make_date(!Name!)",
            expression_type="PYTHON3",
            code_block=(
                "import datetime\n"
                "def make_date(name):\n"
                "    try:\n"
                "        return datetime.datetime(int(name.replace('.tif','').strip()), 7, 1)\n"
                "    except Exception:\n"
                "        return None\n"
            ),
        )
        time_msg = "Time-enabled on AcquiredDate (July 1 of each year)."

    # ── Enable time ───────────────────────────────────────────────────────────
    arcpy.management.SetMosaicDatasetProperties(
        in_mosaic_dataset=mosaic_path,
        use_time="ENABLED",
        start_time_field="AcquiredDate",
        end_time_field="",
        default_mosaic_method="ByAttribute",
        order_field="AcquiredDate",
        sorting_order="ASCENDING",
    )
    print(f"  [{display_label}] {time_msg}")

    return mosaic_path


def save_lyrx(mosaic_path: str, layer_name: str, lyrx_dir: Path) -> str:
    """Make a mosaic layer and save it to a .lyrx file; return the file path."""
    lyr = arcpy.management.MakeMosaicLayer(
        in_mosaic_dataset=mosaic_path,
        out_mosaic_layer=layer_name,
    ).getOutput(0)

    lyrx_path = str(lyrx_dir / f"{layer_name}.lyrx")
    arcpy.management.SaveToLayerFile(
        in_layer=lyr,
        out_layer=lyrx_path,
        is_relative_path="RELATIVE",
    )
    return lyrx_path


def package_layers(lyrx_files: list[str], output_lpkx: Path, summary: str, tags: str) -> None:
    """Package a list of .lyrx files into a single .lpkx."""
    if output_lpkx.exists():
        output_lpkx.unlink()

    arcpy.management.PackageLayer(
        in_layer=";".join(lyrx_files),
        output_file=str(output_lpkx),
        convert_data="CONVERT",
        convert_arcsde_data="CONVERT_ARCSDE",
        extent="#",
        apply_extent_to_arcsde="ALL",
        schema_only="ALL",
        version="CURRENT",
        additional_files="",
        summary=summary,
        tags=tags,
    )
    print(f"\n✓  Layer package written: {output_lpkx}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    arcpy.env.overwriteOutput = True
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LYRX_SCRATCH_DIR.mkdir(exist_ok=True)

    gdb = ensure_gdb(SCRATCH_GDB)

    annual_lyrx:  list[str] = []
    monthly_lyrx: list[str] = []

    # ── Annual: Sentinel-2 ────────────────────────────────────────────────────
    if BUILD_ANNUAL:
        print("\n=== Sentinel-2 Annual Peak Mosaics ===")
        for idx in S2_INDICES:
            for aoi in AOIS:
                mosaic_name   = f"S2_{idx.upper()}_{aoi}"
                display_label = f"S2 {idx.upper()} – {AOI_LABELS[aoi]}"
                raster_dir    = S2_ANNUAL_MAX_DIR / f"{idx}_{aoi}"

                print(f"\n  -- {display_label} --")
                mp = build_mosaic(gdb, mosaic_name, raster_dir, display_label, "annual")
                if mp:
                    lyrx = save_lyrx(mp, mosaic_name, LYRX_SCRATCH_DIR)
                    annual_lyrx.append(lyrx)
                    print(f"  [{display_label}] Layer file saved.")

        # ── Annual: Landsat (skipped if not yet built) ────────────────────────
        if LS_ANNUAL_MAX_DIR.exists():
            print("\n=== Landsat Collection 2 Annual Peak Mosaics ===")
            for idx in LS_INDICES:
                for aoi in AOIS:
                    mosaic_name   = f"LS_{idx.upper()}_{aoi}"
                    display_label = f"Landsat {idx.upper()} – {AOI_LABELS[aoi]}"
                    raster_dir    = LS_ANNUAL_MAX_DIR / f"{idx}_{aoi}_landsat"

                    print(f"\n  -- {display_label} --")
                    mp = build_mosaic(gdb, mosaic_name, raster_dir, display_label, "annual")
                    if mp:
                        lyrx = save_lyrx(mp, mosaic_name, LYRX_SCRATCH_DIR)
                        annual_lyrx.append(lyrx)
                        print(f"  [{display_label}] Layer file saved.")
        else:
            print(
                f"\n[Landsat annual] {LS_ANNUAL_MAX_DIR.name} not yet present — skipping.\n"
                "Re-run after build_landsat_annual_composites.py finishes."
            )
    else:
        print("\n[Skipping annual mosaics — BUILD_ANNUAL = False]")

    # ── Monthly: Sentinel-2 and Landsat ──────────────────────────────────────
    if BUILD_MONTHLY:
        if S2_MONTHLY_MAX_DIR.exists():
            print("\n=== Sentinel-2 Monthly Mosaics ===")
            for idx in S2_INDICES:
                for aoi in AOIS:
                    mosaic_name   = f"S2_{idx.upper()}_{aoi}_monthly"
                    display_label = f"S2 {idx.upper()} – {AOI_LABELS[aoi]} (monthly)"
                    raster_dir    = S2_MONTHLY_MAX_DIR / f"{idx}_{aoi}"

                    print(f"\n  -- {display_label} --")
                    mp = build_mosaic(gdb, mosaic_name, raster_dir, display_label, "monthly")
                    if mp:
                        lyrx = save_lyrx(mp, mosaic_name, LYRX_SCRATCH_DIR)
                        monthly_lyrx.append(lyrx)
                        print(f"  [{display_label}] Layer file saved.")
        else:
            print(
                f"\n[S2 monthly] {S2_MONTHLY_MAX_DIR.name} not yet present — skipping.\n"
                "Re-run after build_monthly_composites.py finishes."
            )

        if LS_MONTHLY_MAX_DIR.exists():
            print("\n=== Landsat Collection 2 Monthly Mosaics ===")
            for idx in LS_INDICES:
                for aoi in AOIS:
                    mosaic_name   = f"LS_{idx.upper()}_{aoi}_monthly"
                    display_label = f"Landsat {idx.upper()} – {AOI_LABELS[aoi]} (monthly)"
                    raster_dir    = LS_MONTHLY_MAX_DIR / f"{idx}_{aoi}_landsat"

                    print(f"\n  -- {display_label} --")
                    mp = build_mosaic(gdb, mosaic_name, raster_dir, display_label, "monthly")
                    if mp:
                        lyrx = save_lyrx(mp, mosaic_name, LYRX_SCRATCH_DIR)
                        monthly_lyrx.append(lyrx)
                        print(f"  [{display_label}] Layer file saved.")
        else:
            print(
                f"\n[Landsat monthly] {LS_MONTHLY_MAX_DIR.name} not yet present — skipping.\n"
                "Re-run after build_landsat_monthly_composites.py finishes."
            )
    else:
        print("\n[Skipping monthly mosaics — BUILD_MONTHLY = False]")

    # ── Package annual ────────────────────────────────────────────────────────
    if annual_lyrx:
        print(f"\n=== Packaging {len(annual_lyrx)} annual layer(s) → {OUTPUT_ANNUAL.name} ===")
        package_layers(
            annual_lyrx,
            OUTPUT_ANNUAL,
            summary=(
                "TNC Appalachian Forest Vegetation Indices — UW GISC Capstone 2026. "
                "Annual peak-season composites (Sentinel-2 2017-2026; "
                "Landsat Collection 2 1984-2025 where available). "
                "Time-enabled: use the ArcGIS Pro time slider to animate by year."
            ),
            tags="TNC, Appalachian, NDVI, NDMI, EVI, Sentinel-2, Landsat, vegetation, annual, time-enabled",
        )
        print("   Set time slider step to 1 Year.")
    else:
        print("\nNo annual layers — annual package skipped.")

    # ── Package monthly ───────────────────────────────────────────────────────
    if monthly_lyrx:
        print(f"\n=== Packaging {len(monthly_lyrx)} monthly layer(s) → {OUTPUT_MONTHLY.name} ===")
        package_layers(
            monthly_lyrx,
            OUTPUT_MONTHLY,
            summary=(
                "TNC Appalachian Forest Vegetation Indices — UW GISC Capstone 2026. "
                "Monthly maximum composites (Sentinel-2 2017-2026; "
                "Landsat Collection 2 1984-2025 where available). "
                "Time-enabled: use the ArcGIS Pro time slider to animate by month."
            ),
            tags="TNC, Appalachian, NDVI, NDMI, EVI, Sentinel-2, Landsat, vegetation, monthly, time-enabled",
        )
        print("   Set time slider step to 1 Month.")
    else:
        print("\nNo monthly layers — monthly package skipped.")

    # Clean up temp .lyrx files
    shutil.rmtree(LYRX_SCRATCH_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
