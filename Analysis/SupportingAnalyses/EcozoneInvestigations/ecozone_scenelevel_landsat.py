#!/usr/bin/env python3
"""
Build per-scene Landsat ecozone summaries for one AOI and year range.
"""

from src.ecozone_scenelevel import run_scenelevel_analysis
from src.landsat import load_landsat_ecozone, load_landsat_scenes
from src.paths import project_path

TABLES_DIR = project_path("results_tables_landsat_dir")


def main() -> None:
    run_scenelevel_analysis(
        sensor_key="landsat",
        sensor_label="Landsat",
        tables_dir=TABLES_DIR,
        load_scenes=load_landsat_scenes,
        load_ecozone=load_landsat_ecozone,
        metadata_columns=[("platform", "Platform"), ("path_row", "Path/Row")],
    )


if __name__ == "__main__":
    main()
