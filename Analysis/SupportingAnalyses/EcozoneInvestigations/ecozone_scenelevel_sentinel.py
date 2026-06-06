#!/usr/bin/env python3
"""
Build per-scene Sentinel ecozone summaries for one AOI and year range.
"""

from src.ecozone_scenelevel import run_scenelevel_analysis
from src.paths import project_path
from src.sentinel import load_sentinel_ecozone, load_sentinel_scenes

TABLES_DIR = project_path("results_tables_dir") / "sentinel"


def main() -> None:
    run_scenelevel_analysis(
        sensor_key="sentinel",
        sensor_label="Sentinel",
        tables_dir=TABLES_DIR,
        load_scenes=load_sentinel_scenes,
        load_ecozone=load_sentinel_ecozone,
        metadata_columns=[("platform", "Platform"), ("tile", "Tile"), ("scene_id", "Scene ID")],
    )


if __name__ == "__main__":
    main()
