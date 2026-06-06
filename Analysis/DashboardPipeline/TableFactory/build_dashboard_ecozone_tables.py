#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.paths import PROJECT_ROOT
from src.table_factory import write_table_with_optional_parquet
from src.table_factory_ecozone import (
    ECOZONE_DICTIONARY_NAME,
    ECOZONE_SCENE_STEM,
    ECOZONE_TEMPORAL_STEM,
    build_ecozone_data_dictionary_markdown,
    build_filtered_scene_catalog,
    build_scene_summary_ecozone,
    build_temporal_summary_ecozone,
    default_ecozone_output_dir,
)

MERGE_KEYS = {
    ECOZONE_SCENE_STEM: [
        "sensor",
        "aoi",
        "index",
        "ecozone_code",
        "source_file_or_composite_id",
        "spatial_percentile",
    ],
    ECOZONE_TEMPORAL_STEM: [
        "sensor",
        "aoi",
        "index",
        "ecozone_code",
        "year",
        "date",
        "season_filter",
        "temporal_agg",
        "temporal_percentile",
        "spatial_percentile",
        "cloud_threshold",
        "pixel_mask_id",
    ],
}


def _read_table(output_dir: Path, stem: str) -> pd.DataFrame:
    parquet_path = output_dir / f"{stem}.parquet"
    csv_path = output_dir / f"{stem}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
        for column in ("date", "time_bin_start", "time_bin_end"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        return frame
    raise FileNotFoundError(f"Required input table not found: {csv_path} or {parquet_path}")


def _archive_existing_file(path: Path) -> None:
    if not path.exists():
        return
    archive_dir = path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    shutil.copy2(path, archive_dir / f"{path.stem}.{timestamp}{path.suffix}")


def _merge_with_existing(output_dir: Path, stem: str, frame: pd.DataFrame) -> pd.DataFrame:
    key_columns = MERGE_KEYS[stem]
    try:
        existing = _read_table(output_dir, stem)
    except FileNotFoundError:
        return frame
    combined = pd.concat([existing, frame], ignore_index=True)
    return combined.drop_duplicates(subset=key_columns, keep="last").reset_index(drop=True)


def _write_named_table(output_dir: Path, stem: str, frame: pd.DataFrame, merge_existing: bool = False) -> None:
    if merge_existing:
        frame = _merge_with_existing(output_dir, stem, frame)
    csv_target = output_dir / f"{stem}.csv"
    parquet_target = output_dir / f"{stem}.parquet"
    _archive_existing_file(csv_target)
    _archive_existing_file(parquet_target)
    csv_path, parquet_path = write_table_with_optional_parquet(frame, csv_target)
    print(f"Wrote {csv_path}")
    if parquet_path is not None:
        print(f"Wrote {parquet_path}")


def _write_dictionary(output_dir: Path) -> None:
    dictionary_path = output_dir / ECOZONE_DICTIONARY_NAME
    _archive_existing_file(dictionary_path)
    dictionary_path.write_text(build_ecozone_data_dictionary_markdown(output_dir), encoding="utf-8")
    print(f"Wrote {dictionary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dashboard-ready ecozone summary tables from aligned raster caches.")
    parser.add_argument(
        "--output-dir",
        default=str(default_ecozone_output_dir(PROJECT_ROOT)),
        help="Directory for CSV/parquet ecozone dashboard table products.",
    )
    parser.add_argument(
        "--limit-scenes-per-group",
        type=int,
        default=None,
        help="Optional dev-mode limiter applied per sensor x AOI x index group before ecozone summaries.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Optional inclusive lower year bound for all downstream ecozone tables.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Optional inclusive upper year bound for all downstream ecozone tables.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("scene-summary", help="Build scene_summary_ecozone from cache manifests and aligned rasters")
    subparsers.add_parser("temporal-summary", help="Build temporal_summary_ecozone from scene_summary_ecozone")
    subparsers.add_parser("data-dictionary", help="Write the ecozone data dictionary only")
    subparsers.add_parser("all", help="Build all ecozone dashboard table products")
    return parser.parse_args()


def build_scene_summary_step(
    output_dir: Path,
    limit_scenes_per_group: int | None,
    start_year: int | None,
    end_year: int | None,
) -> pd.DataFrame:
    print("Building ecozone scene summaries from aligned rasters...")
    scene_catalog = build_filtered_scene_catalog(
        limit_scenes_per_group=limit_scenes_per_group,
        start_year=start_year,
        end_year=end_year,
    )
    frame = build_scene_summary_ecozone(scene_catalog)
    print(f"Ecozone scene summary rows: {len(frame)}")
    _write_named_table(output_dir, ECOZONE_SCENE_STEM, frame, merge_existing=True)
    return frame


def build_temporal_summary_step(
    output_dir: Path,
    limit_scenes_per_group: int | None,
    start_year: int | None,
    end_year: int | None,
) -> pd.DataFrame:
    try:
        scene_summary = _read_table(output_dir, ECOZONE_SCENE_STEM)
        if start_year is not None:
            scene_summary = scene_summary[scene_summary["year"] >= start_year]
        if end_year is not None:
            scene_summary = scene_summary[scene_summary["year"] <= end_year]
    except FileNotFoundError:
        scene_summary = build_scene_summary_step(output_dir, limit_scenes_per_group, start_year, end_year)
    print("Building ecozone temporal summaries...")
    frame = build_temporal_summary_ecozone(scene_summary)
    print(f"Ecozone temporal summary rows: {len(frame)}")
    _write_named_table(output_dir, ECOZONE_TEMPORAL_STEM, frame, merge_existing=True)
    return frame


def main() -> None:
    args = parse_args()
    command = args.command or "all"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if command == "scene-summary":
        build_scene_summary_step(output_dir, args.limit_scenes_per_group, args.start_year, args.end_year)
        return
    if command == "temporal-summary":
        build_temporal_summary_step(output_dir, args.limit_scenes_per_group, args.start_year, args.end_year)
        return
    if command == "data-dictionary":
        _write_dictionary(output_dir)
        return

    build_scene_summary_step(output_dir, args.limit_scenes_per_group, args.start_year, args.end_year)
    build_temporal_summary_step(output_dir, args.limit_scenes_per_group, args.start_year, args.end_year)
    _write_dictionary(output_dir)


if __name__ == "__main__":
    main()
