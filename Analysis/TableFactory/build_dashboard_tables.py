#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.paths import PROJECT_ROOT
from src.table_factory import (
    build_data_dictionary_markdown,
    build_scene_catalog,
    build_scene_summary,
    build_temporal_summary,
    canonical_dashboard_tables_dir,
    write_table_with_optional_parquet,
)

MERGE_KEYS = {
    "scene_catalog": ["sensor", "aoi", "index", "source_file_or_composite_id"],
    "scene_summary": ["sensor", "aoi", "index", "source_file_or_composite_id", "spatial_percentile"],
    "temporal_summary": [
        "sensor",
        "aoi",
        "index",
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


def _apply_scene_limit(scene_catalog: pd.DataFrame, limit_scenes_per_group: int | None) -> pd.DataFrame:
    if limit_scenes_per_group is None:
        return scene_catalog
    limited = (
        scene_catalog.groupby(["sensor", "aoi", "index"], group_keys=False)
        .head(limit_scenes_per_group)
        .reset_index(drop=True)
    )
    print(f"Applied limit: first {limit_scenes_per_group} scenes per sensor/aoi/index group")
    return limited


def _apply_year_filter(
    scene_catalog: pd.DataFrame,
    start_year: int | None,
    end_year: int | None,
) -> pd.DataFrame:
    if start_year is None and end_year is None:
        return scene_catalog
    filtered = scene_catalog.copy()
    if start_year is not None:
        filtered = filtered[filtered["year"] >= start_year]
    if end_year is not None:
        filtered = filtered[filtered["year"] <= end_year]
    print(
        "Applied year filter:"
        f" start_year={start_year if start_year is not None else '-inf'}"
        f" end_year={end_year if end_year is not None else '+inf'}"
    )
    return filtered.reset_index(drop=True)


def _archive_existing_file(path: Path) -> None:
    if not path.exists():
        return
    archive_dir = path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    archived_path = archive_dir / f"{path.stem}.{timestamp}{path.suffix}"
    shutil.copy2(path, archived_path)
    print(f"Archived existing {path.name} -> {archived_path}")


def _merge_with_existing(output_dir: Path, stem: str, frame: pd.DataFrame) -> pd.DataFrame:
    key_columns = MERGE_KEYS.get(stem)
    if key_columns is None:
        return frame
    try:
        existing = _read_table(output_dir, stem)
    except FileNotFoundError:
        return frame
    combined = pd.concat([existing, frame], ignore_index=True)
    combined = combined.drop_duplicates(subset=key_columns, keep="last")
    print(f"Merged with existing {stem}: old_rows={len(existing)} new_rows={len(frame)} combined_rows={len(combined)}")
    return combined.reset_index(drop=True)


def _write_named_table(output_dir: Path, stem: str, frame: pd.DataFrame, merge_existing: bool = False) -> None:
    if merge_existing:
        frame = _merge_with_existing(output_dir, stem, frame)
    csv_target = output_dir / f"{stem}.csv"
    parquet_target = output_dir / f"{stem}.parquet"
    _archive_existing_file(csv_target)
    _archive_existing_file(parquet_target)
    csv_path, parquet_path = write_table_with_optional_parquet(frame, output_dir / f"{stem}.csv")
    print(f"Wrote {csv_path}")
    if parquet_path is None:
        print(f"Skipped parquet for {stem}.csv because no parquet engine is available")
    else:
        print(f"Wrote {parquet_path}")


def _write_dictionary(output_dir: Path) -> None:
    dictionary_path = output_dir / "data_dictionary.md"
    _archive_existing_file(dictionary_path)
    dictionary_path.write_text(build_data_dictionary_markdown(output_dir), encoding="utf-8")
    print(f"Wrote {dictionary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dashboard-ready summary tables from aligned raster caches and manifests.")
    subparsers = parser.add_subparsers(dest="command")

    parser.add_argument(
        "--output-dir",
        default=str(canonical_dashboard_tables_dir(PROJECT_ROOT)),
        help="Directory for CSV/parquet dashboard table products.",
    )
    parser.add_argument(
        "--limit-scenes-per-group",
        type=int,
        default=None,
        help="Optional dev-mode limiter applied per sensor x AOI x index group before raster summaries.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Optional inclusive lower year bound for the scene catalog and all downstream tables.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Optional inclusive upper year bound for the scene catalog and all downstream tables.",
    )
    subparsers.add_parser("scene-catalog", help="Build scene_catalog from cache manifests")
    subparsers.add_parser("scene-summary", help="Build scene_summary from scene_catalog")
    subparsers.add_parser("temporal-summary", help="Build temporal_summary from scene_summary")
    subparsers.add_parser("data-dictionary", help="Write the data dictionary only")
    subparsers.add_parser("all", help="Build all dashboard table products")
    return parser.parse_args()


def build_scene_catalog_step(
    output_dir: Path,
    limit_scenes_per_group: int | None,
    start_year: int | None,
    end_year: int | None,
) -> pd.DataFrame:
    print("Building canonical scene catalog...")
    scene_catalog = build_scene_catalog()
    scene_catalog = _apply_year_filter(scene_catalog, start_year, end_year)
    scene_catalog = _apply_scene_limit(scene_catalog, limit_scenes_per_group)
    print(f"Scene catalog rows: {len(scene_catalog)}")
    _write_named_table(output_dir, "scene_catalog", scene_catalog, merge_existing=True)
    return scene_catalog


def build_scene_summary_step(
    output_dir: Path,
    limit_scenes_per_group: int | None,
    start_year: int | None,
    end_year: int | None,
) -> pd.DataFrame:
    if start_year is not None or end_year is not None or limit_scenes_per_group is not None:
        print("Building filtered scene catalog from manifests for scene-summary step...")
        scene_catalog = build_scene_catalog()
        scene_catalog = _apply_year_filter(scene_catalog, start_year, end_year)
        scene_catalog = _apply_scene_limit(scene_catalog, limit_scenes_per_group)
    else:
        try:
            scene_catalog = _read_table(output_dir, "scene_catalog")
            print("Loaded existing scene_catalog table")
            scene_catalog = _apply_year_filter(scene_catalog, start_year, end_year)
            scene_catalog = _apply_scene_limit(scene_catalog, limit_scenes_per_group)
        except FileNotFoundError:
            scene_catalog = build_scene_catalog_step(output_dir, limit_scenes_per_group, start_year, end_year)
    print("Building scene-level summaries from aligned rasters...")
    scene_summary = build_scene_summary(scene_catalog)
    print(f"Scene summary rows: {len(scene_summary)}")
    _write_named_table(output_dir, "scene_summary", scene_summary, merge_existing=True)
    return scene_summary


def build_temporal_summary_step(
    output_dir: Path,
    limit_scenes_per_group: int | None,
    start_year: int | None,
    end_year: int | None,
) -> pd.DataFrame:
    try:
        scene_summary = _read_table(output_dir, "scene_summary")
        print("Loaded existing scene_summary table")
        scene_summary = _apply_year_filter(scene_summary, start_year, end_year)
    except FileNotFoundError:
        scene_summary = build_scene_summary_step(output_dir, limit_scenes_per_group, start_year, end_year)
    print("Building temporal summaries...")
    temporal_summary = build_temporal_summary(scene_summary)
    print(f"Temporal summary rows: {len(temporal_summary)}")
    _write_named_table(output_dir, "temporal_summary", temporal_summary, merge_existing=True)
    return temporal_summary


def build_all_steps(
    output_dir: Path,
    limit_scenes_per_group: int | None,
    start_year: int | None,
    end_year: int | None,
) -> None:
    build_scene_catalog_step(output_dir, limit_scenes_per_group, start_year, end_year)
    build_scene_summary_step(output_dir, limit_scenes_per_group, start_year, end_year)
    build_temporal_summary_step(output_dir, limit_scenes_per_group, start_year, end_year)
    _write_dictionary(output_dir)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    command = args.command or "all"
    if command == "scene-catalog":
        build_scene_catalog_step(output_dir, args.limit_scenes_per_group, args.start_year, args.end_year)
    elif command == "scene-summary":
        build_scene_summary_step(output_dir, args.limit_scenes_per_group, args.start_year, args.end_year)
    elif command == "temporal-summary":
        build_temporal_summary_step(output_dir, args.limit_scenes_per_group, args.start_year, args.end_year)
    elif command == "data-dictionary":
        _write_dictionary(output_dir)
    else:
        build_all_steps(output_dir, args.limit_scenes_per_group, args.start_year, args.end_year)


if __name__ == "__main__":
    main()
