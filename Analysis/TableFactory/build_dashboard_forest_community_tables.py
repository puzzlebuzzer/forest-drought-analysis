#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.paths import PROJECT_ROOT
from src.table_factory_forest_community import (
    FOREST_ECOZONE_GROUP_SCENE_STEM,
    FOREST_ECOZONE_GROUP_TEMPORAL_STEM,
    FOREST_COMMUNITY_DICTIONARY_NAME,
    FOREST_COMMUNITY_SCENE_STEM,
    FOREST_COMMUNITY_TEMPORAL_STEM,
    build_filtered_scene_catalog,
    build_forest_community_data_dictionary_markdown,
    build_scene_summary_ecozone_group,
    build_scene_summary_forest_community,
    build_temporal_summary_ecozone_group,
    build_temporal_summary_forest_community,
    default_forest_community_output_dir,
    iter_temporal_summary_ecozone_group_chunks,
    iter_temporal_summary_forest_community_chunks,
)

MERGE_KEYS = {
    FOREST_COMMUNITY_SCENE_STEM: [
        "sensor",
        "aoi",
        "index",
        "forest_community_code",
        "source_file_or_composite_id",
        "spatial_percentile",
    ],
    FOREST_COMMUNITY_TEMPORAL_STEM: [
        "sensor",
        "aoi",
        "index",
        "forest_community_code",
        "year",
        "date",
        "season_filter",
        "temporal_agg",
        "temporal_percentile",
        "spatial_percentile",
        "cloud_threshold",
        "pixel_mask_id",
    ],
    FOREST_ECOZONE_GROUP_SCENE_STEM: [
        "sensor",
        "aoi",
        "index",
        "ecozone_group_code",
        "source_file_or_composite_id",
        "spatial_percentile",
    ],
    FOREST_ECOZONE_GROUP_TEMPORAL_STEM: [
        "sensor",
        "aoi",
        "index",
        "ecozone_group_code",
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
MANIFEST_COLUMNS = [
    "sensor",
    "aoi",
    "index",
    "ecozone_code",
    "ecozone_label",
    "forest_community_code",
    "forest_community_display_code",
    "forest_community_label",
    "forest_community_source_dataset",
    "forest_community_source_value",
    "forest_community_source_key",
    "ecozone_group_code",
    "ecozone_group_label",
    "ecozone_group_raw",
    "temporal_agg",
    "temporal_percentile",
    "spatial_percentile",
    "cloud_threshold",
    "season_filter",
]
PARQUET_FAILURE_SAMPLE_ROWS = 1000
PARTITIONED_PARQUET_DIRNAME = "partitioned_parquet"
PARTITION_COLUMNS = ["sensor", "aoi", "index", "ecozone_code", "forest_community_code", "temporal_agg"]
PARTITION_COLUMNS_BY_STEM = {
    FOREST_ECOZONE_GROUP_SCENE_STEM: ["sensor", "aoi", "index", "ecozone_group_code", "temporal_agg"],
    FOREST_ECOZONE_GROUP_TEMPORAL_STEM: ["sensor", "aoi", "index", "ecozone_group_code", "temporal_agg"],
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


def _archive_existing_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    archive_dir = path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    archive_path = archive_dir / f"{path.stem}.{timestamp}{path.suffix}"
    shutil.copy2(path, archive_path)
    return archive_path


def _archive_existing_path(path: Path) -> Path | None:
    if not path.exists():
        return None
    archive_dir = path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    archive_path = archive_dir / f"{path.name}.{timestamp}"
    if path.is_dir():
        if archive_path.exists():
            shutil.rmtree(archive_path)
        shutil.move(str(path), str(archive_path))
    else:
        shutil.copy2(path, archive_path)
    return archive_path


def _write_parquet_failure_report(output_dir: Path, stem: str, frame: pd.DataFrame, exc: Exception) -> None:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    report_path = output_dir / f"{stem}.parquet_failed.{timestamp}.json"
    sample_path = output_dir / f"{stem}.parquet_failed_sample.{timestamp}.csv"
    report = {
        "stem": stem,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "sample_path": sample_path.name,
        "sample_rows": min(PARQUET_FAILURE_SAMPLE_ROWS, len(frame)),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    frame.head(PARQUET_FAILURE_SAMPLE_ROWS).to_csv(sample_path, index=False)
    print(f"Wrote Parquet failure report {report_path}")
    print(f"Wrote capped failure sample {sample_path} rows={report['sample_rows']}")


def _merge_with_existing(output_dir: Path, stem: str, frame: pd.DataFrame) -> pd.DataFrame:
    key_columns = MERGE_KEYS[stem]
    try:
        existing = _read_table(output_dir, stem)
    except FileNotFoundError:
        return frame
    combined = pd.concat([existing, frame], ignore_index=True)
    return combined.drop_duplicates(subset=key_columns, keep="last").reset_index(drop=True)


def _write_manifest(output_dir: Path, stem: str, frame: pd.DataFrame, table_path: Path) -> None:
    if frame.empty:
        return
    group_columns = [column for column in MANIFEST_COLUMNS if column in frame.columns]
    manifest = (
        frame.groupby(group_columns, dropna=False)
        .agg(row_count=("value", "size"), min_year=("year", "min"), max_year=("year", "max"))
        .reset_index()
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    try:
        manifest["path"] = table_path.relative_to(output_dir).as_posix()
    except ValueError:
        manifest["path"] = table_path.name
    csv_path = output_dir / f"{stem}_manifest.csv"
    parquet_path = output_dir / f"{stem}_manifest.parquet"
    _archive_existing_file(csv_path)
    _archive_existing_file(parquet_path)
    manifest.to_csv(csv_path, index=False)
    try:
        manifest.to_parquet(parquet_path, index=False)
    except Exception:
        parquet_path = None
    print(f"Wrote {csv_path}")
    if parquet_path is not None:
        print(f"Wrote {parquet_path}")


def _manifest_for_frame(frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = [column for column in MANIFEST_COLUMNS if column in frame.columns]
    return (
        frame.groupby(group_columns, dropna=False)
        .agg(row_count=("value", "size"), min_year=("year", "min"), max_year=("year", "max"))
        .reset_index()
    )


def _write_manifest_frame(output_dir: Path, stem: str, manifest: pd.DataFrame, table_path: Path) -> None:
    if manifest.empty:
        return
    group_columns = [column for column in MANIFEST_COLUMNS if column in manifest.columns]
    manifest = (
        manifest.groupby(group_columns, dropna=False)
        .agg(row_count=("row_count", "sum"), min_year=("min_year", "min"), max_year=("max_year", "max"))
        .reset_index()
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    try:
        manifest["path"] = table_path.relative_to(output_dir).as_posix()
    except ValueError:
        manifest["path"] = table_path.name
    csv_path = output_dir / f"{stem}_manifest.csv"
    parquet_path = output_dir / f"{stem}_manifest.parquet"
    _archive_existing_file(csv_path)
    _archive_existing_file(parquet_path)
    manifest.to_csv(csv_path, index=False)
    manifest.to_parquet(parquet_path, index=False)
    print(f"Wrote {csv_path} rows={len(manifest):,}")
    print(f"Wrote {parquet_path}")


def _write_named_table(
    output_dir: Path,
    stem: str,
    frame: pd.DataFrame,
    *,
    merge_existing: bool = False,
    write_csv: bool = False,
) -> None:
    if merge_existing:
        frame = _merge_with_existing(output_dir, stem, frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_target = output_dir / f"{stem}.parquet"
    table_path: Path | None = None
    archived_parquet = _archive_existing_file(parquet_target)
    try:
        frame.to_parquet(parquet_target, index=False)
        table_path = parquet_target
        print(f"Wrote {parquet_target}")
    except Exception as exc:
        parquet_target.unlink(missing_ok=True)
        if archived_parquet is not None:
            shutil.copy2(archived_parquet, parquet_target)
            print(f"Restored previous Parquet from {archived_parquet}")
        print(f"Parquet write failed for {stem}: {exc}")
        _write_parquet_failure_report(output_dir, stem, frame, exc)
        raise RuntimeError(
            f"Parquet write failed for {stem}; previous Parquet was preserved and no full CSV fallback was written."
        ) from exc

    if write_csv:
        csv_target = output_dir / f"{stem}.csv"
        _archive_existing_file(csv_target)
        frame.to_csv(csv_target, index=False)
        if table_path is None:
            table_path = csv_target
        print(f"Wrote {csv_target}")
    if table_path is None:
        raise RuntimeError(f"No table output was written for {stem}")
    _write_manifest(output_dir, stem, frame, table_path)


def _write_partitioned_table_chunks(output_dir: Path, stem: str, chunks, *, compression: str = "zstd") -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / PARTITIONED_PARQUET_DIRNAME / stem
    tmp_target = target.with_name(f"{target.name}.tmp")
    if tmp_target.exists():
        shutil.rmtree(tmp_target)
    tmp_target.mkdir(parents=True, exist_ok=True)

    manifest_parts: list[pd.DataFrame] = []
    total_rows = 0
    chunk_count = 0
    try:
        for chunk_count, frame in enumerate(chunks, start=1):
            if frame.empty:
                continue
            stem_partition_columns = PARTITION_COLUMNS_BY_STEM.get(stem, PARTITION_COLUMNS)
            partition_columns = [column for column in stem_partition_columns if column in frame.columns]
            table = pa.Table.from_pandas(frame, preserve_index=False)
            pq.write_to_dataset(
                table,
                root_path=tmp_target,
                partition_cols=partition_columns,
                compression=compression,
                use_dictionary=True,
                write_statistics=True,
                row_group_size=100_000,
            )
            manifest_parts.append(_manifest_for_frame(frame))
            total_rows += len(frame)
            print(f"Wrote partition chunk {chunk_count}: rows={len(frame):,} total_rows={total_rows:,}", flush=True)
    except Exception:
        print(f"Partitioned write failed; partial temp output remains at {tmp_target}", flush=True)
        raise

    if total_rows == 0:
        raise RuntimeError(f"No rows were produced for {stem}")
    if target.exists():
        archived_target = _archive_existing_path(target)
        print(f"Archived previous partitioned dataset to {archived_target}")
    shutil.move(str(tmp_target), str(target))
    manifest = pd.concat(manifest_parts, ignore_index=True) if manifest_parts else pd.DataFrame()
    _write_manifest_frame(output_dir, stem, manifest, target)
    print(f"Wrote partitioned dataset {target} chunks={chunk_count} rows={total_rows:,}")
    return manifest


def _write_dictionary(output_dir: Path) -> None:
    dictionary_path = output_dir / FOREST_COMMUNITY_DICTIONARY_NAME
    _archive_existing_file(dictionary_path)
    dictionary_path.write_text(build_forest_community_data_dictionary_markdown(output_dir), encoding="utf-8")
    print(f"Wrote {dictionary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dashboard-ready forest-community summary tables from aligned raster caches.")
    parser.add_argument(
        "--output-dir",
        default=str(default_forest_community_output_dir(PROJECT_ROOT)),
        help="Directory for forest-community dashboard table products.",
    )
    parser.add_argument(
        "--limit-scenes-per-group",
        type=int,
        default=None,
        help="Optional dev-mode limiter applied per sensor x AOI x index group before forest-community summaries.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Optional inclusive lower year bound for all downstream forest-community tables.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Optional inclusive upper year bound for all downstream forest-community tables.",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write CSV outputs after a successful Parquet write. Parquet is always attempted and is the recommended dashboard source.",
    )
    parser.add_argument(
        "--include-scene-id-list",
        action="store_true",
        help="Store full joined source scene IDs in temporal rows. Off by default to reduce table size.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("scene-summary", help="Build scene_summary_forest_community from cache manifests and aligned rasters")
    subparsers.add_parser("temporal-summary", help="Build temporal_summary_forest_community from scene_summary_forest_community")
    subparsers.add_parser("ecozone-group-scene-summary", help="Build scene_summary_forest_ecozone_group from cache manifests and aligned rasters")
    subparsers.add_parser("ecozone-group-temporal-summary", help="Build temporal_summary_forest_ecozone_group from scene_summary_forest_ecozone_group")
    subparsers.add_parser("ecozone-group-all", help="Build all forest ecozone-group dashboard table products")
    subparsers.add_parser("data-dictionary", help="Write the forest-community data dictionary only")
    subparsers.add_parser("all", help="Build all forest-community dashboard table products")
    return parser.parse_args()


def build_scene_summary_step(
    output_dir: Path,
    limit_scenes_per_group: int | None,
    start_year: int | None,
    end_year: int | None,
    *,
    write_csv: bool,
) -> pd.DataFrame:
    print("Building forest-community scene summaries from aligned rasters...")
    scene_catalog = build_filtered_scene_catalog(
        limit_scenes_per_group=limit_scenes_per_group,
        start_year=start_year,
        end_year=end_year,
    )
    frame = build_scene_summary_forest_community(scene_catalog)
    print(f"Forest-community scene summary rows: {len(frame)}")
    _write_named_table(output_dir, FOREST_COMMUNITY_SCENE_STEM, frame, merge_existing=False, write_csv=write_csv)
    return frame


def build_temporal_summary_step(
    output_dir: Path,
    limit_scenes_per_group: int | None,
    start_year: int | None,
    end_year: int | None,
    *,
    write_csv: bool,
    include_scene_id_list: bool,
) -> pd.DataFrame:
    try:
        scene_summary = _read_table(output_dir, FOREST_COMMUNITY_SCENE_STEM)
        if start_year is not None:
            scene_summary = scene_summary[scene_summary["year"] >= start_year]
        if end_year is not None:
            scene_summary = scene_summary[scene_summary["year"] <= end_year]
    except FileNotFoundError:
        scene_summary = build_scene_summary_step(
            output_dir,
            limit_scenes_per_group,
            start_year,
            end_year,
            write_csv=write_csv,
        )
    print("Building forest-community temporal summaries as partitioned Parquet chunks...")
    chunks = iter_temporal_summary_forest_community_chunks(
        scene_summary,
        include_scene_id_list=include_scene_id_list,
    )
    manifest = _write_partitioned_table_chunks(output_dir, FOREST_COMMUNITY_TEMPORAL_STEM, chunks)
    if write_csv:
        print("Skipping CSV write for partitioned forest-community temporal output; expected row count is too large.")
    return manifest


def build_ecozone_group_scene_summary_step(
    output_dir: Path,
    limit_scenes_per_group: int | None,
    start_year: int | None,
    end_year: int | None,
    *,
    write_csv: bool,
) -> pd.DataFrame:
    print("Building forest ecozone-group scene summaries from aligned rasters...")
    scene_catalog = build_filtered_scene_catalog(
        limit_scenes_per_group=limit_scenes_per_group,
        start_year=start_year,
        end_year=end_year,
    )
    frame = build_scene_summary_ecozone_group(scene_catalog)
    print(f"Forest ecozone-group scene summary rows: {len(frame)}")
    _write_named_table(output_dir, FOREST_ECOZONE_GROUP_SCENE_STEM, frame, merge_existing=False, write_csv=write_csv)
    return frame


def build_ecozone_group_temporal_summary_step(
    output_dir: Path,
    limit_scenes_per_group: int | None,
    start_year: int | None,
    end_year: int | None,
    *,
    write_csv: bool,
    include_scene_id_list: bool,
) -> pd.DataFrame:
    try:
        scene_summary = _read_table(output_dir, FOREST_ECOZONE_GROUP_SCENE_STEM)
        if start_year is not None:
            scene_summary = scene_summary[scene_summary["year"] >= start_year]
        if end_year is not None:
            scene_summary = scene_summary[scene_summary["year"] <= end_year]
    except FileNotFoundError:
        scene_summary = build_ecozone_group_scene_summary_step(
            output_dir,
            limit_scenes_per_group,
            start_year,
            end_year,
            write_csv=write_csv,
        )
    print("Building forest ecozone-group temporal summaries as partitioned Parquet chunks...")
    chunks = iter_temporal_summary_ecozone_group_chunks(
        scene_summary,
        include_scene_id_list=include_scene_id_list,
    )
    manifest = _write_partitioned_table_chunks(output_dir, FOREST_ECOZONE_GROUP_TEMPORAL_STEM, chunks)
    if write_csv:
        print("Skipping CSV write for partitioned forest ecozone-group temporal output; expected row count is large.")
    return manifest


def main() -> None:
    args = parse_args()
    command = args.command or "all"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if command == "scene-summary":
        build_scene_summary_step(output_dir, args.limit_scenes_per_group, args.start_year, args.end_year, write_csv=args.write_csv)
        return
    if command == "temporal-summary":
        build_temporal_summary_step(
            output_dir,
            args.limit_scenes_per_group,
            args.start_year,
            args.end_year,
            write_csv=args.write_csv,
            include_scene_id_list=args.include_scene_id_list,
        )
        return
    if command == "ecozone-group-scene-summary":
        build_ecozone_group_scene_summary_step(output_dir, args.limit_scenes_per_group, args.start_year, args.end_year, write_csv=args.write_csv)
        return
    if command == "ecozone-group-temporal-summary":
        build_ecozone_group_temporal_summary_step(
            output_dir,
            args.limit_scenes_per_group,
            args.start_year,
            args.end_year,
            write_csv=args.write_csv,
            include_scene_id_list=args.include_scene_id_list,
        )
        return
    if command == "ecozone-group-all":
        build_ecozone_group_scene_summary_step(output_dir, args.limit_scenes_per_group, args.start_year, args.end_year, write_csv=args.write_csv)
        build_ecozone_group_temporal_summary_step(
            output_dir,
            args.limit_scenes_per_group,
            args.start_year,
            args.end_year,
            write_csv=args.write_csv,
            include_scene_id_list=args.include_scene_id_list,
        )
        return
    if command == "data-dictionary":
        _write_dictionary(output_dir)
        return

    build_scene_summary_step(output_dir, args.limit_scenes_per_group, args.start_year, args.end_year, write_csv=args.write_csv)
    build_temporal_summary_step(
        output_dir,
        args.limit_scenes_per_group,
        args.start_year,
        args.end_year,
        write_csv=args.write_csv,
        include_scene_id_list=args.include_scene_id_list,
    )
    _write_dictionary(output_dir)


if __name__ == "__main__":
    main()
