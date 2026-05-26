#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.paths import PROJECT_ROOT


SEGMENT_STEMS = (
    "scene_summary_ecozone",
    "temporal_summary_ecozone",
    "scene_summary_forest_community",
    "temporal_summary_forest_community",
)
OPTIMIZED_DIRNAME = "optimized_parquet"
PARTITIONED_DIRNAME = "partitioned_parquet"
DASHBOARD_COLUMNS = [
    "analysis_scope",
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
    "date",
    "year",
    "doy",
    "growing_season_day",
    "season_filter",
    "temporal_agg",
    "temporal_percentile",
    "spatial_percentile",
    "cloud_threshold",
    "cloud_percent",
    "pixel_mask_id",
    "pixel_mask_version",
    "n_pixels",
    "valid_pixel_fraction",
    "n_scenes",
    "value",
    "time_bin_label",
    "time_bin_start",
    "time_bin_end",
    "month_day_label",
]
DATE_COLUMNS = ("date", "time_bin_start", "time_bin_end")
SORT_COLUMNS = [
    "sensor",
    "aoi",
    "index",
    "ecozone_code",
    "forest_community_code",
    "temporal_agg",
    "spatial_percentile",
    "temporal_percentile",
    "cloud_threshold",
    "season_filter",
    "date",
]
PARTITION_COLUMNS = ["sensor", "aoi", "index", "ecozone_code", "forest_community_code", "temporal_agg"]
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


def _read_source_table(data_dir: Path, stem: str) -> pa.Table:
    parquet_path = data_dir / f"{stem}.parquet"
    csv_path = data_dir / f"{stem}.csv"
    if parquet_path.exists():
        dataset = ds.dataset(parquet_path, format="parquet")
        columns = [column for column in DASHBOARD_COLUMNS if column in dataset.schema.names]
        return dataset.to_table(columns=columns)
    if csv_path.exists():
        frame = pd.read_csv(csv_path, usecols=lambda column: column in DASHBOARD_COLUMNS)
        return pa.Table.from_pandas(frame, preserve_index=False)
    raise FileNotFoundError(f"Missing source table for {stem}: {parquet_path} or {csv_path}")


def _has_partitioned_dataset(data_dir: Path, stem: str) -> bool:
    partitioned_path = data_dir / PARTITIONED_DIRNAME / stem
    if not partitioned_path.exists():
        return False
    return any(partitioned_path.rglob("*.parquet"))


def _replace_column(table: pa.Table, column: str, array: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(column)
    return table.set_column(index, column, array)


def _with_analysis_scope(table: pa.Table, stem: str) -> pa.Table:
    if "analysis_scope" in table.schema.names:
        return table
    scope = "forest_community" if "forest_community" in stem else "ecozone"
    return table.append_column("analysis_scope", pa.repeat(scope, table.num_rows))


def _with_typed_dates(table: pa.Table) -> pa.Table:
    for column in DATE_COLUMNS:
        if column not in table.schema.names:
            continue
        parsed = pd.to_datetime(table[column].to_pandas(), errors="coerce", utc=True).dt.tz_localize(None)
        table = _replace_column(table, column, pa.Array.from_pandas(parsed, type=pa.timestamp("ns")))
    return table


def _sorted_table(table: pa.Table) -> pa.Table:
    sort_keys = [(column, "ascending") for column in SORT_COLUMNS if column in table.schema.names]
    if not sort_keys:
        return table
    return table.sort_by(sort_keys)


def _write_manifest(table: pa.Table, data_dir: Path, stem: str) -> None:
    columns = [column for column in [*MANIFEST_COLUMNS, "year", "value"] if column in table.schema.names]
    frame = table.select(columns).to_pandas()
    group_columns = [column for column in MANIFEST_COLUMNS if column in frame.columns]
    manifest = (
        frame.groupby(group_columns, dropna=False)
        .agg(row_count=("value", "size"), min_year=("year", "min"), max_year=("year", "max"))
        .reset_index()
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    manifest["path"] = f"{PARTITIONED_DIRNAME}/{stem}"
    csv_path = data_dir / f"{stem}_manifest.csv"
    parquet_path = data_dir / f"{stem}_manifest.parquet"
    manifest.to_csv(csv_path, index=False)
    manifest.to_parquet(parquet_path, index=False)
    print(f"Wrote {csv_path} rows={len(manifest):,}", flush=True)
    print(f"Wrote {parquet_path}", flush=True)


def _write_optimized_single(table: pa.Table, data_dir: Path, stem: str, *, compression: str, row_group_size: int) -> Path:
    target_dir = data_dir / OPTIMIZED_DIRNAME
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{stem}.parquet"
    tmp_target = target.with_suffix(".parquet.tmp")
    tmp_target.unlink(missing_ok=True)
    pq.write_table(
        table,
        tmp_target,
        compression=compression,
        use_dictionary=True,
        write_statistics=True,
        row_group_size=row_group_size,
    )
    tmp_target.replace(target)
    print(f"Wrote {target} rows={table.num_rows:,}", flush=True)
    return target


def _write_partitioned_dataset(table: pa.Table, data_dir: Path, stem: str, *, compression: str, row_group_size: int) -> Path:
    target = data_dir / PARTITIONED_DIRNAME / stem
    tmp_target = target.with_name(f"{target.name}.tmp")
    if tmp_target.exists():
        shutil.rmtree(tmp_target)
    tmp_target.mkdir(parents=True, exist_ok=True)
    partition_columns = [column for column in PARTITION_COLUMNS if column in table.schema.names]
    pq.write_to_dataset(
        table,
        root_path=tmp_target,
        partition_cols=partition_columns,
        compression=compression,
        use_dictionary=True,
        write_statistics=True,
        row_group_size=row_group_size,
    )
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(tmp_target), str(target))
    print(f"Wrote partitioned dataset {target}", flush=True)
    return target


def optimize_stem(data_dir: Path, stem: str, *, compression: str, row_group_size: int) -> None:
    started = time.perf_counter()
    print(f"Optimizing {stem}...", flush=True)
    parquet_path = data_dir / f"{stem}.parquet"
    csv_path = data_dir / f"{stem}.csv"
    if not parquet_path.exists() and not csv_path.exists() and _has_partitioned_dataset(data_dir, stem):
        manifest_path = data_dir / f"{stem}_manifest.csv"
        if manifest_path.exists():
            print(
                f"Skipped {stem}: partitioned dataset already exists and no single-file source is present.",
                flush=True,
            )
            print(f"Using existing manifest {manifest_path}", flush=True)
            return
        raise FileNotFoundError(
            f"{stem} exists only as a partitioned dataset, but no manifest was found at {manifest_path}."
        )
    table = _read_source_table(data_dir, stem)
    table = _with_analysis_scope(table, stem)
    table = _with_typed_dates(table)
    table = _sorted_table(table)
    _write_optimized_single(table, data_dir, stem, compression=compression, row_group_size=row_group_size)
    _write_partitioned_dataset(table, data_dir, stem, compression=compression, row_group_size=row_group_size)
    _write_manifest(table, data_dir, stem)
    elapsed = time.perf_counter() - started
    print(f"Completed {stem} in {elapsed/60:.1f}m", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build optimized dashboard ecozone and forest-community Parquet products.")
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "Results" / "tables" / "dashboard_data"),
        help="Directory containing dashboard ecozone tables.",
    )
    parser.add_argument("--stem", action="append", choices=SEGMENT_STEMS, help="Specific segment table stem to optimize.")
    parser.add_argument("--compression", default="zstd", help="Parquet compression codec.")
    parser.add_argument("--row-group-size", type=int, default=100_000, help="Rows per Parquet row group.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    stems = tuple(args.stem) if args.stem else SEGMENT_STEMS
    for stem in stems:
        optimize_stem(data_dir, stem, compression=args.compression, row_group_size=args.row_group_size)


if __name__ == "__main__":
    main()
