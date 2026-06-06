#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from src.paths import PROJECT_ROOT

ECOZONE_SCENE_STEM = "scene_summary_ecozone"
ECOZONE_TEMPORAL_STEM = "temporal_summary_ecozone"


COLUMN_TYPES = {
    "analysis_scope": pa.string(),
    "sensor": pa.string(),
    "aoi": pa.string(),
    "index": pa.string(),
    "ecozone_code": pa.int16(),
    "ecozone_label": pa.string(),
    "date": pa.string(),
    "year": pa.int16(),
    "doy": pa.int16(),
    "growing_season_day": pa.float64(),
    "season_filter": pa.string(),
    "temporal_agg": pa.string(),
    "temporal_percentile": pa.string(),
    "spatial_percentile": pa.string(),
    "cloud_threshold": pa.float64(),
    "cloud_percent": pa.float64(),
    "pixel_mask_id": pa.string(),
    "pixel_mask_description": pa.string(),
    "pixel_mask_version": pa.string(),
    "n_pixels": pa.int64(),
    "valid_pixel_fraction": pa.float64(),
    "n_scenes": pa.int32(),
    "value": pa.float64(),
    "source_file_or_composite_id": pa.string(),
    "time_bin_label": pa.string(),
    "time_bin_start": pa.string(),
    "time_bin_end": pa.string(),
    "month_day_label": pa.string(),
}


def _header_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return handle.readline().strip().split(",")


def csv_to_parquet(csv_path: Path, parquet_path: Path, *, block_size_mb: int, compression: str) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV input does not exist: {csv_path}")

    column_types = {column: COLUMN_TYPES[column] for column in _header_columns(csv_path) if column in COLUMN_TYPES}
    read_options = pacsv.ReadOptions(block_size=block_size_mb * 1024 * 1024)
    convert_options = pacsv.ConvertOptions(column_types=column_types, strings_can_be_null=True)
    reader = pacsv.open_csv(csv_path, read_options=read_options, convert_options=convert_options)

    tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    tmp_path.unlink(missing_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    rows_written = 0
    started = time.perf_counter()
    try:
        for batch_idx, batch in enumerate(reader, start=1):
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(
                    tmp_path,
                    table.schema,
                    compression=compression,
                    use_dictionary=True,
                    write_statistics=True,
                )
            writer.write_table(table)
            rows_written += table.num_rows
            if batch_idx == 1 or batch_idx % 10 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"  {csv_path.name}: batch={batch_idx} rows={rows_written:,} elapsed={elapsed/60:.1f}m",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()

    tmp_path.replace(parquet_path)
    elapsed = time.perf_counter() - started
    print(f"Wrote {parquet_path} rows={rows_written:,} elapsed={elapsed/60:.1f}m", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert dashboard CSV tables to Parquet without loading them fully into memory.")
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "SummaryTables" / "dashboard_data"),
        help="Directory containing dashboard CSV tables.",
    )
    parser.add_argument(
        "--stem",
        action="append",
        default=None,
        help="Table stem to convert, for example temporal_summary_ecozone. May be passed multiple times.",
    )
    parser.add_argument(
        "--ecozone-only",
        action="store_true",
        help="Convert scene_summary_ecozone and temporal_summary_ecozone.",
    )
    parser.add_argument("--block-size-mb", type=int, default=64, help="PyArrow CSV reader block size.")
    parser.add_argument("--compression", default="zstd", help="Parquet compression codec.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    stems = args.stem
    if args.ecozone_only or stems is None:
        stems = [ECOZONE_SCENE_STEM, ECOZONE_TEMPORAL_STEM]

    for stem in stems:
        csv_to_parquet(
            data_dir / f"{stem}.csv",
            data_dir / f"{stem}.parquet",
            block_size_mb=args.block_size_mb,
            compression=args.compression,
        )


if __name__ == "__main__":
    main()
