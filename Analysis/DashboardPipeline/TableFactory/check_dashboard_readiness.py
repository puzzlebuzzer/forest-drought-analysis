#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from src.dashboard_data import load_dashboard_data
from src.dashboard_figures import build_timeseries_figure
from src.dashboard_schema import ComparisonConfig


DEFAULT_CHECKS = [
    ComparisonConfig(
        label="overall_s2_north_ndvi_month_p95",
        analysis_scope="overall",
        sensor="s2",
        aoi="north",
        index="ndvi",
        spatial_percentile="p95",
        temporal_agg="month",
        temporal_percentile="p95",
        cloud_threshold=40,
        season_filter="all",
    ),
    ComparisonConfig(
        label="forest_s2_north_ndvi_month_p95_community1",
        analysis_scope="forest_community",
        sensor="s2",
        aoi="north",
        index="ndvi",
        forest_community_code=1,
        spatial_percentile="p95",
        temporal_agg="month",
        temporal_percentile="p95",
        cloud_threshold=40,
        season_filter="all",
    ),
    ComparisonConfig(
        label="forest_s2_north_ndvi_month_p95_all_communities",
        analysis_scope="forest_community",
        sensor="s2",
        aoi="north",
        index="ndvi",
        forest_community_code=None,
        spatial_percentile="p95",
        temporal_agg="month",
        temporal_percentile="p95",
        cloud_threshold=40,
        season_filter="all",
    ),
]


def _timed(label: str, fn):
    started = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - started
    print(f"{label}: {elapsed:.2f}s", flush=True)
    return value, elapsed


def _check_config(bundle, config: ComparisonConfig, year_range: tuple[int, int]) -> dict:
    frame, read_seconds = _timed(f"read {config.label}", lambda: bundle.frame_for_config(config))
    figure, figure_seconds = _timed(
        f"figure {config.label}",
        lambda: build_timeseries_figure(bundle, [config], year_range),
    )
    fig, messages = figure
    result = {
        "label": config.label,
        "config": asdict(config),
        "rows": int(len(frame)),
        "trace_count": int(len(fig.data)),
        "messages": messages,
        "read_seconds": round(read_seconds, 3),
        "figure_seconds": round(figure_seconds, 3),
    }
    if "forest_community_code" in frame.columns:
        result["forest_community_count"] = int(frame["forest_community_code"].dropna().nunique())
    if "year" in frame.columns and not frame.empty:
        result["min_year"] = int(frame["year"].min())
        result["max_year"] = int(frame["year"].max())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dashboard data and plotting smoke checks.")
    parser.add_argument(
        "--data-dir",
        default="SummaryTables/dashboard_data",
        help="Dashboard table directory.",
    )
    parser.add_argument(
        "--output",
        default="SummaryTables/dashboard_data/dashboard_readiness_report.json",
        help="JSON report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_path = Path(args.output).resolve()

    bundle, load_seconds = _timed("load_dashboard_data", lambda: load_dashboard_data(data_dir))
    year_range = bundle.available_year_range()
    report = {
        "data_dir": str(data_dir),
        "load_seconds": round(load_seconds, 3),
        "analysis_scopes": bundle.available_values("analysis_scope"),
        "year_range": list(year_range),
        "forest_community_options": len(bundle.available_values("forest_community_code")),
        "scene_summary_forest_community_manifest_rows": int(len(bundle.scene_summary_forest_community_manifest)),
        "temporal_summary_forest_community_manifest_rows": int(len(bundle.temporal_summary_forest_community_manifest)),
        "checks": [_check_config(bundle, config, year_range) for config in DEFAULT_CHECKS],
    }
    if "row_count" in bundle.temporal_summary_forest_community_manifest.columns:
        report["temporal_summary_forest_community_rows"] = int(
            bundle.temporal_summary_forest_community_manifest["row_count"].sum()
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
