#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.dashboard_data import load_dashboard_data
from src.dashboard_figures import build_growing_season_figure, build_timeseries_figure
from src.dashboard_schema import ComparisonConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export static dashboard figures from precomputed CSV summaries.")
    parser.add_argument(
        "--data-dir",
        default="Results/tables/dashboard_data",
        help="Directory containing scene_summary.csv and temporal_summary.csv",
    )
    parser.add_argument("--figure-type", choices=["timeseries", "growing-season"], required=True)
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--selected-year", type=int, help="Highlighted year for growing-season exports")
    parser.add_argument(
        "--comparison-json",
        action="append",
        required=True,
        help=(
            "JSON object describing one comparison configuration. "
            'Example: {"sensor":"s2","aoi":"north","index":"ndvi","spatial_percentile":"p95",'
            '"temporal_agg":"scene","temporal_percentile":"none","cloud_threshold":40,'
            '"season_filter":"all","label":"Sentinel north"}'
        ),
    )
    return parser.parse_args()


def _comparison_configs(raw_json_items: list[str]) -> list[ComparisonConfig]:
    configs = []
    for raw_item in raw_json_items:
        payload = json.loads(raw_item)
        payload.setdefault("label", "")
        configs.append(ComparisonConfig(**payload))
    return configs


def main() -> None:
    args = parse_args()
    bundle = load_dashboard_data(args.data_dir)
    configs = _comparison_configs(args.comparison_json)
    year_min, year_max = bundle.available_year_range()
    year_range = (args.start_year or year_min, args.end_year or year_max)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.figure_type == "timeseries":
        figure, messages = build_timeseries_figure(bundle, configs, year_range)
        if not figure.data:
            raise SystemExit("No timeseries traces could be generated. Check the filters and input CSVs.")
        for message in messages:
            print(message)
    else:
        selected_year = args.selected_year or year_range[1]
        figure, message = build_growing_season_figure(bundle, configs[0], selected_year)
        if message:
            raise SystemExit(message)

    try:
        figure.write_image(output_path, width=1400, height=800, scale=2)
    except ValueError as exc:
        raise SystemExit("Static image export requires Plotly image support such as `kaleido`.") from exc
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
