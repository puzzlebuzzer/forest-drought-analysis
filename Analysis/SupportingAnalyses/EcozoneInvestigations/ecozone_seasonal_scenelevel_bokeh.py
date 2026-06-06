#!/usr/bin/env python3
"""
ecozone_seasonal_scenelevel_bokeh.py
------------------------------------
Build standalone Bokeh HTML versions of the scene-level ecozone seasonal plots
from the existing spreadsheet output, without re-reading rasters or
recomputing percentiles.

Inputs:
  Results/tables/ecozone_seasonal_scenelevel_summary.xlsx

Outputs:
  Results/figures/ecozone_<index>_seasonal_scenelevel.bokeh.html

Run from the Python directory:
  python Analysis/SupportingAnalyses/EcozoneInvestigations/ecozone_seasonal_scenelevel_bokeh.py
  python Analysis/SupportingAnalyses/EcozoneInvestigations/ecozone_seasonal_scenelevel_bokeh.py --indices NDVI NDMI
  python Analysis/SupportingAnalyses/EcozoneInvestigations/ecozone_seasonal_scenelevel_bokeh.py --points-only
"""

import argparse

import pandas as pd

try:
    from bokeh.layouts import gridplot
    from bokeh.models import Band, ColumnDataSource, HoverTool
    from bokeh.plotting import figure, output_file, save
except ImportError as exc:
    raise SystemExit(
        "Bokeh is not installed. Install it with: pip install bokeh"
    ) from exc

from src.paths import project_path

AOIS = [("north", "GW National Forest"), ("south", "Great Smoky Mtns")]
INDICES = ["NDVI", "NDMI", "EVI"]
ECOZONE_CODES = [1, 2, 3]
ECOZONE_LABELS = {1: "Cool", 2: "Intermediate", 3: "Hot"}
ECOZONE_COLORS = {1: "#4E90C8", 2: "#72B063", 3: "#D9534F"}

TABLE_PATH = project_path("results_tables_dir") / "ecozone_seasonal_scenelevel_summary.xlsx"
FIGURES_DIR = project_path("results_figures_dir")


def build_panel(df: pd.DataFrame, index_name: str, aoi_key: str, aoi_label: str, include_band: bool):
    panel_df = df[(df["AOI Key"] == aoi_key) & (df["Index"] == index_name)].copy()
    panel_df = panel_df.sort_values("Scene Date")

    p = figure(
        title=aoi_label,
        x_axis_type="datetime",
        width=680,
        height=500,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )
    p.xaxis.axis_label = "Scene date"
    p.yaxis.axis_label = f"Scene-level p95 {index_name}"
    p.grid.grid_line_alpha = 0.2
    p.toolbar.logo = None

    hover = HoverTool(
        tooltips=[
            ("Date", "@scene_date{%F}"),
            ("Ecozone", "@ecozone"),
            ("Month", "@month_name"),
            ("p95", "@p95{0.0000}"),
            ("p100", "@p100{0.0000}"),
        ],
        formatters={"@scene_date": "datetime"},
    )
    p.add_tools(hover)

    for ecozone_code in ECOZONE_CODES:
        subset = panel_df[panel_df["Ecozone Code"] == ecozone_code].copy()
        if subset.empty:
            continue

        subset = subset.rename(
            columns={
                "Scene Date": "scene_date",
                "Month Name": "month_name",
                "Ecozone": "ecozone",
                "p100 (max)": "p100",
            }
        )
        source = ColumnDataSource(subset)
        color = ECOZONE_COLORS[ecozone_code]
        legend_label = ECOZONE_LABELS[ecozone_code]

        if include_band:
            band = Band(
                base="scene_date",
                lower="p95",
                upper="p100",
                source=source,
                level="underlay",
                fill_alpha=0.12,
                fill_color=color,
                line_alpha=0,
            )
            p.add_layout(band)

        line = p.line(
            "scene_date",
            "p95",
            source=source,
            line_width=1.6,
            color=color,
            alpha=0.85,
            legend_label=legend_label,
        )
        p.circle(
            "scene_date",
            "p95",
            source=source,
            size=4,
            color=color,
            alpha=0.8,
            legend_label=legend_label,
        )

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    return p


def build_html(df: pd.DataFrame, index_name: str, include_band: bool, out_path):
    left = build_panel(df, index_name, AOIS[0][0], AOIS[0][1], include_band)
    right = build_panel(df, index_name, AOIS[1][0], AOIS[1][1], include_band)

    layout = gridplot([[left, right]], sizing_mode="scale_width", merge_tools=False)
    output_file(
        out_path,
        title=(
            f"Seasonal {index_name} scene-level p95 by ecozone"
            + (" with p95-p100 band" if include_band else " points/lines only")
        ),
    )
    save(layout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Bokeh scene-level ecozone seasonal figures from existing spreadsheet data")
    parser.add_argument(
        "--indices",
        nargs="+",
        choices=INDICES,
        default=INDICES,
        help="Indices to render",
    )
    parser.add_argument(
        "--points-only",
        action="store_true",
        help="Omit the p95 to p100 band and draw only the p95 line/markers",
    )
    args = parser.parse_args()

    if not TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Scene-level seasonal spreadsheet not found at {TABLE_PATH}"
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(TABLE_PATH)
    df["Scene Date"] = pd.to_datetime(df["Scene Date"])

    for index_name in args.indices:
        index_df = df[df["Index"] == index_name].copy()
        if index_df.empty:
            print(f"[{index_name}] Skipping — no rows found in spreadsheet.")
            continue

        suffix = ".bokeh.noband.html" if args.points_only else ".bokeh.html"
        out_path = FIGURES_DIR / f"ecozone_{index_name.lower()}_seasonal_scenelevel{suffix}"
        build_html(index_df, index_name=index_name, include_band=not args.points_only, out_path=out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
