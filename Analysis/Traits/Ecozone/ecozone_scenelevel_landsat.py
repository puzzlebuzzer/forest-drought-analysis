#!/usr/bin/env python3
"""
ecozone_scenelevel_landsat.py
-----------------------------
Build per-scene Landsat NDVI summaries by ecozone for one AOI and year range.

For each cached Landsat NDVI scene, compute ecozone-level p50, p75, p95, and
max values. Write:

- an Excel spreadsheet with one row per scene per ecozone
- a static PNG line plot
- a standalone Bokeh HTML line plot

Default use case:
  python Analysis/Traits/Ecozone/ecozone_scenelevel_landsat.py --aoi north --start-year 2023 --end-year 2023
"""

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

try:
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure, output_file, save
except ImportError:
    ColumnDataSource = None
    HoverTool = None
    figure = None
    output_file = None
    save = None

from src.landsat import load_landsat_ecozone, load_landsat_scenes
from src.paths import project_path

VALID_ECOZONE_CODES = [1, 2, 3]
ECOZONE_LABELS = {1: "Cool", 2: "Intermediate", 3: "Hot"}
ECOZONE_COLORS = {1: "#4E90C8", 2: "#72B063", 3: "#D9534F"}
AOI_DISPLAY = {"north": "GWNF", "south": "Smoky"}
INDEX_OPTIONS = ["NDVI", "NDMI", "EVI"]
SUMMARY_SPECS = [
    ("p50", 50),
    ("p75", 75),
    ("p95", 95),
    ("max", 100),
]
MIN_PIXELS = 100

FIGURES_DIR = project_path("results_figures_landsat_dir")
TABLES_DIR = project_path("results_tables_landsat_dir")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build scene-level Landsat NDVI ecozone summaries."
    )
    parser.add_argument(
        "--aoi",
        choices=sorted(AOI_DISPLAY),
        default="north",
        help="AOI key to process",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2023,
        help="First year to include",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2023,
        help="Last year to include",
    )
    parser.add_argument(
        "--index",
        choices=["ndvi", "ndmi", "evi"],
        help="Index to process. Omit to run all supported indices.",
    )
    parser.add_argument(
        "--plot-zscore-threshold",
        nargs="?",
        type=float,
        const=3.0,
        default=None,
        help=(
            "Plot-only outlier filter applied separately to each ecozone x summary "
            "series. Omit the flag for no filtering. Use the flag with no value for "
            "the default threshold of 3.0, or pass a numeric threshold explicitly. "
            "Spreadsheet rows are unchanged."
        ),
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Recompute the spreadsheet from rasters even if it already exists.",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also generate a static PNG plot. By default only the Bokeh plot is written.",
    )
    return parser.parse_args()


def filter_scenes_by_year(scenes: list[dict], start_year: int, end_year: int) -> list[dict]:
    return [
        scene for scene in scenes
        if start_year <= scene["date"].year <= end_year
    ]


def build_scenelevel_dataframe(aoi: str, index_name: str, scenes: list[dict]) -> pd.DataFrame:
    ecozone_arr, _, _, _ = load_landsat_ecozone(aoi)
    eco_masks = {code: (ecozone_arr == code) for code in VALID_ECOZONE_CODES}

    records: list[dict] = []
    for i, scene in enumerate(scenes, start=1):
        if i == 1 or i % 25 == 0:
            print(f"  Processing scene {i}/{len(scenes)}...", flush=True)

        with rasterio.open(scene["filepath"]) as src:
            data = src.read(1)

        valid = np.isfinite(data)
        for ecozone_code in VALID_ECOZONE_CODES:
            combined = eco_masks[ecozone_code] & valid
            valid_pixels = int(combined.sum())
            if valid_pixels < MIN_PIXELS:
                continue

            pixels = data[combined]
            summaries = {
                "p50": float(np.percentile(pixels, 50)),
                "p75": float(np.percentile(pixels, 75)),
                "p95": float(np.percentile(pixels, 95)),
                "max": float(np.percentile(pixels, 100)),
            }
            record = {
                "AOI": AOI_DISPLAY[aoi],
                "AOI Key": aoi,
                "Index": index_name,
                "Scene Date": scene["date"].date().isoformat(),
                "Year": scene["date"].year,
                "Month": scene["date"].month,
                "Day": scene["date"].day,
                "Ecozone": ECOZONE_LABELS[ecozone_code],
                "Ecozone Code": ecozone_code,
                "Platform": scene.get("platform", "unknown"),
                "Path/Row": scene.get("path_row", ""),
                "Valid Pixels": valid_pixels,
            }
            record.update({name: round(value, 6) for name, value in summaries.items()})
            records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df.sort_values(["Scene Date", "Ecozone Code"]).reset_index(drop=True)


def filter_plot_series(values: pd.Series, zscore_threshold: float) -> pd.Series:
    if zscore_threshold is None:
        return pd.Series(True, index=values.index)
    numeric = pd.to_numeric(values, errors="coerce")
    mean = numeric.mean()
    std = numeric.std(ddof=0)
    if pd.isna(mean) or pd.isna(std) or std == 0:
        return pd.Series(True, index=values.index)
    zscores = ((numeric - mean) / std).abs()
    return zscores <= zscore_threshold


def zscore_suffix(zscore_threshold: float) -> str:
    if zscore_threshold is None:
        return "znone"
    label = str(zscore_threshold).replace(".", "p")
    return f"z{label}"


def scenelevel_figure_dir(aoi: str, index_name: str) -> Path:
    return FIGURES_DIR / "seasonal_curves" / "landsat" / aoi / "by_ecozone" / index_name.lower()


def build_png(df: pd.DataFrame, aoi: str, index_name: str, out_path: Path, zscore_threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(15, 8))
    plot_df = df.copy()
    plot_df["Scene Date"] = pd.to_datetime(plot_df["Scene Date"])

    for ecozone_code in VALID_ECOZONE_CODES:
        ecozone_df = plot_df[plot_df["Ecozone Code"] == ecozone_code].sort_values("Scene Date")
        color = ECOZONE_COLORS[ecozone_code]
        for summary_name, _ in SUMMARY_SPECS:
            series_mask = filter_plot_series(ecozone_df[summary_name], zscore_threshold)
            series_df = ecozone_df.loc[series_mask].copy()
            label = f"{ECOZONE_LABELS[ecozone_code]} {summary_name}"
            ax.plot(
                series_df["Scene Date"],
                series_df[summary_name],
                color=color,
                linestyle="-",
                linewidth=1.8,
                marker="o",
                markersize=3,
                alpha=0.9,
                label=label,
            )

    filter_note = "no plot z-filter" if zscore_threshold is None else f"plot filter: |z| <= {zscore_threshold:g}"
    ax.set_title(
        f"Landsat {index_name} Scene-Level Ecozone Summaries — {AOI_DISPLAY[aoi]} "
        f"({int(plot_df['Year'].min())}-{int(plot_df['Year'].max())}, {filter_note})"
    )
    ax.set_xlabel("Scene date")
    ax.set_ylabel(index_name)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    ax.legend(ncol=3, fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def build_bokeh(df: pd.DataFrame, aoi: str, index_name: str, out_path: Path, zscore_threshold: float) -> None:
    if figure is None:
        raise SystemExit("Bokeh is not installed. Install it with: pip install bokeh")

    plot_df = df.copy()
    plot_df["Scene Date"] = pd.to_datetime(plot_df["Scene Date"])

    p = figure(
        title=(
            f"Landsat {index_name} Scene-Level Ecozone Summaries — {AOI_DISPLAY[aoi]} "
            f"({int(plot_df['Year'].min())}-{int(plot_df['Year'].max())}, "
            f"{'no plot z-filter' if zscore_threshold is None else f'plot filter: |z| <= {zscore_threshold:g}'})"
        ),
        x_axis_type="datetime",
        width=1400,
        height=750,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )
    p.xaxis.axis_label = "Scene date"
    p.yaxis.axis_label = index_name
    p.grid.grid_line_alpha = 0.2
    p.toolbar.logo = None

    hover = HoverTool(
        tooltips=[
            ("Date", "@scene_date{%F}"),
            ("Ecozone", "@ecozone"),
            ("Summary", "@summary_label"),
            ("Value", "@value{0.0000}"),
            ("Platform", "@platform"),
            ("Path/Row", "@path_row"),
            ("Valid Pixels", "@valid_pixels"),
        ],
        formatters={"@scene_date": "datetime"},
    )
    p.add_tools(hover)

    for ecozone_code in VALID_ECOZONE_CODES:
        ecozone_df = plot_df[plot_df["Ecozone Code"] == ecozone_code].sort_values("Scene Date")
        color = ECOZONE_COLORS[ecozone_code]
        for summary_name, _ in SUMMARY_SPECS:
            series_mask = filter_plot_series(ecozone_df[summary_name], zscore_threshold)
            series_df = ecozone_df.loc[series_mask].copy()
            series_df = pd.DataFrame(
                {
                    "scene_date": series_df["Scene Date"],
                    "value": series_df[summary_name],
                    "ecozone": series_df["Ecozone"],
                    "summary_label": [summary_name] * len(series_df),
                    "platform": series_df["Platform"],
                    "path_row": series_df["Path/Row"],
                    "valid_pixels": series_df["Valid Pixels"],
                }
            )
            source = ColumnDataSource(series_df)
            legend_label = f"{ECOZONE_LABELS[ecozone_code]} {summary_name}"
            p.line(
                "scene_date",
                "value",
                source=source,
                line_width=2,
                color=color,
                alpha=0.9,
                legend_label=legend_label,
            )
            p.scatter(
                "scene_date",
                "value",
                source=source,
                size=5,
                color=color,
                alpha=0.75,
                legend_label=legend_label,
            )

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    output_file(out_path, title=p.title.text)
    save(p)
    print(f"Saved: {out_path}")


def main() -> None:
    args = parse_args()
    if args.end_year < args.start_year:
        raise SystemExit("--end-year must be greater than or equal to --start-year")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    year_label = (
        f"{args.start_year}"
        if args.start_year == args.end_year
        else f"{args.start_year}_{args.end_year}"
    )
    indices = [args.index.upper()] if args.index else INDEX_OPTIONS

    for index_name in indices:
        stem = f"landsat_{index_name.lower()}_seasonalcurves_ecozone_{args.aoi}_{year_label}"
        table_path = TABLES_DIR / f"{stem}.xlsx"
        plot_dir = scenelevel_figure_dir(args.aoi, index_name)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_stem = (
            f"{year_label}_{zscore_suffix(args.plot_zscore_threshold)}"
            f"_landsat_{args.aoi}_seasonalcurves_byecozone_{index_name.lower()}"
        )
        png_path = plot_dir / f"{plot_stem}.png"
        bokeh_path = plot_dir / f"{plot_stem}.bokeh.html"

        if table_path.exists() and not args.force_rebuild:
            print(f"[{index_name}] Using existing spreadsheet: {table_path}")
            df = pd.read_excel(table_path)
        else:
            print(f"[{index_name}] Loading Landsat scenes for {args.aoi}...")
            all_scenes = load_landsat_scenes(args.aoi, index_name)
            scenes = filter_scenes_by_year(all_scenes, args.start_year, args.end_year)
            print(f"[{index_name}]   {len(scenes)} scenes matched {args.start_year}-{args.end_year}")

            if not scenes:
                print(f"[{index_name}] Skipping — no matching scenes found for the requested AOI/year range.")
                continue

            df = build_scenelevel_dataframe(args.aoi, index_name, scenes)
            if df.empty:
                print(f"[{index_name}] Skipping — no ecozone summaries were produced.")
                continue

            df.to_excel(table_path, index=False)
            print(f"[{index_name}] Saved: {table_path}")

        if args.png:
            build_png(df, args.aoi, index_name, png_path, args.plot_zscore_threshold)
        build_bokeh(df, args.aoi, index_name, bokeh_path, args.plot_zscore_threshold)


if __name__ == "__main__":
    main()
