"""
src/ecozone_scenelevel.py
-------------------------
Shared scene-level ecozone analysis engine for Landsat and Sentinel scripts.
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

from src.paths import project_path

VALID_ECOZONE_CODES = [1, 2, 3]
ECOZONE_LABELS = {1: "Cool", 2: "Intermediate", 3: "Hot"}
ECOZONE_COLORS = {1: "#4E90C8", 2: "#72B063", 3: "#D9534F"}
AOI_DISPLAY = {"north": "GWNF", "south": "Smoky"}
INDEX_OPTIONS = ["NDVI", "NDMI", "EVI"]
DEFAULT_PERCENTILES = [50, 75, 95, 100]
MIN_PIXELS = 100
FIGURES_DIR = project_path("results_figures_dir")


def build_parser(sensor_label: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Build scene-level {sensor_label} ecozone summaries."
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
        "--percentiles",
        nargs="+",
        type=float,
        default=DEFAULT_PERCENTILES,
        help="Space-separated percentile list, e.g. --percentiles 50 75 95 100.",
    )
    parser.add_argument(
        "--z",
        nargs="?",
        type=float,
        const=1.5,
        default=None,
        help=(
            "Plot-only outlier filter applied separately to each ecozone x summary "
            "series. Omit the flag for no filtering. Use the flag with no value for "
            "the default threshold of 1.5, or pass a numeric threshold explicitly. "
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
    return parser


def filter_scenes_by_year(scenes: list[dict], start_year: int, end_year: int) -> list[dict]:
    return [scene for scene in scenes if start_year <= scene["date"].year <= end_year]


def percentile_label(percentile: float) -> str:
    if float(percentile).is_integer():
        return f"p{int(percentile)}"
    return f"p{str(percentile).replace('.', 'p')}"


def normalize_percentiles(percentiles: list[float]) -> list[float]:
    normalized: list[float] = []
    for percentile in percentiles:
        if percentile < 0 or percentile > 100:
            raise SystemExit("All percentiles must be between 0 and 100.")
        normalized.append(float(percentile))
    if not normalized:
        raise SystemExit("Provide at least one percentile.")
    return sorted(dict.fromkeys(normalized))


def build_scenelevel_dataframe(
    aoi: str,
    index_name: str,
    scenes: list[dict],
    percentiles: list[float],
    load_ecozone,
    metadata_columns: list[tuple[str, str]],
) -> pd.DataFrame:
    ecozone_arr, _, _, _ = load_ecozone(aoi)
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
                "Valid Pixels": valid_pixels,
            }
            for key, label in metadata_columns:
                record[label] = scene.get(key, "")
            for percentile in percentiles:
                record[percentile_label(percentile)] = round(float(np.percentile(pixels, percentile)), 6)
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


def percentiles_suffix(percentiles: list[float]) -> str:
    return "pct" + "-".join(percentile_label(percentile) for percentile in percentiles)


def scenelevel_figure_dir(sensor_key: str, aoi: str, index_name: str) -> Path:
    return FIGURES_DIR / "seasonal_curves" / "by_ecozone" / sensor_key / aoi / index_name.lower()


def build_png(
    df: pd.DataFrame,
    aoi: str,
    index_name: str,
    out_path: Path,
    zscore_threshold: float,
    percentiles: list[float],
    sensor_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(15, 8))
    plot_df = df.copy()
    plot_df["Scene Date"] = pd.to_datetime(plot_df["Scene Date"])

    for ecozone_code in VALID_ECOZONE_CODES:
        ecozone_df = plot_df[plot_df["Ecozone Code"] == ecozone_code].sort_values("Scene Date")
        color = ECOZONE_COLORS[ecozone_code]
        for percentile in percentiles:
            summary_name = percentile_label(percentile)
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
        f"{sensor_label} {index_name} Scene-Level Ecozone Summaries — {AOI_DISPLAY[aoi]} "
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


def build_bokeh(
    df: pd.DataFrame,
    aoi: str,
    index_name: str,
    out_path: Path,
    zscore_threshold: float,
    percentiles: list[float],
    sensor_label: str,
    hover_fields: list[tuple[str, str, str]],
) -> None:
    if figure is None:
        raise SystemExit("Bokeh is not installed. Install it with: pip install bokeh")

    plot_df = df.copy()
    plot_df["Scene Date"] = pd.to_datetime(plot_df["Scene Date"])

    p = figure(
        title=(
            f"{sensor_label} {index_name} Scene-Level Ecozone Summaries — {AOI_DISPLAY[aoi]} "
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

    hover_tooltips = [
        ("Date", "@scene_date{%F}"),
        ("Ecozone", "@ecozone"),
        ("Summary", "@summary_label"),
        ("Value", "@value{0.0000}"),
    ]
    for _, label, formatter in hover_fields:
        hover_tooltips.append((label, formatter))
    hover = HoverTool(
        tooltips=hover_tooltips,
        formatters={"@scene_date": "datetime"},
    )
    p.add_tools(hover)

    for ecozone_code in VALID_ECOZONE_CODES:
        ecozone_df = plot_df[plot_df["Ecozone Code"] == ecozone_code].sort_values("Scene Date")
        color = ECOZONE_COLORS[ecozone_code]
        for percentile in percentiles:
            summary_name = percentile_label(percentile)
            series_mask = filter_plot_series(ecozone_df[summary_name], zscore_threshold)
            series_df = ecozone_df.loc[series_mask].copy()
            source_data = {
                "scene_date": series_df["Scene Date"],
                "value": series_df[summary_name],
                "ecozone": series_df["Ecozone"],
                "summary_label": [summary_name] * len(series_df),
            }
            for _, label, _ in hover_fields:
                source_key = label.lower().replace(" ", "_").replace("/", "_")
                source_data[source_key] = series_df[label]
            source = ColumnDataSource(pd.DataFrame(source_data))
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


def run_scenelevel_analysis(
    *,
    sensor_key: str,
    sensor_label: str,
    tables_dir: Path,
    load_scenes,
    load_ecozone,
    metadata_columns: list[tuple[str, str]],
) -> None:
    args = build_parser(sensor_label).parse_args()
    if args.end_year < args.start_year:
        raise SystemExit("--end-year must be greater than or equal to --start-year")
    percentiles = normalize_percentiles(args.percentiles)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    year_label = (
        f"{args.start_year}"
        if args.start_year == args.end_year
        else f"{args.start_year}_{args.end_year}"
    )
    indices = [args.index.upper()] if args.index else INDEX_OPTIONS
    hover_fields = [(key, label, f"@{label.lower().replace(' ', '_').replace('/', '_')}") for key, label in metadata_columns]
    hover_fields.append(("valid_pixels", "Valid Pixels", "@valid_pixels"))

    for index_name in indices:
        pct_suffix = percentiles_suffix(percentiles)
        stem = f"{sensor_key}_{index_name.lower()}_seasonalcurves_ecozone_{args.aoi}_{year_label}_{pct_suffix}"
        table_path = tables_dir / f"{stem}.xlsx"
        plot_dir = scenelevel_figure_dir(sensor_key, args.aoi, index_name)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_stem = (
            f"{year_label}_{pct_suffix}_{zscore_suffix(args.z)}"
            f"_seasonalcurves_byecozone_{sensor_key}_{args.aoi}_{index_name.lower()}"
        )
        png_path = plot_dir / f"{plot_stem}.png"
        bokeh_path = plot_dir / f"{plot_stem}.bokeh.html"

        if table_path.exists() and not args.force_rebuild:
            print(f"[{index_name}] Using existing spreadsheet: {table_path}")
            df = pd.read_excel(table_path)
        else:
            print(f"[{index_name}] Loading {sensor_label} scenes for {args.aoi}...")
            all_scenes = load_scenes(args.aoi, index_name)
            scenes = filter_scenes_by_year(all_scenes, args.start_year, args.end_year)
            print(f"[{index_name}]   {len(scenes)} scenes matched {args.start_year}-{args.end_year}")

            if not scenes:
                print(f"[{index_name}] Skipping — no matching scenes found for the requested AOI/year range.")
                continue

            df = build_scenelevel_dataframe(
                args.aoi,
                index_name,
                scenes,
                percentiles,
                load_ecozone,
                metadata_columns,
            )
            if df.empty:
                print(f"[{index_name}] Skipping — no ecozone summaries were produced.")
                continue

            df.to_excel(table_path, index=False)
            print(f"[{index_name}] Saved: {table_path}")

        if args.png:
            build_png(df, args.aoi, index_name, png_path, args.z, percentiles, sensor_label)
        build_bokeh(df, args.aoi, index_name, bokeh_path, args.z, percentiles, sensor_label, hover_fields)
