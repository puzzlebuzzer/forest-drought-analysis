#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.aoi import valid_aois
from src.paths import PROJECT_ROOT, project_path


INDICES = ["ndmi", "ndvi", "evi"]
PRIMARY_INDEX = "ndmi"
GROWING_SEASON_DEFINITION = "Team-defined May 15-Sep 15 growing season."
NO_EXTERNAL_DATA_NOTE = "Classification based only on vegetation-index summaries; no external climate data used."
CLASSIFICATION_METHOD = "AOI/sensor/index percentile rank: top 20% wet/canopy-moist, bottom 20% dry/canopy-stressed."
CLASS_COLORS = {
    "wet/canopy-moist": "#2f7ebc",
    "neutral": "#8a8a8a",
    "dry/canopy-stressed": "#c46a3a",
}
SENSOR_LABELS = {"ls": "Landsat", "s2": "Sentinel-2"}
AOI_LABELS = {
    "north": "George Washington National Forest / Virginia AOI",
    "south": "Smoky Mountains / Nantahala-region AOI",
}


@dataclass(frozen=True)
class SourceSelection:
    path: Path
    source_table: str


def _existing_table(path: Path) -> Path | None:
    parquet_path = path.with_suffix(".parquet")
    csv_path = path.with_suffix(".csv")
    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    return None


def default_source_table() -> SourceSelection:
    dashboard_dir = project_path("results_tables_dir") / "dashboard_data"
    candidates = [
        dashboard_dir / "temporal_summary",
        dashboard_dir / "growing_season_summary",
        dashboard_dir / "scene_summary",
    ]
    for stem in candidates:
        table_path = _existing_table(stem)
        if table_path is not None:
            return SourceSelection(table_path, table_path.name)
    raise FileNotFoundError(
        "No dashboard vegetation summary table found. Expected one of "
        "temporal_summary, growing_season_summary, or scene_summary as CSV/parquet "
        f"under {dashboard_dir}."
    )


def read_source_table(path: Path) -> pd.DataFrame:
    columns = [
        "sensor",
        "aoi",
        "index",
        "date",
        "year",
        "doy",
        "season_filter",
        "temporal_agg",
        "temporal_percentile",
        "spatial_percentile",
        "cloud_threshold",
        "n_scenes",
        "value",
    ]
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            available_columns = pq.ParquetFile(path).schema.names
        except ImportError:
            available_columns = columns
        usecols = [column for column in columns if column in available_columns]
        frame = pd.read_parquet(path, columns=usecols)
    else:
        header = pd.read_csv(path, nrows=0).columns
        usecols = [column for column in columns if column in header]
        frame = pd.read_csv(path, usecols=usecols)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def _is_growing_season(frame: pd.DataFrame) -> pd.Series:
    if "season_filter" in frame.columns:
        season = frame["season_filter"].astype(str).str.lower()
        growing = season.eq("growing")
        if growing.any():
            return growing
    if "date" not in frame.columns:
        raise ValueError("Source table has neither season_filter='growing' nor a date column for May 15-Sep 15 filtering.")
    month_day = frame["date"].dt.month * 100 + frame["date"].dt.day
    return month_day.between(515, 915)


def _filter_if_column(frame: pd.DataFrame, column: str, value) -> pd.DataFrame:
    if value is None or column not in frame.columns:
        return frame
    return frame[frame[column].astype(str).str.lower() == str(value).lower()]


def prepare_observations(
    frame: pd.DataFrame,
    *,
    temporal_agg: str,
    temporal_percentile: str,
    spatial_percentile: str,
    cloud_threshold: int | None,
) -> pd.DataFrame:
    required = {"sensor", "aoi", "index", "year", "value"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Source table is missing required columns: {sorted(missing)}")

    data = frame.copy()
    data["sensor"] = data["sensor"].astype(str).str.lower()
    data["aoi"] = data["aoi"].astype(str).str.lower()
    data["index"] = data["index"].astype(str).str.lower()
    data = data[data["aoi"].isin(valid_aois())]
    data = data[data["index"].isin(INDICES)]
    data = data[_is_growing_season(data)]
    data = _filter_if_column(data, "temporal_agg", temporal_agg)
    data = _filter_if_column(data, "temporal_percentile", temporal_percentile)
    data = _filter_if_column(data, "spatial_percentile", spatial_percentile)
    if cloud_threshold is not None and "cloud_threshold" in data.columns:
        data = data[pd.to_numeric(data["cloud_threshold"], errors="coerce") == cloud_threshold]
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data[np.isfinite(data["value"])]
    if data.empty:
        raise ValueError(
            "No vegetation-index observations remained after filtering. "
            "Try different temporal/spatial percentile or cloud-threshold options."
        )
    return data


def annual_summary(observations: pd.DataFrame, method: str) -> pd.DataFrame:
    method = method.lower()
    if method in ("median", "p50"):
        agg_func = "median"
        method_label = "median across growing-season bins"
    elif method in ("p75", "p95"):
        quantile = float(method[1:]) / 100.0
        agg_func = lambda values: values.quantile(quantile)
        method_label = f"{method} across growing-season bins"
    else:
        raise ValueError("--annual-aggregation must be one of median, p50, p75, or p95")

    annual = (
        observations.groupby(["aoi", "sensor", "index", "year"], dropna=False)
        .agg(
            annual_index_value=("value", agg_func),
            n_observations=("value", "count"),
            n_scenes=("n_scenes", "sum") if "n_scenes" in observations.columns else ("value", "count"),
        )
        .reset_index()
    )
    annual["annual_aggregation_method"] = method_label
    annual["growing_season_definition"] = GROWING_SEASON_DEFINITION
    return annual


def _rank_percentile(series: pd.Series) -> pd.Series:
    n = int(series.notna().sum())
    if n <= 1:
        return pd.Series([50.0 if pd.notna(value) else np.nan for value in series], index=series.index)
    ranks = series.rank(method="average", ascending=True)
    return (ranks - 1.0) / (n - 1.0) * 100.0


def classify_annual(annual: pd.DataFrame) -> pd.DataFrame:
    classified = annual.copy()
    group_cols = ["aoi", "sensor", "index"]
    stats = (
        classified.groupby(group_cols)["annual_index_value"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "long_term_mean", "std": "long_term_std"})
        .reset_index()
    )
    classified = classified.merge(stats, on=group_cols, how="left")
    classified["index_anomaly"] = classified["annual_index_value"] - classified["long_term_mean"]
    classified["index_zscore"] = classified["index_anomaly"] / classified["long_term_std"].replace(0, np.nan)
    classified["index_percentile"] = classified.groupby(group_cols)["annual_index_value"].transform(_rank_percentile)
    classified["index_rank"] = (
        classified.groupby(group_cols)["annual_index_value"]
        .rank(method="first", ascending=False)
        .astype("Int64")
    )
    classified["classification"] = np.select(
        [
            classified["index_percentile"] >= 80.0,
            classified["index_percentile"] <= 20.0,
        ],
        ["wet/canopy-moist", "dry/canopy-stressed"],
        default="neutral",
    )
    classified["classification_method"] = CLASSIFICATION_METHOD
    classified["notes"] = np.where(
        classified["index"].eq(PRIMARY_INDEX),
        f"Primary internal moisture-response index. {NO_EXTERNAL_DATA_NOTE}",
        f"Supporting vegetation-response index. {NO_EXTERNAL_DATA_NOTE}",
    )
    return classified


def output_columns() -> list[str]:
    return [
        "aoi",
        "sensor",
        "index",
        "year",
        "growing_season_definition",
        "annual_index_value",
        "annual_aggregation_method",
        "index_anomaly",
        "index_zscore",
        "index_percentile",
        "index_rank",
        "classification",
        "classification_method",
        "n_observations",
        "source_table",
        "notes",
    ]


def write_outputs(classified: pd.DataFrame, output_csv: Path, annual_output: Path | None) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    classified[output_columns()].sort_values(["aoi", "sensor", "index", "year"]).to_csv(output_csv, index=False)
    print(f"Wrote {output_csv} rows={len(classified):,}")
    if annual_output is not None:
        annual_output.parent.mkdir(parents=True, exist_ok=True)
        if annual_output.suffix == ".parquet":
            classified.to_parquet(annual_output, index=False)
        else:
            classified.to_csv(annual_output, index=False)
        print(f"Wrote {annual_output} rows={len(classified):,}")


def print_validation(classified: pd.DataFrame, min_years: int, min_observations: int) -> None:
    group_cols = ["aoi", "sensor", "index"]
    year_counts = classified.groupby(group_cols)["year"].nunique()
    sparse_groups = year_counts[year_counts < min_years]
    if not sparse_groups.empty:
        print(f"Warning: some AOI/sensor/index groups have fewer than {min_years} years:")
        print(sparse_groups.to_string())

    sparse_years = classified[classified["n_observations"] < min_observations]
    if not sparse_years.empty:
        print(f"Warning: some annual summaries have fewer than {min_observations} growing-season observations:")
        print(sparse_years[["aoi", "sensor", "index", "year", "n_observations"]].to_string(index=False))

    print("Classification counts by AOI/sensor/index:")
    counts = classified.groupby(group_cols + ["classification"])["year"].count().unstack(fill_value=0)
    print(counts.to_string())

    print("Wet/dry year lists by AOI/sensor/index:")
    subset = classified[classified["classification"].isin(["wet/canopy-moist", "dry/canopy-stressed"])]
    for (aoi, sensor, index, classification), rows in subset.groupby(group_cols + ["classification"]):
        years = ", ".join(str(int(year)) for year in sorted(rows["year"]))
        print(f"  {aoi} {sensor} {index} {classification}: {years}")


def plot_classification(classified: pd.DataFrame, output_path: Path) -> None:
    aois = [aoi for aoi in valid_aois() if aoi in set(classified["aoi"])]
    indices = [index for index in INDICES if index in set(classified["index"])]
    fig = make_subplots(
        rows=len(aois),
        cols=len(indices),
        shared_yaxes=False,
        subplot_titles=[f"{AOI_LABELS.get(aoi, aoi)} - {index.upper()}" for aoi in aois for index in indices],
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )
    marker_symbols = {"ls": "circle", "s2": "diamond"}
    for row_idx, aoi in enumerate(aois, start=1):
        for col_idx, index in enumerate(indices, start=1):
            panel = classified[(classified["aoi"] == aoi) & (classified["index"] == index)]
            for sensor, rows in panel.groupby("sensor"):
                rows = rows.sort_values("year")
                fig.add_trace(
                    go.Scatter(
                        x=rows["year"],
                        y=rows["index_anomaly"],
                        mode="lines+markers",
                        name=f"{SENSOR_LABELS.get(sensor, sensor)} {index.upper()}",
                        legendgroup=f"{sensor}-{index}",
                        marker={
                            "color": [CLASS_COLORS.get(value, "#444444") for value in rows["classification"]],
                            "symbol": marker_symbols.get(sensor, "circle"),
                            "size": 8,
                            "line": {"color": "white", "width": 0.5},
                        },
                        line={"color": "#666666", "width": 1},
                        customdata=np.stack(
                            [
                                rows["annual_index_value"].round(4),
                                rows["index_percentile"].round(1),
                                rows["classification"],
                                rows["n_observations"],
                            ],
                            axis=-1,
                        ),
                        hovertemplate=(
                            "Year=%{x}<br>"
                            "Anomaly=%{y:.4f}<br>"
                            "Annual value=%{customdata[0]:.4f}<br>"
                            "Percentile=%{customdata[1]:.1f}<br>"
                            "Class=%{customdata[2]}<br>"
                            "Observations=%{customdata[3]}<extra></extra>"
                        ),
                    ),
                    row=row_idx,
                    col=col_idx,
                )
            fig.add_hline(y=0, line_color="#555555", line_width=1, opacity=0.5, row=row_idx, col=col_idx)

    fig.update_layout(
        title={
            "text": (
                "Satellite-index-based growing-season year classification<br>"
                "<sup>Classification based only on interannual vegetation-index differences; no external climate data used.</sup>"
            )
        },
        width=1500,
        height=430 * max(1, len(aois)),
        template="plotly_white",
        legend_title_text="Sensor / index",
    )
    fig.update_yaxes(title_text="Index anomaly")
    fig.update_xaxes(title_text="Year")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(output_path, scale=2)
    except Exception as exc:
        html_path = output_path.with_suffix(".html")
        fig.write_html(html_path)
        print(f"Warning: Plotly PNG export failed ({exc}). Wrote HTML fallback: {html_path}", flush=True)
        _plot_classification_pillow(classified, output_path)


def _plot_classification_pillow(classified: pd.DataFrame, output_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    rows = []
    for aoi in valid_aois():
        for index in INDICES:
            panel = classified[(classified["aoi"] == aoi) & (classified["index"] == index)]
            if not panel.empty:
                rows.append((aoi, index, panel))

    width = 1500
    panel_height = 235
    left_margin = 105
    right_margin = 45
    top_margin = 115
    bottom_margin = 70
    height = top_margin + panel_height * len(rows) + bottom_margin
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((left_margin, 30), "Satellite-index-based growing-season year classification", fill="black", font=font)
    draw.text(
        (left_margin, 55),
        "Classification based only on interannual vegetation-index differences; no external climate data used.",
        fill="#444444",
        font=font,
    )

    for idx, (aoi, index, panel) in enumerate(rows):
        panel = panel.sort_values(["sensor", "year"])
        panel_top = top_margin + idx * panel_height
        panel_bottom = panel_top + panel_height - 45
        plot_left = left_margin
        plot_right = width - right_margin
        finite = panel["index_anomaly"].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        max_abs = max(float(np.max(np.abs(finite))) if len(finite) else 1.0, 0.001)
        y_zero = int((panel_top + panel_bottom) / 2)
        y_scale = (panel_bottom - panel_top) / (2.2 * max_abs)
        years = sorted(panel["year"].astype(int).unique())
        x_positions = {
            year: int(plot_left + (i + 0.5) * (plot_right - plot_left) / max(1, len(years)))
            for i, year in enumerate(years)
        }

        draw.text((plot_left, panel_top - 22), f"{AOI_LABELS.get(aoi, aoi)} - {index.upper()}", fill="black", font=font)
        draw.line((plot_left, y_zero, plot_right, y_zero), fill="#777777", width=1)
        draw.line((plot_left, panel_top, plot_left, panel_bottom), fill="#999999", width=1)

        for sensor, sensor_rows in panel.groupby("sensor"):
            previous = None
            for _, row in sensor_rows.sort_values("year").iterrows():
                x = x_positions[int(row["year"])] + (-3 if sensor == "ls" else 3)
                y = int(y_zero - float(row["index_anomaly"]) * y_scale)
                if previous is not None:
                    draw.line((previous[0], previous[1], x, y), fill="#666666", width=1)
                color = CLASS_COLORS.get(row["classification"], "#444444")
                if sensor == "s2":
                    draw.polygon([(x, y - 5), (x + 5, y), (x, y + 5), (x - 5, y)], fill=color)
                else:
                    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
                previous = (x, y)

        if idx == len(rows) - 1:
            for year in years[:: max(1, len(years) // 12)]:
                draw.text((x_positions[year] - 12, panel_bottom + 8), str(year), fill="#333333", font=font)

    legend_x = left_margin
    legend_y = height - 42
    for label, color in CLASS_COLORS.items():
        draw.rectangle((legend_x, legend_y, legend_x + 18, legend_y + 18), fill=color)
        draw.text((legend_x + 24, legend_y + 2), label, fill="black", font=font)
        legend_x += 190
    draw.ellipse((legend_x, legend_y + 4, legend_x + 10, legend_y + 14), fill="#555555")
    draw.text((legend_x + 16, legend_y + 2), "Landsat", fill="black", font=font)
    legend_x += 90
    draw.polygon(
        [(legend_x + 5, legend_y + 4), (legend_x + 10, legend_y + 9), (legend_x + 5, legend_y + 14), (legend_x, legend_y + 9)],
        fill="#555555",
    )
    draw.text((legend_x + 16, legend_y + 2), "Sentinel-2", fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def parse_args() -> argparse.Namespace:
    default_source = default_source_table()
    parser = argparse.ArgumentParser(description="Classify wet/neutral/dry years from vegetation-index differences only.")
    parser.add_argument("--input-table", type=Path, default=default_source.path)
    parser.add_argument("--temporal-agg", default="half_month")
    parser.add_argument("--temporal-percentile", default="p50")
    parser.add_argument("--spatial-percentile", default="p50")
    parser.add_argument("--cloud-threshold", type=int, default=30)
    parser.add_argument("--annual-aggregation", default="median", choices=["median", "p50", "p75", "p95"])
    parser.add_argument("--min-years-for-warning", type=int, default=10)
    parser.add_argument("--min-observations-for-warning", type=int, default=3)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=project_path("results_tables_dir") / "index_based_moisture_year_classification.csv",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=project_path("results_tables_dir") / "index_based_moisture_year_classification.parquet",
    )
    parser.add_argument(
        "--annual-output",
        type=Path,
        default=project_path("results_tables_dir") / "index_based_moisture_annual_summary.csv",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=project_path("results_figures_dir") / "index_based_moisture_year_classification.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.input_table.resolve()
    print("Satellite-index-only moisture/productivity year classification")
    print(f"Source table: {source_path}")
    print(f"External climate inputs: none")

    source = read_source_table(source_path)
    observations = prepare_observations(
        source,
        temporal_agg=args.temporal_agg,
        temporal_percentile=args.temporal_percentile,
        spatial_percentile=args.spatial_percentile,
        cloud_threshold=args.cloud_threshold,
    )
    print(f"Filtered growing-season observations: {len(observations):,}")

    annual = annual_summary(observations, args.annual_aggregation)
    annual["source_table"] = source_path.name
    annual_path = args.annual_output.resolve()
    annual_path.parent.mkdir(parents=True, exist_ok=True)
    annual.to_csv(annual_path, index=False)
    print(f"Wrote {annual_path} rows={len(annual):,}")

    classified = classify_annual(annual)
    write_outputs(classified, args.output_csv.resolve(), args.output_parquet.resolve())
    print_validation(classified, args.min_years_for_warning, args.min_observations_for_warning)
    plot_classification(classified, args.figure_output.resolve())
    print(f"Wrote {args.figure_output.resolve()}")

    metadata_path = args.output_csv.resolve().with_suffix(".metadata.json")
    metadata = {
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "created": date.today().isoformat(),
        "arguments": {key: str(value) for key, value in vars(args).items()},
        "primary_index": PRIMARY_INDEX,
        "external_climate_inputs": "none",
        "classification_method": CLASSIFICATION_METHOD,
        "growing_season_definition": GROWING_SEASON_DEFINITION,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
