#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import rasterio
from rasterio.mask import mask

from src.aoi import get_aoi_shapefile, valid_aois
from src.paths import PROJECT_ROOT, project_path


PRISM_VARIABLE = "ppt"
PRISM_RESOLUTION = "4km"
PRECIP_PERIOD_MONTHS = list(range(1, 13))
PRECIP_PERIOD_MONTHS_LABEL = "January,February,March,April,May,June,July,August,September,October,November,December"
PRECIP_PERIOD_DEFINITION = "Calendar-year total using all monthly PRISM ppt totals for Jan-Dec."
PRISM_SOURCE = "PRISM Climate Group monthly ppt via services.nacse.org"
DEFAULT_DOWNLOAD_TEMPLATE = "https://services.nacse.org/prism/data/get/us/{resolution}/{variable}/{yyyymm}"
CLASS_COLORS = {"wet": "#3A7FC1", "neutral": "#8C8C8C", "dry": "#C97834"}
AOI_DISPLAY = {
    "north": "George Washington National Forest / Virginia AOI",
    "south": "Smoky Mountains / Nantahala-region AOI",
}


@dataclass(frozen=True)
class PrismRaster:
    year: int
    month: int
    path: Path
    source_url: str | None
    downloaded: bool


def _import_geopandas():
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise SystemExit(
            "This script needs geopandas to read AOI polygons from the TNC shapefile. "
            "Install it in the project environment, or provide a pre-clipped AOI workflow before running. "
            "Example: pip install geopandas"
        ) from exc
    return gpd


def _geopandas_available() -> bool:
    try:
        __import__("geopandas")
    except ImportError:
        return False
    return True


def _default_end_year() -> int:
    today = date.today()
    return today.year - 1 if today.month < 10 else today.year


def _infer_year_range() -> tuple[int, int]:
    dashboard_dir = project_path("results_tables_dir") / "dashboard_data"
    scene_catalog_parquet = dashboard_dir / "scene_catalog.parquet"
    scene_catalog_csv = dashboard_dir / "scene_catalog.csv"
    years: list[int] = []
    if scene_catalog_parquet.exists():
        years = pd.read_parquet(scene_catalog_parquet, columns=["year"])["year"].dropna().astype(int).tolist()
    elif scene_catalog_csv.exists():
        years = pd.read_csv(scene_catalog_csv, usecols=["year"])["year"].dropna().astype(int).tolist()
    if years:
        return min(years), min(max(years), _default_end_year())

    wet_dry_path = project_path("config_dir") / "wet_dry_years.csv"
    if wet_dry_path.exists():
        df = pd.read_csv(wet_dry_path)
        years = df["year"].dropna().astype(int).tolist()
        return min(years), max(years)
    return 2017, _default_end_year()


def _aoi_key_from_values(values) -> str | None:
    text = " ".join(str(value).lower() for value in values if value is not None and not pd.isna(value))
    if any(token in text for token in ("north", "gwnf", "gw national", "george washington", "virginia")):
        return "north"
    if any(token in text for token in ("south", "smoky", "nantahala", "tennessee", "north carolina")):
        return "south"
    return None


def _aoi_key_from_row(row: pd.Series) -> str | None:
    return _aoi_key_from_values(row.values)


def _load_aoi_geometries_geopandas(path: Path) -> dict[str, list[dict]]:
    gpd = _import_geopandas()
    frame = gpd.read_file(path).to_crs("EPSG:4326")
    frame["_aoi_key"] = frame.apply(_aoi_key_from_row, axis=1)
    missing = frame[frame["_aoi_key"].isna()]
    if not missing.empty:
        print(
            "Warning: some AOI shapefile rows could not be mapped to north/south and will be ignored: "
            f"{len(missing)} rows",
            flush=True,
        )
    geometries: dict[str, list[dict]] = {}
    for aoi in valid_aois():
        subset = frame[frame["_aoi_key"] == aoi]
        if subset.empty:
            raise ValueError(
                f"No AOI polygon matched '{aoi}' in {path}. "
                "Update _aoi_key_from_row or provide a shapefile with north/south-identifying attributes."
            )
        geometries[aoi] = [geom.__geo_interface__ for geom in subset.geometry if geom is not None]
    return geometries


def _load_aoi_geometries_ogr2ogr(path: Path) -> dict[str, list[dict]]:
    if shutil.which("ogr2ogr") is None:
        raise SystemExit(
            "AOI polygon loading requires either geopandas or the GDAL ogr2ogr command. "
            "Neither is available in this environment."
        )
    with tempfile.TemporaryDirectory(prefix="prism_aoi_geojson_") as tmp_dir_name:
        geojson_path = Path(tmp_dir_name) / "aoi.geojson"
        subprocess.run(
            [
                "ogr2ogr",
                "-f",
                "GeoJSON",
                "-t_srs",
                "EPSG:4326",
                str(geojson_path),
                str(path),
            ],
            check=True,
        )
        payload = json.loads(geojson_path.read_text(encoding="utf-8"))

    geometries: dict[str, list[dict]] = {aoi: [] for aoi in valid_aois()}
    ignored = 0
    for feature in payload.get("features", []):
        properties = feature.get("properties") or {}
        aoi = _aoi_key_from_values(properties.values())
        if aoi in geometries and feature.get("geometry"):
            geometries[aoi].append(feature["geometry"])
        else:
            ignored += 1
    if ignored:
        print(f"Warning: ignored {ignored} AOI features that could not be mapped to north/south.", flush=True)
    for aoi, values in geometries.items():
        if not values:
            raise ValueError(
                f"No AOI polygon matched '{aoi}' in {path}. "
                "Update AOI key matching or provide a shapefile with north/south-identifying attributes."
            )
    return geometries


def load_aoi_geometries(aoi_path: Path | None = None) -> dict[str, list[dict]]:
    path = aoi_path or get_aoi_shapefile()
    if not path.exists():
        raise FileNotFoundError(f"AOI shapefile not found: {path}")
    if _geopandas_available():
        return _load_aoi_geometries_geopandas(path)
    return _load_aoi_geometries_ogr2ogr(path)


def yyyymm(year: int, month: int) -> str:
    return f"{year}{month:02d}"


def prism_cache_dir(root: Path, resolution: str, variable: str) -> Path:
    return root / "prism" / resolution / variable / "monthly"


def _find_cached_raster(cache_dir: Path, year: int, month: int) -> Path | None:
    stamp = yyyymm(year, month)
    patterns = [
        f"*{stamp}*.bil",
        f"*{stamp}*.tif",
        f"*{stamp}*.tiff",
    ]
    for pattern in patterns:
        matches = sorted(cache_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None


def _download_prism_zip(url: str, target: Path) -> None:
    import requests

    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "zip" not in content_type.lower():
            preview = response.text[:500]
            raise RuntimeError(
                f"PRISM response was not a ZIP file for {url}. "
                f"Content-Type={content_type!r}. Response preview: {preview!r}"
            )
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def _extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)


def get_prism_raster(
    year: int,
    month: int,
    *,
    cache_dir: Path,
    input_dir: Path | None,
    download: bool,
    download_template: str,
    resolution: str,
    variable: str,
) -> PrismRaster | None:
    search_dirs = [directory for directory in (input_dir, cache_dir) if directory is not None]
    for directory in search_dirs:
        raster = _find_cached_raster(directory, year, month)
        if raster is not None:
            return PrismRaster(year, month, raster, None, False)

    if not download:
        return None

    stamp = yyyymm(year, month)
    url = download_template.format(resolution=resolution, variable=variable, yyyymm=stamp, year=year, month=f"{month:02d}")
    zip_path = cache_dir / f"PRISM_{variable}_{resolution}_{stamp}.zip"
    extract_dir = cache_dir / stamp
    if not zip_path.exists():
        print(f"Downloading PRISM {variable} {stamp}: {url}", flush=True)
        try:
            _download_prism_zip(url, zip_path)
        except Exception as exc:
            print(f"Warning: PRISM {variable} {stamp} is unavailable or could not be downloaded: {exc}", flush=True)
            return None
    elif not zipfile.is_zipfile(zip_path):
        print(f"Cached PRISM file is not a valid ZIP, re-downloading: {zip_path}", flush=True)
        zip_path.unlink()
        try:
            _download_prism_zip(url, zip_path)
        except Exception as exc:
            print(f"Warning: PRISM {variable} {stamp} is unavailable or could not be downloaded: {exc}", flush=True)
            return None
    _extract_zip(zip_path, extract_dir)
    raster = _find_cached_raster(extract_dir, year, month)
    if raster is None:
        raise FileNotFoundError(f"Downloaded {zip_path}, but no {stamp} raster was found after extraction.")
    return PrismRaster(year, month, raster, url, True)


def mean_precip_for_aoi(raster_path: Path, geometries: list[dict]) -> float:
    with rasterio.open(raster_path) as src:
        raster_geometries = geometries
        if src.crs and src.crs.to_string() != "EPSG:4326":
            try:
                from rasterio.warp import transform_geom
                raster_geometries = [
                    transform_geom("EPSG:4326", src.crs, geom, precision=6)
                    for geom in geometries
                ]
            except Exception as exc:
                raise RuntimeError(f"Failed to transform AOI geometry to raster CRS for {raster_path}") from exc
        data, _ = mask(src, raster_geometries, crop=True, filled=False)
        values = np.asarray(data[0], dtype=float)
        if np.ma.isMaskedArray(data):
            values = data[0].astype(float).filled(np.nan)
        values[~np.isfinite(values)] = np.nan
        return float(np.nanmean(values))


def build_monthly_inventory(
    years: range,
    *,
    cache_dir: Path,
    input_dir: Path | None,
    download: bool,
    download_template: str,
    resolution: str,
    variable: str,
) -> tuple[list[PrismRaster], list[dict]]:
    rasters: list[PrismRaster] = []
    missing: list[dict] = []
    for year in years:
        for month in PRECIP_PERIOD_MONTHS:
            raster = get_prism_raster(
                year,
                month,
                cache_dir=cache_dir,
                input_dir=input_dir,
                download=download,
                download_template=download_template,
                resolution=resolution,
                variable=variable,
            )
            if raster is None:
                missing.append({"year": year, "month": month})
            else:
                rasters.append(raster)
    return rasters, missing


def extract_monthly_precip(rasters: list[PrismRaster], aoi_geometries: dict[str, list[dict]]) -> pd.DataFrame:
    rows: list[dict] = []
    for raster in rasters:
        for aoi, geometries in aoi_geometries.items():
            mean_mm = mean_precip_for_aoi(raster.path, geometries)
            rows.append(
                {
                    "aoi": aoi,
                    "aoi_label": AOI_DISPLAY.get(aoi, aoi),
                    "year": raster.year,
                    "month": raster.month,
                    "month_label": date(raster.year, raster.month, 1).strftime("%b"),
                    "mean_precip_mm": mean_mm,
                    "raster_path": str(raster.path),
                    "source_url": raster.source_url,
                    "downloaded": raster.downloaded,
                }
            )
    return pd.DataFrame(rows).sort_values(["aoi", "year", "month"]).reset_index(drop=True)


def _rank_percentile(series: pd.Series) -> pd.Series:
    n = int(series.notna().sum())
    if n <= 1:
        return pd.Series([50.0 if pd.notna(value) else np.nan for value in series], index=series.index)
    ranks = series.rank(method="average", ascending=True)
    return (ranks - 1.0) / (n - 1.0) * 100.0


def classify_years(annual: pd.DataFrame) -> pd.DataFrame:
    classified = annual.copy()
    complete = classified["months_present"] == len(PRECIP_PERIOD_MONTHS)
    classified["_precip_for_classification"] = classified["annual_precip_mm"].where(complete)
    stats = (
        classified.groupby("aoi")["_precip_for_classification"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "aoi_mean_precip_mm", "std": "aoi_std_precip_mm"})
        .reset_index()
    )
    classified = classified.merge(stats, on="aoi", how="left")
    classified["precip_anomaly_mm"] = classified["_precip_for_classification"] - classified["aoi_mean_precip_mm"]
    classified["precip_zscore"] = classified["precip_anomaly_mm"] / classified["aoi_std_precip_mm"].replace(0, np.nan)
    classified["precip_percentile"] = classified.groupby("aoi")["_precip_for_classification"].transform(_rank_percentile)
    classified["precip_rank"] = (
        classified.groupby("aoi")["_precip_for_classification"]
        .rank(method="first", ascending=False)
        .astype("Int64")
    )
    classified["classification"] = np.select(
        [
            classified["precip_percentile"] >= 80.0,
            classified["precip_percentile"] <= 20.0,
        ],
        ["wet", "dry"],
        default="neutral",
    )
    classified.loc[classified["months_present"] < len(PRECIP_PERIOD_MONTHS), "classification"] = "incomplete"
    classified = classified.drop(columns=["_precip_for_classification"])
    return classified


def aggregate_annual(monthly: pd.DataFrame, years: range) -> pd.DataFrame:
    annual = (
        monthly.groupby(["aoi", "aoi_label", "year"], dropna=False)
        .agg(
            annual_precip_mm=("mean_precip_mm", lambda values: values.sum(min_count=1)),
            months_present=("month", "nunique"),
            missing_months=("month", lambda values: ",".join(str(month) for month in sorted(set(PRECIP_PERIOD_MONTHS) - set(values)))),
            monthly_mean_precip_mm=("mean_precip_mm", "mean"),
        )
        .reset_index()
    )
    expected = pd.MultiIndex.from_product(
        [valid_aois(), list(years)],
        names=["aoi", "year"],
    ).to_frame(index=False)
    expected["aoi_label"] = expected["aoi"].map(AOI_DISPLAY).fillna(expected["aoi"])
    annual = expected.merge(annual, on=["aoi", "aoi_label", "year"], how="left")
    annual["months_present"] = annual["months_present"].fillna(0).astype(int)
    annual["missing_months"] = annual.apply(
        lambda row: ""
        if row["months_present"] == len(PRECIP_PERIOD_MONTHS)
        else (
            row["missing_months"]
            if isinstance(row["missing_months"], str) and row["missing_months"]
            else ",".join(str(month) for month in PRECIP_PERIOD_MONTHS)
        ),
        axis=1,
    )
    annual["precip_period_months"] = PRECIP_PERIOD_MONTHS_LABEL
    annual["precip_period_definition"] = PRECIP_PERIOD_DEFINITION
    return annual


def add_metadata_columns(frame: pd.DataFrame, *, resolution: str, variable: str, notes: str) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["prism_resolution"] = resolution
    enriched["prism_variable"] = variable
    enriched["source"] = PRISM_SOURCE
    enriched["notes"] = notes
    return enriched


def plot_classification(classified: pd.DataFrame, output_path: Path) -> None:
    aois = [aoi for aoi in valid_aois() if aoi in set(classified["aoi"])]
    fig = make_subplots(
        rows=len(aois),
        cols=1,
        shared_xaxes=False,
        subplot_titles=[AOI_DISPLAY.get(aoi, aoi) for aoi in aois],
        vertical_spacing=0.12,
    )
    for row_idx, aoi in enumerate(aois, start=1):
        subset = classified[classified["aoi"] == aoi].sort_values("year")
        colors = [CLASS_COLORS.get(value, "#444444") for value in subset["classification"]]
        months_present = (
            subset["months_present"]
            if "months_present" in subset.columns
            else pd.Series([len(PRECIP_PERIOD_MONTHS)] * len(subset), index=subset.index)
        )
        fig.add_trace(
            go.Bar(
                x=subset["year"],
                y=subset["precip_anomaly_mm"],
                marker_color=colors,
                text=subset["classification"],
                customdata=np.stack(
                    [
                        subset["annual_precip_mm"].round(1),
                        subset["precip_percentile"].round(1),
                        months_present,
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "Year=%{x}<br>"
                    "Anomaly=%{y:.1f} mm<br>"
                    "Total=%{customdata[0]:.1f} mm<br>"
                    "Percentile=%{customdata[1]:.1f}<br>"
                    "Months=%{customdata[2]}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row_idx,
            col=1,
        )
        fig.add_hline(y=0, line_color="#555555", line_width=1, opacity=0.5, row=row_idx, col=1)

    for classification, color in CLASS_COLORS.items():
        fig.add_trace(
            go.Bar(x=[None], y=[None], marker_color=color, name=classification.title()),
            row=1,
            col=1,
        )
    fig.update_layout(
        title={
            "text": (
                "PRISM-derived annual precipitation classification<br>"
                "<sup>Calendar-year totals: January-December PRISM monthly precipitation</sup>"
            )
        },
        barmode="relative",
        width=1200,
        height=420 * max(1, len(aois)),
        template="plotly_white",
        legend_title_text="Classification",
    )
    fig.update_yaxes(title_text="Precipitation anomaly (mm)")
    fig.update_xaxes(title_text="Year", dtick=1)
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

    aois = [aoi for aoi in valid_aois() if aoi in set(classified["aoi"])]
    width = 1400
    panel_height = 360
    top_margin = 115
    left_margin = 115
    right_margin = 40
    bottom_margin = 70
    height = top_margin + panel_height * len(aois) + bottom_margin
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    title = "PRISM-derived annual precipitation classification"
    subtitle = "Calendar-year totals: January-December PRISM monthly precipitation"
    draw.text((left_margin, 30), title, fill="black", font=font)
    draw.text((left_margin, 55), subtitle, fill="#444444", font=font)

    for idx, aoi in enumerate(aois):
        subset = classified[classified["aoi"] == aoi].sort_values("year").reset_index(drop=True)
        panel_top = top_margin + idx * panel_height
        panel_bottom = panel_top + panel_height - 55
        plot_left = left_margin
        plot_right = width - right_margin
        values = subset["precip_anomaly_mm"].astype(float).to_numpy()
        finite = values[np.isfinite(values)]
        max_abs = max(float(np.max(np.abs(finite))) if len(finite) else 1.0, 1.0)
        y_zero = int((panel_top + panel_bottom) / 2)
        y_scale = (panel_bottom - panel_top) / (2.2 * max_abs)
        years = subset["year"].astype(int).tolist()
        bar_gap = 6
        bar_width = max(8, int((plot_right - plot_left) / max(1, len(years)) - bar_gap))

        draw.text((plot_left, panel_top - 25), AOI_DISPLAY.get(aoi, aoi), fill="black", font=font)
        draw.line((plot_left, y_zero, plot_right, y_zero), fill="#666666", width=1)
        draw.line((plot_left, panel_top, plot_left, panel_bottom), fill="#999999", width=1)
        draw.text((20, panel_top + 10), "Anomaly (mm)", fill="#333333", font=font)

        for i, row in subset.iterrows():
            x_center = int(plot_left + (i + 0.5) * (plot_right - plot_left) / max(1, len(subset)))
            value = float(row["precip_anomaly_mm"]) if pd.notna(row["precip_anomaly_mm"]) else 0.0
            y_value = int(y_zero - value * y_scale)
            color = CLASS_COLORS.get(row["classification"], "#444444")
            x0 = x_center - bar_width // 2
            x1 = x_center + bar_width // 2
            y0, y1 = sorted((y_zero, y_value))
            draw.rectangle((x0, y0, x1, y1), fill=color, outline="white")
            draw.text((x_center - 12, panel_bottom + 8), str(int(row["year"])), fill="#333333", font=font)

    legend_x = left_margin
    legend_y = height - 40
    for label, color in CLASS_COLORS.items():
        draw.rectangle((legend_x, legend_y, legend_x + 18, legend_y + 18), fill=color)
        draw.text((legend_x + 24, legend_y + 2), label.title(), fill="black", font=font)
        legend_x += 115

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def output_columns() -> list[str]:
    return [
        "aoi",
        "year",
        "precip_period_months",
        "precip_period_definition",
        "annual_precip_mm",
        "precip_anomaly_mm",
        "precip_zscore",
        "precip_percentile",
        "precip_rank",
        "classification",
        "prism_resolution",
        "prism_variable",
        "source",
        "notes",
    ]


def parse_args() -> argparse.Namespace:
    inferred_start, inferred_end = _infer_year_range()
    parser = argparse.ArgumentParser(description="Build PRISM annual precipitation wet/neutral/dry classifications.")
    parser.add_argument("--start-year", type=int, default=inferred_start)
    parser.add_argument("--end-year", type=int, default=inferred_end)
    parser.add_argument("--download", action="store_true", help="Download missing PRISM monthly rasters into the cache.")
    parser.add_argument("--input-dir", type=Path, help="Directory containing already-downloaded PRISM monthly rasters.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=project_path("results_rasters_dir") / "climate",
        help="Root directory for cached PRISM downloads/extractions.",
    )
    parser.add_argument("--aoi-shapefile", type=Path, default=get_aoi_shapefile())
    parser.add_argument("--resolution", default=PRISM_RESOLUTION)
    parser.add_argument("--variable", default=PRISM_VARIABLE)
    parser.add_argument("--download-template", default=DEFAULT_DOWNLOAD_TEMPLATE)
    parser.add_argument("--config-output", type=Path, default=project_path("config_dir") / "prism_growing_season_year_classes.csv")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=project_path("results_tables_dir") / "prism_growing_season_precip_summary.csv",
    )
    parser.add_argument(
        "--monthly-output",
        type=Path,
        default=project_path("results_tables_dir") / "prism_monthly_precip_extractions.csv",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=project_path("results_figures_dir") / "prism_growing_season_precip_classification.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.end_year < args.start_year:
        raise SystemExit("--end-year must be greater than or equal to --start-year")

    cache_dir = prism_cache_dir(args.cache_dir.resolve(), args.resolution, args.variable)
    notes = "Top/bottom 20% rank classification within each AOI; Jan-Dec annual PRISM monthly ppt totals."
    print(f"PRISM annual precipitation classification {args.start_year}-{args.end_year}")
    print(f"Cache directory: {cache_dir}")
    print(f"Download missing rasters: {args.download}")

    aoi_geometries = load_aoi_geometries(args.aoi_shapefile.resolve())
    rasters, missing = build_monthly_inventory(
        range(args.start_year, args.end_year + 1),
        cache_dir=cache_dir,
        input_dir=args.input_dir.resolve() if args.input_dir else None,
        download=args.download,
        download_template=args.download_template,
        resolution=args.resolution,
        variable=args.variable,
    )
    if missing:
        print(f"Warning: missing {len(missing)} monthly PRISM inputs.")
        missing_by_year = pd.DataFrame(missing).groupby("year")["month"].apply(list)
        for year, months in missing_by_year.items():
            print(f"  {year}: missing months {months}")
    if not rasters:
        raise SystemExit(
            "No PRISM rasters found. Re-run with --download or provide --input-dir containing monthly PRISM rasters."
        )

    monthly = extract_monthly_precip(rasters, aoi_geometries)
    monthly = add_metadata_columns(monthly, resolution=args.resolution, variable=args.variable, notes=notes)
    annual = aggregate_annual(monthly, range(args.start_year, args.end_year + 1))
    classified = classify_years(annual)
    classified = add_metadata_columns(classified, resolution=args.resolution, variable=args.variable, notes=notes)

    config_frame = classified[output_columns()].sort_values(["aoi", "year"]).reset_index(drop=True)
    summary_frame = classified.sort_values(["aoi", "year"]).reset_index(drop=True)

    for path, frame in (
        (args.monthly_output.resolve(), monthly),
        (args.summary_output.resolve(), summary_frame),
        (args.config_output.resolve(), config_frame),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"Wrote {path} rows={len(frame):,}")

    incomplete = summary_frame[summary_frame["months_present"] != len(PRECIP_PERIOD_MONTHS)]
    if not incomplete.empty:
        print("Warning: some AOI-years do not have exactly twelve monthly inputs:")
        print(incomplete[["aoi", "year", "months_present", "missing_months"]].to_string(index=False))

    counts = summary_frame.groupby(["aoi", "classification"])["year"].count().unstack(fill_value=0)
    print("Classification counts by AOI:")
    print(counts.to_string())

    try:
        plot_classification(summary_frame, args.figure_output.resolve())
        print(f"Wrote {args.figure_output.resolve()}")
    except Exception:
        print("Figure PNG export failed; tables were still written.")

    metadata_path = args.summary_output.resolve().with_suffix(".metadata.json")
    metadata = {
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "arguments": {key: str(value) for key, value in vars(args).items()},
        "precip_period_months": PRECIP_PERIOD_MONTHS,
        "precip_period_definition": PRECIP_PERIOD_DEFINITION,
        "missing_months": missing,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
