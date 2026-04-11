#!/usr/bin/env python3
"""
ecozone_seasonal_scenelevel_plotly.py
-------------------------------------
Build standalone Plotly HTML versions of the scene-level ecozone seasonal plots
from the existing spreadsheet output, without re-reading rasters or
recomputing percentiles.

Inputs:
  Results/tables/ecozone_seasonal_scenelevel_summary.xlsx

Outputs:
  Results/figures/ecozone_<index>_seasonal_scenelevel.interactive.html

Run from the Python directory:
  python Analysis/Traits/Ecozone/ecozone_seasonal_scenelevel_plotly.py
  python Analysis/Traits/Ecozone/ecozone_seasonal_scenelevel_plotly.py --indices NDVI NDMI
  python Analysis/Traits/Ecozone/ecozone_seasonal_scenelevel_plotly.py --no-band
"""

import argparse

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.paths import project_path

AOIS = [("north", "GW National Forest"), ("south", "Great Smoky Mtns")]
INDICES = ["NDVI", "NDMI", "EVI"]
ECOZONE_CODES = [1, 2, 3]
ECOZONE_LABELS = {1: "Cool", 2: "Intermediate", 3: "Hot"}
ECOZONE_COLORS = {1: "#4E90C8", 2: "#72B063", 3: "#D9534F"}

TABLE_PATH = project_path("results_tables_dir") / "ecozone_seasonal_scenelevel_summary.xlsx"
FIGURES_DIR = project_path("results_figures_dir")


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_figure(df: pd.DataFrame, index_name: str, include_band: bool) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[label for _, label in AOIS],
        shared_yaxes=False,
        horizontal_spacing=0.08,
    )

    for col_idx, (aoi_key, aoi_label) in enumerate(AOIS, start=1):
        for ecozone_code in ECOZONE_CODES:
            subset = df[
                (df["AOI Key"] == aoi_key)
                & (df["Index"] == index_name)
                & (df["Ecozone Code"] == ecozone_code)
            ].sort_values("Scene Date")
            if subset.empty:
                continue

            color = ECOZONE_COLORS[ecozone_code]
            fill_color = hex_to_rgba(color, 0.12)
            ecozone_label = ECOZONE_LABELS[ecozone_code]
            customdata = subset[["Month Name", "p100 (max)"]].to_numpy()

            if include_band:
                fig.add_trace(
                    go.Scatter(
                        x=subset["Scene Date"],
                        y=subset["p100 (max)"],
                        mode="lines",
                        line={"width": 0, "color": color},
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=ecozone_label,
                    ),
                    row=1,
                    col=col_idx,
                )

            fig.add_trace(
                go.Scatter(
                    x=subset["Scene Date"],
                    y=subset["p95"],
                    mode="lines+markers",
                    line={"width": 1.6, "color": color},
                    marker={"size": 4, "color": color},
                    fill="tonexty" if include_band else None,
                    fillcolor=fill_color if include_band else None,
                    name=ecozone_label,
                    legendgroup=ecozone_label,
                    showlegend=(col_idx == 1),
                    customdata=customdata,
                    hovertemplate=(
                        "Date=%{x|%Y-%m-%d}<br>"
                        "Ecozone=" + ecozone_label + "<br>"
                        "Month=%{customdata[0]}<br>"
                        "p95=%{y:.4f}<br>"
                        "p100=%{customdata[1]:.4f}<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )

        fig.update_xaxes(
            title_text="Scene date",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            tickformat="%Y",
            row=1,
            col=col_idx,
        )
        fig.update_yaxes(
            title_text="Scene-level p95 " + index_name if col_idx == 1 else "",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        title=(
            f"Seasonal {index_name} — Scene-level p95 by Ecozone"
            + (" (shaded band = p95 to p100)" if include_band else " (line/marker only)")
        ),
        template="plotly_white",
        width=1400,
        height=650,
        hovermode="x unified",
        legend_title_text="Ecozone",
        margin={"l": 60, "r": 30, "t": 80, "b": 60},
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Plotly scene-level ecozone seasonal figures from existing spreadsheet data")
    parser.add_argument(
        "--indices",
        nargs="+",
        choices=INDICES,
        default=INDICES,
        help="Indices to render",
    )
    parser.add_argument(
        "--no-band",
        action="store_true",
        help="Omit the p95 to p100 shaded band and draw only the p95 line/markers",
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

        fig = build_figure(index_df, index_name=index_name, include_band=not args.no_band)
        suffix = ".interactive.noband.html" if args.no_band else ".interactive.html"
        out_path = FIGURES_DIR / f"ecozone_{index_name.lower()}_seasonal_scenelevel{suffix}"
        fig.write_html(out_path, include_plotlyjs=True, full_html=True)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
