from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go

from src.dashboard_data import DashboardDataBundle, filter_frame
from src.dashboard_schema import ComparisonConfig


def _apply_stddev_filters(frame: pd.DataFrame, config: ComparisonConfig) -> pd.DataFrame:
    if frame.empty:
        return frame
    values = pd.to_numeric(frame["value"], errors="coerce")
    mean = values.mean()
    std = values.std(ddof=0)
    if pd.isna(mean) or pd.isna(std) or std == 0:
        return frame

    keep_mask = pd.Series(True, index=frame.index)
    z_scores = (values - mean) / std
    if config.exclude_below_stddev is not None:
        keep_mask &= z_scores >= -config.exclude_below_stddev
    if config.exclude_above_stddev is not None:
        keep_mask &= z_scores <= config.exclude_above_stddev
    return frame.loc[keep_mask].copy()


def _series_label(config: ComparisonConfig) -> str:
    return config.label or " / ".join(
        [
            config.sensor,
            config.aoi,
            config.index,
            config.temporal_agg,
            config.temporal_percentile,
            config.spatial_percentile,
            config.season_filter,
        ]
    )


def _resolve_x_axis(frame: pd.DataFrame, temporal_agg: str) -> str:
    if temporal_agg == "scene":
        return "date"
    if frame["time_bin_start"].notna().any():
        return "time_bin_start"
    return "date"


def _with_year_breaks(frame: pd.DataFrame, x_axis: str) -> pd.DataFrame:
    if frame.empty or "year" not in frame.columns:
        return frame
    segments = []
    for _, year_frame in frame.groupby("year", dropna=False):
        segments.append(year_frame)
        segments.append(pd.DataFrame([{column: None for column in frame.columns}]))
    return pd.concat(segments[:-1], ignore_index=True) if segments else frame


def build_timeseries_figure(
    bundle: DashboardDataBundle,
    configs: list[ComparisonConfig],
    year_range: tuple[int, int],
) -> tuple[go.Figure, list[str]]:
    fig = go.Figure()
    messages: list[str] = []

    for config in configs:
        frame = bundle.frame_for_config(config)
        filtered = filter_frame(
            frame,
            filters=asdict(config) | {"label": None},
            year_range=year_range,
        )
        if filtered.empty:
            messages.append(f"No data for `{_series_label(config)}` in the selected year range.")
            continue

        filtered = _apply_stddev_filters(filtered, config)
        if filtered.empty:
            messages.append(f"All rows for `{_series_label(config)}` were removed by the standard-deviation filters.")
            continue

        x_axis = _resolve_x_axis(filtered, config.temporal_agg)
        filtered = filtered.sort_values(x_axis)
        if config.season_filter == "growing":
            filtered = _with_year_breaks(filtered, x_axis)

        fig.add_trace(
            go.Scatter(
                x=filtered[x_axis],
                y=filtered["value"],
                mode="lines+markers",
                name=_series_label(config),
                customdata=filtered[
                    ["sensor", "aoi", "index", "temporal_agg", "spatial_percentile", "temporal_percentile", "pixel_mask_id"]
                ],
                hovertemplate=(
                    "Date=%{x|%Y-%m-%d}<br>"
                    "Value=%{y:.4f}<br>"
                    "Sensor=%{customdata[0]}<br>"
                    "AOI=%{customdata[1]}<br>"
                    "Index=%{customdata[2]}<br>"
                    "Temporal agg=%{customdata[3]}<br>"
                    "Spatial pct=%{customdata[4]}<br>"
                    "Temporal pct=%{customdata[5]}<br>"
                    "Mask=%{customdata[6]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template="plotly_white",
        title="Terrain–Vegetation Time Series",
        xaxis_title="Date",
        yaxis_title="Summary value",
        hovermode="x unified",
        showlegend=False,
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    return fig, messages


def build_growing_season_figure(
    bundle: DashboardDataBundle,
    config: ComparisonConfig,
    selected_year: int,
) -> tuple[go.Figure, str | None]:
    filtered = filter_frame(bundle.scene_summary, filters=asdict(config) | {"label": None, "season_filter": "growing"})
    if filtered.empty:
        return go.Figure(), "No growing-season rows match the selected configuration."
    filtered = _apply_stddev_filters(filtered, config)
    if filtered.empty:
        return go.Figure(), "All growing-season rows were removed by the standard-deviation filters."
    filtered = filtered.sort_values(["year", "growing_season_day", "date"])
    fig = go.Figure()

    for year, year_frame in filtered.groupby("year", dropna=True):
        if pd.isna(year):
            continue
        year_int = int(year)
        if year_int == selected_year:
            line = {"color": "#145a32", "width": 3.0}
            opacity = 0.95
        elif year_int == selected_year - 1:
            line = {"color": "#7d3c98", "width": 2.0}
            opacity = 0.65
        elif year_int == selected_year + 1:
            line = {"color": "#f4d03f", "width": 2.0}
            opacity = 0.65
        else:
            line = {"color": "#bfc9ca", "width": 1.0}
            opacity = 0.25

        fig.add_trace(
            go.Scatter(
                x=year_frame["growing_season_day"],
                y=year_frame["value"],
                mode="lines",
                line=line,
                opacity=opacity,
                name=str(year_int),
                showlegend=year_int in (selected_year - 1, selected_year, selected_year + 1),
                hovertemplate="Year=" + str(year_int) + "<br>Growing day=%{x}<br>Value=%{y:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_white",
        title=f"Growing Season Overlay: {_series_label(config)}",
        xaxis_title="Day of growing season (May 15 = 1)",
        yaxis_title="Summary value",
        hovermode="closest",
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    fig.update_xaxes(range=[1, 124])
    return fig, None
