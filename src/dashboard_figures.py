from __future__ import annotations

from functools import lru_cache

import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go

from src.dashboard_data import DashboardDataBundle, ECOZONE_LABELS, filter_frame, filters_for_config
from src.dashboard_schema import ComparisonConfig
from src.paths import project_path

AOI_LABELS = {
    "north": "GW-Jeff",
    "south": "Smoky",
}
SENSOR_LABELS = {
    "ls": "landsat",
    "s2": "sentinel-2",
}
ECOZONE_TRACE_COLORS = {
    1: "#2f80ed",
    2: "#f2c94c",
    3: "#eb5757",
}
ALL_SEGMENT_VALUES = {None, "all", "All", "ALL"}
PRISM_CLASSIFICATION_PATH = project_path("config_dir") / "prism_growing_season_year_classes.csv"
PRISM_STRIPE_ZSCORE_THRESHOLD = 1.0
PRISM_STRIPE_RAMPS = {
    "dry": {
        "low": (255, 220, 220),
        "high": (215, 35, 35),
    },
    "wet": {
        "low": (85, 165, 255),
        "high": (0, 90, 220),
    },
}
PRISM_STRIPE_OPACITY = 0.30
PRISM_STRIPE_RAMP_FLOOR = 0.20
PRISM_STRIPE_LABEL_FONT = {"size": 12, "color": "#ffffff"}


def _cloud_threshold_label(value) -> str:
    if value is None:
        return "none"
    try:
        return f"{int(float(value))}%"
    except (TypeError, ValueError):
        return str(value)


def _is_all_segment_value(value) -> bool:
    return value in ALL_SEGMENT_VALUES


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


def _segment_label_from_frame(frame: pd.DataFrame, label_column: str) -> str | None:
    if frame is None or frame.empty or label_column not in frame.columns:
        return None
    labels = frame[label_column].dropna().astype(str)
    labels = labels[labels != ""]
    if labels.empty:
        return None
    return labels.iloc[0]


def _series_label(config: ComparisonConfig, frame: pd.DataFrame | None = None) -> str:
    parts = [
        AOI_LABELS.get(config.aoi, config.aoi),
        SENSOR_LABELS.get(config.sensor, config.sensor),
        config.index,
    ]
    analysis_scope = getattr(config, "analysis_scope", "overall")
    if analysis_scope == "ecozone":
        if _is_all_segment_value(config.ecozone_code) and frame is None:
            label = "All"
        else:
            label = _segment_label_from_frame(frame, "ecozone_label") or str(
                ECOZONE_LABELS.get(config.ecozone_code, f"ecozone {config.ecozone_code}")
            )
        parts.append(label.lower())
    elif analysis_scope == "forest_community":
        if _is_all_segment_value(config.forest_community_code) and frame is None:
            label = "All"
        else:
            label = _segment_label_from_frame(frame, "forest_community_label") or f"forest community {config.forest_community_code}"
        parts.append(label)
    parts.extend(
        [
            _cloud_threshold_label(config.cloud_threshold),
            config.spatial_percentile,
            config.temporal_agg,
            config.temporal_percentile,
        ]
    )
    return config.label or " / ".join(parts)


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


def _ecozone_trace_color(config: ComparisonConfig, frame: pd.DataFrame | None = None) -> str | None:
    if getattr(config, "analysis_scope", "overall") != "ecozone":
        return None
    ecozone_code = config.ecozone_code
    if _is_all_segment_value(ecozone_code) and frame is not None and "ecozone_code" in frame.columns:
        codes = pd.to_numeric(frame["ecozone_code"], errors="coerce").dropna()
        if not codes.empty:
            ecozone_code = int(codes.iloc[0])
    try:
        return ECOZONE_TRACE_COLORS.get(int(ecozone_code))
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=4)
def _load_prism_year_classes(path_str: str = str(PRISM_CLASSIFICATION_PATH)) -> pd.DataFrame:
    path = pd.io.common.stringify_path(path_str)
    try:
        header = pd.read_csv(path, nrows=0).columns
        usecols = [
            column
            for column in ["aoi", "year", "annual_precip_mm", "growing_season_precip_mm", "precip_zscore"]
            if column in header
        ]
        frame = pd.read_csv(path, usecols=usecols)
    except FileNotFoundError:
        return pd.DataFrame(columns=["aoi", "year", "annual_precip_mm", "precip_zscore"])
    if "annual_precip_mm" not in frame.columns and "growing_season_precip_mm" in frame.columns:
        frame["annual_precip_mm"] = frame["growing_season_precip_mm"]
    if not {"aoi", "year", "annual_precip_mm", "precip_zscore"}.issubset(frame.columns):
        return pd.DataFrame(columns=["aoi", "year", "annual_precip_mm", "precip_zscore"])
    frame["aoi"] = frame["aoi"].astype(str).str.lower()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["annual_precip_mm"] = pd.to_numeric(frame["annual_precip_mm"], errors="coerce")
    frame["precip_zscore"] = pd.to_numeric(frame["precip_zscore"], errors="coerce")
    frame = frame.dropna(subset=["year", "annual_precip_mm", "precip_zscore"])
    frame["year"] = frame["year"].astype(int)
    return frame


def _stripe_class_from_zscore(zscore: float) -> str | None:
    if zscore >= PRISM_STRIPE_ZSCORE_THRESHOLD:
        return "wet"
    if zscore <= -PRISM_STRIPE_ZSCORE_THRESHOLD:
        return "dry"
    return None


def _stripe_strength(zscore: float, max_excess: float) -> float:
    classification = _stripe_class_from_zscore(zscore)
    if classification == "wet":
        excess = zscore - PRISM_STRIPE_ZSCORE_THRESHOLD
    elif classification == "dry":
        excess = abs(zscore) - PRISM_STRIPE_ZSCORE_THRESHOLD
    else:
        return 0.0
    if max_excess <= 0:
        return 1.0
    return max(0.0, min(1.0, excess / max_excess))


def _interpolate_rgb(low: tuple[int, int, int], high: tuple[int, int, int], strength: float) -> tuple[int, int, int]:
    return tuple(int(round(low_value + (high_value - low_value) * strength)) for low_value, high_value in zip(low, high))


def _stripe_fillcolor(classification: str, strength: float) -> str:
    ramp = PRISM_STRIPE_RAMPS[classification]
    strength = max(0.0, min(1.0, strength))
    ramp_position = PRISM_STRIPE_RAMP_FLOOR + (1.0 - PRISM_STRIPE_RAMP_FLOOR) * strength
    red, green, blue = _interpolate_rgb(ramp["low"], ramp["high"], ramp_position)
    return f"rgba({red}, {green}, {blue}, {PRISM_STRIPE_OPACITY:.3f})"


def _prism_stripes_for_configs(
    configs: list[ComparisonConfig],
    year_range: tuple[int, int],
    prism_classes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if not configs:
        return pd.DataFrame(columns=["year", "classification", "stripe_strength", "annual_precip_mm", "precip_zscore"])
    aois = sorted({config.aoi for config in configs if config.aoi})
    if not aois:
        return pd.DataFrame(columns=["year", "classification", "stripe_strength", "annual_precip_mm", "precip_zscore"])
    if len(aois) != 1:
        return pd.DataFrame(columns=["year", "classification", "stripe_strength", "annual_precip_mm", "precip_zscore"])

    classes = _load_prism_year_classes() if prism_classes is None else prism_classes.copy()
    if classes.empty:
        return pd.DataFrame(columns=["year", "classification", "stripe_strength", "annual_precip_mm", "precip_zscore"])
    if "annual_precip_mm" not in classes.columns and "growing_season_precip_mm" in classes.columns:
        classes["annual_precip_mm"] = classes["growing_season_precip_mm"]
    classes["aoi"] = classes["aoi"].astype(str).str.lower()
    classes["year"] = pd.to_numeric(classes["year"], errors="coerce")
    classes["annual_precip_mm"] = pd.to_numeric(classes["annual_precip_mm"], errors="coerce")
    classes["precip_zscore"] = pd.to_numeric(classes["precip_zscore"], errors="coerce")
    classes = classes.dropna(subset=["year", "annual_precip_mm", "precip_zscore"])
    classes["year"] = classes["year"].astype(int)
    classes = classes[
        classes["aoi"].isin(aois)
        & classes["year"].between(int(year_range[0]), int(year_range[1]))
    ].copy()
    if classes.empty:
        return pd.DataFrame(columns=["year", "classification", "stripe_strength", "annual_precip_mm", "precip_zscore"])

    classes["classification"] = classes["precip_zscore"].map(_stripe_class_from_zscore)
    classes = classes.dropna(subset=["classification"]).copy()
    if classes.empty:
        return pd.DataFrame(columns=["year", "classification", "stripe_strength", "annual_precip_mm", "precip_zscore"])

    classes["excess"] = (classes["precip_zscore"].abs() - PRISM_STRIPE_ZSCORE_THRESHOLD).clip(lower=0.0)
    max_excess = classes.groupby(["aoi", "classification"])["excess"].transform("max")
    classes["stripe_strength"] = [
        _stripe_strength(zscore, group_max)
        for zscore, group_max in zip(classes["precip_zscore"], max_excess)
    ]

    return classes[["year", "classification", "stripe_strength", "annual_precip_mm", "precip_zscore"]].drop_duplicates().sort_values("year")


def _stripe_label(row) -> str:
    return f"{float(row.annual_precip_mm):.0f}"


def _add_prism_year_stripes(fig: go.Figure, configs: list[ComparisonConfig], year_range: tuple[int, int]) -> None:
    stripes = _prism_stripes_for_configs(configs, year_range)
    if stripes.empty:
        return
    for row in stripes.itertuples(index=False):
        fig.add_vrect(
            x0=f"{int(row.year)}-01-01",
            x1=f"{int(row.year) + 1}-01-01",
            fillcolor=_stripe_fillcolor(row.classification, row.stripe_strength),
            line_width=0,
            layer="below",
        )
        fig.add_annotation(
            x=f"{int(row.year)}-07-01",
            y=1.035,
            xref="x",
            yref="paper",
            text=_stripe_label(row),
            showarrow=False,
            yanchor="bottom",
            font=PRISM_STRIPE_LABEL_FONT,
            opacity=0.85,
        )
    fig.add_annotation(
        x=1.0,
        y=1.035,
        xref="paper",
        yref="paper",
        text="mm precip.<br>(PRISM)",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=PRISM_STRIPE_LABEL_FONT,
        opacity=0.85,
    )


def _add_timeseries_trace(
    fig: go.Figure,
    config: ComparisonConfig,
    filtered: pd.DataFrame,
    color_override: str | None = None,
) -> None:
    x_axis = _resolve_x_axis(filtered, config.temporal_agg)
    filtered = filtered.sort_values(x_axis)
    if config.season_filter == "growing":
        filtered = _with_year_breaks(filtered, x_axis)
    color = color_override or _ecozone_trace_color(config, filtered)

    fig.add_trace(
        go.Scatter(
            x=filtered[x_axis],
            y=filtered["value"],
            mode="lines+markers",
            name=_series_label(config, filtered),
            line={"color": color} if color else None,
            marker={"color": color} if color else None,
            customdata=filtered[
                [
                    "sensor",
                    "aoi",
                    "index",
                    "analysis_scope",
                    "ecozone_label",
                    "forest_community_label",
                    "temporal_agg",
                    "spatial_percentile",
                    "temporal_percentile",
                    "pixel_mask_id",
                ]
            ],
            hovertemplate=(
                "Date=%{x|%Y-%m-%d}<br>"
                "Value=%{y:.4f}<br>"
                "Sensor=%{customdata[0]}<br>"
                "AOI=%{customdata[1]}<br>"
                "Index=%{customdata[2]}<br>"
                "Scope=%{customdata[3]}<br>"
                "Ecozone=%{customdata[4]}<br>"
                "Forest community=%{customdata[5]}<br>"
                "Temporal agg=%{customdata[6]}<br>"
                "Spatial pct=%{customdata[7]}<br>"
                "Temporal pct=%{customdata[8]}<br>"
                "Mask=%{customdata[9]}<extra></extra>"
            ),
        )
    )


def _visible_y_axis_range(fig: go.Figure) -> list[float] | None:
    values = []
    for trace in fig.data:
        values.extend(pd.to_numeric(pd.Series(trace.y), errors="coerce").dropna().tolist())
    if not values:
        return None

    y_min = min(values)
    y_max = max(values)
    y_pad = max((y_max - y_min) * 0.08, 0.01)
    lower = y_min - y_pad
    upper = y_max + y_pad
    if y_max <= 1.0 < upper:
        upper = 1.0
    return [lower, upper]


def build_timeseries_figure(
    bundle: DashboardDataBundle,
    configs: list[ComparisonConfig],
    year_range: tuple[int, int],
    visible_segments_by_layer: dict[int, set[int]] | None = None,
    combined_group_frames_by_layer: dict[int, list[pd.DataFrame | tuple[pd.DataFrame, str | None]]] | None = None,
    segment_color_offsets_by_layer: dict[int, dict[int, int]] | None = None,
    config_color_overrides_by_layer: dict[int, str] | None = None,
) -> tuple[go.Figure, list[str]]:
    fig = go.Figure()
    messages: list[str] = []
    visible_segments_by_layer = visible_segments_by_layer or {}
    combined_group_frames_by_layer = combined_group_frames_by_layer or {}
    segment_color_offsets_by_layer = segment_color_offsets_by_layer or {}
    config_color_overrides_by_layer = config_color_overrides_by_layer or {}

    for layer_idx, config in enumerate(configs):
        frame = bundle.frame_for_config(config)
        filtered = filter_frame(
            frame,
            filters=filters_for_config(config),
            year_range=year_range,
        )
        if filtered.empty:
            messages.append(f"No data for `{_series_label(config)}` in the selected year range.")
            continue

        if getattr(config, "analysis_scope", "overall") == "ecozone" and _is_all_segment_value(config.ecozone_code):
            any_trace = False
            visible_codes = visible_segments_by_layer.get(layer_idx)
            color_offsets = segment_color_offsets_by_layer.get(layer_idx, {})
            for ecozone_code, ecozone_frame in filtered.groupby("ecozone_code", dropna=True):
                if visible_codes is not None and int(ecozone_code) not in visible_codes:
                    continue
                ecozone_frame = _apply_stddev_filters(ecozone_frame, config)
                if ecozone_frame.empty:
                    continue
                color_offset = color_offsets.get(int(ecozone_code))
                _add_timeseries_trace(
                    fig,
                    config,
                    ecozone_frame,
                    color_override=(
                        pc.qualitative.Plotly[color_offset % len(pc.qualitative.Plotly)]
                        if color_offset is not None
                        else None
                    ),
                )
                any_trace = True
            if not any_trace:
                messages.append(f"All rows for `{_series_label(config)}` were removed by the standard-deviation filters.")
            continue

        if getattr(config, "analysis_scope", "overall") == "forest_community" and _is_all_segment_value(config.forest_community_code):
            any_trace = False
            visible_codes = visible_segments_by_layer.get(layer_idx)
            color_idx = 0
            color_offsets = segment_color_offsets_by_layer.get(layer_idx, {})
            for community_code, community_frame in filtered.groupby("forest_community_code", dropna=True):
                if visible_codes is not None and int(community_code) not in visible_codes:
                    continue
                community_frame = _apply_stddev_filters(community_frame, config)
                if community_frame.empty:
                    continue
                color_offset = color_offsets.get(int(community_code), color_idx)
                _add_timeseries_trace(
                    fig,
                    config,
                    community_frame,
                    color_override=pc.qualitative.Plotly[color_offset % len(pc.qualitative.Plotly)],
                )
                color_idx += 1
                any_trace = True
            if not any_trace and not combined_group_frames_by_layer.get(layer_idx):
                messages.append(f"All rows for `{_series_label(config)}` were removed by the standard-deviation filters.")
            continue

        filtered = _apply_stddev_filters(filtered, config)
        if filtered.empty:
            messages.append(f"All rows for `{_series_label(config)}` were removed by the standard-deviation filters.")
            continue

        _add_timeseries_trace(fig, config, filtered, color_override=config_color_overrides_by_layer.get(layer_idx))

    for layer_idx, group_frames in combined_group_frames_by_layer.items():
        if layer_idx >= len(configs):
            continue
        config = configs[layer_idx]
        for group_item in group_frames:
            if isinstance(group_item, tuple):
                group_frame, color = group_item
            else:
                group_frame = group_item
                color = None
            group_frame = _apply_stddev_filters(group_frame, config)
            if group_frame.empty:
                continue
            _add_timeseries_trace(fig, config, group_frame, color_override=color)

    fig.update_layout(
        template="plotly_white",
        title="Ecozone–Vegetation Time Series",
        xaxis_title="Date",
        yaxis_title="Summary value",
        hovermode="x unified",
        showlegend=False,
        margin={"l": 40, "r": 70, "t": 80, "b": 40},
    )
    y_axis_range = _visible_y_axis_range(fig)
    if y_axis_range is not None:
        fig.update_yaxes(range=y_axis_range)
    _add_prism_year_stripes(fig, configs, year_range)
    return fig, messages


def build_growing_season_figure(
    bundle: DashboardDataBundle,
    config: ComparisonConfig,
    selected_year: int,
    year_range: tuple[int, int] | None = None,
) -> tuple[go.Figure, str | None]:
    data_config = ComparisonConfig(
        **{
            **config.__dict__,
            "temporal_agg": "scene",
            "temporal_percentile": "none",
            "season_filter": "growing",
        }
    )
    filtered = filter_frame(
        bundle.frame_for_config(data_config),
        filters=filters_for_config(data_config),
        year_range=year_range,
    )
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


def available_growing_season_years(
    bundle: DashboardDataBundle,
    config: ComparisonConfig,
    year_range: tuple[int, int] | None = None,
) -> list[int]:
    data_config = ComparisonConfig(
        **{
            **config.__dict__,
            "temporal_agg": "scene",
            "temporal_percentile": "none",
            "season_filter": "growing",
        }
    )
    filtered = filter_frame(
        bundle.frame_for_config(data_config),
        filters=filters_for_config(data_config),
        year_range=year_range,
    )
    if filtered.empty or "year" not in filtered.columns:
        return []
    years = pd.to_numeric(filtered["year"], errors="coerce").dropna().astype(int)
    return sorted(years.unique().tolist())
