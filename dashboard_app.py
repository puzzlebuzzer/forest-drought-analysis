from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.colors as pc
import plotly.io as pio
import streamlit as st

from src.dashboard_data import filter_frame, load_dashboard_data
from src.dashboard_figures import build_growing_season_figure, build_timeseries_figure
from src.dashboard_schema import ComparisonConfig

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "Results" / "tables" / "dashboard_data"
ENABLE_GROWING_SEASON_OVERLAY = False
SPATIAL_PERCENTILE_LABELS = {
    "p50": "p50 (median)",
    "p75": "p75 (upper quartile)",
    "p100": "p100 (max)",
}
STDDEV_FILTER_OPTIONS = ["none", 1, 1.5, 2]


def _inject_ui_css() -> None:
    st.markdown(
        """
        <style>
        div[data-baseweb="select"] * {
            cursor: pointer !important;
        }
        div[data-baseweb="select"] input {
            caret-color: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    if "comparison_configs" not in st.session_state:
        st.session_state.comparison_configs = []
    if "default_overlay_seeded" not in st.session_state:
        st.session_state.default_overlay_seeded = False


def _select_or_default(options: list, preferred):
    if preferred in options:
        return preferred
    if options:
        return options[0]
    return None


def _safe_selectbox(label: str, options: list, preferred, format_func=None):
    if not options:
        st.sidebar.caption(f"No available values for {label.lower()} in the loaded CSVs.")
        return None
    selected = _select_or_default(options, preferred)
    try:
        return st.sidebar.selectbox(
            label,
            options,
            index=options.index(selected),
            filter_mode=None,
            format_func=format_func,
        )
    except TypeError:
        if format_func is not None:
            return st.sidebar.selectbox(label, options, index=options.index(selected), format_func=format_func)
        return st.sidebar.selectbox(label, options, index=options.index(selected))


def _safe_plain_selectbox(label: str, options: list, preferred):
    if not options:
        st.sidebar.caption(f"No available values for {label.lower()} in the loaded CSVs.")
        return None
    selected = _select_or_default(options, preferred)
    return st.sidebar.selectbox(label, options, index=options.index(selected))


def _stddev_option(value):
    return None if value == "none" else float(value)


def _config_display_label(config_dict: dict, idx: int | None = None) -> str:
    label = config_dict.get("label") or " / ".join(
        [
            str(config_dict["sensor"]),
            str(config_dict["aoi"]),
            str(config_dict["index"]),
            str(config_dict["temporal_agg"]),
            str(config_dict["temporal_percentile"]),
            str(config_dict["spatial_percentile"]),
            str(config_dict["cloud_threshold"]),
        ]
    )
    return f"{idx + 1}. {label}" if idx is not None else label


def _render_data_dir_control() -> Path:
    st.sidebar.header("Data")
    data_dir_input = st.sidebar.text_input("Summary CSV directory", value=str(DEFAULT_DATA_DIR))
    return Path(data_dir_input)


@st.cache_data(show_spinner=False)
def _load_dashboard_data_cached(data_dir: str):
    return load_dashboard_data(data_dir)


def _build_sidebar(bundle) -> tuple[tuple[int, int], ComparisonConfig | None]:
    year_min, year_max = bundle.available_year_range()
    if year_min == year_max:
        st.sidebar.caption(f"Only one year is available in the loaded tables: {year_min}")
        selected_year_range = (year_min, year_max)
    else:
        selected_year_range = st.sidebar.slider(
            "Year range",
            min_value=year_min,
            max_value=year_max,
            value=(year_min, year_max),
        )

    sensors = bundle.available_values("sensor")
    aois = bundle.available_values("aoi")
    indices = bundle.available_values("index")
    spatial_percentiles = bundle.available_values("spatial_percentile")
    temporal_aggs = bundle.available_values("temporal_agg")
    temporal_percentiles = bundle.available_values("temporal_percentile")
    cloud_thresholds = bundle.available_values("cloud_threshold")
    season_filters = bundle.available_values("season_filter")

    st.sidebar.header("Comparison builder")
    builder_modes = ["Create new overlay", "Edit existing overlay"]
    builder_mode = st.sidebar.radio("Builder mode", builder_modes, index=0 if not st.session_state.comparison_configs else 0)
    selected_existing = None
    if builder_mode == "Edit existing overlay" and st.session_state.comparison_configs:
        selected_existing = st.sidebar.selectbox(
            "Overlay to edit",
            list(range(len(st.session_state.comparison_configs))),
            format_func=lambda idx: _config_display_label(st.session_state.comparison_configs[idx], idx),
        )
    existing_defaults = st.session_state.comparison_configs[selected_existing] if selected_existing is not None else {}

    sensor = _safe_selectbox("Sensor", sensors, existing_defaults.get("sensor", "s2"))
    aoi = _safe_selectbox("AOI", aois, existing_defaults.get("aoi", "north"))
    index_name = _safe_selectbox("Index", indices, existing_defaults.get("index", "ndvi"))
    spatial_percentile = _safe_selectbox(
        "Spatial aggregation percentile",
        spatial_percentiles,
        existing_defaults.get("spatial_percentile", "p95"),
        format_func=lambda value: SPATIAL_PERCENTILE_LABELS.get(value, value),
    )
    temporal_agg = _safe_selectbox("Temporal aggregation", temporal_aggs, existing_defaults.get("temporal_agg", "scene"))
    if temporal_agg == "scene":
        temporal_percentile = "none"
    else:
        visible_temporal_percentiles = list(spatial_percentiles)
        temporal_percentile = _safe_selectbox(
            "Temporal aggregation percentile",
            visible_temporal_percentiles,
            existing_defaults.get("temporal_percentile", "p95"),
        )
    cloud_threshold = _safe_plain_selectbox("Cloud threshold", cloud_thresholds, existing_defaults.get("cloud_threshold", 40))
    season_filter = _safe_selectbox("Season filter", season_filters, existing_defaults.get("season_filter", "all"))
    exclude_below_default = existing_defaults.get("exclude_below_stddev")
    exclude_above_default = existing_defaults.get("exclude_above_stddev")
    exclude_below_stddev = _safe_plain_selectbox(
        "Exclude below z-score",
        STDDEV_FILTER_OPTIONS,
        "none" if exclude_below_default is None else exclude_below_default,
    )
    exclude_above_stddev = _safe_plain_selectbox(
        "Exclude above z-score",
        STDDEV_FILTER_OPTIONS,
        "none" if exclude_above_default is None else exclude_above_default,
    )
    label = st.sidebar.text_input("Optional custom label", value=existing_defaults.get("label", ""))

    if not all([sensor, aoi, index_name, spatial_percentile, temporal_agg, temporal_percentile, season_filter]):
        return selected_year_range, None

    config = ComparisonConfig(
        label=label.strip(),
        sensor=sensor,
        aoi=aoi,
        index=index_name,
        spatial_percentile=spatial_percentile,
        temporal_agg=temporal_agg,
        temporal_percentile=temporal_percentile,
        cloud_threshold=int(cloud_threshold),
        season_filter=season_filter,
        exclude_below_stddev=_stddev_option(exclude_below_stddev),
        exclude_above_stddev=_stddev_option(exclude_above_stddev),
    )
    return selected_year_range, config, builder_mode, selected_existing


def _render_config_table() -> None:
    if not st.session_state.comparison_configs:
        st.info("Add at least one comparison configuration to draw an overlay plot.")
        return
    st.subheader("Selected overlays")
    palette = pc.qualitative.Plotly
    for idx, config_dict in enumerate(st.session_state.comparison_configs):
        columns = st.columns([0.6, 4.4, 1])
        label = _config_display_label(config_dict, idx).split(". ", 1)[1]
        color = palette[idx % len(palette)]
        columns[0].markdown(
            f"""
            <div style="width: 0.9rem; height: 0.9rem; background:{color}; border-radius: 2px; margin-top: 0.2rem;"></div>
            """,
            unsafe_allow_html=True,
        )
        columns[1].markdown(f"`{idx + 1}` {label}")
        if columns[2].button("Remove", key=f"remove_{idx}"):
            st.session_state.comparison_configs.pop(idx)
            st.rerun()


def _build_export_subset(bundle, configs: list[ComparisonConfig], year_range: tuple[int, int]) -> pd.DataFrame:
    subsets = []
    for idx, config in enumerate(configs, start=1):
        frame = bundle.frame_for_config(config)
        filtered = filter_frame(frame, filters=asdict(config) | {"label": None}, year_range=year_range).copy()
        if filtered.empty:
            continue
        filtered.insert(0, "overlay_label", config.label or _config_display_label(asdict(config)))
        filtered.insert(1, "overlay_order", idx)
        subsets.append(filtered)
    if not subsets:
        return pd.DataFrame()
    return pd.concat(subsets, ignore_index=True)


def _render_export_controls(figure, export_frame: pd.DataFrame) -> None:
    st.subheader("Exports")
    col1, col2, col3, col4 = st.columns(4)
    csv_bytes = export_frame.to_csv(index=False).encode("utf-8") if not export_frame.empty else b""
    xlsx_buffer = BytesIO()
    if not export_frame.empty:
        with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
            export_frame.to_excel(writer, sheet_name="subset", index=False)
    html_bytes = figure.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")
    try:
        png_bytes = pio.to_image(figure, format="png", width=1400, height=800, scale=2)
    except Exception:
        png_bytes = None

    col1.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="dashboard_subset.csv",
        mime="text/csv",
        disabled=export_frame.empty,
        use_container_width=True,
    )
    col2.download_button(
        "Download Excel",
        data=xlsx_buffer.getvalue(),
        file_name="dashboard_subset.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        disabled=export_frame.empty,
        use_container_width=True,
    )
    col3.download_button(
        "Download PNG",
        data=png_bytes or b"",
        file_name="dashboard_plot.png",
        mime="image/png",
        disabled=png_bytes is None,
        use_container_width=True,
    )
    col4.download_button(
        "Download HTML",
        data=html_bytes,
        file_name="dashboard_plot.html",
        mime="text/html",
        use_container_width=True,
    )
    if png_bytes is None:
        st.caption("PNG export requires Plotly static image support such as `kaleido` in the active environment.")


def main() -> None:
    st.set_page_config(page_title="Appalachian Terrain–Vegetation Dashboard", layout="wide")
    _inject_ui_css()
    _init_state()

    st.title("Appalachian Terrain–Vegetation Analysis Dashboard")

    data_dir = _render_data_dir_control()
    bundle = _load_dashboard_data_cached(str(data_dir))
    year_range, config, builder_mode, selected_existing = _build_sidebar(bundle)

    if config and not st.session_state.default_overlay_seeded and not st.session_state.comparison_configs:
        st.session_state.comparison_configs.append(asdict(config))
        st.session_state.default_overlay_seeded = True
        st.rerun()

    action_label = "Update overlay" if builder_mode == "Edit existing overlay" and selected_existing is not None else "Add overlay"
    if config and st.sidebar.button(action_label, type="primary"):
        if builder_mode == "Edit existing overlay" and selected_existing is not None:
            st.session_state.comparison_configs[selected_existing] = asdict(config)
        else:
            st.session_state.comparison_configs.append(asdict(config))
        st.rerun()

    st.subheader("Time-series overlay")
    config_objects = [ComparisonConfig(**cfg) for cfg in st.session_state.comparison_configs]
    figure, messages = build_timeseries_figure(bundle, config_objects, year_range)
    if figure.data:
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.warning("No lines could be drawn from the current configurations and year range.")
    for message in messages:
        st.info(message)

    _render_config_table()
    export_frame = _build_export_subset(bundle, config_objects, year_range)
    _render_export_controls(figure, export_frame)


if __name__ == "__main__":
    main()
