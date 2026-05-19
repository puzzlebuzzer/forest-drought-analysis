from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import plotly.colors as pc
import plotly.io as pio
import streamlit as st

from src.dashboard_data import filter_frame, filters_for_config, load_dashboard_data
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
    if "prepared_exports" not in st.session_state:
        st.session_state.prepared_exports = {}
    if "builder_mode" not in st.session_state:
        st.session_state.builder_mode = "edit"
    if "builder_target_index" not in st.session_state:
        st.session_state.builder_target_index = None
    if "builder_pending_values" not in st.session_state:
        st.session_state.builder_pending_values = None


def _select_or_default(options: list, preferred):
    if preferred in options:
        return preferred
    if options:
        return options[0]
    return None


def _safe_selectbox(label: str, options: list, preferred, format_func=None, scope=st.sidebar, key: str | None = None):
    if not options:
        scope.caption(f"No available values for {label.lower()} in the loaded CSVs.")
        return None
    selected = _select_or_default(options, st.session_state.get(key, preferred) if key else preferred)
    try:
        return scope.selectbox(
            label,
            options,
            index=options.index(selected),
            filter_mode=None,
            format_func=format_func,
            key=key,
            disabled=False,
        )
    except TypeError:
        if format_func is not None:
            return scope.selectbox(label, options, index=options.index(selected), format_func=format_func, key=key)
        return scope.selectbox(label, options, index=options.index(selected), key=key)


def _safe_plain_selectbox(label: str, options: list, preferred, scope=st.sidebar, key: str | None = None):
    if not options:
        scope.caption(f"No available values for {label.lower()} in the loaded CSVs.")
        return None
    selected = _select_or_default(options, st.session_state.get(key, preferred) if key else preferred)
    return scope.selectbox(label, options, index=options.index(selected), key=key)


def _safe_selectbox_disabled(
    label: str,
    options: list,
    preferred,
    scope=st.sidebar,
    key: str | None = None,
):
    if not options:
        scope.caption(f"No available values for {label.lower()} in the loaded CSVs.")
        return None
    selected = _select_or_default(options, st.session_state.get(key, preferred) if key else preferred)
    try:
        return scope.selectbox(
            label,
            options,
            index=options.index(selected),
            filter_mode=None,
            key=key,
            disabled=True,
        )
    except TypeError:
        return scope.selectbox(label, options, index=options.index(selected), key=key, disabled=True)


def _stddev_option(value):
    return None if value == "none" else float(value)


def _config_display_label(config_dict: dict, idx: int | None = None) -> str:
    label = config_dict.get("label") or " / ".join(
        [
            str(config_dict["aoi"]),
            str(config_dict["sensor"]),
            str(config_dict["index"]),
            str(config_dict["cloud_threshold"]),
            str(config_dict["spatial_percentile"]),
            str(config_dict["temporal_agg"]),
            str(config_dict["temporal_percentile"]),
        ]
    )
    return f"{idx + 1}. {label}" if idx is not None else label


def _render_data_dir_control() -> Path:
    data_dir_input = st.sidebar.text_input("Summary CSV directory", value=str(DEFAULT_DATA_DIR))
    return Path(data_dir_input)


@st.cache_data(show_spinner=False)
def _load_dashboard_data_cached(data_dir: str):
    return load_dashboard_data(data_dir)


def _default_config_dict(bundle) -> dict:
    spatial_percentiles = bundle.available_values("spatial_percentile")
    return {
        "label": "",
        "sensor": _select_or_default(bundle.available_values("sensor"), "s2"),
        "aoi": _select_or_default(bundle.available_values("aoi"), "north"),
        "index": _select_or_default(bundle.available_values("index"), "ndvi"),
        "spatial_percentile": _select_or_default(spatial_percentiles, "p95"),
        "temporal_agg": _select_or_default(bundle.available_values("temporal_agg"), "scene"),
        "temporal_percentile": _select_or_default(spatial_percentiles, "p95"),
        "cloud_threshold": _select_or_default(bundle.available_values("cloud_threshold"), 40),
        "season_filter": _select_or_default(bundle.available_values("season_filter"), "all"),
        "exclude_below_stddev": None,
        "exclude_above_stddev": None,
    }


def _load_builder_values(config_dict: dict) -> None:
    st.session_state.builder_pending_values = {
        "builder_sensor": config_dict.get("sensor"),
        "builder_aoi": config_dict.get("aoi"),
        "builder_index": config_dict.get("index"),
        "builder_spatial_percentile": config_dict.get("spatial_percentile"),
        "builder_temporal_agg": config_dict.get("temporal_agg"),
        "builder_temporal_percentile": config_dict.get("temporal_percentile"),
        "builder_cloud_threshold": config_dict.get("cloud_threshold"),
        "builder_season_filter": config_dict.get("season_filter"),
        "builder_exclude_below_stddev": "none" if config_dict.get("exclude_below_stddev") is None else config_dict.get("exclude_below_stddev"),
        "builder_exclude_above_stddev": "none" if config_dict.get("exclude_above_stddev") is None else config_dict.get("exclude_above_stddev"),
        "builder_label": config_dict.get("label", ""),
    }


def _apply_pending_builder_values() -> None:
    pending_values = st.session_state.builder_pending_values
    if not pending_values:
        return
    for key, value in pending_values.items():
        st.session_state[key] = value
    st.session_state.builder_pending_values = None


def _start_new_overlay(bundle) -> None:
    st.session_state.builder_mode = "new"
    st.session_state.builder_target_index = None
    _load_builder_values(_default_config_dict(bundle))


def _start_edit_overlay(config_dict: dict, index: int) -> None:
    st.session_state.builder_mode = "edit"
    st.session_state.builder_target_index = index
    _load_builder_values(config_dict)


def _ensure_builder_state(bundle) -> None:
    if st.session_state.builder_mode == "edit":
        target_index = st.session_state.builder_target_index
        if target_index is None or not (0 <= target_index < len(st.session_state.comparison_configs)):
            if st.session_state.comparison_configs:
                target_index = len(st.session_state.comparison_configs) - 1
                _start_edit_overlay(st.session_state.comparison_configs[target_index], target_index)
            else:
                _start_new_overlay(bundle)
                return
    required_keys = [
        "builder_sensor",
        "builder_aoi",
        "builder_index",
        "builder_spatial_percentile",
        "builder_temporal_agg",
        "builder_temporal_percentile",
        "builder_cloud_threshold",
        "builder_season_filter",
        "builder_exclude_below_stddev",
        "builder_exclude_above_stddev",
        "builder_label",
    ]
    if all(key in st.session_state for key in required_keys):
        return
    if st.session_state.comparison_configs:
        target_index = len(st.session_state.comparison_configs) - 1
        _start_edit_overlay(st.session_state.comparison_configs[target_index], target_index)
    else:
        _start_new_overlay(bundle)


def _build_sidebar(bundle) -> tuple[tuple[int, int], ComparisonConfig | None, str | None, int | None]:
    _apply_pending_builder_values()

    if st.sidebar.button("Add layer"):
        _start_new_overlay(bundle)
        st.rerun()

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
    cloud_thresholds = bundle.available_values("cloud_threshold")
    season_filters = bundle.available_values("season_filter")

    _ensure_builder_state(bundle)
    with st.sidebar.form("comparison_builder_form", enter_to_submit=False):
        aoi = _safe_selectbox("AOI", aois, "north", scope=st, key="builder_aoi")
        sensor = _safe_selectbox("Sensor", sensors, "s2", scope=st, key="builder_sensor")
        index_name = _safe_selectbox("Index", indices, "ndvi", scope=st, key="builder_index")
        cloud_threshold = _safe_plain_selectbox("Cloud threshold", cloud_thresholds, 40, scope=st, key="builder_cloud_threshold")
        spatial_percentile = _safe_selectbox(
            "Spatial aggregation percentile",
            spatial_percentiles,
            "p95",
            format_func=lambda value: SPATIAL_PERCENTILE_LABELS.get(value, value),
            scope=st,
            key="builder_spatial_percentile",
        )
        temporal_agg = _safe_selectbox("Interval", temporal_aggs, "scene", scope=st, key="builder_temporal_agg")
        temporal_percentile = _safe_selectbox(
            "Interval aggregation percentile",
            list(spatial_percentiles),
            "p95",
            scope=st,
            key="builder_temporal_percentile",
        ) if temporal_agg != "scene" else _safe_selectbox_disabled(
            "Interval aggregation percentile",
            ["none"],
            "none",
            scope=st,
            key="builder_temporal_percentile",
        )
        season_filter = _safe_selectbox("Season filter", season_filters, "all", scope=st, key="builder_season_filter")
        exclude_below_stddev = _safe_plain_selectbox(
            "Exclude below z-score",
            STDDEV_FILTER_OPTIONS,
            "none",
            scope=st,
            key="builder_exclude_below_stddev",
        )
        exclude_above_stddev = _safe_plain_selectbox(
            "Exclude above z-score",
            STDDEV_FILTER_OPTIONS,
            "none",
            scope=st,
            key="builder_exclude_above_stddev",
        )
        label = st.text_input("Optional custom label", key="builder_label")
        action_label = "Add layer" if st.session_state.builder_mode == "new" else "Apply changes"
        submitted = st.form_submit_button(action_label, type="primary", width="stretch")

    if not all([sensor, aoi, index_name, spatial_percentile, temporal_agg, temporal_percentile, season_filter]):
        return selected_year_range, None, None, st.session_state.builder_target_index

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
    action = None
    if submitted:
        action = "add" if st.session_state.builder_mode == "new" else "update"
    return selected_year_range, config, action, st.session_state.builder_target_index


def _render_config_table() -> None:
    if not st.session_state.comparison_configs:
        st.info("Add at least one comparison configuration to draw a comparison plot.")
        return
    st.subheader("Layers")
    palette = pc.qualitative.Plotly
    for idx, config_dict in enumerate(st.session_state.comparison_configs):
        columns = st.columns([0.6, 4.0, 0.8, 1])
        label = _config_display_label(config_dict, idx).split(". ", 1)[1]
        color = palette[idx % len(palette)]
        columns[0].markdown(
            f"""
            <div style="width: 0.9rem; height: 0.9rem; background:{color}; border-radius: 2px; margin-top: 0.2rem;"></div>
            """,
            unsafe_allow_html=True,
        )
        columns[1].markdown(f"`{idx + 1}` {label}")
        if columns[2].button("Edit", key=f"edit_{idx}"):
            _start_edit_overlay(config_dict, idx)
            st.rerun()
        if columns[3].button("Remove", key=f"remove_{idx}"):
            st.session_state.comparison_configs.pop(idx)
            if st.session_state.comparison_configs:
                next_index = min(idx, len(st.session_state.comparison_configs) - 1)
                _start_edit_overlay(st.session_state.comparison_configs[next_index], next_index)
            else:
                st.session_state.builder_target_index = None
                st.session_state.builder_mode = "new"
            st.rerun()


def _build_export_subset(bundle, configs: list[ComparisonConfig], year_range: tuple[int, int]) -> pd.DataFrame:
    subsets = []
    for idx, config in enumerate(configs, start=1):
        frame = bundle.frame_for_config(config)
        filtered = filter_frame(frame, filters=filters_for_config(config), year_range=year_range).copy()
        if filtered.empty:
            continue
        filtered.insert(0, "comparison_label", config.label or _config_display_label(asdict(config)))
        filtered.insert(1, "comparison_order", idx)
        subsets.append(filtered)
    if not subsets:
        return pd.DataFrame()
    return pd.concat(subsets, ignore_index=True)


def _export_signature(configs: list[ComparisonConfig], year_range: tuple[int, int]) -> str:
    payload = {
        "year_range": list(year_range),
        "configs": [asdict(config) for config in configs],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _prepare_data_export(bundle, configs: list[ComparisonConfig], year_range: tuple[int, int]) -> bytes:
    export_frame = _build_export_subset(bundle, configs, year_range)
    return export_frame.to_csv(index=False).encode("utf-8") if not export_frame.empty else b""


def _prepare_plot_export(figure) -> bytes | None:
    try:
        png_bytes = pio.to_image(figure, format="png", width=1400, height=800, scale=2)
    except Exception:
        png_bytes = None
    return png_bytes


def _render_staged_download_button(
    container,
    label: str,
    export_key: str,
    prepared: dict,
    signature: str,
    file_name: str,
    mime: str,
    data: bytes | None,
    prepare_fn,
    unavailable_reason: str | None = None,
) -> None:
    if data is not None:
        container.download_button(
            label,
            data=data,
            file_name=file_name,
            mime=mime,
            width="content",
        )
        return
    if container.button(label, width="content", disabled=unavailable_reason is not None):
        prepared = {**prepared, export_key: prepare_fn()}
        st.session_state.prepared_exports[signature] = prepared
        st.rerun()


def _render_export_controls(bundle, configs: list[ComparisonConfig], year_range: tuple[int, int], figure) -> None:
    st.subheader("Exports")
    signature = _export_signature(configs, year_range)
    prepared = st.session_state.prepared_exports.get(signature, {})

    col1, col2, _ = st.columns([1, 1, 8])
    csv_bytes = prepared.get("csv")
    png_bytes = prepared.get("png")
    _render_staged_download_button(
        col1,
        "CSV",
        "csv",
        prepared,
        signature,
        "dashboard_subset.csv",
        "text/csv",
        csv_bytes,
        lambda: _prepare_data_export(bundle, configs, year_range),
    )
    _render_staged_download_button(
        col2,
        "PNG",
        "png",
        prepared,
        signature,
        "dashboard_plot.png",
        "image/png",
        png_bytes,
        lambda: _prepare_plot_export(figure),
    )


def main() -> None:
    st.set_page_config(page_title="Appalachian Terrain–Vegetation Dashboard", layout="wide")
    _inject_ui_css()
    _init_state()

    st.title("Appalachian Terrain–Vegetation Analysis Dashboard")

    data_dir = _render_data_dir_control()
    bundle = _load_dashboard_data_cached(str(data_dir))
    year_range, config, builder_action, selected_existing = _build_sidebar(bundle)

    if config and not st.session_state.default_overlay_seeded and not st.session_state.comparison_configs:
        seeded_config = asdict(config)
        st.session_state.comparison_configs.append(seeded_config)
        st.session_state.default_overlay_seeded = True
        _start_edit_overlay(seeded_config, 0)
        st.rerun()

    if config and builder_action is not None:
        if builder_action == "update" and selected_existing is not None:
            updated_config = asdict(config)
            st.session_state.comparison_configs[selected_existing] = updated_config
            _start_edit_overlay(updated_config, selected_existing)
        else:
            new_config = asdict(config)
            st.session_state.comparison_configs.append(new_config)
            _start_edit_overlay(new_config, len(st.session_state.comparison_configs) - 1)
        st.rerun()

    config_objects = [ComparisonConfig(**cfg) for cfg in st.session_state.comparison_configs]
    figure, messages = build_timeseries_figure(bundle, config_objects, year_range)
    if figure.data:
        st.plotly_chart(figure, width="stretch")
    else:
        st.warning("No lines could be drawn from the current configurations and year range.")
    for message in messages:
        st.info(message)

    _render_config_table()
    _render_export_controls(bundle, config_objects, year_range, figure)


if __name__ == "__main__":
    main()
