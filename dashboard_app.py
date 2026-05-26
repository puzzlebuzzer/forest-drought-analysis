from __future__ import annotations

import copy
from io import BytesIO
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import textwrap

import pandas as pd
import plotly.colors as pc
import plotly.io as pio
import streamlit as st

from src.dashboard_data import ECOZONE_LABELS, filter_frame, filters_for_config, load_dashboard_data
from src.dashboard_figures import ECOZONE_TRACE_COLORS, build_growing_season_figure, build_timeseries_figure
from src.dashboard_schema import ComparisonConfig

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "Results" / "tables" / "dashboard_data"
ENABLE_GROWING_SEASON_OVERLAY = False
DEFAULT_SPATIAL_PERCENTILE = "p99"
DEFAULT_TEMPORAL_AGG = "half_month"
DEFAULT_TEMPORAL_PERCENTILE = "p99"
SPATIAL_PERCENTILE_LABELS = {
    "p50": "p50 (median)",
    "p75": "p75 (upper quartile)",
    "p100": "p100 (max)",
}
AOI_LABELS = {
    "north": "GW-Jeff",
    "south": "Smoky",
}
SENSOR_LABELS = {
    "ls": "landsat",
    "s2": "sentinel-2",
}
STDDEV_FILTER_OPTIONS = ["none", 1, 1.5, 2]
DEFAULT_EXCLUDE_BELOW_STDDEV = 2
ECOZONE_ALL_OPTION = "all"
ANALYSIS_SCOPE_LABELS = {
    "overall": "Overall",
    "ecozone": "Ecozone",
    "forest_community": "Forest community",
}
BUILDER_WIDGET_KEYS = [
    "builder_analysis_scope",
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
    "builder_ecozone_code",
    "builder_forest_community_code",
]
REQUIRED_BUILDER_WIDGET_KEYS = [
    key for key in BUILDER_WIDGET_KEYS if key not in {"builder_ecozone_code", "builder_forest_community_code"}
]


def _cloud_threshold_label(value) -> str:
    if value is None:
        return "none"
    try:
        return f"{int(float(value))}%"
    except (TypeError, ValueError):
        return str(value)


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
    if "builder_defaults" not in st.session_state:
        st.session_state.builder_defaults = {}
    if "builder_state_migrated" not in st.session_state:
        st.session_state.builder_defaults = {
            key: st.session_state[key]
            for key in BUILDER_WIDGET_KEYS
            if key in st.session_state
        }
        for key in BUILDER_WIDGET_KEYS:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.builder_state_migrated = True


def _select_or_default(options: list, preferred):
    if preferred in options:
        return preferred
    if options:
        return options[0]
    return None


def _selectbox_kwargs(
    options: list,
    selected,
    *,
    key: str | None,
    format_func=None,
    disabled: bool = False,
) -> dict:
    kwargs = {
        "options": options,
        "key": key,
        "disabled": disabled,
    }
    if format_func is not None:
        kwargs["format_func"] = format_func
    if not key or key not in st.session_state:
        kwargs["index"] = options.index(selected)
    return kwargs


def _safe_selectbox(
    label: str,
    options: list,
    preferred,
    format_func=None,
    scope=st.sidebar,
    key: str | None = None,
    disabled: bool = False,
):
    if not options:
        scope.caption(f"No available values for {label.lower()} in the loaded CSVs.")
        return None
    selected = _select_or_default(options, st.session_state.get(key, preferred) if key else preferred)
    kwargs = _selectbox_kwargs(options, selected, key=key, format_func=format_func, disabled=disabled)
    try:
        return scope.selectbox(
            label,
            kwargs.pop("options"),
            filter_mode=None,
            **kwargs,
        )
    except TypeError:
        kwargs = _selectbox_kwargs(options, selected, key=key, format_func=format_func, disabled=disabled)
        return scope.selectbox(label, kwargs.pop("options"), **kwargs)


def _safe_plain_selectbox(
    label: str,
    options: list,
    preferred,
    scope=st.sidebar,
    key: str | None = None,
    format_func=None,
):
    if not options:
        scope.caption(f"No available values for {label.lower()} in the loaded CSVs.")
        return None
    selected = _select_or_default(options, st.session_state.get(key, preferred) if key else preferred)
    kwargs = _selectbox_kwargs(options, selected, key=key, format_func=format_func)
    return scope.selectbox(label, kwargs.pop("options"), **kwargs)


def _stddev_option(value):
    return None if value == "none" else float(value)


def _ecozone_option_label(value) -> str:
    if value == ECOZONE_ALL_OPTION or value is None:
        return "All"
    return str(ECOZONE_LABELS.get(value, value)).lower()


def _forest_community_option_label(value, bundle=None) -> str:
    if value == ECOZONE_ALL_OPTION or value is None:
        return "All"
    if bundle is None:
        return f"community {value}"
    return bundle.label_for_code(
        "forest_community_code",
        "forest_community_label",
        value,
        f"Forest community {value}",
    )


def _config_display_label(config_dict: dict, idx: int | None = None, bundle=None) -> str:
    parts = [
        str(AOI_LABELS.get(config_dict["aoi"], config_dict["aoi"])),
        str(SENSOR_LABELS.get(config_dict["sensor"], config_dict["sensor"])),
        str(config_dict["index"]),
    ]
    analysis_scope = config_dict.get("analysis_scope", "overall")
    if analysis_scope == "ecozone":
        ecozone_code = config_dict.get("ecozone_code")
        parts.append(_ecozone_option_label(ecozone_code))
    elif analysis_scope == "forest_community":
        forest_community_code = config_dict.get("forest_community_code")
        parts.append(_forest_community_option_label(forest_community_code, bundle))
    parts.extend(
        [
            _cloud_threshold_label(config_dict["cloud_threshold"]),
            str(config_dict["spatial_percentile"]),
            str(config_dict["temporal_agg"]),
            str(config_dict["temporal_percentile"]),
        ]
    )
    label = config_dict.get("label") or " / ".join(
        parts
    )
    return f"{idx + 1}. {label}" if idx is not None else label


def _is_all_segment_config(config_dict: dict) -> bool:
    analysis_scope = config_dict.get("analysis_scope", "overall")
    if analysis_scope == "ecozone":
        return config_dict.get("ecozone_code") is None
    if analysis_scope == "forest_community":
        return config_dict.get("forest_community_code") is None
    return False


def _segment_legend_entries(bundle, config: ComparisonConfig, year_range: tuple[int, int]) -> list[tuple[int, str]]:
    analysis_scope = getattr(config, "analysis_scope", "overall")
    if analysis_scope == "ecozone" and config.ecozone_code is None:
        code_column = "ecozone_code"
        label_column = "ecozone_label"
        fallback_prefix = "Ecozone"
    elif analysis_scope == "forest_community" and config.forest_community_code is None:
        code_column = "forest_community_code"
        label_column = "forest_community_label"
        fallback_prefix = "Forest community"
    else:
        return []

    frame = bundle.frame_for_config(config)
    filtered = filter_frame(frame, filters=filters_for_config(config), year_range=year_range)
    if filtered.empty or code_column not in filtered.columns:
        return []

    entries = []
    for code, segment_frame in filtered.groupby(code_column, dropna=True):
        labels = segment_frame[label_column].dropna().astype(str) if label_column in segment_frame.columns else pd.Series(dtype=str)
        labels = labels[labels != ""]
        label = labels.iloc[0] if not labels.empty else f"{fallback_prefix} {int(code)}"
        entries.append((int(code), label))
    return sorted(entries, key=lambda item: item[0])


def _segment_checkbox_key(layer_idx: int, segment_code: int) -> str:
    return f"layer_{layer_idx}_segment_{segment_code}_visible"


def _visible_segments_by_layer(
    bundle,
    configs: list[ComparisonConfig],
    year_range: tuple[int, int],
) -> dict[int, set[int]]:
    visible: dict[int, set[int]] = {}
    for idx, config in enumerate(configs):
        entries = _segment_legend_entries(bundle, config, year_range)
        if not entries:
            continue
        selected_codes = set()
        for code, _ in entries:
            key = _segment_checkbox_key(idx, code)
            if key not in st.session_state:
                st.session_state[key] = True
            if st.session_state.get(key, True):
                selected_codes.add(code)
        visible[idx] = selected_codes
    return visible


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
        "analysis_scope": "overall",
        "sensor": _select_or_default(bundle.available_values("sensor"), "ls"),
        "aoi": _select_or_default(bundle.available_values("aoi"), "north"),
        "index": _select_or_default(bundle.available_values("index"), "ndvi"),
        "ecozone_code": None,
        "forest_community_code": None,
        "spatial_percentile": _select_or_default(spatial_percentiles, DEFAULT_SPATIAL_PERCENTILE),
        "temporal_agg": _select_or_default(bundle.available_values("temporal_agg"), DEFAULT_TEMPORAL_AGG),
        "temporal_percentile": _select_or_default(spatial_percentiles, DEFAULT_TEMPORAL_PERCENTILE),
        "cloud_threshold": _select_or_default(bundle.available_values("cloud_threshold"), 40),
        "season_filter": _select_or_default(bundle.available_values("season_filter"), "growing"),
        "exclude_below_stddev": DEFAULT_EXCLUDE_BELOW_STDDEV,
        "exclude_above_stddev": None,
    }


def _load_builder_values(config_dict: dict) -> None:
    st.session_state.builder_pending_values = {
        "builder_analysis_scope": config_dict.get("analysis_scope", "overall"),
        "builder_sensor": config_dict.get("sensor"),
        "builder_aoi": config_dict.get("aoi"),
        "builder_index": config_dict.get("index"),
        "builder_ecozone_code": (
            config_dict.get("ecozone_code") if config_dict.get("ecozone_code") is not None else ECOZONE_ALL_OPTION
        ),
        "builder_forest_community_code": (
            config_dict.get("forest_community_code")
            if config_dict.get("forest_community_code") is not None
            else ECOZONE_ALL_OPTION
        ),
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
    st.session_state.builder_defaults = dict(pending_values)
    for key in pending_values:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.builder_pending_values = None


def _builder_default(key: str, fallback):
    return st.session_state.builder_defaults.get(key, fallback)


def _start_new_overlay(bundle) -> None:
    st.session_state.builder_mode = "new"
    st.session_state.builder_target_index = None
    _load_builder_values(_default_config_dict(bundle))


def _start_edit_overlay(config_dict: dict, index: int) -> None:
    st.session_state.builder_mode = "edit"
    st.session_state.builder_target_index = index
    _load_builder_values(config_dict)


def _ensure_builder_state(bundle) -> None:
    if st.session_state.builder_mode == "new":
        if not st.session_state.builder_defaults and not st.session_state.builder_pending_values:
            _load_builder_values(_default_config_dict(bundle))
        return
    if st.session_state.builder_mode == "edit":
        target_index = st.session_state.builder_target_index
        if target_index is None or not (0 <= target_index < len(st.session_state.comparison_configs)):
            if st.session_state.comparison_configs:
                target_index = len(st.session_state.comparison_configs) - 1
                _start_edit_overlay(st.session_state.comparison_configs[target_index], target_index)
            else:
                _start_new_overlay(bundle)
                return
    if all(key in st.session_state for key in REQUIRED_BUILDER_WIDGET_KEYS):
        return
    if st.session_state.comparison_configs:
        target_index = len(st.session_state.comparison_configs) - 1
        _start_edit_overlay(st.session_state.comparison_configs[target_index], target_index)
    else:
        _start_new_overlay(bundle)


def _build_sidebar(bundle) -> tuple[tuple[int, int], ComparisonConfig | None, str | None, int | None]:
    _apply_pending_builder_values()

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

    if st.sidebar.button("Add new layer"):
        _start_new_overlay(bundle)
        st.rerun()

    sensors = bundle.available_values("sensor")
    aois = bundle.available_values("aoi")
    indices = bundle.available_values("index")
    analysis_scopes = bundle.available_values("analysis_scope")
    ecozone_codes = [ECOZONE_ALL_OPTION, *bundle.available_values("ecozone_code")]
    forest_community_codes = [ECOZONE_ALL_OPTION, *bundle.available_values("forest_community_code")]
    spatial_percentiles = bundle.available_values("spatial_percentile")
    temporal_aggs = bundle.available_values("temporal_agg")
    cloud_thresholds = bundle.available_values("cloud_threshold")
    season_filters = bundle.available_values("season_filter")

    _ensure_builder_state(bundle)
    _apply_pending_builder_values()
    with st.sidebar.form("comparison_builder_form", enter_to_submit=False):
        analysis_scope = _safe_selectbox(
            "Scope",
            analysis_scopes,
            _builder_default("builder_analysis_scope", "overall"),
            scope=st,
            key="builder_analysis_scope",
            format_func=lambda value: ANALYSIS_SCOPE_LABELS.get(value, value),
        )
        aoi = _safe_selectbox(
            "AOI",
            aois,
            _builder_default("builder_aoi", "north"),
            scope=st,
            key="builder_aoi",
            format_func=lambda value: AOI_LABELS.get(value, value),
        )
        sensor = _safe_selectbox(
            "Sensor",
            sensors,
            _builder_default("builder_sensor", "ls"),
            scope=st,
            key="builder_sensor",
            format_func=lambda value: SENSOR_LABELS.get(value, value),
        )
        index_name = _safe_selectbox("Index", indices, _builder_default("builder_index", "ndvi"), scope=st, key="builder_index")
        ecozone_code = None
        forest_community_code = None
        if analysis_scope == "ecozone":
            ecozone_code = _safe_selectbox(
                "Ecozone",
                ecozone_codes,
                _builder_default("builder_ecozone_code", ECOZONE_ALL_OPTION),
                scope=st,
                key="builder_ecozone_code",
                format_func=_ecozone_option_label,
            )
        elif analysis_scope == "forest_community":
            forest_community_code = _safe_selectbox(
                "Forest community",
                forest_community_codes,
                _builder_default(
                    "builder_forest_community_code",
                    ECOZONE_ALL_OPTION,
                ),
                scope=st,
                key="builder_forest_community_code",
                format_func=lambda value: _forest_community_option_label(value, bundle),
            )
        cloud_threshold = _safe_plain_selectbox(
            "Cloud threshold",
            cloud_thresholds,
            _builder_default("builder_cloud_threshold", 40),
            scope=st,
            key="builder_cloud_threshold",
            format_func=_cloud_threshold_label,
        )
        spatial_percentile = _safe_selectbox(
            "Spatial aggregation percentile",
            spatial_percentiles,
            _builder_default("builder_spatial_percentile", DEFAULT_SPATIAL_PERCENTILE),
            format_func=lambda value: SPATIAL_PERCENTILE_LABELS.get(value, value),
            scope=st,
            key="builder_spatial_percentile",
        )
        temporal_agg = _safe_selectbox(
            "Interval",
            temporal_aggs,
            _builder_default("builder_temporal_agg", DEFAULT_TEMPORAL_AGG),
            scope=st,
            key="builder_temporal_agg",
        )
        temporal_percentile = _safe_selectbox(
            "Interval aggregation percentile",
            list(spatial_percentiles),
            _builder_default("builder_temporal_percentile", DEFAULT_TEMPORAL_PERCENTILE),
            scope=st,
            key="builder_temporal_percentile",
            format_func=lambda value: SPATIAL_PERCENTILE_LABELS.get(value, value),
        )
        season_filter = _safe_selectbox(
            "Season filter",
            season_filters,
            _builder_default("builder_season_filter", "growing"),
            scope=st,
            key="builder_season_filter",
        )
        exclude_above_stddev = _safe_plain_selectbox(
            "Exclude above z-score",
            STDDEV_FILTER_OPTIONS,
            _builder_default("builder_exclude_above_stddev", "none"),
            scope=st,
            key="builder_exclude_above_stddev",
        )
        exclude_below_stddev = _safe_plain_selectbox(
            "Exclude below z-score",
            STDDEV_FILTER_OPTIONS,
            _builder_default("builder_exclude_below_stddev", DEFAULT_EXCLUDE_BELOW_STDDEV),
            scope=st,
            key="builder_exclude_below_stddev",
        )
        label_kwargs = {"key": "builder_label"}
        if "builder_label" not in st.session_state:
            label_kwargs["value"] = _builder_default("builder_label", "")
        label = st.text_input("Optional custom label", **label_kwargs)
        action_label = "Add new layer" if st.session_state.builder_mode == "new" else "Apply changes"
        submitted = st.form_submit_button(action_label, type="primary", width="stretch")

    if not all([analysis_scope, sensor, aoi, index_name, spatial_percentile, temporal_agg, temporal_percentile, season_filter]):
        return selected_year_range, None, None, st.session_state.builder_target_index
    if analysis_scope == "ecozone" and ecozone_code is None:
        return selected_year_range, None, None, st.session_state.builder_target_index
    if analysis_scope == "forest_community" and forest_community_code is None:
        return selected_year_range, None, None, st.session_state.builder_target_index

    config = ComparisonConfig(
        label=label.strip(),
        analysis_scope=analysis_scope,
        sensor=sensor,
        aoi=aoi,
        index=index_name,
        ecozone_code=(
            int(ecozone_code)
            if analysis_scope == "ecozone" and ecozone_code != ECOZONE_ALL_OPTION
            else None
        ),
        forest_community_code=(
            int(forest_community_code)
            if analysis_scope == "forest_community" and forest_community_code != ECOZONE_ALL_OPTION
            else None
        ),
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


def _segment_legend_color(config: ComparisonConfig, code: int, palette: list[str], color_idx: int) -> str:
    if getattr(config, "analysis_scope", "overall") == "ecozone":
        return ECOZONE_TRACE_COLORS.get(int(code), palette[color_idx % len(palette)])
    return palette[color_idx % len(palette)]


def _render_segment_legend(
    entries: list[tuple[int, str]],
    config: ComparisonConfig,
    palette: list[str],
    start_color_idx: int,
    layer_idx: int,
    selected_color_offsets: dict[int, int],
) -> None:
    if not entries:
        return
    for offset, (code, entry) in enumerate(entries):
        checkbox_key = _segment_checkbox_key(layer_idx, code)
        is_selected = st.session_state.get(checkbox_key, True)
        color_offset = selected_color_offsets.get(code, offset)
        palette_idx = color_offset if getattr(config, "analysis_scope", "overall") == "forest_community" else start_color_idx + color_offset
        color = _segment_legend_color(config, code, palette, palette_idx)
        if not is_selected:
            color = "#c9c9c9"
        spacer, checkbox_col, swatch, label_col, _, _ = st.columns([0.6, 0.35, 0.3, 3.35, 0.8, 1])
        spacer.empty()
        checkbox_col.checkbox("", key=checkbox_key, label_visibility="collapsed")
        swatch.markdown(
            f"""
            <div style="width: 0.65rem; height: 0.65rem; background:{color}; border-radius: 2px; margin-top: 0.35rem;"></div>
            """,
            unsafe_allow_html=True,
        )
        label_col.markdown(f"<span style='font-size: 0.9rem;'>{entry}</span>", unsafe_allow_html=True)


def _render_config_table(bundle, year_range: tuple[int, int]) -> None:
    if not st.session_state.comparison_configs:
        st.info("Add at least one comparison configuration to draw a comparison plot.")
        return
    st.subheader("Layers")
    palette = pc.qualitative.Plotly
    trace_color_idx = 0
    for idx, config_dict in enumerate(st.session_state.comparison_configs):
        config = ComparisonConfig(**config_dict)
        segment_entries = _segment_legend_entries(bundle, config, year_range)
        selected_codes = [
            code
            for code, _ in segment_entries
            if st.session_state.get(_segment_checkbox_key(idx, code), True)
        ]
        selected_color_offsets = {code: offset for offset, code in enumerate(selected_codes)}
        is_all_segment = _is_all_segment_config(config_dict)
        columns = st.columns([0.6, 4.0, 0.8, 1])
        label = _config_display_label(config_dict, idx, bundle).split(". ", 1)[1]
        color = "#888888" if is_all_segment and segment_entries else palette[trace_color_idx % len(palette)]
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
        if is_all_segment:
            _render_segment_legend(segment_entries, config, palette, trace_color_idx, idx, selected_color_offsets)
        trace_color_idx += max(1, len(segment_entries) if is_all_segment else 1)


def _build_export_subset(bundle, configs: list[ComparisonConfig], year_range: tuple[int, int]) -> pd.DataFrame:
    subsets = []
    for idx, config in enumerate(configs, start=1):
        frame = bundle.frame_for_config(config)
        filtered = filter_frame(frame, filters=filters_for_config(config), year_range=year_range).copy()
        if filtered.empty:
            continue
        filtered.insert(0, "comparison_label", config.label or _config_display_label(asdict(config), bundle=bundle))
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


def _coerce_datetime(value):
    if value is None:
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.to_pydatetime()


def _plotly_rgba_to_pillow(color: str, default=(235, 235, 235, 255)) -> tuple[int, int, int, int]:
    if not isinstance(color, str):
        return default
    color = color.strip()
    if color.startswith("rgba(") and color.endswith(")"):
        parts = [part.strip() for part in color[5:-1].split(",")]
        if len(parts) == 4:
            return (
                int(float(parts[0])),
                int(float(parts[1])),
                int(float(parts[2])),
                int(float(parts[3]) * 255),
            )
    if color.startswith("rgb(") and color.endswith(")"):
        parts = [part.strip() for part in color[4:-1].split(",")]
        if len(parts) == 3:
            return (int(float(parts[0])), int(float(parts[1])), int(float(parts[2])), 255)
    if color.startswith("#") and len(color) == 7:
        try:
            return (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16), 255)
        except ValueError:
            return default
    return default


def _trace_values(values) -> list:
    return list(values) if values is not None else []


def _trace_color(trace, fallback: str) -> str:
    for container_name in ("line", "marker"):
        container = getattr(trace, container_name, None)
        color = getattr(container, "color", None) if container is not None else None
        if color:
            return str(color)
    return fallback


def _export_metadata_lines(bundle, configs: list[ComparisonConfig], year_range: tuple[int, int]) -> list[str]:
    lines = [f"Year range: {year_range[0]}-{year_range[1]}"]
    for idx, config in enumerate(configs, start=1):
        config_dict = asdict(config)
        label = _config_display_label(config_dict, idx - 1, bundle).split(". ", 1)[1]
        below = config.exclude_below_stddev if config.exclude_below_stddev is not None else "none"
        above = config.exclude_above_stddev if config.exclude_above_stddev is not None else "none"
        lines.append(
            f"Layer {idx}: {label}; season={config.season_filter}; "
            f"exclude below z={below}; exclude above z={above}"
        )
    return lines


def _wrap_export_metadata(lines: list[str], width: int = 150) -> list[str]:
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=width) or [""])
    return wrapped


def _figure_with_export_metadata(figure, metadata_lines: list[str]):
    export_figure = copy.deepcopy(figure)
    wrapped_lines = _wrap_export_metadata(metadata_lines, width=165)
    margin = export_figure.layout.margin.to_plotly_json() if export_figure.layout.margin else {}
    margin["b"] = max(int(margin.get("b") or 80), 165 + 18 * max(0, len(wrapped_lines) - 1))
    export_figure.update_layout(margin=margin)
    export_figure.add_annotation(
        x=-0.03,
        y=-0.30,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        text="<br>".join(wrapped_lines),
        font={"size": 11, "color": "#444444"},
    )
    return export_figure


def _prepare_plot_export_pillow(figure, metadata_lines: list[str] | None = None, width: int = 1400, height: int = 860) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    left = 90
    right = 45
    top = 100
    metadata_lines = _wrap_export_metadata(metadata_lines or [], width=165)
    metadata_height = 16 * len(metadata_lines) + (16 if metadata_lines else 0)
    bottom = 105 + metadata_height
    plot_left = left
    plot_right = width - right
    plot_top = top
    plot_bottom = height - bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    x_values = []
    y_values = []
    for trace in figure.data:
        for x, y in zip(_trace_values(trace.x), _trace_values(trace.y)):
            x_dt = _coerce_datetime(x)
            if x_dt is None or y is None or pd.isna(y):
                continue
            x_values.append(x_dt)
            y_values.append(float(y))
    if not x_values or not y_values:
        image = Image.new("RGB", (width, height), "white")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    y_pad = max((y_max - y_min) * 0.08, 0.01)
    y_min -= y_pad
    y_max += y_pad
    if max(y_values) <= 1.0 < y_max:
        y_max = 1.0
    x_span = max((x_max - x_min).total_seconds(), 1.0)
    y_span = max(y_max - y_min, 1e-9)

    def x_to_px(value) -> int | None:
        x_dt = _coerce_datetime(value)
        if x_dt is None:
            return None
        return int(plot_left + ((x_dt - x_min).total_seconds() / x_span) * plot_width)

    def y_to_px(value) -> int | None:
        if value is None or pd.isna(value):
            return None
        return int(plot_bottom - ((float(value) - y_min) / y_span) * plot_height)

    image = Image.new("RGB", (width, height), "white")
    overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    palette = pc.qualitative.Plotly

    for shape in figure.layout.shapes or []:
        if getattr(shape, "type", None) != "rect":
            continue
        x0 = x_to_px(shape.x0)
        x1 = x_to_px(shape.x1)
        if x0 is None or x1 is None:
            continue
        draw_overlay.rectangle(
            (min(x0, x1), plot_top, max(x0, x1), plot_bottom),
            fill=_plotly_rgba_to_pillow(shape.fillcolor),
        )
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)

    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="#dddddd", width=1)
    for i in range(6):
        y = plot_top + int(i * plot_height / 5)
        value = y_max - i * y_span / 5
        draw.line((plot_left, y, plot_right, y), fill="#eeeeee", width=1)
        draw.text((10, y - 7), f"{value:.2f}", fill="#333333", font=font)

    start_year = x_min.year
    end_year = x_max.year
    step = max(1, (end_year - start_year) // 12 + 1)
    for year in range(start_year, end_year + 1, step):
        x = x_to_px(f"{year}-01-01")
        if x is None:
            continue
        draw.line((x, plot_bottom, x, plot_bottom + 5), fill="#333333", width=1)
        draw.text((x - 12, plot_bottom + 10), str(year), fill="#333333", font=font)

    for annotation in figure.layout.annotations or []:
        xref = annotation.xref or "paper"
        x = x_to_px(annotation.x) if xref == "x" else int(plot_left + float(annotation.x) * plot_width)
        if x is None:
            continue
        y = int(plot_top - 25) if float(annotation.y) > 1 else int(plot_top + (1 - float(annotation.y)) * plot_height)
        text = str(annotation.text).replace("<br>", "\n")
        fill = annotation.font.color if annotation.font and annotation.font.color else "#333333"
        draw.multiline_text((x - 18, y), text, fill=fill, font=font, spacing=1)

    for trace_idx, trace in enumerate(figure.data):
        color = _trace_color(trace, palette[trace_idx % len(palette)])
        points = []
        for x, y in zip(_trace_values(trace.x), _trace_values(trace.y)):
            x_px = x_to_px(x)
            y_px = y_to_px(y)
            if x_px is None or y_px is None:
                if len(points) > 1:
                    draw.line(points, fill=color, width=2)
                points = []
                continue
            points.append((x_px, y_px))
            draw.ellipse((x_px - 3, y_px - 3, x_px + 3, y_px + 3), fill=color, outline="white")
        if len(points) > 1:
            draw.line(points, fill=color, width=2)

    title = "Terrain-Vegetation Time Series"
    try:
        title_value = figure.layout.title.text
        if title_value:
            title = str(title_value).replace("<br>", " ")
    except Exception:
        pass
    draw.text((plot_left, 35), title, fill="#111111", font=font)
    draw.text((plot_left, plot_bottom + 34), "Date", fill="#333333", font=font)
    draw.text((10, 35), "Summary value", fill="#333333", font=font)
    if metadata_lines:
        metadata_y = plot_bottom + 62
        for line in metadata_lines:
            draw.text((10, metadata_y), line, fill="#444444", font=font)
            metadata_y += 16

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _prepare_plot_export(figure, metadata_lines: list[str] | None = None) -> tuple[bytes | None, str | None]:
    export_figure = _figure_with_export_metadata(figure, metadata_lines or [])
    try:
        png_bytes = pio.to_image(export_figure, format="png", width=1400, height=860, scale=2)
    except Exception as exc:
        try:
            return _prepare_plot_export_pillow(figure, metadata_lines), None
        except Exception as fallback_exc:
            return None, f"PNG export unavailable: {fallback_exc or exc}"
    return png_bytes, None


def _render_export_controls(bundle, configs: list[ComparisonConfig], year_range: tuple[int, int], figure) -> None:
    st.subheader("Exports")
    signature = _export_signature(configs, year_range)
    prepared = dict(st.session_state.prepared_exports.get(signature, {}))
    metadata_lines = _export_metadata_lines(bundle, configs, year_range)

    if "csv" not in prepared:
        prepared["csv"] = _prepare_data_export(bundle, configs, year_range)
    if "png" not in prepared and "png_error" not in prepared:
        with st.spinner("Preparing PNG export..."):
            png_bytes, png_error = _prepare_plot_export(figure, metadata_lines)
        prepared["png"] = png_bytes
        if png_error:
            prepared["png_error"] = png_error
    st.session_state.prepared_exports[signature] = prepared

    col1, col2, _ = st.columns([1, 1, 8])
    csv_bytes = prepared.get("csv")
    png_bytes = prepared.get("png")
    png_error = prepared.get("png_error")

    col1.download_button(
        "CSV",
        data=csv_bytes,
        file_name="dashboard_subset.csv",
        mime="text/csv",
        width="content",
    )
    if png_bytes is not None:
        col2.download_button(
            "PNG",
            data=png_bytes,
            file_name="dashboard_plot.png",
            mime="image/png",
            width="content",
        )
    elif png_error:
        col2.button("PNG", width="content", disabled=True)
        col2.caption(png_error.splitlines()[0])


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
    visible_segments_by_layer = _visible_segments_by_layer(bundle, config_objects, year_range)
    figure, messages = build_timeseries_figure(bundle, config_objects, year_range, visible_segments_by_layer)
    if figure.data:
        st.plotly_chart(figure, width="stretch")
    else:
        st.warning("No lines could be drawn from the current configurations and year range.")
    for message in messages:
        st.info(message)

    _render_config_table(bundle, year_range)
    _render_export_controls(bundle, config_objects, year_range, figure)


if __name__ == "__main__":
    main()
