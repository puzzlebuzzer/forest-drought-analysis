from __future__ import annotations

import copy
from dataclasses import asdict
from datetime import datetime
import html
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
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
STACK_GROWING_SEASON_FILTER = "stack_growing"
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
    "ecozone": "Broad ecozone",
    "forest_community": "Forest communities",
}
SEASON_FILTER_LABELS = {
    "growing": "growing",
    STACK_GROWING_SEASON_FILTER: "stack growing",
    "all": "all",
}
BUILDER_WIDGET_KEYS = [
    "builder_analysis_scope",
    "builder_select_broad_ecozone",
    "builder_select_forest_communities",
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


def _season_filter_options(values: list) -> list:
    options = list(values)
    if STACK_GROWING_SEASON_FILTER not in options:
        insert_at = options.index("growing") + 1 if "growing" in options else 0
        options.insert(insert_at, STACK_GROWING_SEASON_FILTER)
    return options


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
        div[data-testid="stCheckbox"] {
            min-height: 1rem;
            margin-bottom: -0.45rem;
        }
        div[data-testid="stCheckbox"] label {
            align-items: center;
            min-height: 1rem;
            padding-top: 0;
            padding-bottom: 0;
        }
        div[data-testid="stCheckbox"] label > div:first-child {
            background-color: #eeeeee !important;
            border-color: #b8b8b8 !important;
            color: #111111 !important;
            margin-top: 0 !important;
        }
        div[data-testid="stCheckbox"] svg {
            color: #111111 !important;
            fill: #111111 !important;
        }
        .plot-selection-legend {
            display: flex;
            flex-direction: column;
            gap: 0.28rem;
            align-items: flex-start;
            margin: -0.25rem 0 0.85rem 0;
            padding: 0.45rem 0 0.1rem 0;
            color: #f2f2f2;
            font-size: 0.88rem;
            line-height: 1.25;
        }
        .plot-selection-legend-item {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            white-space: nowrap;
        }
        .plot-selection-legend-swatch {
            display: inline-block;
            width: 0.68rem;
            height: 0.68rem;
            border-radius: 2px;
            flex: 0 0 auto;
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
        st.session_state.builder_defaults["builder_exclude_below_stddev"] = DEFAULT_EXCLUDE_BELOW_STDDEV
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


def _download_filename_stem(label: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
    return stem or "dashboard_layer"


def _timestamp_for_download() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def _layer_combined_checkbox_key(layer_idx: int) -> str:
    return f"layer_{layer_idx}_combined_visible"


def _layer_expanded_key(layer_idx: int) -> str:
    return f"layer_{layer_idx}_expanded"


def _broad_ecozone_checkbox_key(layer_idx: int, ecozone_code: int) -> str:
    return f"layer_{layer_idx}_broad_ecozone_{ecozone_code}_visible"


def _broad_ecozone_group_checkbox_key(layer_idx: int) -> str:
    return f"layer_{layer_idx}_broad_ecozone_group_visible"


def _segment_group_checkbox_key(layer_idx: int, group_key: str) -> str:
    return f"layer_{layer_idx}_segment_group_{group_key}_visible"


def _segment_group_combined_checkbox_key(layer_idx: int, group_key: str) -> str:
    return f"layer_{layer_idx}_segment_group_{group_key}_combined_visible"


def _layer_segmentation_checkbox_key(layer_idx: int, selected_scope: str, current_scope: str) -> str:
    return f"layer_{layer_idx}_select_{selected_scope}_{current_scope}"


def _segment_all_checkbox_key(layer_idx: int) -> str:
    return f"layer_{layer_idx}_segments_all_visible"


def _clear_layer_segment_state(layer_idx: int) -> None:
    prefixes = (
        f"layer_{layer_idx}_combined_visible",
        f"layer_{layer_idx}_broad_ecozone_",
        f"layer_{layer_idx}_segment_",
        f"layer_{layer_idx}_segment_group_",
        f"layer_{layer_idx}_segments_all_visible",
    )
    for key in list(st.session_state.keys()):
        if any(str(key).startswith(prefix) for prefix in prefixes):
            del st.session_state[key]


def _set_only_layer_expanded(layer_idx: int) -> None:
    for key in list(st.session_state.keys()):
        if str(key).startswith("layer_") and str(key).endswith("_expanded"):
            st.session_state[key] = False
    st.session_state[_layer_expanded_key(layer_idx)] = True


def _render_indented_checkbox(
    label: str,
    *,
    key: str,
    level: int,
    disabled: bool = False,
    value: bool | None = None,
    on_change=None,
    args: tuple = (),
) -> None:
    indent = max(0.04, 0.18 + (0.32 * level) - 0.25)
    swatch_start = 1.60
    total_width = 5.75
    checkbox_width = max(0.35, swatch_start - indent)
    trailing_width = max(1.0, total_width - indent - checkbox_width - 0.22)
    _, checkbox_col, _, _ = st.columns([indent, checkbox_width, 0.22, trailing_width])
    kwargs = {
        "key": key,
        "disabled": disabled,
        "on_change": on_change,
        "args": args,
    }
    if value is not None:
        kwargs["value"] = value
    checkbox_col.checkbox(label, **kwargs)


def _render_indented_segment_checkbox(
    label: str,
    *,
    key: str,
    level: int,
    color: str | None = None,
    disabled: bool = False,
    value: bool | None = None,
    on_change=None,
    args: tuple = (),
) -> None:
    indent = max(0.04, 0.18 + (0.32 * level) - 0.25)
    swatch_start = 1.60
    total_width = 5.75
    checkbox_width = max(0.35, swatch_start - indent)
    trailing_width = max(1.0, total_width - indent - checkbox_width - 0.22)
    _, checkbox_col, swatch_col, _ = st.columns([indent, checkbox_width, 0.22, trailing_width])
    kwargs = {
        "key": key,
        "disabled": disabled,
        "on_change": on_change,
        "args": args,
    }
    if value is not None:
        kwargs["value"] = value
    checkbox_col.checkbox(label, **kwargs)
    if color:
        swatch_col.markdown(
            f"""
            <div style="height: 1rem; display: flex; align-items: flex-end; justify-content: flex-start; margin-top: 0.42rem;">
                <div style="width: 0.65rem; height: 0.65rem; background:{color}; border-radius: 2px;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _set_all_segment_checkboxes(all_key: str, child_keys: list[str]) -> None:
    value = bool(st.session_state.get(all_key, True))
    for child_key in child_keys:
        st.session_state[child_key] = value


def _set_layer_segmentation(layer_idx: int, selected_scope: str, checkbox_key: str) -> None:
    if not (0 <= layer_idx < len(st.session_state.comparison_configs)):
        return
    config_dict = dict(st.session_state.comparison_configs[layer_idx])
    current_scope = config_dict.get("analysis_scope", "overall")
    if st.session_state.get(checkbox_key, False):
        config_dict["analysis_scope"] = selected_scope
    elif current_scope == selected_scope:
        config_dict["analysis_scope"] = "overall"
    config_dict["ecozone_code"] = None
    config_dict["forest_community_code"] = None
    st.session_state.comparison_configs[layer_idx] = config_dict
    if st.session_state.get("builder_target_index") == layer_idx:
        _load_builder_values(config_dict)


def _config_with_scope(config: ComparisonConfig, analysis_scope: str) -> ComparisonConfig:
    config_dict = asdict(config)
    config_dict["analysis_scope"] = analysis_scope
    config_dict["ecozone_code"] = None
    config_dict["forest_community_code"] = None
    return ComparisonConfig(**config_dict)


def _forest_community_grouped_legend_entries(
    entries: list[tuple[int, str]],
    bundle,
    config: ComparisonConfig,
) -> list[tuple[str, str, list[tuple[int, str]]]]:
    if not entries:
        return []
    entry_by_code = {int(code): label for code, label in entries}
    metadata: dict[int, tuple[float | None, str]] = {}
    source_frames = [
        bundle.scene_summary_forest_community_manifest,
        bundle.temporal_summary_forest_community_manifest,
        bundle.scene_summary,
        bundle.temporal_summary,
    ]
    for frame in source_frames:
        required = {"forest_community_code", "ecozone_group_label"}
        if frame.empty or not required.issubset(frame.columns):
            continue
        candidates = frame
        if "aoi" in candidates.columns:
            candidates = candidates[candidates["aoi"] == config.aoi]
        if "sensor" in candidates.columns:
            candidates = candidates[candidates["sensor"] == config.sensor]
        if "index" in candidates.columns:
            candidates = candidates[candidates["index"] == config.index]
        if candidates.empty:
            continue
        for code in entry_by_code:
            if code in metadata:
                continue
            code_matches = candidates[pd.to_numeric(candidates["forest_community_code"], errors="coerce") == code]
            if code_matches.empty:
                continue
            group_code = None
            if "ecozone_group_code" in code_matches.columns:
                group_codes = pd.to_numeric(code_matches["ecozone_group_code"], errors="coerce").dropna()
                if not group_codes.empty:
                    group_code = float(group_codes.iloc[0])
            labels = code_matches["ecozone_group_label"].dropna().astype(str)
            labels = labels[labels != ""]
            metadata[code] = (group_code, labels.iloc[0] if not labels.empty else "Unlabeled group")
        if len(metadata) == len(entry_by_code):
            break

    grouped: dict[str, dict] = {}
    for code, label in entries:
        group_code, group_label = metadata.get(int(code), (None, "Unlabeled group"))
        group_key = str(int(group_code)) if group_code is not None else "unlabeled"
        group = grouped.setdefault(
            group_key,
            {
                "sort": group_code if group_code is not None else 9999,
                "label": group_label,
                "entries": [],
            },
        )
        group["entries"].append((int(code), label))
    ordered_groups = sorted(grouped.items(), key=lambda item: (item[1]["sort"], item[1]["label"]))
    return [(group_key, group["label"], group["entries"]) for group_key, group in ordered_groups]


def _visible_segments_by_layer(
    bundle,
    configs: list[ComparisonConfig],
    year_range: tuple[int, int],
) -> dict[int, set[int]]:
    visible: dict[int, set[int]] = {}
    for idx, config in enumerate(configs):
        entries = _segment_legend_entries(bundle, _config_with_scope(config, "forest_community"), year_range)
        if not entries:
            continue
        selected_codes = set()
        for code, _ in entries:
            key = _segment_checkbox_key(idx, code)
            if key not in st.session_state:
                st.session_state[key] = False
            if st.session_state.get(key, False):
                selected_codes.add(code)
        visible[idx] = selected_codes
    return visible


def _visible_broad_ecozones_by_layer(
    bundle,
    configs: list[ComparisonConfig],
    year_range: tuple[int, int],
) -> dict[int, set[int]]:
    visible: dict[int, set[int]] = {}
    for idx, config in enumerate(configs):
        entries = _segment_legend_entries(bundle, _config_with_scope(config, "ecozone"), year_range)
        if not entries:
            continue
        selected_codes = set()
        for code, _ in entries:
            key = _broad_ecozone_checkbox_key(idx, code)
            if key not in st.session_state:
                st.session_state[key] = False
            if st.session_state.get(key, False):
                selected_codes.add(code)
        visible[idx] = selected_codes
    return visible


def _combined_segment_id(group_key: str) -> str:
    return f"combined:{group_key}"


def _broad_ecozone_segment_id(ecozone_code: int) -> str:
    return f"broad:{ecozone_code}"


def _overall_combined_segment_id() -> str:
    return "combined:overall"


def _combined_group_frames_by_layer(
    bundle,
    configs: list[ComparisonConfig],
    year_range: tuple[int, int],
    selected_color_offsets_by_layer: dict[int, dict[object, int]] | None = None,
    palette: list[str] | None = None,
) -> dict[int, list[tuple[pd.DataFrame, str | None]]]:
    selected_color_offsets_by_layer = selected_color_offsets_by_layer or {}
    palette = palette or pc.qualitative.Plotly
    combined_frames: dict[int, list[tuple[pd.DataFrame, str | None]]] = {}
    for idx, config in enumerate(configs):
        forest_config = _config_with_scope(config, "forest_community")
        entries = _segment_legend_entries(bundle, forest_config, year_range)
        grouped_entries = _forest_community_grouped_legend_entries(entries, bundle, forest_config)
        selected_groups = []
        for group_key, group_label, group_entries in grouped_entries:
            if len(group_entries) <= 1:
                continue
            combined_key = _segment_group_combined_checkbox_key(idx, group_key)
            if combined_key not in st.session_state:
                st.session_state[combined_key] = False
            if st.session_state.get(combined_key, False):
                try:
                    group_code = int(group_key)
                except ValueError:
                    continue
                group_frame = bundle.frame_for_forest_community_group(forest_config, group_code)
                group_frame = filter_frame(group_frame, filters={}, year_range=year_range).copy()
                if group_frame.empty:
                    continue
                group_frame["forest_community_label"] = f"{group_label} Combined"
                group_frame["forest_community_code"] = group_code
                color_offset = selected_color_offsets_by_layer.get(idx, {}).get(_combined_segment_id(group_key))
                color = palette[color_offset % len(palette)] if color_offset is not None else None
                selected_groups.append((group_frame, color))
        if selected_groups:
            combined_frames[idx] = selected_groups
    return combined_frames


def _selected_color_offsets_by_layer(
    bundle,
    configs: list[ComparisonConfig],
    year_range: tuple[int, int],
) -> dict[int, dict[object, int]]:
    selected_color_offsets_by_layer: dict[int, dict[object, int]] = {}
    next_color_offset = 0
    for idx, config in enumerate(configs):
        selected_segments: list[object] = []
        if st.session_state.get(_layer_combined_checkbox_key(idx), True):
            selected_segments.append(_overall_combined_segment_id())

        broad_entries = _segment_legend_entries(bundle, _config_with_scope(config, "ecozone"), year_range)
        for code, _ in broad_entries:
            if st.session_state.get(_broad_ecozone_checkbox_key(idx, code), False):
                selected_segments.append(_broad_ecozone_segment_id(code))

        forest_config = _config_with_scope(config, "forest_community")
        segment_entries = _segment_legend_entries(bundle, forest_config, year_range)
        grouped_entries = _forest_community_grouped_legend_entries(segment_entries, bundle, forest_config)
        for group_key, _, group_entries in grouped_entries:
            combined_key = _segment_group_combined_checkbox_key(idx, group_key)
            if len(group_entries) > 1 and st.session_state.get(combined_key, False):
                selected_segments.append(_combined_segment_id(group_key))
            for code, _ in group_entries:
                if st.session_state.get(_segment_checkbox_key(idx, code), False):
                    selected_segments.append(code)
        selected_color_offsets_by_layer[idx] = {
            segment_id: next_color_offset + offset
            for offset, segment_id in enumerate(selected_segments)
        }
        next_color_offset += len(selected_segments)
    return selected_color_offsets_by_layer


def _build_plot_layers(
    bundle,
    configs: list[ComparisonConfig],
    year_range: tuple[int, int],
    selected_color_offsets_by_layer: dict[int, dict[object, int]],
    palette: list[str] | None = None,
) -> tuple[
    list[ComparisonConfig],
    dict[int, set[int]],
    dict[int, list[tuple[pd.DataFrame, str | None]]],
    dict[int, dict[int, int]],
    dict[int, str],
]:
    palette = palette or pc.qualitative.Plotly
    broad_visible_by_base = _visible_broad_ecozones_by_layer(bundle, configs, year_range)
    forest_visible_by_base = _visible_segments_by_layer(bundle, configs, year_range)
    combined_frames_by_base = _combined_group_frames_by_layer(
        bundle,
        configs,
        year_range,
        selected_color_offsets_by_layer,
        palette,
    )

    plot_configs: list[ComparisonConfig] = []
    visible_segments_by_plot: dict[int, set[int]] = {}
    combined_frames_by_plot: dict[int, list[tuple[pd.DataFrame, str | None]]] = {}
    segment_color_offsets_by_plot: dict[int, dict[int, int]] = {}
    config_color_overrides_by_plot: dict[int, str] = {}

    for base_idx, config in enumerate(configs):
        if st.session_state.get(_layer_combined_checkbox_key(base_idx), True):
            plot_idx = len(plot_configs)
            plot_configs.append(_config_with_scope(config, "overall"))
            color_offset = selected_color_offsets_by_layer.get(base_idx, {}).get(_overall_combined_segment_id())
            if color_offset is not None:
                config_color_overrides_by_plot[plot_idx] = palette[color_offset % len(palette)]

        broad_visible = broad_visible_by_base.get(base_idx, set())
        if broad_visible:
            plot_idx = len(plot_configs)
            plot_configs.append(_config_with_scope(config, "ecozone"))
            visible_segments_by_plot[plot_idx] = broad_visible
            base_offsets = selected_color_offsets_by_layer.get(base_idx, {})
            segment_color_offsets_by_plot[plot_idx] = {
                int(code): offset
                for code in broad_visible
                if (offset := base_offsets.get(_broad_ecozone_segment_id(code))) is not None
            }

        forest_visible = forest_visible_by_base.get(base_idx, set())
        group_frames = combined_frames_by_base.get(base_idx, [])
        if forest_visible or group_frames:
            plot_idx = len(plot_configs)
            plot_configs.append(_config_with_scope(config, "forest_community"))
            visible_segments_by_plot[plot_idx] = forest_visible
            if group_frames:
                combined_frames_by_plot[plot_idx] = group_frames
            base_offsets = selected_color_offsets_by_layer.get(base_idx, {})
            segment_color_offsets_by_plot[plot_idx] = {
                int(code): offset
                for code, offset in base_offsets.items()
                if isinstance(code, int)
            }

    return (
        plot_configs,
        visible_segments_by_plot,
        combined_frames_by_plot,
        segment_color_offsets_by_plot,
        config_color_overrides_by_plot,
    )


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
    analysis_scope = config_dict.get("analysis_scope", "overall")
    st.session_state.builder_pending_values = {
        "builder_analysis_scope": analysis_scope,
        "builder_select_broad_ecozone": analysis_scope == "ecozone",
        "builder_select_forest_communities": analysis_scope == "forest_community",
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
    for key, value in pending_values.items():
        st.session_state[key] = value
    st.session_state.builder_pending_values = None


def _builder_default(key: str, fallback):
    return st.session_state.builder_defaults.get(key, fallback)


def _clear_builder_widget_state() -> None:
    for key in BUILDER_WIDGET_KEYS:
        if key in st.session_state:
            del st.session_state[key]


def _start_new_overlay(bundle) -> None:
    st.session_state.builder_mode = "new"
    st.session_state.builder_target_index = None
    _clear_builder_widget_state()
    _load_builder_values(_default_config_dict(bundle))


def _append_default_layer(bundle) -> None:
    config_dict = _default_config_dict(bundle)
    st.session_state.comparison_configs.append(config_dict)
    st.session_state.default_overlay_seeded = True
    new_index = len(st.session_state.comparison_configs) - 1
    _set_only_layer_expanded(new_index)
    _start_edit_overlay(config_dict, new_index)


def _start_edit_overlay(config_dict: dict, index: int) -> None:
    st.session_state.builder_mode = "edit"
    st.session_state.builder_target_index = index
    _set_only_layer_expanded(index)
    _load_builder_values(config_dict)


def _ensure_builder_state(bundle) -> None:
    if not st.session_state.comparison_configs and not st.session_state.default_overlay_seeded:
        _start_new_overlay(bundle)
        return
    if st.session_state.builder_mode == "new":
        if not st.session_state.builder_defaults and not st.session_state.builder_pending_values:
            _load_builder_values(_default_config_dict(bundle))
        return
    valid_edit_target = False
    if st.session_state.builder_mode == "edit":
        target_index = st.session_state.builder_target_index
        if target_index is None or not (0 <= target_index < len(st.session_state.comparison_configs)):
            if st.session_state.comparison_configs:
                target_index = len(st.session_state.comparison_configs) - 1
                _start_edit_overlay(st.session_state.comparison_configs[target_index], target_index)
            else:
                _start_new_overlay(bundle)
                return
        valid_edit_target = True
        if st.session_state.builder_defaults or st.session_state.builder_pending_values:
            return
    if all(key in st.session_state for key in REQUIRED_BUILDER_WIDGET_KEYS):
        return
    if valid_edit_target:
        _start_edit_overlay(
            st.session_state.comparison_configs[st.session_state.builder_target_index],
            st.session_state.builder_target_index,
        )
    elif st.session_state.comparison_configs:
        target_index = len(st.session_state.comparison_configs) - 1
        _start_edit_overlay(st.session_state.comparison_configs[target_index], target_index)
    else:
        _start_new_overlay(bundle)


def _build_sidebar(bundle) -> tuple[tuple[int, int], ComparisonConfig | None, int | None]:
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

    sensors = bundle.available_values("sensor")
    aois = bundle.available_values("aoi")
    indices = bundle.available_values("index")
    spatial_percentiles = bundle.available_values("spatial_percentile")
    temporal_aggs = bundle.available_values("temporal_agg")
    cloud_thresholds = bundle.available_values("cloud_threshold")
    season_filters = _season_filter_options(bundle.available_values("season_filter"))

    _ensure_builder_state(bundle)
    _apply_pending_builder_values()
    target_index = st.session_state.builder_target_index
    if target_index is not None and 0 <= target_index < len(st.session_state.comparison_configs):
        st.sidebar.markdown(f"### **Layer {target_index + 1}**")
    aoi = _safe_selectbox(
        "AOI",
        aois,
        _builder_default("builder_aoi", "north"),
        scope=st.sidebar,
        key="builder_aoi",
        format_func=lambda value: AOI_LABELS.get(value, value),
    )
    sensor = _safe_selectbox(
        "Sensor",
        sensors,
        _builder_default("builder_sensor", "ls"),
        scope=st.sidebar,
        key="builder_sensor",
        format_func=lambda value: SENSOR_LABELS.get(value, value),
    )
    index_name = _safe_selectbox("Index", indices, _builder_default("builder_index", "ndvi"), scope=st.sidebar, key="builder_index")
    cloud_threshold = _safe_plain_selectbox(
        "Cloud threshold",
        cloud_thresholds,
        _builder_default("builder_cloud_threshold", 40),
        scope=st.sidebar,
        key="builder_cloud_threshold",
        format_func=_cloud_threshold_label,
    )
    spatial_percentile = _safe_selectbox(
        "Spatial aggregation percentile",
        spatial_percentiles,
        _builder_default("builder_spatial_percentile", DEFAULT_SPATIAL_PERCENTILE),
        format_func=lambda value: SPATIAL_PERCENTILE_LABELS.get(value, value),
        scope=st.sidebar,
        key="builder_spatial_percentile",
    )
    temporal_agg = _safe_selectbox(
        "Interval",
        temporal_aggs,
        _builder_default("builder_temporal_agg", DEFAULT_TEMPORAL_AGG),
        scope=st.sidebar,
        key="builder_temporal_agg",
    )
    temporal_percentile = _safe_selectbox(
        "Interval aggregation percentile",
        list(spatial_percentiles),
        _builder_default("builder_temporal_percentile", DEFAULT_TEMPORAL_PERCENTILE),
        scope=st.sidebar,
        key="builder_temporal_percentile",
        format_func=lambda value: SPATIAL_PERCENTILE_LABELS.get(value, value),
    )
    season_filter = _safe_selectbox(
        "Season filter",
        season_filters,
        _builder_default("builder_season_filter", "growing"),
        scope=st.sidebar,
        key="builder_season_filter",
        format_func=lambda value: SEASON_FILTER_LABELS.get(value, value),
    )
    exclude_above_stddev = _safe_plain_selectbox(
        "Exclude above z-score",
        STDDEV_FILTER_OPTIONS,
        _builder_default("builder_exclude_above_stddev", "none"),
        scope=st.sidebar,
        key="builder_exclude_above_stddev",
    )
    exclude_below_stddev = _safe_plain_selectbox(
        "Exclude below z-score",
        STDDEV_FILTER_OPTIONS,
        _builder_default("builder_exclude_below_stddev", DEFAULT_EXCLUDE_BELOW_STDDEV),
        scope=st.sidebar,
        key="builder_exclude_below_stddev",
    )
    label_kwargs = {"key": "builder_label"}
    if "builder_label" not in st.session_state:
        label_kwargs["value"] = _builder_default("builder_label", "")
    label = st.sidebar.text_input("Optional custom label", **label_kwargs)

    if not all([sensor, aoi, index_name, spatial_percentile, temporal_agg, temporal_percentile, season_filter]):
        return selected_year_range, None, st.session_state.builder_target_index

    config = ComparisonConfig(
        label=label.strip(),
        analysis_scope="overall",
        sensor=sensor,
        aoi=aoi,
        index=index_name,
        ecozone_code=None,
        forest_community_code=None,
        spatial_percentile=spatial_percentile,
        temporal_agg=temporal_agg,
        temporal_percentile=temporal_percentile,
        cloud_threshold=int(cloud_threshold),
        season_filter=season_filter,
        exclude_below_stddev=_stddev_option(exclude_below_stddev),
        exclude_above_stddev=_stddev_option(exclude_above_stddev),
    )
    return selected_year_range, config, st.session_state.builder_target_index


def _segment_legend_color(config: ComparisonConfig, code: int, palette: list[str], color_idx: int) -> str:
    if getattr(config, "analysis_scope", "overall") == "ecozone":
        return ECOZONE_TRACE_COLORS.get(int(code), palette[color_idx % len(palette)])
    return palette[color_idx % len(palette)]


def _render_segment_legend(
    config: ComparisonConfig,
    bundle,
    palette: list[str],
    start_color_idx: int,
    layer_idx: int,
    selected_color_offsets: dict[object, int],
    year_range: tuple[int, int],
) -> None:
    broad_config = _config_with_scope(config, "ecozone")
    forest_config = _config_with_scope(config, "forest_community")
    forest_entries = _segment_legend_entries(bundle, forest_config, year_range)
    grouped_entries = _forest_community_grouped_legend_entries(forest_entries, bundle, forest_config)

    broad_entries = _segment_legend_entries(bundle, broad_config, year_range)
    combined_key = _layer_combined_checkbox_key(layer_idx)
    broad_child_keys = [_broad_ecozone_checkbox_key(layer_idx, code) for code, _ in broad_entries]
    forest_child_keys = [_segment_checkbox_key(layer_idx, code) for code, _ in forest_entries]
    group_combined_keys = [
        _segment_group_combined_checkbox_key(layer_idx, group_key)
        for group_key, _, group_entries in grouped_entries
        if len(group_entries) > 1
    ]
    child_keys = [combined_key, *broad_child_keys, *forest_child_keys, *group_combined_keys]
    for child_key in child_keys:
        if child_key not in st.session_state:
            st.session_state[child_key] = child_key == combined_key

    overall_color_offset = selected_color_offsets.get(_overall_combined_segment_id(), start_color_idx)
    overall_color = palette[overall_color_offset % len(palette)]
    if not st.session_state.get(combined_key, True):
        overall_color = "#c9c9c9"
    _render_indented_segment_checkbox(
        "Overall Combined",
        key=combined_key,
        level=1,
        color=overall_color,
    )

    if broad_entries:
        broad_group_key = _broad_ecozone_group_checkbox_key(layer_idx)
        st.session_state[broad_group_key] = all(st.session_state.get(child_key, False) for child_key in broad_child_keys)
        _render_indented_checkbox(
            "Broad ecozones",
            key=broad_group_key,
            level=1,
            on_change=_set_all_segment_checkboxes,
            args=(broad_group_key, broad_child_keys),
        )
        for code, entry in broad_entries:
            checkbox_key = _broad_ecozone_checkbox_key(layer_idx, code)
            is_selected = st.session_state.get(checkbox_key, False)
            color_offset = selected_color_offsets.get(_broad_ecozone_segment_id(code), start_color_idx)
            color = palette[color_offset % len(palette)]
            if not is_selected:
                color = "#c9c9c9"
            _render_indented_segment_checkbox(
                entry,
                key=checkbox_key,
                level=2,
                color=color,
            )

    for group_key, group_label, group_entries in grouped_entries:
        group_child_keys = [_segment_checkbox_key(layer_idx, code) for code, _ in group_entries]
        group_combined_key = _segment_group_combined_checkbox_key(layer_idx, group_key)
        if len(group_entries) > 1:
            group_child_keys = [group_combined_key, *group_child_keys]
        checkbox_key = _segment_group_checkbox_key(layer_idx, group_key)
        st.session_state[checkbox_key] = all(st.session_state.get(child_key, False) for child_key in group_child_keys)
        _render_indented_checkbox(
            group_label,
            key=checkbox_key,
            level=1,
            on_change=_set_all_segment_checkboxes,
            args=(checkbox_key, group_child_keys),
        )
        if len(group_entries) > 1:
            is_combined_selected = st.session_state.get(group_combined_key, False)
            combined_segment_id = _combined_segment_id(group_key)
            color_offset = selected_color_offsets.get(combined_segment_id, 0)
            color = palette[color_offset % len(palette)]
            if not is_combined_selected:
                color = "#c9c9c9"
            _render_indented_segment_checkbox(
                "Combined",
                key=group_combined_key,
                level=2,
                color=color,
            )
        for code, entry in group_entries:
            checkbox_key = _segment_checkbox_key(layer_idx, code)
            is_selected = st.session_state.get(checkbox_key, False)
            color_offset = selected_color_offsets.get(code, 0)
            color = palette[color_offset % len(palette)]
            if not is_selected:
                color = "#c9c9c9"
            _render_indented_segment_checkbox(
                entry,
                key=checkbox_key,
                level=2,
                color=color,
            )


def _selected_plot_legend_entries(
    bundle,
    configs: list[ComparisonConfig],
    year_range: tuple[int, int],
    selected_color_offsets_by_layer: dict[int, dict[object, int]],
    palette: list[str] | None = None,
) -> list[tuple[str, str, str]]:
    palette = palette or pc.qualitative.Plotly
    entries: list[tuple[str, str, str]] = []
    for idx, config in enumerate(configs):
        layer_label = _config_display_label(asdict(_config_with_scope(config, "overall")), idx, bundle).split(". ", 1)[1]
        selected_offsets = selected_color_offsets_by_layer.get(idx, {})

        if st.session_state.get(_layer_combined_checkbox_key(idx), True):
            color_offset = selected_offsets.get(_overall_combined_segment_id(), 0)
            entries.append((layer_label, "Overall Combined", palette[color_offset % len(palette)]))

        broad_entries = _segment_legend_entries(bundle, _config_with_scope(config, "ecozone"), year_range)
        for code, label in broad_entries:
            if not st.session_state.get(_broad_ecozone_checkbox_key(idx, code), False):
                continue
            color_offset = selected_offsets.get(_broad_ecozone_segment_id(code), 0)
            entries.append((layer_label, label, palette[color_offset % len(palette)]))

        forest_config = _config_with_scope(config, "forest_community")
        forest_entries = _segment_legend_entries(bundle, forest_config, year_range)
        grouped_entries = _forest_community_grouped_legend_entries(forest_entries, bundle, forest_config)
        for group_key, group_label, group_entries in grouped_entries:
            if len(group_entries) > 1 and st.session_state.get(_segment_group_combined_checkbox_key(idx, group_key), False):
                color_offset = selected_offsets.get(_combined_segment_id(group_key), 0)
                entries.append((layer_label, f"{group_label} Combined", palette[color_offset % len(palette)]))
            for code, label in group_entries:
                if not st.session_state.get(_segment_checkbox_key(idx, code), False):
                    continue
                color_offset = selected_offsets.get(code, 0)
                entries.append((layer_label, label, palette[color_offset % len(palette)]))
    return entries


def _render_plot_selection_legend(
    bundle,
    configs: list[ComparisonConfig],
    year_range: tuple[int, int],
    selected_color_offsets_by_layer: dict[int, dict[object, int]],
) -> None:
    entries = _selected_plot_legend_entries(bundle, configs, year_range, selected_color_offsets_by_layer)
    if not entries:
        return
    chips = []
    for layer_label, item_label, color in entries:
        safe_layer_label = html.escape(str(layer_label))
        safe_item_label = html.escape(str(item_label))
        chips.append(
            (
                '<div class="plot-selection-legend-item">'
                f'<span class="plot-selection-legend-swatch" style="background:{color};"></span>'
                f"<span><strong>{safe_layer_label}</strong> / {safe_item_label}</span>"
                "</div>"
            )
        )
    st.markdown(
        f'<div class="plot-selection-legend">{"".join(chips)}</div>',
        unsafe_allow_html=True,
    )


def _render_layer_segmentation_controls(bundle, config_dict: dict, layer_idx: int) -> None:
    available_scopes = bundle.available_values("analysis_scope")
    current_scope = config_dict.get("analysis_scope", "overall")
    can_select_broad_ecozone = "ecozone" in available_scopes
    can_select_forest_communities = "forest_community" in available_scopes
    broad_key = _layer_segmentation_checkbox_key(layer_idx, "ecozone", current_scope)
    forest_key = _layer_segmentation_checkbox_key(layer_idx, "forest_community", current_scope)
    _render_indented_checkbox(
        "Select broad ecozone",
        key=broad_key,
        level=1,
        value=current_scope == "ecozone",
        disabled=current_scope == "forest_community" or not can_select_broad_ecozone,
        on_change=_set_layer_segmentation,
        args=(layer_idx, "ecozone", broad_key),
    )
    _render_indented_checkbox(
        "Select forest communities",
        key=forest_key,
        level=1,
        value=current_scope == "forest_community",
        disabled=current_scope == "ecozone" or not can_select_forest_communities,
        on_change=_set_layer_segmentation,
        args=(layer_idx, "forest_community", forest_key),
    )


def _render_config_table(
    bundle,
    year_range: tuple[int, int],
    selected_color_offsets_by_layer: dict[int, dict[object, int]],
) -> None:
    st.subheader("Layers")
    if st.button("Add new", key="add_layer_header"):
        _append_default_layer(bundle)
        st.rerun()
    if not st.session_state.comparison_configs:
        st.info("Add at least one comparison configuration to draw a comparison plot.")
        return
    palette = pc.qualitative.Plotly
    trace_color_idx = 0
    for idx, config_dict in enumerate(st.session_state.comparison_configs):
        config = ComparisonConfig(**config_dict)
        selected_color_offsets = selected_color_offsets_by_layer.get(idx, {})
        expanded_key = _layer_expanded_key(idx)
        if expanded_key not in st.session_state:
            st.session_state[expanded_key] = len(st.session_state.comparison_configs) == 1
        is_expanded = bool(st.session_state.get(expanded_key, False))
        columns = st.columns([0.28, 4.15, 0.55, 0.55, 0.7])
        display_config = asdict(_config_with_scope(config, "overall"))
        label = _config_display_label(display_config, idx, bundle).split(". ", 1)[1]
        if columns[0].button("▾" if is_expanded else "▸", key=f"expand_{idx}", help="Show or hide layer details"):
            st.session_state[expanded_key] = not is_expanded
            st.rerun()
        columns[1].markdown(
            f"""
            <div style="height: 2.35rem; display: flex; align-items: center;">
                <span><code>{idx + 1}</code> {label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        csv_bytes = _prepare_layer_data_export(bundle, config, idx, year_range)
        csv_filename = f"{_download_filename_stem(label)}_{_timestamp_for_download()}.csv"
        columns[2].download_button(
            "CSV",
            data=csv_bytes,
            file_name=csv_filename,
            mime="text/csv",
            key=f"csv_{idx}",
            width="content",
            disabled=not bool(csv_bytes),
        )
        if columns[3].button("Edit", key=f"edit_{idx}"):
            _start_edit_overlay(config_dict, idx)
            st.rerun()
        if columns[4].button("Remove", key=f"remove_{idx}"):
            st.session_state.comparison_configs.pop(idx)
            if st.session_state.comparison_configs:
                next_index = min(idx, len(st.session_state.comparison_configs) - 1)
                _start_edit_overlay(st.session_state.comparison_configs[next_index], next_index)
            else:
                st.session_state.builder_target_index = None
                st.session_state.builder_mode = "new"
                st.session_state.default_overlay_seeded = False
            st.rerun()
        if st.session_state.get(expanded_key, False):
            _render_segment_legend(config, bundle, palette, trace_color_idx, idx, selected_color_offsets, year_range)
        segment_entries = _segment_legend_entries(bundle, _config_with_scope(config, "forest_community"), year_range)
        trace_color_idx += max(1, len(segment_entries))


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


def _export_frame_with_label(frame: pd.DataFrame, label: str, order: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    exported = frame.copy()
    exported.insert(0, "comparison_label", label)
    exported.insert(1, "comparison_order", order)
    return exported


def _build_layer_export_subset(
    bundle,
    config: ComparisonConfig,
    layer_idx: int,
    year_range: tuple[int, int],
) -> pd.DataFrame:
    subsets = []
    export_order = 1

    if st.session_state.get(_layer_combined_checkbox_key(layer_idx), True):
        overall_config = _config_with_scope(config, "overall")
        frame = bundle.frame_for_config(overall_config)
        filtered = filter_frame(frame, filters=filters_for_config(overall_config), year_range=year_range)
        label = overall_config.label or _config_display_label(asdict(overall_config), bundle=bundle)
        subsets.append(_export_frame_with_label(filtered, label, export_order))
        export_order += 1

    broad_config = _config_with_scope(config, "ecozone")
    broad_entries = _segment_legend_entries(bundle, broad_config, year_range)
    selected_broad_codes = [
        code
        for code, _ in broad_entries
        if st.session_state.get(_broad_ecozone_checkbox_key(layer_idx, code), False)
    ]
    if selected_broad_codes:
        frame = bundle.frame_for_config(broad_config)
        filtered = filter_frame(frame, filters=filters_for_config(broad_config), year_range=year_range)
        filtered = filtered[filtered["ecozone_code"].isin(selected_broad_codes)]
        label = broad_config.label or _config_display_label(asdict(broad_config), bundle=bundle)
        subsets.append(_export_frame_with_label(filtered, label, export_order))
        export_order += 1

    forest_config = _config_with_scope(config, "forest_community")
    forest_entries = _segment_legend_entries(bundle, forest_config, year_range)
    grouped_entries = _forest_community_grouped_legend_entries(forest_entries, bundle, forest_config)
    selected_community_codes = [
        code
        for code, _ in forest_entries
        if st.session_state.get(_segment_checkbox_key(layer_idx, code), False)
    ]
    if selected_community_codes:
        frame = bundle.frame_for_config(forest_config)
        filtered = filter_frame(frame, filters=filters_for_config(forest_config), year_range=year_range)
        filtered = filtered[filtered["forest_community_code"].isin(selected_community_codes)]
        label = forest_config.label or _config_display_label(asdict(forest_config), bundle=bundle)
        subsets.append(_export_frame_with_label(filtered, label, export_order))
        export_order += 1

    for group_key, group_label, group_entries in grouped_entries:
        if len(group_entries) <= 1:
            continue
        if not st.session_state.get(_segment_group_combined_checkbox_key(layer_idx, group_key), False):
            continue
        try:
            group_code = int(group_key)
        except ValueError:
            continue
        group_frame = bundle.frame_for_forest_community_group(forest_config, group_code)
        filtered = filter_frame(group_frame, filters={}, year_range=year_range).copy()
        filtered["forest_community_label"] = f"{group_label} Combined"
        filtered["forest_community_code"] = group_code
        label = f"{forest_config.label or _config_display_label(asdict(forest_config), bundle=bundle)} / {group_label} Combined"
        subsets.append(_export_frame_with_label(filtered, label, export_order))
        export_order += 1

    nonempty_subsets = [subset for subset in subsets if not subset.empty]
    if not nonempty_subsets:
        return pd.DataFrame()
    return pd.concat(nonempty_subsets, ignore_index=True)


def _prepare_layer_data_export(
    bundle,
    config: ComparisonConfig,
    layer_idx: int,
    year_range: tuple[int, int],
) -> bytes:
    export_frame = _build_layer_export_subset(bundle, config, layer_idx, year_range)
    return export_frame.to_csv(index=False).encode("utf-8") if not export_frame.empty else b""


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

    title = "Ecozone-Vegetation Time Series"
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


def main() -> None:
    st.set_page_config(page_title="Appalachian Ecozone–Vegetation Dashboard", layout="wide")
    _inject_ui_css()
    _init_state()

    st.title("Appalachian Ecozone–Vegetation Analysis Dashboard")

    data_dir = _render_data_dir_control()
    bundle = _load_dashboard_data_cached(str(data_dir))
    year_range, config, selected_existing = _build_sidebar(bundle)

    if not st.session_state.default_overlay_seeded and not st.session_state.comparison_configs:
        seeded_config = _default_config_dict(bundle)
        st.session_state.comparison_configs.append(seeded_config)
        st.session_state.default_overlay_seeded = True
        _start_edit_overlay(seeded_config, 0)
        st.rerun()

    if config and selected_existing is not None:
        previous_config = st.session_state.comparison_configs[selected_existing]
        updated_config = asdict(config)
        st.session_state.comparison_configs[selected_existing] = updated_config
        if previous_config.get("aoi") != updated_config.get("aoi"):
            _clear_layer_segment_state(selected_existing)

    config_objects = [ComparisonConfig(**cfg) for cfg in st.session_state.comparison_configs]
    selected_color_offsets_by_layer = _selected_color_offsets_by_layer(bundle, config_objects, year_range)
    stack_growing_configs = [
        config for config in config_objects if config.season_filter == STACK_GROWING_SEASON_FILTER
    ]
    if stack_growing_configs:
        selected_year = int(year_range[1])
        figure, message = build_growing_season_figure(
            bundle,
            stack_growing_configs[0],
            selected_year,
            year_range=year_range,
        )
        messages = [message] if message else []
        if len(stack_growing_configs) > 1:
            messages.append("Stack growing mode uses the first layer with `stack growing` selected.")
    else:
        (
            plot_configs,
            visible_segments_by_layer,
            combined_group_frames_by_layer,
            segment_color_offsets_by_layer,
            config_color_overrides_by_layer,
        ) = _build_plot_layers(
            bundle,
            config_objects,
            year_range,
            selected_color_offsets_by_layer,
        )
        figure, messages = build_timeseries_figure(
            bundle,
            plot_configs,
            year_range,
            visible_segments_by_layer,
            combined_group_frames_by_layer,
            segment_color_offsets_by_layer,
            config_color_overrides_by_layer,
        )
    if figure.data:
        st.plotly_chart(figure, width="stretch")
    else:
        st.warning("No lines could be drawn from the current configurations and year range.")
    for message in messages:
        st.info(message)

    if not stack_growing_configs:
        _render_plot_selection_legend(bundle, config_objects, year_range, selected_color_offsets_by_layer)
    _render_config_table(bundle, year_range, selected_color_offsets_by_layer)


if __name__ == "__main__":
    main()
