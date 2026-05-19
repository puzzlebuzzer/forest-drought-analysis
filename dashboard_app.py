from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.colors as pc
import streamlit as st

from src.dashboard_data import load_dashboard_data
from src.dashboard_figures import build_growing_season_figure, build_timeseries_figure
from src.dashboard_schema import ComparisonConfig

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "Results" / "tables" / "dashboard_data"
OVERLAY_SETS_PATH = Path(__file__).resolve().parent / "Results" / "tables" / "dashboard_overlay_sets.xlsx"
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
    if "saved_overlay_sets" not in st.session_state:
        st.session_state.saved_overlay_sets = _load_saved_overlay_sets()


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


def _serialize_overlay_sets(saved_sets: dict[str, list[dict]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sets_rows = []
    overlay_rows = []
    timestamp = datetime.now().isoformat(timespec="seconds")
    for set_name, overlays in saved_sets.items():
        sets_rows.append({"set_name": set_name, "saved_at": timestamp, "overlay_count": len(overlays)})
        for overlay_order, overlay in enumerate(overlays, start=1):
            row = {"set_name": set_name, "overlay_order": overlay_order}
            row.update(overlay)
            overlay_rows.append(row)
    return pd.DataFrame(sets_rows), pd.DataFrame(overlay_rows)


def _load_saved_overlay_sets() -> dict[str, list[dict]]:
    if not OVERLAY_SETS_PATH.exists():
        return {}
    try:
        overlays = pd.read_excel(OVERLAY_SETS_PATH, sheet_name="overlays")
    except Exception:
        return {}
    saved_sets: dict[str, list[dict]] = {}
    if overlays.empty:
        return saved_sets
    overlays = overlays.sort_values(["set_name", "overlay_order"])
    for set_name, group in overlays.groupby("set_name", dropna=True):
        saved_sets[str(set_name)] = []
        for row in group.to_dict(orient="records"):
            row.pop("set_name", None)
            row.pop("overlay_order", None)
            for key in ("cloud_threshold",):
                if pd.notna(row.get(key)):
                    row[key] = int(row[key])
            for key in ("exclude_below_stddev", "exclude_above_stddev"):
                if pd.isna(row.get(key)):
                    row[key] = None
                elif row.get(key) is not None:
                    row[key] = float(row[key])
            saved_sets[str(set_name)].append(row)
    return saved_sets


def _persist_saved_overlay_sets() -> None:
    OVERLAY_SETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    sets_frame, overlays_frame = _serialize_overlay_sets(st.session_state.saved_overlay_sets)
    with pd.ExcelWriter(OVERLAY_SETS_PATH, engine="openpyxl") as writer:
        sets_frame.to_excel(writer, sheet_name="sets", index=False)
        overlays_frame.to_excel(writer, sheet_name="overlays", index=False)


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


def _render_saved_set_controls() -> None:
    st.sidebar.header("Saved overlay sets")
    saved_names = sorted(st.session_state.saved_overlay_sets.keys())
    if saved_names:
        selected_set = st.sidebar.selectbox("Saved set", saved_names, index=0, key="saved_set_picker")
        load_col, delete_col = st.sidebar.columns(2)
        if load_col.button("Load set", use_container_width=True):
            st.session_state.comparison_configs = [dict(item) for item in st.session_state.saved_overlay_sets[selected_set]]
            st.rerun()
        if delete_col.button("Delete set", use_container_width=True):
            st.session_state.saved_overlay_sets.pop(selected_set, None)
            _persist_saved_overlay_sets()
            st.rerun()
    else:
        st.sidebar.caption("No saved overlay sets yet.")

    save_name = st.sidebar.text_input("Save current overlays as", value="", key="save_overlay_set_name")
    if st.sidebar.button("Save overlay set", use_container_width=True):
        if not save_name.strip():
            st.sidebar.warning("Enter a set name first.")
        elif not st.session_state.comparison_configs:
            st.sidebar.warning("Add at least one overlay before saving a set.")
        else:
            st.session_state.saved_overlay_sets[save_name.strip()] = [dict(item) for item in st.session_state.comparison_configs]
            _persist_saved_overlay_sets()
            st.sidebar.success(f"Saved overlay set `{save_name.strip()}`")


def _render_data_dir_control() -> Path:
    st.sidebar.header("Data")
    data_dir_input = st.sidebar.text_input("Summary CSV directory", value=str(DEFAULT_DATA_DIR))
    return Path(data_dir_input)


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


def main() -> None:
    st.set_page_config(page_title="Appalachian Terrain–Vegetation Dashboard", layout="wide")
    _inject_ui_css()
    _init_state()

    st.title("Appalachian Terrain–Vegetation Analysis Dashboard")
    st.caption("Loads precomputed summary CSVs only. No raster statistics are recomputed in the app.")

    data_dir = _render_data_dir_control()
    _render_saved_set_controls()
    bundle = load_dashboard_data(data_dir)
    year_range, config, builder_mode, selected_existing = _build_sidebar(bundle)

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

    st.subheader("Growing-season overlay")
    if not ENABLE_GROWING_SEASON_OVERLAY:
        st.caption("This panel is currently disabled behind a feature flag. Planned future work includes a story-map-friendly animated export workflow.")
    else:
        if not st.session_state.comparison_configs:
            st.info("Add a comparison configuration first. The first saved configuration is used for the growing-season view.")
            return

        selected_config = ComparisonConfig(**st.session_state.comparison_configs[0])
        growing_years = sorted(
            bundle.scene_summary[
                (bundle.scene_summary["sensor"] == selected_config.sensor)
                & (bundle.scene_summary["aoi"] == selected_config.aoi)
                & (bundle.scene_summary["index"] == selected_config.index)
                & (bundle.scene_summary["growing_season_day"].notna())
            ]["year"].dropna().astype(int).unique().tolist()
        )
        if not growing_years:
            st.warning("No growing-season data are available for the first saved configuration.")
            return

        if min(growing_years) == max(growing_years):
            st.caption(f"Only one growing-season year is available for this configuration: {growing_years[0]}")
            focus_year = growing_years[0]
        else:
            focus_year = st.slider(
                "Highlighted year",
                min_value=min(growing_years),
                max_value=max(growing_years),
                value=max(growing_years),
            )
        growing_figure, growing_message = build_growing_season_figure(bundle, selected_config, focus_year)
        if growing_message:
            st.warning(growing_message)
        else:
            st.plotly_chart(growing_figure, use_container_width=True)

    st.subheader("Data availability")
    st.write(
        {
            "data_dir": str(bundle.data_dir),
            "scene_rows": len(bundle.scene_summary),
            "temporal_rows": len(bundle.temporal_summary),
            "growing_season_scene_rows": int(bundle.scene_summary["growing_season_day"].notna().sum()) if "growing_season_day" in bundle.scene_summary.columns else 0,
            "available_years": bundle.available_year_range(),
            "pixel_masks": bundle.available_values("pixel_mask_id"),
        }
    )
    # Future drought/ecozone extensions can add derived selectors here while
    # continuing to reuse the same comparison-config and figure plumbing.
    st.caption("Future drought and ecozone comparison controls can plug into this same configuration model.")


if __name__ == "__main__":
    main()
