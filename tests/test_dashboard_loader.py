from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.dashboard_data import filter_frame, load_dashboard_data, normalize_summary_frame
from src.dashboard_figures import ECOZONE_TRACE_COLORS, build_timeseries_figure, _stripe_fillcolor, _prism_stripes_for_configs
from src.dashboard_schema import ComparisonConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PROJECT_ROOT / "Results" / "tables" / "dashboard_samples"


class DashboardLoaderTests(unittest.TestCase):
    def test_load_dashboard_data_reads_fixture_tables(self) -> None:
        bundle = load_dashboard_data(SAMPLE_DIR)
        self.assertGreater(len(bundle.scene_summary), 0)
        self.assertGreater(len(bundle.temporal_summary), 0)
        self.assertEqual(bundle.available_year_range(), (2018, 2024))
        self.assertIn("ecozone", bundle.available_values("analysis_scope"))
        self.assertIn("forest_community", bundle.available_values("analysis_scope"))

    def test_normalization_handles_alias_columns(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "AOI Key": "north",
                    "Index": "NDVI",
                    "Scene Date": "2024-06-20",
                    "Platform": "landsat-8",
                    "Valid Pixels": 10,
                    "p95": 0.75,
                }
            ]
        )
        normalized = normalize_summary_frame(raw, "scene_summary")
        self.assertEqual(normalized.loc[0, "sensor"], "ls")
        self.assertEqual(normalized.loc[0, "aoi"], "north")
        self.assertEqual(normalized.loc[0, "index"], "ndvi")
        self.assertAlmostEqual(normalized.loc[0, "value"], 0.75)

    def test_normalization_corrects_forest_community_label_typo(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "sensor": "s2",
                    "aoi": "north",
                    "index": "ndvi",
                    "forest_community_code": 801,
                    "forest_community_label": "Northern Hardwood Slop",
                    "date": "2024-07-01",
                    "value": 0.75,
                }
            ]
        )
        normalized = normalize_summary_frame(raw, "temporal_summary_forest_community")
        self.assertEqual(normalized.loc[0, "forest_community_label"], "Northern Hardwood Slope")

    def test_normalization_preserves_forest_community_tier_fields(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "sensor": "s2",
                    "aoi": "north",
                    "index": "ndvi",
                    "forest_community_code": 116,
                    "forest_community_display_code": "16a",
                    "forest_community_label": "Dry-mesic oak",
                    "forest_community_source_dataset": "NBlueRidge",
                    "forest_community_source_value": 16,
                    "forest_community_source_key": "north:NBlueRidge:16",
                    "ecozone_group_code": 6,
                    "ecozone_group_label": "Dry-mesic oak",
                    "ecozone_group_raw": "6-Dry-mesic oak",
                    "date": "2024-07-01",
                    "value": 0.75,
                }
            ]
        )
        normalized = normalize_summary_frame(raw, "temporal_summary_forest_community")
        self.assertEqual(int(normalized.loc[0, "forest_community_code"]), 116)
        self.assertEqual(normalized.loc[0, "forest_community_display_code"], "16a")
        self.assertEqual(int(normalized.loc[0, "forest_community_source_value"]), 16)
        self.assertEqual(int(normalized.loc[0, "ecozone_group_code"]), 6)
        self.assertEqual(normalized.loc[0, "ecozone_group_raw"], "6-Dry-mesic oak")

    def test_filter_frame_applies_filters_and_year_range(self) -> None:
        bundle = load_dashboard_data(SAMPLE_DIR)
        filtered = filter_frame(
            bundle.scene_summary,
            filters={"sensor": "s2", "aoi": "north", "index": "ndvi"},
            year_range=(2023, 2024),
        )
        self.assertTrue((filtered["sensor"] == "s2").all())
        self.assertTrue((filtered["aoi"] == "north").all())
        self.assertTrue(filtered["year"].between(2023, 2024).all())

    def test_frame_for_config_loads_filtered_ecozone_table(self) -> None:
        bundle = load_dashboard_data(SAMPLE_DIR)
        config = ComparisonConfig(
            label="",
            analysis_scope="ecozone",
            sensor="s2",
            aoi="north",
            index="ndvi",
            ecozone_code=1,
            spatial_percentile="p95",
            temporal_agg="month",
            temporal_percentile="p95",
            cloud_threshold=40,
            season_filter="growing",
        )
        filtered = bundle.frame_for_config(config)
        self.assertEqual(len(filtered), 2)
        self.assertTrue((filtered["analysis_scope"] == "ecozone").all())
        self.assertTrue((filtered["ecozone_code"] == 1).all())
        self.assertTrue((filtered["ecozone_label"] == "Cool").all())

    def test_ecozone_all_draws_one_trace_per_ecozone(self) -> None:
        bundle = load_dashboard_data(SAMPLE_DIR)
        config = ComparisonConfig(
            label="",
            analysis_scope="ecozone",
            sensor="s2",
            aoi="north",
            index="ndvi",
            ecozone_code=None,
            spatial_percentile="p95",
            temporal_agg="month",
            temporal_percentile="p95",
            cloud_threshold=40,
            season_filter="growing",
        )
        figure, messages = build_timeseries_figure(bundle, [config], (2023, 2024))
        self.assertEqual(messages, [])
        self.assertEqual(len(figure.data), 2)
        self.assertEqual(
            {trace.name for trace in figure.data},
            {
                "GW-Jeff / sentinel-2 / ndvi / cool / 40% / p95 / month / p95",
                "GW-Jeff / sentinel-2 / ndvi / intermediate / 40% / p95 / month / p95",
            },
        )
        self.assertEqual([trace.line.color for trace in figure.data], [ECOZONE_TRACE_COLORS[1], ECOZONE_TRACE_COLORS[2]])

    def test_all_segment_visibility_filters_traces(self) -> None:
        bundle = load_dashboard_data(SAMPLE_DIR)
        config = ComparisonConfig(
            label="",
            analysis_scope="ecozone",
            sensor="s2",
            aoi="north",
            index="ndvi",
            ecozone_code=None,
            spatial_percentile="p95",
            temporal_agg="month",
            temporal_percentile="p95",
            cloud_threshold=40,
            season_filter="growing",
        )
        figure, messages = build_timeseries_figure(bundle, [config], (2023, 2024), {0: {2}})
        self.assertEqual(messages, [])
        self.assertEqual(len(figure.data), 1)
        self.assertEqual(figure.data[0].name, "GW-Jeff / sentinel-2 / ndvi / intermediate / 40% / p95 / month / p95")

    def test_forest_community_all_uses_unique_colors_for_selected_traces(self) -> None:
        bundle = load_dashboard_data(SAMPLE_DIR)
        config = ComparisonConfig(
            label="",
            analysis_scope="forest_community",
            sensor="s2",
            aoi="north",
            index="ndvi",
            forest_community_code=None,
            spatial_percentile="p95",
            temporal_agg="month",
            temporal_percentile="p95",
            cloud_threshold=40,
            season_filter="growing",
        )
        figure, messages = build_timeseries_figure(bundle, [config], (2023, 2024), {0: {502, 503}})
        self.assertEqual(messages, [])
        colors = [trace.line.color for trace in figure.data]
        self.assertEqual(len(colors), 2)
        self.assertEqual(len(set(colors)), 2)

    def test_frame_for_config_loads_filtered_forest_community_table(self) -> None:
        bundle = load_dashboard_data(SAMPLE_DIR)
        config = ComparisonConfig(
            label="",
            analysis_scope="forest_community",
            sensor="s2",
            aoi="north",
            index="ndvi",
            forest_community_code=502,
            spatial_percentile="p95",
            temporal_agg="month",
            temporal_percentile="p95",
            cloud_threshold=40,
            season_filter="growing",
        )
        filtered = bundle.frame_for_config(config)
        self.assertEqual(len(filtered), 2)
        self.assertTrue((filtered["analysis_scope"] == "forest_community").all())
        self.assertTrue((filtered["forest_community_code"] == 502).all())
        self.assertTrue((filtered["forest_community_label"] == "Chestnut oak").all())

    def test_scene_filter_uses_cloud_percent_when_cloud_threshold_column_is_missing(self) -> None:
        frame = pd.DataFrame(
            [
                {"sensor": "s2", "cloud_percent": 20.0, "value": 1.0},
                {"sensor": "s2", "cloud_percent": 45.0, "value": 2.0},
            ]
        )
        filtered = filter_frame(frame, filters={"cloud_threshold": 30})
        self.assertEqual(len(filtered), 1)
        self.assertEqual(float(filtered.iloc[0]["value"]), 1.0)

    def test_season_all_is_a_real_filter_value(self) -> None:
        frame = pd.DataFrame(
            [
                {"season_filter": "all", "growing_season_day": pd.NA, "value": 1.0},
                {"season_filter": "growing", "growing_season_day": 10, "value": 2.0},
            ]
        )
        filtered = filter_frame(frame, filters={"season_filter": "all"})
        self.assertEqual(filtered["value"].tolist(), [1.0])

    def test_growing_filter_falls_back_to_growing_day_for_scene_rows(self) -> None:
        frame = pd.DataFrame(
            [
                {"season_filter": "all", "growing_season_day": pd.NA, "value": 1.0},
                {"season_filter": "all", "growing_season_day": 10, "value": 2.0},
            ]
        )
        filtered = filter_frame(frame, filters={"season_filter": "growing"})
        self.assertEqual(filtered["value"].tolist(), [2.0])

    def test_prism_stripes_use_selected_aoi_classes(self) -> None:
        config = ComparisonConfig(
            label="",
            sensor="ls",
            aoi="north",
            index="ndmi",
            spatial_percentile="p50",
            temporal_agg="month",
            temporal_percentile="p50",
            cloud_threshold=30,
            season_filter="growing",
        )
        prism = pd.DataFrame(
            [
                {"aoi": "north", "year": 2000, "annual_precip_mm": 650.0, "precip_zscore": 2.0},
                {"aoi": "north", "year": 2001, "annual_precip_mm": 584.0, "precip_zscore": 0.9},
                {"aoi": "north", "year": 2002, "annual_precip_mm": 410.0, "precip_zscore": -2.0},
                {"aoi": "south", "year": 2000, "annual_precip_mm": 350.0, "precip_zscore": -2.0},
            ]
        )
        stripes = _prism_stripes_for_configs([config], (1999, 2002), prism)
        self.assertEqual(
            stripes.to_dict("records"),
            [
                {
                    "year": 2000,
                    "classification": "wet",
                    "stripe_strength": 1.0,
                    "annual_precip_mm": 650.0,
                    "precip_zscore": 2.0,
                },
                {
                    "year": 2002,
                    "classification": "dry",
                    "stripe_strength": 1.0,
                    "annual_precip_mm": 410.0,
                    "precip_zscore": -2.0,
                },
            ],
        )

    def test_prism_stripes_for_mixed_aois_are_hidden(self) -> None:
        north = ComparisonConfig(
            label="",
            sensor="ls",
            aoi="north",
            index="ndmi",
            spatial_percentile="p50",
            temporal_agg="month",
            temporal_percentile="p50",
            cloud_threshold=30,
            season_filter="growing",
        )
        south = ComparisonConfig(
            label="",
            sensor="ls",
            aoi="south",
            index="ndmi",
            spatial_percentile="p50",
            temporal_agg="month",
            temporal_percentile="p50",
            cloud_threshold=30,
            season_filter="growing",
        )
        prism = pd.DataFrame(
            [
                {"aoi": "north", "year": 2000, "annual_precip_mm": 620.0, "precip_zscore": 1.7},
                {"aoi": "south", "year": 2000, "annual_precip_mm": 390.0, "precip_zscore": -1.7},
                {"aoi": "north", "year": 2001, "annual_precip_mm": 440.0, "precip_zscore": -1.5},
                {"aoi": "south", "year": 2001, "annual_precip_mm": 300.0, "precip_zscore": -2.0},
            ]
        )
        stripes = _prism_stripes_for_configs([north, south], (2000, 2001), prism)
        self.assertTrue(stripes.empty)

    def test_prism_stripe_color_ramp_increases_with_extremity(self) -> None:
        self.assertEqual(_stripe_fillcolor("wet", 0), "rgba(68, 150, 248, 0.300)")
        self.assertEqual(_stripe_fillcolor("wet", 1), "rgba(0, 90, 220, 0.300)")
        self.assertEqual(_stripe_fillcolor("dry", 0), "rgba(247, 183, 183, 0.300)")
        self.assertEqual(_stripe_fillcolor("dry", 1), "rgba(215, 35, 35, 0.300)")


if __name__ == "__main__":
    unittest.main()
