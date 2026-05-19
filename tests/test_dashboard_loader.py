from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.dashboard_data import filter_frame, load_dashboard_data, normalize_summary_frame


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PROJECT_ROOT / "Results" / "tables" / "dashboard_samples"


class DashboardLoaderTests(unittest.TestCase):
    def test_load_dashboard_data_reads_fixture_tables(self) -> None:
        bundle = load_dashboard_data(SAMPLE_DIR)
        self.assertGreater(len(bundle.scene_summary), 0)
        self.assertGreater(len(bundle.temporal_summary), 0)
        self.assertEqual(bundle.available_year_range(), (2018, 2024))

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


if __name__ == "__main__":
    unittest.main()
