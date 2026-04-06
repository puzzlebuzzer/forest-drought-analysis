# Ecozone Year-Class Comparison

This folder is intended for wet / normal / dry comparison outputs built from
existing ecozone-level Sentinel summary CSVs.

The script to generate these outputs is:

`Python/Analysis/Traits/Ecozone/ecozone_yearclass_comparison.py`

Run it manually from the project `Python/` directory:

```bash
python ./Analysis/Traits/Ecozone/ecozone_yearclass_comparison.py
```

Inputs read by the script:

- `Results/2_Anomaly_Onset/onset_timing/onset_timing_summary_all_percentiles.csv`
- `Results/2_Anomaly_Onset/onset_timing/onset_timing_monthly_details_all_percentiles.csv`
- `Results/3_Anomaly_Progression/monthly_trajectories/trajectory_monthly_anomalies_all_percentiles.csv`
- `Results/2_Anomaly_Onset/fraction_below_baseline/fraction_below_baseline.csv`
- `Results/Other/ecozone_comparative_dynamics/ecozone_reference_metrics.csv`

Planned outputs:

- `yearclass_unified_comparison.csv`
- `yearclass_summary_by_aoi_index.csv`
- `yearclass_summary_by_ecozone.csv`
- `yearclass_comparison_heatmaps.png`
- `yearclass_onset_months.png`

Metric rules:

- `normal` means the neutral baseline.
- For anomaly metrics, `value_normal = 0.0` because the source anomaly values
  are already defined relative to neutral.
- For spatial extent metrics (`fraction_below_*`), `value_normal` comes from the
  actual `neutral` rows in `fraction_below_baseline.csv`.
- For `onset_month`, `value_normal` remains `NA` because neutral does not have a
  divergence month in the source onset outputs.

Main metric types:

- `onset_month`
- `onset_anomaly`
- `cumulative_anomaly`
- `early_season_anomaly`
- `late_season_anomaly`
- `late_minus_early_anomaly`
- `fraction_below_04`, `fraction_below_06`, `fraction_below_08`, `fraction_below_10`

These outputs are descriptive and meant to support statements such as:

- wet diverges earlier than dry in a given ecozone
- dry shows stronger suppression relative to normal
- wet and dry differ in late-season persistence relative to normal
- dry occupies more below-baseline area than neutral in key months
