# Ecozone Comparative Dynamics

This folder is intended for explicit ecozone-vs-ecozone comparisons built from
existing Sentinel ecozone summary CSV outputs.

The script to generate these outputs is:

`Python/Analysis/Traits/Ecozone/ecozone_comparative_dynamics.py`

Run it manually from the project `Python/` directory:

```bash
python ./Analysis/Traits/Ecozone/ecozone_comparative_dynamics.py
```

Inputs read by the script:

- `Results/2_Anomaly_Onset/onset_timing/onset_timing_summary_all_percentiles.csv`
- `Results/2_Anomaly_Onset/onset_timing/onset_timing_monthly_details_all_percentiles.csv`
- `Results/2_Anomaly_Onset/fraction_below_baseline/fraction_below_baseline.csv`
- `Results/3_Anomaly_Progression/monthly_trajectories/trajectory_summary_all_percentiles.csv`
- `Results/3_Anomaly_Progression/monthly_trajectories/trajectory_monthly_anomalies_all_percentiles.csv`
- `Results/4_Anomaly_Recovery/simple_recovery/simple_recovery_by_year_all_percentiles.csv`

Planned outputs:

- `ecozone_reference_metrics.csv`
- `ecozone_pairwise_comparisons.csv`
- `ecozone_pairwise_summary.csv`
- `ecozone_onset_heatmaps.png`
- `ecozone_magnitude_heatmaps.png`
- `ecozone_trajectory_heatmaps.png`
- `ecozone_recovery_heatmaps.png`

Comparison metrics:

- onset timing: mean onset month and mean onset anomaly by ecozone
- magnitude context: mean fraction below baseline at Apr, Jun, Aug, and Oct
- trajectory shape: peak month, peak anomaly, cumulative Apr-Oct anomaly, and
  late-minus-early season anomaly
- recovery: dominant recovery label, mean late-season anomaly, mean following
  spring anomaly, and mean net change

The outputs are descriptive and deterministic. They are intended to support
statements such as:

- one ecozone diverges earlier or later than another
- one ecozone shows stronger suppression than another
- one ecozone has a more sustained late-season response
- one ecozone shows a larger recovery shift than another

Known limitations:

- comparisons rely on existing summary CSVs rather than raw raster re-analysis
- recovery remains limited by the coarse existing recovery categories
- differences are descriptive; they are not formal statistical tests
