# Index Role Summary

This folder is intended for a compact cross-index comparison built from the
existing ecozone investigation CSV outputs.

The script to generate these outputs is:

`Python/Analysis/Traits/Ecozone/ecozone_index_role_summary.py`

Run it manually from the project `Python/` directory:

```bash
python ./Analysis/Traits/Ecozone/ecozone_index_role_summary.py
```

Expected inputs:

- `Results/2_Anomaly_Onset/onset_timing/onset_timing_summary_all_percentiles.csv`
- `Results/3_Anomaly_Progression/monthly_trajectories/trajectory_summary_all_percentiles.csv`
- `Results/3_Anomaly_Progression/monthly_trajectories/trajectory_monthly_anomalies_all_percentiles.csv`
- `Results/3_Anomaly_Progression/percentile_spread/percentile_spread_group_summary.csv`
- `Results/4_Anomaly_Recovery/simple_recovery/simple_recovery_by_year_all_percentiles.csv`

Planned outputs:

- `index_role_comparison_by_class.csv`
- `index_role_comparison_overall.csv`
- `index_role_overall_ranking.csv`
- `index_role_metric_heatmaps.png`
- `index_role_metric_bars.png`

Heuristic definitions:

- `onset_month_stability`: higher when onset month is similar across available
  summary percentiles.
- `direction_sign_consistency`: higher when monthly progression anomalies mostly
  keep the same sign once they exceed a small anomaly threshold.
- `trajectory_clarity`: average of direction consistency, peak-month stability,
  and inverse core spread reliability.
- `recovery_label_consistency`: share of evaluable dry-year recovery cases that
  fall into the dominant recovery label.

These metrics are intended to support interpretation, not to claim formal
certainty.
