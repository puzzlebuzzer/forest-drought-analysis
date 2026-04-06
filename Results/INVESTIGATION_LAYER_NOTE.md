# Investigation Layer Note

## Files Created

- `Analysis/Traits/Ecozone/investigation_common.py`
- `Analysis/Traits/Ecozone/ecozone_onset_timing_investigation.py`
- `Analysis/Traits/Ecozone/ecozone_monthly_trajectory_investigation.py`
- `Analysis/Traits/Ecozone/ecozone_simple_recovery_investigation.py`

## Input Assumptions

- Sentinel monthly composites are read from `Results/0-CacheBaseData/monthly_max`.
- Wet/dry AOI year labels are reused from `config/wet_dry_years.csv`.
- Ecozone grouping is reused from each AOI's `tnc_ecozone_simplified_snapped.tif`.
- These scripts do not edit or rebuild any existing cache, analysis script, or result.

## Analysis Assumptions

- Baseline seasonal behavior is defined from `neutral` years only.
- Monthly ecozone summaries now iterate across `SUMMARY_PERCENTILES`, currently `50` and `75`.
- Onset timing uses a descriptive divergence rule:
  `abs(dry anomaly) >= max(0.03, 1.0 * baseline monthly std)`.
- Onset timing records both `wet` and `dry` anomalies relative to the neutral baseline.
- Progression uses monthly `wet` and `dry` anomalies relative to the neutral baseline.
- Recovery uses a simple next-spring check:
  late-season stress in Aug-Oct of a dry year is compared against Apr-Jun of the following year.
- Recovery also includes a `wet_reference` track for comparison.
- Recovery classes are descriptive only:
  `quick`, `partial`, `slow`, `insufficient_data`.

## How To Run

From the `Python` project root:

```bash
python Analysis/Traits/Ecozone/ecozone_onset_timing_investigation.py
python Analysis/Traits/Ecozone/ecozone_monthly_trajectory_investigation.py
python Analysis/Traits/Ecozone/ecozone_simple_recovery_investigation.py
```

## Outputs

### Onset Timing

- Folder: `Results/2_Anomaly_Onset/onset_timing/`
- Outputs:
  - `onset_ndvi_p50.png`, `onset_ndvi_p75.png`
  - `onset_ndmi_p50.png`, `onset_ndmi_p75.png`
  - `onset_evi_p50.png`, `onset_evi_p75.png`
  - `onset_timing_summary_p50.csv`, `onset_timing_summary_p75.csv`
  - `onset_timing_monthly_details_p50.csv`, `onset_timing_monthly_details_p75.csv`

### Monthly Trajectories

- Folder: `Results/3_Anomaly_Progression/monthly_trajectories/`
- Outputs:
  - `trajectories_ndvi_p50.png`, `trajectories_ndvi_p75.png`
  - `trajectories_ndmi_p50.png`, `trajectories_ndmi_p75.png`
  - `trajectories_evi_p50.png`, `trajectories_evi_p75.png`
  - `trajectory_monthly_anomalies_p50.csv`, `trajectory_monthly_anomalies_p75.csv`
  - `trajectory_summary_p50.csv`, `trajectory_summary_p75.csv`

### Simple Recovery

- Folder: `Results/4_Anomaly_Recovery/simple_recovery/`
- Outputs:
  - `recovery_ndvi_p50.png`, `recovery_ndvi_p75.png`
  - `recovery_ndmi_p50.png`, `recovery_ndmi_p75.png`
  - `recovery_evi_p50.png`, `recovery_evi_p75.png`
  - `simple_recovery_by_year_p50.csv`, `simple_recovery_by_year_p75.csv`
  - `simple_recovery_summary_p50.csv`, `simple_recovery_summary_p75.csv`
