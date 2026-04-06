# Ecozone Year-Type Cross Comparison

This folder is intended for explicit ecozone x year-type comparison outputs
built from existing summary tables.

The script to generate these outputs is:

`Python/Analysis/Traits/Ecozone/ecozone_yeartype_cross_comparison.py`

Run it manually from the project `Python/` directory:

```bash
python ./Analysis/Traits/Ecozone/ecozone_yeartype_cross_comparison.py
```

Inputs read by the script:

- `Results/Other/ecozone_yearclass_comparison/yearclass_summary_by_ecozone.csv`
- `Results/Other/ecozone_comparative_dynamics/ecozone_reference_metrics.csv`

Main metric choices:

- `onset`: `onset_month`
- `magnitude`: `cumulative_anomaly`
- `trajectory`: `late_minus_early_anomaly`
- `spatial_extent`: `fraction_below_10`

Planned outputs:

- `ecozone_yeartype_table.csv`
- `ecozone_yeartype_pairwise_comparisons.csv`
- `ecozone_yeartype_crossed_differences.csv`
- `ecozone_yeartype_crossed_pairwise.csv`
- `ecozone_yeartype_statements.csv`
- `ecozone_yeartype_statements.md`

These outputs are descriptive and support statements such as:

- ecozone A vs ecozone B under wet conditions
- ecozone A vs ecozone B under dry conditions
- how the wet-vs-dry contrast differs across ecozones
