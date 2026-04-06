# Anomaly From Normal Demo

This folder is intended for a small presentation-ready set of Sentinel anomaly
rasters generated from existing monthly composites.

The script to generate these outputs is:

`Python/Analysis/Diagnostics/anomaly_from_normal_demo.py`

Run it manually from the project `Python/` directory:

```bash
python ./Analysis/Diagnostics/anomaly_from_normal_demo.py
```

Planned demo set:

- north NDVI dry October 2023
- north NDMI dry October 2023
- south NDVI wet July 2020
- south NDMI wet July 2020
- north NDVI dry Apr-Oct 2023 mean anomaly composite
- south NDMI wet Apr-Oct 2018-2021 mean anomaly composite

Outputs written by the script:

- baseline GeoTIFFs for each selected example
- anomaly GeoTIFFs for each selected example
- quicklook PNG previews
- `demo_manifest.csv`

Baseline rule:

- per-pixel median across available neutral-year monthly Sentinel composites
- computed separately by AOI, index, and month

The intent is demonstration only. This is not a full anomaly production archive.
