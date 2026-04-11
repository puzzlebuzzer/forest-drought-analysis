# Resume Brief — 2026-04-11

## Purpose

Reconstruct the missing end-of-session context after the close action happened before `Cache/save_session_snapshot.sh` was run.

## Recovered Source Files

- `CODEX.md`
- `docs/PROJECT_OVERVIEW.md`
- `docs/DATA_PROVENANCE_AND_CACHE_CHARACTERISTICS.md`
- `docs/DERIVATION_ECOZONE_SEASONAL_CURVES_NDVI.md`
- `docs/DERIVATION_MONTHLY_P99_ANOMALY_TRAJECTORY_NDVI.md`
- `Results/INVESTIGATION_LAYER_NOTE.md`
- `logs/2026-03-28.md`
- `docs/PROJECT_TODOS.md`
- `docs/ANALYSIS_GAP_REVIEW.md`

## Confirmed Project State

- Date of reconstruction: `2026-04-11`
- The git repo root is `Python/`
- The active cache roots on disk are:
  - `AOI/NorthAOI/GWNF_cache`
  - `AOI/SouthAOI/Smoky_cache`
- No `_3_24` cache directories were found during this reconstruction pass
- The March 28 stabilization decision appears to have been carried out on disk
- `Results_cache1/` remains the documented validated archive from the historical `_3_4` baseline
- `Results/` remains the active/current output area for newer work and investigation outputs

## Most Relevant Prior Decisions

- Treat `_3_4` as the authoritative baseline unless the docs are explicitly updated
- Treat later rebuild orchestration and one-off support scripts with more caution than the historical core analysis lineage
- Prefer simple, inspectable workflows over complex automated restart logic
- Maintain session-level documentation because project state has become hard to preserve across AI-assisted sessions

## Most Recent Recoverable Work

The latest recoverable authored changes are documentation-focused and date to April 7 through April 11:

- `docs/DATA_PROVENANCE_AND_CACHE_CHARACTERISTICS.md`
  - separates trusted historical `_3_4` Sentinel behavior from the current rebuild code
  - documents that the historical Sentinel baseline used `SCL == 4`, AOI bounding-box framing, and no AOI polygon mask
- `docs/DERIVATION_ECOZONE_SEASONAL_CURVES_NDVI.md`
  - explains the raw scene-level pooled seasonal NDVI workflow
- `docs/DERIVATION_MONTHLY_P99_ANOMALY_TRAJECTORY_NDVI.md`
  - documents the anomaly-trajectory workflow and notes that the current repo script uses `p50` and `p75`, not `p99`
- `Results/INVESTIGATION_LAYER_NOTE.md`
  - documents the newer investigation layer scripts for anomaly onset, monthly trajectories, and simple recovery

## Current Operational Traces Still On Disk

### Build Progress

- `AOI/SouthAOI/Smoky_cache/landsat/indices/build_progress_south.txt`
  - updated `2026-04-09 11:15:49`
  - status: `finished with 3 failures — re-run to retry`
- `AOI/NorthAOI/GWNF_cache/landsat/indices/build_progress_north.txt`
  - updated `2026-04-06 10:40:04`
  - status: `finished with 9 failures — re-run to retry`
  - includes at least one logged `403` rate-limit event
- `AOI/SouthAOI/Smoky_cache/s2/indices/build_progress_south.txt`
  - updated `2026-04-07 17:00:14`
  - status: `starting`
  - `NDVI`, `NDMI`, and `EVI` show `Needed 0`
  - `SCL` shows `1618` remaining, so that run did not progress past startup

### Older Restart Evidence

- `Cache/restart_log_north.txt`
  - preserves the March 25-26 `_3_24` north Sentinel restart loop
  - shows repeated `403`-triggered restarts against the now-abandoned `_3_24` suffix path

## What Was Not Recoverable

- No session snapshot exists under `logs/session_snapshots/`
- No `/tmp/cache_builder_runner_landsat.log` was present
- No `/tmp/cache_builder_runner_sentinel.log` was present
- The missing terminal state could not be reconstructed beyond the surviving docs and build-progress artifacts

## Working Interpretation

The last visible work before this reconstruction was primarily documentation and provenance cleanup, not an active full cache rebuild. The project appears to have shifted from rebuild/recovery work toward:

- documenting trustworthy baseline behavior
- writing derivation notes for key NDVI deliverables
- preserving the distinction between validated archive outputs and newer investigation outputs

## Suggested Reopen Order

1. `CODEX.md`
2. `logs/2026-04-11_resume_brief.md`
3. `docs/PROJECT_OVERVIEW.md`
4. `docs/DATA_PROVENANCE_AND_CACHE_CHARACTERISTICS.md`
5. Latest `build_progress_*.txt` if cache work resumes
6. Figure-specific derivation notes when working on a deliverable
