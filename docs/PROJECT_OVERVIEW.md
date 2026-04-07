NDVI / NDMI Drought Analysis — Project Overview

## Purpose

This project investigates vegetation productivity and drought dynamics across ecozones using Sentinel-2-derived indices, with current emphasis on NDVI and NDMI. The goal is to understand how different ecozones respond to moisture variability, including drought stress, resilience, and recovery patterns.

The analysis is comparative and ecological rather than purely predictive. The focus is on differences between ecozones and on seasonal and annual patterns.

## Current Status

The project is in a recovery and stabilization phase after tooling and pipeline issues.

Recent problems included:

- Unreliable script generation, including rate limiting, auth, and syntax issues
- Confusion around data deletion and cache state
- Loss of confidence in what data is present and usable
- Lack of persistent session-level logging

Current priorities are:

- Verify and stabilize data sources
- Rebuild a minimal, trustworthy pipeline
- Establish clear documentation and logging practices

This documentation push is not separate from the technical work. It is a direct response to difficulty managing project state across a large, evolving AI-assisted workflow and to the need to communicate clearly with other people involved in the project.

## Script Lineage

The current codebase is not a clean split between "old trusted scripts" and "new untrusted scripts."

More accurate interpretation:

- the core cache-building and analysis scripts descend from the `_3_4` working pipeline
- some of those scripts were later renamed, generalized, or modified in preparation for a future `_3_24` rerun
- the `_3_24` rerun never completed, so those later modifications were not validated end-to-end
- one-off repair and support scripts should be treated separately from the historical core pipeline

Practical trust categories:

- historical core pipeline
- historical core pipeline with later unvalidated modifications
- one-off recovery or support tooling

## Lessons Learned / Rebuild Guardrails

The trust breakdown during the `_3_24` effort was not a blanket failure of the whole codebase. It was concentrated in the rebuild and orchestration layer.

Observed failure modes:

- repeated rebuild coaching around rate-limit handling and signed-token refresh did not reliably complete more than one pass
- restart and retry logic became too complex relative to what could be confidently verified
- the rebuild effort proceeded under an incorrect assumption that `_3_4` had already been deleted
- that incorrect storage assumption contributed directly to the full-disk failure
- project state became too hard to track mentally, especially while coordinating with AI tools and trying to preserve documentation for others

Guardrails for future rebuild attempts:

- verify disk state before any large build
- verify which cache directories actually exist before assuming deletions happened
- prefer simple, inspectable operational logic over elaborate automated recovery behavior
- treat rebuild orchestration as separate from trusted analysis logic
- make project-state updates in documentation as decisions happen, not later from memory
- maintain orientation documents that are understandable to collaborators, not only to the current working session

## Authoritative Data State

This section is the current source of truth for cache status.

- The authoritative working cache baseline is `_3_4`
- The `_3_24` rebuild is being abandoned and deleted due to disk pressure and low trust in the partial rebuild outputs
- Stabilization work should assume analysis runs against `_3_4` unless this file is updated

Current cache interpretation:

- `GWNF_cache` and `Smoky_cache` are the stable analysis base
- `GWNF_cache_3_24` was a partial north Sentinel rebuild that stopped when disk space filled
- `Smoky_cache_3_24` was effectively empty and not analysis-ready

Rationale for abandoning `_3_24`:

- Disk space is full and must be reclaimed
- The partial rebuild is not trusted enough to justify preserving it
- Stabilization is more important than pursuing updated data immediately
- Updated data with broader SCL handling is still a future goal, but not at the expense of a stable baseline

## Data Context

Primary source: Sentinel-2 imagery

Current accepted constraint:

- the present working Sentinel analysis appears to reflect `SCL = 4` vegetation pixels only

Implications:

- transitional or stressed vegetation states may be underrepresented
- early drought signals may be smoothed
- the data is still usable for relative comparisons across ecozones

This limitation is known and accepted for the stabilization phase, but may be revisited later.

## Results Provenance

The validated historical deliverables from the `_3_4` analysis runs live in `Results_cache1/`.

Important interpretation:

- `Results_cache1/` is the last known-good output archive from `_3_4`
- `Results/` was later repurposed as the intended destination for a future `_3_24` rerun
- that `_3_24` rerun never completed, so `Results/` should not be treated as the validated archive

For stabilization work:

- use `Results_cache1/` when asking what has already been successfully produced
- use `_3_4` caches when asking what data the validated outputs came from
- treat `Results/` as a planned rerun location, not as evidence of completed analysis

## Analysis Direction

Planned or in-progress analyses include:

- Ecozone productivity using NDVI patterns
- Moisture dynamics using NDMI trends
- NDVI vs NDMI relationships
- Seasonal peak timing by ecozone
- Identification of wet vs dry years using annual NDMI context
- Peak vegetation response in wet vs dry years
- Moisture resilience by ecozone
- Growing season moisture stress
- Elevation gradients and drought response

These analyses may be reprioritized as the pipeline stabilizes.

## Minimal Trustworthy Pipeline

Until the project is stable, the smallest trustworthy path is:

1. Use `_3_4` Sentinel caches
2. Use existing `_3_4` ecozone traits
3. Treat `Results_cache1/` as the reference output archive
4. Only re-run one NDMI-focused ecozone analysis if verification requires it

The preferred first stabilization target is documentation and verification of the existing `_3_4` baseline rather than any rebuild or large multi-stage automation.

## Known Issues / Risks

- Uncertainty about historical cache decisions and what was intended versus what actually exists
- Over-reliance on generated scripts without verification
- Support and recovery scripts may be weaker than the core analysis scripts
- Missing documentation of what has been run versus what has only been planned
- Disk space constraints directly affect pipeline decisions
- Current scripts write to `Results/`, while validated historical outputs live in `Results_cache1/`
- Some core scripts have working `_3_4` lineage but later `_3_24`-era edits that were never fully revalidated

## Working Principles

- Prioritize clarity and reproducibility over speed
- Use minimal, inspectable scripts instead of complex automation
- Do not delete or modify data without verification
- Treat AI tools as assistants, not autonomous agents
- Maintain a continuous log of actions and decisions
- Record which cache version is considered authoritative in every working session
- Prefer verified filesystem state and existing outputs over remembered tool claims
- Write process documentation for collaborators as well as for personal state management

## Project Structure

- `logs/` — session-based working logs
- `docs/` — higher-level summaries and cleaned documentation
- `Analysis/` — analysis scripts
- `Cache/` — cache-building and cache-audit scripts
- `config/` — paths and year classification inputs
- `PROJECT_OVERVIEW.md` — source of truth for current project state

## Definition of Stable

The project is considered stable when:

- Data location and contents are verified and documented
- A minimal pipeline runs successfully end-to-end
- Each analysis step is understandable and reproducible
- Logs clearly reflect what has been done and why
- The authoritative cache baseline is explicit and consistent

## Immediate Next Steps

- Delete `_3_24` caches to recover disk space
- Treat `_3_4` as the only authoritative cache baseline
- Begin logging each session in `logs/`
- Record that `Results_cache1/` is the validated `_3_4` deliverable archive
- Inventory which scripts are historically validated versus later modified for `_3_24`
- Refine this overview as understanding improves
