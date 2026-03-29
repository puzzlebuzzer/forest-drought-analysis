
NDVI / NDMI Drought Analysis — Project Overview


Purpose


This project investigates vegetation productivity and drought dynamics across ecozones using Sentinel-2–derived indices (NDVI, NDMI). The goal is to understand how different ecozones respond to moisture variability, including identifying drought stress, resilience, and recovery patterns.



The analysis is comparative and ecological rather than purely predictive—focused on differences between ecozones and seasonal/annual patterns.

Current Status


The project is in a recovery and stabilization phase after tooling and pipeline issues.



Recent problems included:

Unreliable script generation (rate limiting, auth, syntax issues)

Confusion around data deletion and cache state

Loss of confidence in what data is present and usable



At present, the priority is:

Verify and stabilize data sources

Rebuild a minimal, trustworthy pipeline

Establish clear documentation and logging practices

Data Context
Primary source: Sentinel-2 imagery

Current constraint: data appears to include SCL = 4 (vegetation) pixels only



Implications:

Likely excludes transitional or stressed vegetation states

May smooth or underrepresent early drought signals

Still valid for relative comparisons across ecozones



This limitation is known and accepted for now, but may be revisited.

Analysis Direction


Planned / in-progress analyses include:

Ecozone productivity (NDVI patterns)

Moisture dynamics (NDMI trends)

NDVI vs NDMI relationships (scatter analysis)

Seasonal peak timing by ecozone

Identification of wet vs dry years (annual NDMI)

Peak vegetation response in wet vs dry years

Moisture resilience by ecozone

Growing season moisture stress

Elevation gradients and drought response



These analyses may be reprioritized as the pipeline stabilizes.

Known Issues / Risks
Uncertainty about completeness and location of cached data

Over-reliance on generated scripts without verification

Lack of persistent system state tracking

Missing documentation of what has been run vs planned

Working Principles (Going Forward)
Prioritize clarity and reproducibility over speed

Use minimal, inspectable scripts instead of complex automation

Do not delete or modify data without verification

Treat AI tools as assistants, not autonomous agents

Maintain a continuous log of actions and decisions

Project Structure (Planned)
/logs/ — session-based working logs (unfiltered thinking + actions)

/docs/ — higher-level summaries and cleaned documentation

/analysis/ — scripts and outputs

PROJECT_OVERVIEW.md — this file; source of truth for project state

Definition of “Stable”


The project is considered stable when:

Data location and contents are verified and documented

A minimal pipeline runs successfully end-to-end

Each analysis step is understandable and reproducible

Logs clearly reflect what has been done and why

Next Steps
Verify Sentinel-2 cache contents and location

Identify one minimal analysis to run successfully

Begin logging each session in /logs/

Refine this overview as understanding improves

