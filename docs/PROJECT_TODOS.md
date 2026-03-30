# Project TODOs

Last updated: 2026-03-29

This file is the project-level task list for work that spans sessions or cuts across multiple scripts.
Use `logs/` for session actions and `PROJECT_OVERVIEW.md` for current project state.

## Active

- [ ] Create a maintained inventory of roadmap analyses with three statuses: implemented, validated in `Results_cache1`, and re-run verified against `_3_4`.
- [ ] Record which scripts are historically trusted `_3_4` lineage versus later `_3_24`-era modifications that still need validation.
- [ ] Decide whether to keep `Results_cache1` as archive-only or mirror its key summary tables into a versioned `docs/` or `Results/` location.
- [ ] Verify whether the `_3_24` cache deletion described in the 2026-03-28 log has actually been completed on disk, then document the post-deletion disk state.
- [ ] Add a short analysis index document that tells collaborators which script produces which figure/table/package.

## Analysis Gaps To Close

- [ ] Confirm whether long-term Landsat trend outputs were ever generated and preserved.
- [ ] Confirm whether any validated tables exist for ecozone analyses; none are currently visible in `Results_cache1`.
- [ ] Decide whether inter-trait crosstab scripts are exploratory utilities or part of the formal project deliverables.
- [ ] Add provenance notes for archived figures so each figure can be tied to a script, cache version, and approximate run date.
- [ ] Verify that current scripts writing to `Results/` still match the archived `_3_4` outputs in `Results_cache1`.

## Lower Priority

- [ ] Standardize naming between archived deliverables and current script outputs, especially annual vs monthly layer-package names.
- [ ] Add a lightweight reproducibility checklist for any future rerun against `_3_4`.
- [ ] Capture a collaborator-facing summary of the current ecological storyline supported by the validated analyses.
