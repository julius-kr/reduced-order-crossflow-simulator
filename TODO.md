# TODO

Working list for the rewrite. The plan and its acceptance gates are in `docs/implementation_plan.md`.

## M0: layout, legacy isolation, baseline
- [x] Freeze `python/` as `legacy/python/` with a hash manifest
- [x] Scaffold `src/`: typed project config, units, geometry
- [ ] Legacy adapter and isolated worker (never import the snapshot in-process)
- [ ] Representative example projects (work package A)
- [ ] Record the legacy behaviour baseline
- [ ] Remove `python/` once nothing references it

## M1: pressure, profiles, caching
- [ ] Single-series duct profile instead of the legacy double loop
- [ ] Sparse lubrication pressure on physical spacings
- [ ] Hydraulic network for pressure-driven operation
- [ ] Bounded caches with explicit keys
- [ ] Performance report

## M2: transport and maps
- [ ] 3D Stokes per feature group
- [ ] Continuous streamline tracing (no per-step resampling)
- [ ] Conservative remap

## M3/M4: junctions and pillars
- [ ] Flux-constrained inlet patterns
- [ ] Serial-inlet junction maps
- [ ] Asymmetric partial-height pillars, interaction grouping
- [ ] Reference data (CFD/experiment): none in the repo yet, ask for it

## M5/M6: application
- [ ] GUI with a side-by-side solver comparison
- [ ] Packaging (PyInstaller) and smoke test
- [ ] Validation and performance report
