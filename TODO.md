# TODO

Working list for the rewrite. The plan and its acceptance gates are in `docs/implementation_plan.md`.

## M0: layout, legacy isolation, baseline
- [x] Freeze `python/` as `legacy/python/` with a hash manifest
- [x] Scaffold `src/`: typed project config, units, geometry
- [x] Legacy adapter and isolated worker (never import the snapshot in-process)
- [x] Representative example projects (work package A)
- [x] Record the legacy behaviour baseline
- [x] Remove `python/` once nothing references it

## M1: pressure, profiles, caching
- [x] Single-series duct profile instead of the legacy double loop
- [x] Sparse lubrication pressure on physical spacings
- [x] Hydraulic network for pressure-driven operation
- [x] Bounded caches with explicit keys
- [x] Performance report

## M2: transport and maps
- [x] 3D Stokes per feature group
- [x] Continuous streamline tracing (no per-step resampling)
- [x] Conservative remap

## M3/M4: junctions and pillars
- [x] Flux-constrained inlet patterns
- [x] Serial-inlet junction maps
- [x] Asymmetric partial-height pillars, interaction grouping
- [ ] Reference data (CFD/experiment): none in the repo yet, ask for it

## M5/M6: application
- [ ] GUI with a side-by-side solver comparison
- [ ] Packaging (PyInstaller) and smoke test
- [ ] Validation and performance report

## Follow-ups
- [x] Pin the environments (app, dev, legacy)
- [x] Reuse donor maps when only the inlet pattern changes
- [x] Pathlines for a 3D view
- [x] CFD reference to measure accuracy against (no external data available)
- [x] Legacy CFD (3D-hydrodynamic-flow-sculpting map library) as a fourth comparison
- [ ] Grey cells: fill small gaps without valid trajectories
- [ ] 3D flow view in the GUI
