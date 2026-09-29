# Reduced-order crossflow simulator

Predicts material cross-sections in low-Reynolds-number microfluidic flow sculpting: serial and
parallel inlets, prescribed inlet arrays, and asymmetric partial-height pillars. The new solver
computes a 3D Stokes flow field for each interacting group of features and transports material
along continuous streamlines. Legacy is the frozen original heuristic solver. It is kept unchanged
and runs next to the new solver in its own isolated worker process.

Status: work in progress on `docs/implementation_plan.md`. M0 is complete: the original top-level
`python/` tree was removed after the snapshot in `legacy/python/` had been hash-verified.

## Install (Windows, Python 3.11+)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/dev.txt   # pinned; app.txt for runtime only
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

`pyamg` (AMG preconditioner) and `numba` (compiled streamline kernel) are accelerators. The code
falls back to block LU and to the NumPy tracer without them.

The legacy solver can run in an environment of its own: install `requirements/legacy.txt` there and
set `CROSSFLOW_LEGACY_PYTHON` to its interpreter.

## Use

```powershell
crossflow run examples/right_triangle_bottom.json --figure sections.png
crossflow run examples/right_triangle_bottom.json --backend legacy
crossflow run examples/right_triangle_bottom.json --backend cfd --cfd-cells 24   # CFD reference
crossflow compare examples/parallel_unequal.json --export report/      # paired legacy/new run
crossflow compare examples/right_triangle_bottom.json --cfd            # legacy, new, and accuracy vs CFD
crossflow compare examples/parallel_unequal.json --benchmark 3         # repeatable benchmark protocol
crossflow sweep examples/serial_three_inlets.json --set inlets.1.flow_rate_m3_s --values 1e-11,3e-11 --csv sweep.csv
crossflow validate-references reference_data/analytical
crossflow verify-snapshot                        # legacy hashes and migration state
crossflow legacy-tests                           # the six original tests against the frozen snapshot
```

Project files are versioned JSON with SI numbers under unit-suffixed keys (`width_m`,
`flow_rate_m3_s`). Strings such as `"300 um"` or `"2 uL/min"` are accepted on input. The
`examples/` directory holds the representative cases. `examples/index.json` marks which ones legacy
supports (the shared-capability comparison presets).

## Repository layout

```text
legacy/            frozen snapshot of the original python/ tree (manifest.json holds hashes); never edited
src/crossflow_simulator/
  config.py        typed project contracts, units, validation      results.py   result contract, NPZ I/O
  physics.py       duct profiles, groups                            geometry.py  polygons, rasterization
  inlets.py        flux-constrained inlet patterns                  materials.py stable colours
  solvers/         stokes.py (3D MAC Stokes), section.py (variable viscosity), pressure.py (lubrication),
                   network.py (pressure-driven inlets)
  models/          domain.py (feature groups), stokes_map.py (flows, caching, viscosity coupling), validity.py
  transport.py     continuous streamline integration
  maps.py          chained donor maps, composition, conservative remap, sub-cell fractions
  simulator.py     simulate(config, progress, cancel) — the application service
  backends/        isolated new and legacy workers, protocol, legacy adapter
  comparison.py    paired legacy/new runs, metrics, benchmark protocol, export
  calibration.py   reference-data contract, holdout, envelopes
  cli.py           command line
tests/             new-solver, comparison, and legacy-isolation tests
benchmarks/        legacy baseline capture
reference_data/    analytical fixtures and the legacy behaviour baseline
examples/          project files and presets
tools/             provenance and generator scripts
docs/              implementation plan and review record
```

## Tests and benchmarks

```powershell
python -m pytest -q                       # full suite
python -m pytest -q -m "not legacy"       # without the legacy worker
python benchmarks/capture_legacy_baseline.py   # re-record legacy behaviour fixtures
```
