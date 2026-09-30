# Reduced-order crossflow simulator

Predicts material cross-sections in low-Reynolds-number microfluidic flow sculpting: serial and
parallel inlets, prescribed inlet arrays, and asymmetric partial-height pillars. The new solver
computes a 3D Stokes flow field for each interacting group of features and transports material
along continuous streamlines. Legacy is the frozen original heuristic solver. It is kept unchanged
and runs next to the new solver in its own isolated worker process. The CFD reference solves the
whole channel in one 3D Navier–Stokes computation, without the new solver's reduced-order
assumptions. It serves as the ground truth that both solvers are measured against. The legacy CFD is
the vendored [3D-hydrodynamic-flow-sculpting](https://github.com/yiyingzou/3D-hydrodynamic-flow-sculpting)
tool. It chains index maps precomputed from COMSOL streamlines, and it is measured against the same
CFD reference.

Status: implementation of `docs/implementation_plan.md` in progress. The solvers, the paired comparison,
and both CFD references work from the command line; the desktop application is next. Measured
performance is in [docs/validation_and_performance.md](docs/validation_and_performance.md).

## Install (Windows, Python 3.11+)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/dev.txt   # pinned; app.txt for runtime only
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

`pyamg` (AMG preconditioner) and `numba` (compiled streamline kernel) are accelerators. The code
falls back to block LU and to the NumPy tracer without them.

The legacy CFD needs its map libraries only for pillars. They are too large for the repository
(pillar library 214 MB) and are git-ignored. Unpack the published archives (link in
[3D-hydrodynamic-flow-sculpting/README.md](3D-hydrodynamic-flow-sculpting/README.md)) into
`3D-hydrodynamic-flow-sculpting/generated_library_*/`, and the raw COMSOL streamlines, if you have
them, into `streamline_files_*/`. `crossflow verify-snapshot` reports whether the pillar library
is found. It is loaded with a restricted unpickler that accepts NumPy arrays only.

## Use

```powershell
crossflow run examples/right_triangle_bottom.json --figure sections.png
crossflow run examples/right_triangle_bottom.json --backend legacy
crossflow run examples/right_triangle_bottom.json --backend cfd --cfd-cells 24   # CFD reference
crossflow compare examples/parallel_unequal.json --export report/      # paired legacy/new run
crossflow compare examples/right_triangle_bottom.json --cfd            # legacy, new, and accuracy vs CFD
crossflow compare examples/square_library_pillars.json --cfd --legacy-cfd   # all four solvers
crossflow run examples/parallel_unequal.json --backend legacy_cfd       # legacy CFD (map library) only
crossflow compare examples/parallel_unequal.json --benchmark 3         # repeatable benchmark protocol
crossflow sweep examples/serial_three_inlets.json --set inlets.1.flow_rate_m3_s --values 1e-11,3e-11 --csv sweep.csv
crossflow validate-references reference_data/analytical
crossflow verify-snapshot                        # legacy hashes and migration state
crossflow legacy-tests                           # the six original tests against the frozen snapshot
```

The CFD takes 50–90 s on its standard grid (about 11 minutes with a viscosity contrast). Results
are stored, so running a computed case again loads its CFD immediately.

Project files are versioned JSON with SI numbers under unit-suffixed keys (`width_m`,
`flow_rate_m3_s`). Strings such as `"300 um"` or `"2 uL/min"` are accepted on input. The
`examples/` directory holds the representative cases. `examples/index.json` marks which ones legacy
supports (the shared-capability comparison presets).

## Repository layout

```text
legacy/            frozen snapshot of the original python/ tree (manifest.json holds hashes); never edited
3D-hydrodynamic-flow-sculpting/
                   legacy CFD: byte-exact copy of the upstream tool (GPL-3.0; hashes pinned in legacy_cfd.py);
                   its map libraries go into generated_library_*/ (git-ignored)
src/crossflow_simulator/
  config.py        typed project contracts, units, validation      results.py   result contract, NPZ I/O
  physics.py       duct profiles, groups                            geometry.py  polygons, rasterization
  inlets.py        flux-constrained inlet patterns                  materials.py stable colours
  solvers/         stokes.py (3D MAC Stokes), section.py (variable viscosity), pressure.py (lubrication),
                   network.py (pressure-driven inlets), convection.py (inertia term for the CFD)
  models/          domain.py (feature groups), stokes_map.py (flows, caching, viscosity coupling), validity.py
  transport.py     continuous streamline integration (transport_kernel.py: optional compiled kernel)
  maps.py          chained donor maps, composition, conservative remap, sub-cell fractions
  pathlines.py     forward pathlines (display only)
  simulator.py     simulate(config, progress, cancel) — the application service
  cfd.py           run_cfd(config, ...) — full-channel Navier-Stokes CFD reference, mesh uncertainty, cache
  legacy_cfd.py    translation of projects to the legacy CFD's library entries (or the reasons why not)
  backends/        isolated new, legacy, CFD, and legacy CFD workers, protocol, legacy adapter
  comparison.py    legacy/new/CFD/legacy-CFD runs, metrics, accuracy against the CFD, benchmark protocol, export
  calibration.py   reference-data contract, holdout, envelopes
  cli.py           command line
tests/             new-solver, comparison, and legacy-isolation tests
benchmarks/        run_benchmarks.py, legacy baseline capture
reference_data/    analytical fixtures and the legacy behaviour baseline
examples/          project files and presets
cpp/               reserved for a compiled kernel if a future performance gate requires it
docs/              implementation plan, review record, performance report
```

## Tests and benchmarks

```powershell
python -m pytest -q                       # full suite
python -m pytest -q -m "not legacy"
python benchmarks/run_benchmarks.py --trials 3
python benchmarks/capture_legacy_baseline.py   # re-record legacy behaviour fixtures
```

## Known open items

* No external CFD or experimental reference data exist in this repository, so every prediction is
  labelled *not externally validated*. The in-app CFD reference removes the new solver's
  reduced-order assumptions, but it shares that solver's discretization, so it is a numerical
  reference, not independent validation. See the validation report for what is verified and how
  to add reference data.
* The legacy CFD covers only its library geometries: 900 triangular pillars in a square channel.
  None of the original examples (300 x 150 um channels) has library pillars, so the legacy CFD runs
  them only without pillars. Five square-channel examples, marked "(legacy CFD)" in the example list,
  use library pillars, and all four solvers run them. Inlet ports are not translated,
  because the upstream inlet library's parameters have no documented reference frame. Its maps do
  not conserve flux: each pillar moves about 0.75 pp of each wall-side stream into the middle
  stream (see the validation report).
