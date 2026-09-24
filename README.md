# Reduced-order crossflow simulator

Predicts material cross-sections in low-Reynolds-number microfluidic flow sculpting: serial and
parallel inlets, prescribed inlet arrays, and asymmetric partial-height pillars.

Status: work in progress on `docs/implementation_plan.md`. The original heuristic solver is frozen
in `legacy/python/` (see [legacy/README.md](legacy/README.md)); the new implementation is being
built under `src/`. The original top-level `python/` tree stays until nothing depends on it.

## Development (Windows, Python 3.11+)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pytest -q
```

## Repository layout

```text
legacy/            frozen snapshot of the original python/ tree (manifest.json holds hashes); never edited
python/            original tree; removed after the migration (M0)
src/crossflow_simulator/
                   new implementation
tests/             tests of the new implementation
tools/             provenance and generator scripts
docs/              implementation plan and review record
```
