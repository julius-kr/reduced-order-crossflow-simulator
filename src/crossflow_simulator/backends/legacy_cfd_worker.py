"""Isolated worker that runs the vendored 3D-hydrodynamic-flow-sculpting tool ("legacy CFD").

This file is executed as a *script* in its own interpreter (``python -I``), never imported
with the upstream code into the new package's process. It imports only the standard library,
NumPy, and the upstream ``computation.py`` (which itself needs NumPy, Pillow, and SciPy),
after checking that file against its pinned SHA-256.

The upstream tool applies precomputed index maps (from COMSOL streamline exports) to a
250 x 250 section. The adapter in ``legacy_cfd.py`` chooses the library entries; this worker
only loads the libraries, registers the chosen maps, and calls the unchanged
``flowsculpting`` operators. The one addition is exact symmetry: a top-mounted or mirrored
pillar uses its library map conjugated by the z or y reflection (the square channel is
symmetric under both), and the reflected map is still applied by the upstream operator.

Library files are pickled NumPy dictionaries. :func:`load_library_file` never calls
``pickle.load``: its unpickler resolves only NumPy array reconstruction and plain containers,
so a manipulated library cannot execute code. The parent process imports this module for that
loader alone (it has no import-time side effects).

Protocol: one JSON object per line on stdin/stdout (see ``backends/protocol`` in the new
package; kept independent here on purpose).
"""
from __future__ import annotations

import argparse
import codecs
import gzip
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import threading
import time
import traceback
from pathlib import Path

PROTOCOL = "crossflow-worker/1"
GRID = 250  # computation.flowsculpting: nx = nz = 250


# --------------------------------------------------------------------------- safe library loading
def _allowed_globals() -> dict:
    import numpy as np

    try:
        from numpy._core import multiarray
    except ImportError:  # NumPy < 2
        from numpy.core import multiarray  # type: ignore[no-redef]
    allowed = {("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype, ("_codecs", "encode"): codecs.encode}
    for module in ("numpy.core.multiarray", "numpy._core.multiarray"):
        allowed[(module, "_reconstruct")] = multiarray._reconstruct
        allowed[(module, "scalar")] = multiarray.scalar
    return allowed


class _ArrayUnpickler(pickle.Unpickler):
    """Resolves NumPy array reconstruction only; any other global is refused."""

    def __init__(self, file) -> None:
        super().__init__(file)
        self._allowed = _allowed_globals()

    def find_class(self, module: str, name: str):
        try:
            return self._allowed[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(
                f"library file references {module}.{name}; only NumPy arrays and plain containers are accepted"
            ) from None


def load_library_file(path: str | Path) -> dict:
    """Load a pickled dictionary saved with ``numpy.save`` (``.npy`` or gzip ``.npy.gz``) without running code."""
    import numpy as np

    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as file:
        version = np.lib.format.read_magic(file)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(file)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(file)
        else:
            raise ValueError(f"{path.name}: unsupported .npy format version {version}")
        if dtype != np.dtype(object) or shape != ():
            raise ValueError(f"{path.name}: expected a pickled dictionary, found an array {shape} of {dtype}")
        value = _ArrayUnpickler(file).load()
    if isinstance(value, np.ndarray):
        value = value.item()
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: expected a dictionary, found {type(value).__name__}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- worker plumbing
def _peak_memory() -> dict:
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return {"peak_bytes": None, "method": "unavailable (GetProcessMemoryInfo failed)"}
            return {"peak_bytes": int(counters.PeakWorkingSetSize), "current_bytes": int(counters.WorkingSetSize),
                    "method": "process-lifetime peak working set (GetProcessMemoryInfo)"}
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        scale = 1 if sys.platform == "darwin" else 1024
        return {"peak_bytes": int(peak * scale), "method": "process-lifetime max RSS (getrusage)"}
    except Exception:  # noqa: BLE001
        return {"peak_bytes": None, "method": "unavailable"}


class Channel:
    def __init__(self) -> None:
        self._out = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def send(self, message: dict) -> None:
        with self._lock:
            self._out.write(json.dumps(message) + "\n")
            self._out.flush()


def _load_upstream(upstream: Path, expected_sha256: str, revision: str) -> tuple[object, dict]:
    sys.dont_write_bytecode = True
    source = upstream / "computation.py"
    if not source.exists():
        raise SystemExit(f"Upstream computation.py not found in {upstream}")
    digest = sha256_file(source)
    if digest != expected_sha256:
        raise SystemExit(f"Upstream computation.py differs from the pinned revision (sha256 {digest[:12]}, "
                         f"expected {expected_sha256[:12]})")
    spec = importlib.util.spec_from_file_location("computation", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["computation"] = module
    spec.loader.exec_module(module)
    identity = {"name": "3D-hydrodynamic-flow-sculpting", "source_revision": revision,
                "module": str(source), "computation_sha256": digest, "python": sys.version.split()[0],
                "executable": sys.executable}
    for name, label in (("numpy", "numpy"), ("scipy", "scipy"), ("PIL", "pillow")):
        try:
            identity[label] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            identity[label] = None
    return module, identity


def _check_isolation() -> list[str]:
    """The new package must never be loaded next to the upstream code."""
    return [name for name in sys.modules if name == "crossflow_simulator" or name.startswith("crossflow_simulator.")]


_LIBRARIES: dict[str, dict] = {}


def _library(path: Path) -> tuple[dict, dict, bool]:
    """The library at ``path`` (kept in memory between jobs), its provenance, and whether it was loaded now."""
    stat = path.stat()
    token = f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
    cached = _LIBRARIES.get(str(path.resolve()))
    if cached is not None and cached["token"] == token:
        return cached["data"], cached["info"], False
    started = time.perf_counter()
    data = load_library_file(path)
    info = {"file": str(path), "bytes": stat.st_size, "sha256": sha256_file(path), "entries": len(data),
            "load_seconds": time.perf_counter() - started}
    _LIBRARIES[str(path.resolve())] = {"token": token, "data": data, "info": info}
    return data, info, True


def _index_map(entry, key: str):
    import numpy as np

    index_map = entry["map"] if isinstance(entry, dict) else entry
    index_map = np.asarray(index_map)
    if index_map.shape != (GRID * GRID,) or not np.issubdtype(index_map.dtype, np.integer):
        raise ValueError(f"library entry {key!r} is not a {GRID * GRID}-cell index map")
    if index_map.min() < 0 or index_map.max() >= GRID * GRID:
        raise ValueError(f"library entry {key!r} points outside the {GRID} x {GRID} section")
    return index_map.astype(np.int64)


def reflected(index_map, flip_z: bool, mirror_y: bool):
    """``R o map o R`` for the reflection ``R`` (a map of the reflected geometry)."""
    import numpy as np

    cells = np.arange(GRID * GRID)
    z, y = np.divmod(cells, GRID)
    reflect = (((GRID - 1 - z) if flip_z else z) * GRID + ((GRID - 1 - y) if mirror_y else y))
    return reflect[index_map[reflect]]


def _codes(outlet, channel_codes):
    import numpy as np

    flat = outlet.reshape(outlet.shape[0], -1)
    occupied = flat.max(axis=0) > 0
    codes = np.asarray(channel_codes, dtype=np.int32)[np.argmax(flat, axis=0)]
    return np.where(occupied, codes, -1).reshape(GRID, GRID).astype(np.int32)


def _execute(computation, plan: dict, cancel: threading.Event, send, job_id) -> dict:
    import numpy as np

    timings: dict[str, float] = {}
    started = time.perf_counter()
    fs = computation.flowsculpting()
    if (fs.nx, fs.nz) != (GRID, GRID):
        raise ValueError(f"upstream grid is {fs.nx} x {fs.nz}, adapter expects {GRID} x {GRID}")
    initial = plan["initial"]
    channel_codes = initial["channel_codes"]
    codes = np.load(initial["file"], allow_pickle=False).reshape(-1)
    if codes.shape != (GRID * GRID,):
        raise ValueError(f"inlet profile has {codes.size} nodes, expected {GRID * GRID}")
    fs.inlet = np.stack([(codes == c).astype(np.float32) for c in channel_codes])  # one channel per stream
    fs.inlet_color = fs.generate_color_image(fs.inlet)
    timings["initial_profile"] = time.perf_counter() - started

    operations = plan["operations"]
    library_info, loaded = None, None
    fs.map_lib = {}
    if operations:
        send({"type": "progress", "job_id": job_id, "stage": "library", "fraction": 0.05,
              "message": "loading the pillar map library"})
        t0 = time.perf_counter()
        library, library_info, loaded = _library(Path(plan["library"]))
        timings["library_load" if loaded else "library_lookup"] = time.perf_counter() - t0
        for op in operations:
            if op["key"] not in library:
                raise KeyError(f"library has no entry {op['key']!r}")
            index_map = _index_map(library[op["key"]], op["key"])
            if op["flip_z"] or op["mirror_y"]:
                index_map = reflected(index_map, op["flip_z"], op["mirror_y"])
            fs.map_lib[op["map_key"]] = index_map

    t0 = time.perf_counter()
    states = []  # codes after the first k operations, each computed by the upstream operator chain
    log = []
    for k in range(len(operations) + 1):
        if cancel.is_set():
            raise KeyboardInterrupt("cancelled")
        if k:
            send({"type": "progress", "job_id": job_id, "stage": "maps", "fraction": 0.1 + 0.9 * k / len(operations),
                  "message": f"library map {k}/{len(operations)}: {operations[k - 1]['pillar']}"})
        fs.compute_and_generate_new_flow_profile([("pillar", op["map_key"]) for op in operations[:k]])
        states.append(_codes(fs.outlet, channel_codes))
        if k:
            log.append({"op": "pillar", "pillar": operations[k - 1]["pillar"], "key": operations[k - 1]["key"],
                        "flip_z": operations[k - 1]["flip_z"], "mirror_y": operations[k - 1]["mirror_y"],
                        "changed_fraction": float(np.mean(states[k] != states[k - 1]))})
    timings["apply_maps"] = time.perf_counter() - t0
    sections = np.stack([states[station["operations"]] for station in plan["stations"]])
    return {"codes": sections, "timings": timings, "log": log, "library": library_info,
            "library_cache": "not needed" if loaded is None else ("loaded" if loaded else "in memory")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--expect-sha256", required=True)
    parser.add_argument("--revision", default="")
    args = parser.parse_args(argv)
    channel = Channel()
    sys.stdout = sys.stderr  # stray prints must never corrupt the protocol stream
    try:
        computation, identity = _load_upstream(args.upstream, args.expect_sha256, args.revision)
        import numpy as np
    except SystemExit as exc:
        channel.send({"type": "fatal", "protocol": PROTOCOL, "message": str(exc)})
        return 2
    started = time.perf_counter()
    channel.send({"type": "hello", "protocol": PROTOCOL, "backend": "legacy_cfd", "identity": identity,
                  "pid": os.getpid(), "isolation_problems": _check_isolation()})
    jobs: list[dict] = []
    cancel_flags: dict = {}
    available = threading.Condition()
    stop = threading.Event()

    def reader() -> None:
        for line in sys.stdin:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            with available:
                if message.get("type") == "cancel":
                    cancel_flags.setdefault(message.get("job_id"), threading.Event()).set()
                elif message.get("type") == "shutdown":
                    stop.set()
                else:
                    jobs.append(message)
                available.notify_all()
        stop.set()
        with available:
            available.notify_all()

    threading.Thread(target=reader, daemon=True).start()
    while True:
        with available:
            while not jobs and not stop.is_set():
                available.wait()
            if not jobs and stop.is_set():
                break
            job = jobs.pop(0)
        job_id = job.get("job_id")
        cancel = cancel_flags.setdefault(job_id, threading.Event())
        if job.get("type") != "run":
            continue
        t0 = time.perf_counter()
        try:
            output = _execute(computation, job["plan"], cancel, channel.send, job_id)
            meta = {"identity": identity, "timings": output["timings"], "log": output["log"],
                    "library": output["library"], "library_cache": output["library_cache"],
                    "isolation_problems": _check_isolation()}
            path = Path(job["output"])
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, codes=output["codes"], meta=np.array(json.dumps(meta)))
            channel.send({"type": "result", "job_id": job_id, "status": "success", "output": str(path),
                          "solve_seconds": time.perf_counter() - t0, "timings": output["timings"],
                          "memory": _peak_memory(), "worker_uptime": time.perf_counter() - started})
        except KeyboardInterrupt:
            channel.send({"type": "result", "job_id": job_id, "status": "cancelled", "message": "Cancelled",
                          "solve_seconds": time.perf_counter() - t0})
        except Exception as exc:  # noqa: BLE001 - reported to the parent as a failed run
            channel.send({"type": "result", "job_id": job_id, "status": "failed",
                          "message": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                          "solve_seconds": time.perf_counter() - t0})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
