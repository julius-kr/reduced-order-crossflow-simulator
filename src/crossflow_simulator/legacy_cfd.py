"""Legacy CFD: the vendored 3D-hydrodynamic-flow-sculpting tool as a comparison backend.

The upstream tool (``3D-hydrodynamic-flow-sculpting/`` at the repository root, pinned below)
predicts sections by chaining precomputed index maps. Each map was built from COMSOL
streamlines of one geometry on a 250 x 250 section grid, so the tool is exact only for
geometries in its libraries, and it applies them as independent operators in streamwise order.
This module decides whether a project can be expressed with those libraries and which entries
it needs, or says why not; ``backends/legacy_cfd_worker.py`` runs the upstream code.

What the libraries cover (``library_*_dict.npy``, pinned parameter names from the upstream code):

* pillars: triangles in a square channel (``cw = ch = 1.5`` library units) with the base
  edge A-B across the channel at ``x = 4.0`` and the tip C at ``(xc, yc)``, height ``ph``;
* inlets: top ports with an operator width ``ow`` and position ``ip`` whose reference frame the
  upstream code does not define, so ports are not translated;
* outlets: shaped outlet channels, which projects do not describe.

Adapter conventions (reported in the effective settings):

* lengths are scaled by ``W / 1.5`` (Stokes streamline maps are scale-free); a pillar matches
  an entry when every parameter agrees within ``MATCH_TOLERANCE`` library units;
* the library pillars stand against the channel's z = ch wall, i.e. they are top-mounted in this
  project's frame (measured from the published COMSOL streamlines, see ``LIBRARY_MOUNT``); a
  bottom-mounted pillar uses the z-reflected map, and a pillar whose mirror image (y -> W - y) is in
  the library uses the y-reflected map, both exact symmetries of the channel;
* the inlet pattern is the new solver's flux-constrained partition evaluated at the upstream nodes
  and set as the upstream inlet profile (see :func:`initial_profile`);
* sections are reported at the new solver's stations (inlet plane and after each feature group),
  on the upstream node grid (250 x 250, walls included).
"""
from __future__ import annotations

import hashlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .config import PillarSpec, ProjectConfig

UPSTREAM_FOLDER = "3D-hydrodynamic-flow-sculpting"
UPSTREAM = {
    "name": "3D-hydrodynamic-flow-sculpting",
    "url": "https://github.com/yiyingzou/3D-hydrodynamic-flow-sculpting",
    "revision": "b6912d31051d904c3cc075a155a0bc41ff711a5d",
    "license": "GPL-3.0 (3D-hydrodynamic-flow-sculpting/LICENSE)",
    "libraries_url": "https://drive.google.com/drive/folders/16BtjLV74uk7A9ZAgGJoo5kOc5aUSpx3p",
}
# What a library map really costs: the COMSOL simulation behind it. The published pillar streamline
# files record their export time ("% Date", minute resolution). All 900 were exported as one batch,
# from 2026-05-21 02:59 to 2026-05-22 12:42, with no gap over 30 min: 899 intervals of 2.25 min,
# each one COMSOL solve plus its streamline export on the library authors' machine. Applying a
# stored map takes well under a second; that time excludes the simulation.
COMSOL_PILLAR = {"seconds_per_map": 135.0, "maps": 900, "first_export": "2026-05-21 02:59",
                 "last_export": "2026-05-22 12:42",
                 "basis": "mean interval between the export stamps of the published pillar streamline files"}


def comsol_seconds(maps: int) -> float:
    """Estimated COMSOL time behind ``maps`` pillar library maps."""
    return maps * COMSOL_PILLAR["seconds_per_map"]


def comsol_export_intervals(folder: Path) -> dict:
    """Export stamps of COMSOL streamline files (``% Date`` header lines): count, span, mean interval."""
    from datetime import datetime

    stamps = []
    for path in sorted(Path(folder).glob("*.txt")):
        with open(path, encoding="utf-8", errors="replace") as handle:
            for _ in range(7):
                line = handle.readline()
                if line.startswith("% Date:"):
                    stamps.append(datetime.strptime(line.split(":", 1)[1].strip(), "%b %d %Y, %H:%M"))
                    break
    stamps.sort()
    gaps = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]
    return {"files": len(stamps), "first": stamps[0] if stamps else None, "last": stamps[-1] if stamps else None,
            "mean_seconds": sum(gaps) / len(gaps) if gaps else None, "max_gap_seconds": max(gaps, default=None)}


# SHA-256 of every file at the pinned revision (byte-exact git blobs).
UPSTREAM_FILES = {
    ".gitignore": "fe07aeae2cfd1ff2ce4c0c5911dfbb550a3f1065e213175039abadfc3a5a15bb",
    "LICENSE": "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986",
    "README.md": "77c8ea7ddb588fddb597edb0aa38886378cae71f91ae597218fb0f86ae45965e",
    "computation.py": "9e8c19b18762e488b84a53a43e71cc688a932f68ca295a3d6da4a50e9813fdc5",
    "computation_3D.py": "96eccc0016ef128e01e68b4237bcb98397070436c46af324c560688014ac05ce",
    "generated_library_inlet/.gitkeep": "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
    "generated_library_outlet/.gitkeep": "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
    "generated_library_pillar/.gitkeep": "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
    "library_generation_inlet.py": "cb0052c1fd51bf68cd54f5a2e00119626ebaee7b1865096a235e6a79d85dcc43",
    "library_generation_outlet.py": "85151014ae5169d247f656c0ee2566fc0205524001a23ba6578bd1befe55ab92",
    "library_generation_pillar.py": "2a706bf5ffeae7029ecdb3fab88224818184149ef177fca520ad60fff247378d",
    "streamline_files_inlet/.gitkeep": "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
    "streamline_files_outlet/.gitkeep": "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
    "streamline_files_pillar/.gitkeep": "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
    "visualization_v2.py": "4a6876dc41784543078fd90dc1e9a4ca792c0d361d3f39fe54a54efad74e04de",
    "visualization_v3.py": "11ac2dfdd1e7ae7c536482ed86d062fc493384023ebce8ca5f14c465f1ae8974",
}
GRID = 250  # upstream section grid (nodes, walls included)
LIBRARY_SIZE = 1.5  # library channel width and height (cw = ch)
BASE_X = 4.0  # library pillars: base edge A-B on x = 4.0 (upstream visualization_v2.draw_pillar_geometry)
# The published streamlines put every library pillar against z = ch: for ph = 0.25 ch the backward
# displacement behind the base edge is about 5x larger next to z = ch than next to z = 0, and on a pillar of
# 0.75 ch the top-mount reading agrees with the CFD reference (IoU 0.90) where the other gives 0.71.
LIBRARY_MOUNT = "top"
PILLAR_KEYS = ("cw", "ch", "ph", "ya", "yb", "xc", "yc")
MATCH_TOLERANCE = 0.01  # library units: 0.67% of the channel width
LIBRARY_ENV = "CROSSFLOW_LEGACY_CFD_LIBRARIES"
ROOT_ENV = "CROSSFLOW_LEGACY_CFD_ROOT"


class LegacyCFDError(RuntimeError):
    pass


def upstream_root() -> Path:
    """The vendored upstream folder (``3D-hydrodynamic-flow-sculpting/``)."""
    from .backends.base import bundle_root

    override = os.environ.get(ROOT_ENV)
    root = Path(override) if override else bundle_root() / UPSTREAM_FOLDER
    if not (root / "computation.py").exists():
        raise LegacyCFDError(f"Upstream code not found in {root}; set {ROOT_ENV} to the {UPSTREAM_FOLDER} folder.")
    return root


def library_dirs() -> list[Path]:
    """Where the ``generated_library_*`` folders are looked up, in order.

    The upstream folder (as its README describes), unless overridden; the packaged application also
    looks next to its executables.
    """
    from .backends.base import frozen

    override = os.environ.get(LIBRARY_ENV)
    if override:
        return [Path(override)]
    dirs = [upstream_root()]
    if frozen():
        dirs.append(Path(sys.executable).parent / UPSTREAM_FOLDER)
    return dirs


def library_files(kind: str = "pillar") -> dict[str, Path] | None:
    """The data and index files of one library, or None when it is not installed."""
    for directory in library_dirs():
        folder = directory / f"generated_library_{kind}"
        index = folder / f"library_{kind}_dict.npy"
        for name in (f"library_{kind}.npy", f"library_{kind}.npy.gz"):
            if (folder / name).exists() and index.exists():
                return {"data": folder / name, "index": index}
    return None


def verify_upstream(root: Path | None = None) -> dict[str, Any]:
    """Compare the vendored files with the pinned hashes; list files that are not library data."""
    root = root or upstream_root()
    mismatched, missing = [], []
    for name, digest in UPSTREAM_FILES.items():
        path = root / name
        if not path.exists():
            missing.append(name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            mismatched.append(name)
    data_folders = ("generated_library_", "streamline_files_")
    extra = sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.relative_to(root).as_posix() not in UPSTREAM_FILES
        and not p.relative_to(root).parts[0].startswith(data_folders)
    )
    return {"ok": not (mismatched or missing or extra), "mismatched": mismatched, "missing": missing,
            "unexpected": extra, "files": len(UPSTREAM_FILES), "source_revision": UPSTREAM["revision"]}


# --------------------------------------------------------------------------- library index
@lru_cache(maxsize=4)
def _load_index(path: str, token: str) -> dict:
    from .backends.legacy_cfd_worker import load_library_file

    data = load_library_file(path)
    tree = data.get("tree")
    if not isinstance(tree, dict):
        raise ValueError(f"{Path(path).name} has no parameter tree")
    entries: list[tuple[float, ...]] = []

    def walk(node: dict, prefix: tuple[float, ...]) -> None:
        if not node:
            entries.append(prefix)
        for value, child in node.items():
            walk(child, prefix + (float(value),))

    walk(tree, ())
    entries = [e for e in entries if len(e) == len(PILLAR_KEYS)]
    return {"entries": np.array(entries, dtype=float).reshape(-1, len(PILLAR_KEYS)),
            "values": {k: [float(v) for v in data.get(k, [])] for k in PILLAR_KEYS}}


def pillar_index() -> dict | None:
    """Parameter combinations of the installed pillar library (None when it is not installed)."""
    files = library_files("pillar")
    if files is None:
        return None
    stat = files["index"].stat()
    return _load_index(str(files["index"]), f"{stat.st_mtime_ns}|{stat.st_size}")


def library_key(values) -> str:
    """The upstream key of a pillar entry (``visualization_v2.get_current_pillar_key``)."""
    return "_".join(f"{name}{float(value):.4f}" for name, value in zip(PILLAR_KEYS, values))


# --------------------------------------------------------------------------- pillar translation
def pillar_parameters(pillar: PillarSpec, width: float, height: float) -> tuple[np.ndarray | None, str | None]:
    """The pillar in library units (``cw, ch, ph, ya, yb, xc, yc``), or the reason it has no library form."""
    footprint = pillar.footprint()
    if len(footprint) != 3:
        return None, f"a {len(footprint)}-vertex footprint (library pillars are triangles)"
    scale = width / LIBRARY_SIZE
    across = [(i, (i + 1) % 3) for i in range(3)
              if abs(footprint[i, 0] - footprint[(i + 1) % 3, 0]) <= MATCH_TOLERANCE * scale]
    if len(across) != 1:
        return None, "no edge across the channel (library pillars have their base A-B perpendicular to the flow)"
    i, j = across[0]
    tip = footprint[3 - i - j]
    a, b = sorted((footprint[i], footprint[j]), key=lambda p: p[1])
    base_x = 0.5 * (a[0] + b[0])
    return np.array([width / scale, height / scale, pillar.height / scale, a[1] / scale, b[1] / scale,
                     BASE_X + (tip[0] - base_x) / scale, tip[1] / scale]), None


def _mirrored(values: np.ndarray) -> np.ndarray:
    """Parameters of the geometry reflected about the channel's y midplane."""
    cw, ch, ph, ya, yb, xc, yc = values
    return np.array([cw, ch, ph, cw - yb, cw - ya, xc, cw - yc])


def match_pillar(values: np.ndarray, index: dict) -> dict:
    """The library entry for these parameters (directly or mirrored), or the nearest one."""
    entries = index["entries"]
    best = None
    for mirror_y, target in ((False, values), (True, _mirrored(values))):
        if len(entries) == 0:
            break
        deviation = np.max(np.abs(entries - target), axis=1)
        k = int(np.argmin(deviation))
        if best is None or deviation[k] < best["deviation"]:
            best = {"entry": entries[k], "mirror_y": mirror_y, "deviation": float(deviation[k])}
    if best is None:
        return {"found": False, "nearest": None}
    best["found"] = best["deviation"] <= MATCH_TOLERANCE
    best["key"] = library_key(best["entry"])
    return best


def _describe(values: np.ndarray, scale: float, unit: float = 1e-6) -> str:
    cw, ch, ph, ya, yb, xc, yc = (float(v) * scale / unit for v in values)
    base = BASE_X * scale / unit
    return (f"height {ph:.4g} um, base y {ya:.4g}-{yb:.4g} um, tip {xc - base:+.4g} um downstream at y "
            f"{yc:.4g} um")


def _pillar_operations(cfg: ProjectConfig, pillars: list[PillarSpec], index: dict | None,
                       unsupported: list[str], conventions: list[str]) -> list[dict]:
    W, H = cfg.channel.width, cfg.channel.height
    scale = W / LIBRARY_SIZE
    operations = []
    for pillar in sorted(pillars, key=lambda p: float(p.footprint()[:, 0].min())):
        if pillar.group is not None and any(p.group == pillar.group and p.id != pillar.id
                                            for p in cfg.active_pillars()):
            unsupported.append(f"pillar {pillar.id!r}: composite pillar groups (library pillars are single "
                               "triangles)")
            continue
        values, reason = pillar_parameters(pillar, W, H)
        if values is None:
            unsupported.append(f"pillar {pillar.id!r}: {reason}")
            continue
        if index is None:
            continue
        match = match_pillar(values, index)
        if not match["found"]:
            nearest = ""
            if match.get("entry") is not None:
                entry = _mirrored(match["entry"]) if match["mirror_y"] else match["entry"]
                nearest = (f"; nearest library pillar: {_describe(entry, scale)} (differs by up to "
                           f"{match['deviation'] * scale * 1e6:.3g} um)")
            unsupported.append(f"pillar {pillar.id!r}: no library entry for {_describe(values, scale)}{nearest}")
            continue
        flip_z = pillar.mount != LIBRARY_MOUNT
        suffix = ("|flip_z" if flip_z else "") + ("|mirror_y" if match["mirror_y"] else "")
        operations.append({"op": "pillar", "pillar": pillar.id, "key": match["key"], "map_key": match["key"] + suffix,
                           "flip_z": flip_z, "mirror_y": bool(match["mirror_y"]), "deviation": match["deviation"]})
        notes = [n for n, on in (("z-reflected map (top mount)", flip_z),
                                 ("y-reflected map (mirror image in the library)", match["mirror_y"])) if on]
        conventions.append(f"pillar {pillar.id!r}: library entry {match['key']}"
                           + (f", {', '.join(notes)}" if notes else ""))
    return operations


# --------------------------------------------------------------------------- inlet profile and plan
def initial_profile(cfg: ProjectConfig) -> dict[str, Any]:
    """The inlet profile on the upstream node grid (the new solver's partition evaluated at every node).

    Not the upstream ``create_initial_flow_profile``: it takes side-by-side layers only and rounds each
    boundary down to a whole node of a 250-wide index (on 1:2:3 layers a 0.8 pp flux bias before any map).
    """
    from .inlets import partition_pattern

    partition = partition_pattern(cfg)
    y = np.linspace(0.0, cfg.channel.width, GRID)
    z = np.linspace(0.0, cfg.channel.height, GRID)
    Y, Z = np.meshgrid(y, z)
    profile = partition.evaluate(Y, Z).astype(np.int32)
    return {"kind": "profile", "codes": profile, "channel_codes": sorted(int(c) for c in np.unique(profile))}


def build_plan(cfg: ProjectConfig) -> dict[str, Any]:
    """Express a project with the upstream libraries, or explain why it cannot be."""
    from .cfd import stations
    from .models.domain import plan_groups

    W, H = cfg.channel.width, cfg.channel.height
    unsupported: list[str] = []
    ignored: list[str] = []
    conventions = [
        f"lengths scaled by W / {LIBRARY_SIZE} = {W / LIBRARY_SIZE * 1e6:.4g} um per library unit; parameters "
        f"match within {MATCH_TOLERANCE} library units",
        f"sections on the upstream {GRID} x {GRID} node grid (walls included), at the new solver's stations",
        "pillars applied as independent library maps in streamwise order (upstream model: no interaction "
        "between nearby pillars)",
    ]
    try:
        verified = verify_upstream()
        if verified["mismatched"] or verified["missing"]:
            unsupported.append("vendored upstream code differs from the pinned revision "
                               f"({', '.join(verified['mismatched'] + verified['missing'])})")
    except LegacyCFDError as exc:
        unsupported.append(str(exc))
    active = cfg.active_inlets()
    if cfg.operating.drive != "flow_rate":
        unsupported.append("pressure-driven operation (the map library has no hydraulic network)")
    if len({cfg.material(i.material).viscosity for i in active}) > 1:
        unsupported.append("viscosity contrast (the library maps were computed for one fluid)")
    if len({cfg.material(i.material).density for i in active}) > 1:
        ignored.append("density differences (no effect in creeping flow)")
    if cfg.port_inlets():
        unsupported.append("inlet ports (the upstream inlet library's operator width 'ow' and port position 'ip' "
                           "have no documented reference frame to map a port onto)")
    pillars = cfg.active_pillars()
    index = None
    if pillars:
        if abs(W - H) > MATCH_TOLERANCE / LIBRARY_SIZE * W:
            unsupported.append(f"pillar maps exist only for a square channel (library {LIBRARY_SIZE} x "
                               f"{LIBRARY_SIZE}); this channel is {W * 1e6:.4g} x {H * 1e6:.4g} um")
        else:
            try:
                index = pillar_index()
            except Exception as exc:  # noqa: BLE001 - a broken library is reported, not raised
                unsupported.append(f"pillar library unreadable: {exc}")
            else:
                if index is None:
                    folder = f"{UPSTREAM_FOLDER}/generated_library_pillar"
                    unsupported.append(f"pillar map library not installed: put library_pillar.npy (or .npy.gz) and "
                                       f"library_pillar_dict.npy into {folder} (download: {UPSTREAM_FOLDER}/README.md)")
            conventions.append(f"library pillars are {LIBRARY_MOUNT}-mounted (against z = ch in the published COMSOL "
                               "streamlines)")
    operations: list[dict] = []
    station_plan: list[dict] = []
    if not cfg.port_inlets():
        lookup = index if not unsupported else None  # entries are only matched for an otherwise supported case
        station_plan.append({"name": "inlet plane", "x": 0.0, "operations": 0})
        groups = plan_groups(cfg)
        for group, (name, x) in zip(groups, stations(cfg)[1:]):
            operations += _pillar_operations(cfg, group.pillars, lookup, unsupported, conventions)
            station_plan.append({"name": name, "x": x, "operations": len(operations)})
        if not groups:
            name, x = stations(cfg)[-1]
            station_plan.append({"name": name, "x": x, "operations": 0})
    initial = None
    if not unsupported:
        try:
            initial = initial_profile(cfg)
        except ValueError as exc:
            unsupported.append(f"inlet pattern: {exc}")
        else:
            conventions.append("inlet pattern: new solver's flux-constrained partition evaluated at the upstream "
                               "nodes and set as the upstream inlet profile (its divider helper rounds boundaries "
                               "down to whole nodes)")
    files = library_files("pillar") if operations else None
    return {
        "operations": operations,
        "stations": station_plan,
        "initial": initial,
        "library": str(files["data"]) if files else None,
        "unsupported": unsupported,
        "ignored": ignored,
        "conventions": conventions,
    }


__all__ = ["UPSTREAM", "UPSTREAM_FOLDER", "build_plan", "library_files", "match_pillar", "pillar_index",
           "pillar_parameters", "upstream_root", "verify_upstream"]
