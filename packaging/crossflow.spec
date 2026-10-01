# PyInstaller specification for the Windows application.
#
# Build from the repository root (in the pinned environment of requirements/app.txt):
#     pyinstaller --noconfirm packaging/crossflow.spec
# Output: dist/crossflow/ with crossflow-gui.exe (desktop) and crossflow.exe (CLI and isolated workers).
#
# The frozen legacy snapshot is shipped as *data* (legacy/), never as importable modules of the new
# package; the legacy worker script is shipped separately (workers/) and executed with runpy. The
# vendored 3D-hydrodynamic-flow-sculpting code (legacy CFD) is shipped the same way. Of its published
# data only the pillar map library is bundled, compressed (87 MB), when it is installed in the source
# tree; the inlet and outlet libraries and the raw COMSOL streamlines (GBs) are not used at run time.
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).parent
SRC = ROOT / "src"


def pillar_library():
    """The installed pillar map library (compressed data and index) for the bundle, if present."""
    folder = ROOT / "3D-hydrodynamic-flow-sculpting" / "generated_library_pillar"
    files = [folder / "library_pillar.npy.gz", folder / "library_pillar_dict.npy"]
    if not all(f.exists() for f in files):
        return []
    return [(str(f), "3D-hydrodynamic-flow-sculpting/generated_library_pillar") for f in files]


def tree(source: Path, target: str, skip=()):
    items = []
    for path in source.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(skip):
            items.append((str(path), str(Path(target) / path.relative_to(source).parent)))
    return items


datas = (
    tree(ROOT / "legacy", "legacy")
    + tree(ROOT / "examples", "examples")
    + tree(ROOT / "3D-hydrodynamic-flow-sculpting", "3D-hydrodynamic-flow-sculpting", skip=(".npy", ".npy.gz", ".txt"))
    + pillar_library()
    + [(str(SRC / "crossflow_simulator" / "backends" / name), "workers")
       for name in ("legacy_worker.py", "legacy_cfd_worker.py")]
)
hiddenimports = (
    collect_submodules("crossflow_simulator")
    + collect_submodules("pyamg")
    + ["scipy.sparse.linalg", "scipy.optimize", "scipy.ndimage", "matplotlib.backends.backend_qtagg"]
)
excludes = ["tkinter", "pytest", "IPython"]

common = dict(pathex=[str(SRC)], datas=datas, hiddenimports=hiddenimports, excludes=excludes, noarchive=False)
gui_analysis = Analysis([str(ROOT / "packaging" / "crossflow_app.py")], **common)
cli_analysis = Analysis([str(ROOT / "packaging" / "crossflow_app.py")], **common)
MERGE((gui_analysis, "crossflow_app", "crossflow-gui"), (cli_analysis, "crossflow_app", "crossflow"))

gui_pyz = PYZ(gui_analysis.pure)
cli_pyz = PYZ(cli_analysis.pure)
gui_exe = EXE(gui_pyz, gui_analysis.scripts, [], exclude_binaries=True, name="crossflow-gui", console=False)
cli_exe = EXE(cli_pyz, cli_analysis.scripts, [], exclude_binaries=True, name="crossflow", console=True)
COLLECT(gui_exe, gui_analysis.binaries, gui_analysis.datas, cli_exe, cli_analysis.binaries, cli_analysis.datas,
        name="crossflow")
