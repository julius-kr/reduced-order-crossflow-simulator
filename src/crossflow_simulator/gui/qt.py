"""Qt/Matplotlib import shim (PySide6 only)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyside6")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402,F401
from PySide6.QtCore import QObject, Qt, QTimer, Signal  # noqa: E402,F401

import matplotlib  # noqa: E402

matplotlib.use("QtAgg", force=True)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT  # noqa: E402,F401
from matplotlib.figure import Figure  # noqa: E402,F401
