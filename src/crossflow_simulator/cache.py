"""Bounded, versioned caches for profiles, operators, flow solutions, and maps.

Invalidation is by key: every key includes the cache format version plus the
hash of every input the cached object depends on (geometry, grid, viscosity
field, boundary structure). Reuse is therefore only possible when those inputs
are identical; see ``simulator`` for what each key contains.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

CACHE_FORMAT = "crossflow-cache/1"


def estimate_nbytes(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (list, tuple)):
        return sum(estimate_nbytes(v) for v in value) + 64
    if isinstance(value, dict):
        return sum(estimate_nbytes(v) for v in value.values()) + 64
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, (int, np.integer)):
        return int(nbytes)
    return 1024


class MemoryCache:
    """Thread-safe least-recently-used cache with an explicit memory budget."""

    def __init__(self, max_bytes: int, name: str = "cache"):
        self.max_bytes = int(max_bytes)
        self.name = name
        self._items: OrderedDict[Any, tuple[Any, int]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: Any) -> Any:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return item[0]

    def put(self, key: Any, value: Any, nbytes: int | None = None) -> None:
        size = estimate_nbytes(value) if nbytes is None else int(nbytes)
        with self._lock:
            if key in self._items:
                self._bytes -= self._items.pop(key)[1]
            if size > self.max_bytes:
                return
            self._items[key] = (value, size)
            self._bytes += size
            while self._bytes > self.max_bytes and self._items:
                _, (_, evicted) = self._items.popitem(last=False)
                self._bytes -= evicted
                self.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "entries": len(self._items),
                "bytes": self._bytes,
                "max_bytes": self.max_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }


def default_cache_dir() -> Path:
    override = os.environ.get("CROSSFLOW_CACHE_DIR")
    if override:
        return Path(override)
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "crossflow_simulator" / "cache"
    return Path.home() / ".cache" / "crossflow_simulator"


class DiskCache:
    """NPZ store (no pickles) with atomic writes and size-bounded LRU eviction by access time."""

    def __init__(self, root: str | Path | None = None, max_bytes: int = 2 * 2**30):
        self.root = Path(root) if root is not None else default_cache_dir()
        self.max_bytes = int(max_bytes)
        self.enabled = os.environ.get("CROSSFLOW_DISABLE_DISK_CACHE", "") == ""

    def _path(self, namespace: str, key: str) -> Path:
        return self.root / CACHE_FORMAT.replace("/", "-") / namespace / f"{key}.npz"

    def load(self, namespace: str, key: str) -> dict[str, np.ndarray] | None:
        if not self.enabled:
            return None
        path = self._path(namespace, key)
        try:
            with np.load(path, allow_pickle=False) as data:
                arrays = {name: data[name] for name in data.files}
            os.utime(path)
            return arrays
        except (FileNotFoundError, OSError, ValueError):
            return None

    def save(self, namespace: str, key: str, arrays: dict[str, np.ndarray]) -> None:
        if not self.enabled:
            return
        path = self._path(namespace, key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp.npz")
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, path)
            self._evict()
        except OSError:
            pass

    def _evict(self) -> None:
        files = []
        for path in (self.root / CACHE_FORMAT.replace("/", "-")).rglob("*.npz"):
            try:
                stat = path.stat()
                files.append((stat.st_atime, stat.st_size, path))
            except OSError:
                continue
        total = sum(size for _, size, _ in files)
        for _, size, path in sorted(files):
            if total <= self.max_bytes:
                break
            try:
                path.unlink()
                total -= size
            except OSError:
                pass

    def clear(self) -> None:
        for path in (self.root / CACHE_FORMAT.replace("/", "-")).rglob("*.npz"):
            try:
                path.unlink()
            except OSError:
                pass


_memory = MemoryCache(max_bytes=int(os.environ.get("CROSSFLOW_MEMORY_CACHE_MB", "1024")) * 2**20, name="solver")
_disk = DiskCache()


def memory_cache() -> MemoryCache:
    return _memory


def disk_cache() -> DiskCache:
    return _disk
