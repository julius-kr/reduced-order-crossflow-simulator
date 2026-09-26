"""Wall-clock timings per named stage (reported with every result)."""
from __future__ import annotations

import time
from typing import Any


class Stopwatch:
    """Accumulates wall-clock timings per named stage."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    def time(self, stage: str) -> "_StageTimer":
        return _StageTimer(self, stage)


class _StageTimer:
    def __init__(self, owner: Stopwatch, stage: str):
        self.owner, self.stage = owner, stage

    def __enter__(self) -> "_StageTimer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        elapsed = time.perf_counter() - self.start
        self.owner.timings[self.stage] = self.owner.timings.get(self.stage, 0.0) + elapsed
