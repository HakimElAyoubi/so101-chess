"""One-line progress bar written to chess_sim/data/progress.txt (viewed with progress_view.command).

Format: '[####------] 40%  12/30  elapsed 1m05s  ETA 1m40s | <state>'
"""
from __future__ import annotations

import os
import time

from .config import PROGRESS_TXT


def _fmt(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


class ProgressBar:
    """Writes the bar at most once per `period` seconds (and on done/fail)."""

    def __init__(self, total: int, path: str = PROGRESS_TXT, period: float = 1.0, width: int = 10):
        self.total = max(1, int(total))
        self.path = path
        self.period = period
        self.width = width
        self.t0 = time.time()
        self._last = 0.0
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def line(self, count: int, state: str = "") -> str:
        frac = min(1.0, count / self.total)
        filled = int(round(frac * self.width))
        elapsed = time.time() - self.t0
        eta = (elapsed / frac - elapsed) if frac > 0 else float("nan")
        eta_s = _fmt(eta) if frac > 0 else "--"
        return (f"[{'#' * filled}{'-' * (self.width - filled)}] {int(frac * 100):3d}%  "
                f"{count}/{self.total}  elapsed {_fmt(elapsed)}  ETA {eta_s} | {state}")

    def update(self, count: int, state: str = "", force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last < self.period:
            return
        self._last = now
        try:
            with open(self.path, "w") as f:
                f.write(self.line(count, state) + "\n")
        except OSError:
            pass

    def done(self, state: str = "DONE") -> None:
        self.update(self.total, state, force=True)

    def fail(self, state: str = "FAILED") -> None:
        self.update(0, state, force=True)
