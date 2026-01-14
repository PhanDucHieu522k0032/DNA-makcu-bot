import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class TelemetrySnapshot:
    window_s: float
    frames_ok: int
    frames_drop: int
    actions: int
    avg_loop_ms: float
    avg_capture_ms: float
    avg_match_ms: float


class Telemetry:
    def __init__(self, report_interval_s: float = 1.0, enabled: bool = True):
        self.enabled = enabled
        self.report_interval_s = max(0.2, float(report_interval_s))
        self._t0 = time.perf_counter()
        self._last_report_t = self._t0
        self._frames_ok = 0
        self._frames_drop = 0
        self._actions = 0
        self._sum_loop = 0.0
        self._sum_capture = 0.0
        self._sum_match = 0.0

    def on_frame(self, *, ok: bool, loop_s: float = 0.0, capture_s: float = 0.0, match_s: float = 0.0):
        if not self.enabled:
            return
        if ok:
            self._frames_ok += 1
            self._sum_loop += float(loop_s)
            self._sum_capture += float(capture_s)
            self._sum_match += float(match_s)
        else:
            self._frames_drop += 1

    def on_action(self):
        if not self.enabled:
            return
        self._actions += 1

    def maybe_report(self) -> Optional[TelemetrySnapshot]:
        if not self.enabled:
            return None
        now = time.perf_counter()
        if (now - self._last_report_t) < self.report_interval_s:
            return None

        window = now - self._last_report_t
        frames_ok = self._frames_ok
        frames_drop = self._frames_drop
        actions = self._actions

        def _avg_ms(total_s: float, n: int) -> float:
            if n <= 0:
                return 0.0
            return (total_s / n) * 1000.0

        snap = TelemetrySnapshot(
            window_s=window,
            frames_ok=frames_ok,
            frames_drop=frames_drop,
            actions=actions,
            avg_loop_ms=_avg_ms(self._sum_loop, frames_ok),
            avg_capture_ms=_avg_ms(self._sum_capture, frames_ok),
            avg_match_ms=_avg_ms(self._sum_match, frames_ok),
        )

        self._last_report_t = now
        self._frames_ok = 0
        self._frames_drop = 0
        self._actions = 0
        self._sum_loop = 0.0
        self._sum_capture = 0.0
        self._sum_match = 0.0
        return snap
