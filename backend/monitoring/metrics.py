"""In-process operational metrics (reset on restart; no raw content is stored)."""

import threading
from collections import Counter, defaultdict, deque

import numpy as np


class Metrics:
    def __init__(self, window: int = 1000):
        self._lock = threading.Lock()
        self._counts: Counter = Counter()
        self._series: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    def inc(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counts[name] += n

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._series[name].append(float(value))

    def record_request(self, endpoint: str, status: int, latency_ms: float) -> None:
        self.inc("requests_total")
        self.inc("requests_success" if status < 400 else "requests_failed")
        self.inc(f"endpoint:{endpoint}")
        self.observe("response_ms", latency_ms)

    def snapshot(self) -> dict:
        with self._lock:
            counts, series = dict(self._counts), {k: list(v) for k, v in self._series.items()}
        summary = {}
        for name, vals in series.items():
            a = np.asarray(vals)
            summary[name] = {
                "count": int(a.size),
                "mean": round(float(a.mean()), 2),
                "p50": round(float(np.percentile(a, 50)), 2),
                "p95": round(float(np.percentile(a, 95)), 2),
                "max": round(float(a.max()), 2),
            }
        return {"counters": counts, "latency_ms": summary}
