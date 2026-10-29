from backend.monitoring.metrics import Metrics


def test_request_counters_and_latency_summary():
    m = Metrics()
    m.record_request("/chat", 200, 100.0)
    m.record_request("/chat", 500, 300.0)
    s = m.snapshot()
    assert s["counters"]["requests_total"] == 2 and s["counters"]["requests_failed"] == 1
    assert s["latency_ms"]["response_ms"]["max"] == 300.0
    assert s["latency_ms"]["response_ms"]["p50"] == 200.0


def test_empty_snapshot_and_window_bound():
    assert Metrics().snapshot() == {"counters": {}, "latency_ms": {}}
    m = Metrics(window=3)
    for v in range(10):
        m.observe("x", v)
    assert m.snapshot()["latency_ms"]["x"]["count"] == 3
