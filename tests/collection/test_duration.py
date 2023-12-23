import time

from greptimeai.collector import _DurationTable


def test_without_name():
    table = _DurationTable()

    id = "fake_trace_id"
    table.set(id)
    time.sleep(0.01)
    assert table.latency_in_ms(id)
    assert table.latency_in_ms(id) is None
    assert table.latency_in_ms("non_exist") is None


def test_with_name():
    table = _DurationTable()

    id, name = "fake_trace_id", "fake_name"
    table.set(id, name)
    time.sleep(0.01)
    assert table.latency_in_ms(id, name)
    assert table.latency_in_ms(id, name) is None
    assert table.latency_in_ms("non_exist", "non_exist") is None
