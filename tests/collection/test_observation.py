from opentelemetry.metrics import CallbackOptions

from greptimeai.collector.collection import _Observation


def test_cost():
    obs = _Observation("cost")
    cost, attrs = 0.01, {"model": "gpt-4"}

    obs.put(cost, attrs)
    assert len(obs._value) == 1

    options = CallbackOptions()
    obs.observation_callback()(options)
    assert len(obs._value) == 0
