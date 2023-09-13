import time
from typing import Union, Dict, Tuple

import prometheus_client

from opentelemetry.metrics import Observation, CallbackOptions

# export PROMETHEUS_DISABLE_CREATED_SERIES=True
prometheus_client.REGISTRY.unregister(prometheus_client.GC_COLLECTOR)
prometheus_client.REGISTRY.unregister(prometheus_client.PLATFORM_COLLECTOR)
prometheus_client.REGISTRY.unregister(prometheus_client.PROCESS_COLLECTOR)


class _TimeTable:

    def __init__(self):
        "track multiple timer specified by key"
        self.__tables = {}

    def set(self, key: str) -> None:
        self.__tables[key] = time.time()

    def remove(self, key: str) -> None:
        self.__tables.pop(key, None)

    def latency_in_millisecond(self, key: str) -> Union[float, None]:
        """
        return latency in milli second if key exist, None if not exist.
        """
        now = time.time()

        start = self.__tables.pop(key, None)
        if start is None:
            return None

        return 1000 * (now - start)


class _Observation:
    """
    # FIXME(yuanbohan): concurrent conflict. refer to Mutex or other solutions
    """

    def __init__(self, name: str = ""):
        self._name = name
        self._cost = {}

    def _reset(self):
        self._cost = {}

    def _dict_to_tuple(self, m: Dict) -> Union[Tuple, None]:
        """
        sort keys first
        """
        if len(m) == 0:
            return None

        l = []
        for key in sorted(m.keys()):
            val = m[key]
            l.append((key, val))
        return tuple(l)

    def put(self, val: float, attrs: Dict) -> None:
        """
        attrs contains even elements
        """
        tuple_key = self._dict_to_tuple(attrs)
        self._cost.setdefault(tuple_key, 0)
        self._cost[tuple_key] += val

    def observation_callback(self):

        def fn(_: CallbackOptions):
            obs = []
            for tuple_key, val in self._cost.items():
                attrs = dict(tuple_key)
                ob = Observation(val, attrs)
                obs.append(ob)

            self._reset()
            return obs

        return fn
