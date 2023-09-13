import time
from queue import Queue
from typing import Union, Dict

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


class _ObservationQueue:

    def __init__(self):
        self._queue = Queue()

    def put(self, val: float, attrs: Dict) -> None:
        self._queue.put(Observation(val, attributes=attrs))

    def observation_callback(self):

        def fn(_: CallbackOptions):
            if not self._queue.empty():
                yield self._queue.get()

        return fn
