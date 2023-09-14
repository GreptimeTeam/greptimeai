import time
from typing import Union, Dict, Tuple

from opentelemetry.metrics import Observation, CallbackOptions
from opentelemetry.trace import Span

_SOURCE = "langchain"


class _TraceTable:

    def __init__(self):
        "maintain the OTel span object of run_id"
        self._traces: Dict[str, Span] = {}

    def put_span(self, run_id: str, span: Span):
        self._traces[run_id] = span
        print(f"{ self._traces = }")

    def get_span(self, run_id: str) -> Span:
        return self._traces.get(run_id, None)

    def pop_span(self, run_id: str) -> Span:
        return self._traces.pop(run_id, None)


class _TimeTable:

    def __init__(self):
        "track multiple timer specified by key"
        self.__tables = {}

    def set(self, key: str) -> None:
        self.__tables[key] = time.time()

    def remove(self, key: str) -> None:
        self.__tables.pop(key, None)

    def latency_in_ms(self, key: str) -> Union[float, None]:
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

        l = [(key, m[key]) for key in sorted(m.keys())]
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
            l = [
                Observation(val, dict(tuple_key))
                for tuple_key, val in self._cost.items()
            ]
            self._reset()
            return l

        return fn
