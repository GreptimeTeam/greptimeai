import time
from typing import Union, Dict, Tuple, Any, List

from opentelemetry.metrics import Observation, CallbackOptions
from opentelemetry.trace import Span
from langchain.schema.messages import BaseMessage
from langchain.schema import Generation


def _is_valid_otel_attributes_value_type(val: Any) -> bool:
    """
    check if value is valid type for OTel attributes
    """
    return isinstance(val, (bool, str, int, float, bytes))


def _sanitate_attributes(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """
    prepare attributes value to any of ['bool', 'str', 'bytes', 'int', 'float']
    or a sequence of these types

    when value is List[Dict] or Dict, try to use the key,value pair of Dict
    otherwise use str function directly

    """
    result = {}

    def put_dict_to_result(prefix: str, attrs: Dict[str, Any]):
        for key, val in attrs.items():
            new_key = f"{prefix}.{key}"
            if _is_valid_otel_attributes_value_type(val):
                result[new_key] = val
            else:
                result[new_key] = str(val)

    for key, val in attrs.items():
        if _is_valid_otel_attributes_value_type(val):
            result[key] = val
        elif isinstance(val, list):
            new_list = []
            dict_count = 0
            for list_val in val:
                if _is_valid_otel_attributes_value_type(list_val):
                    new_list.append(list_val)
                elif isinstance(list_val, dict):
                    put_dict_to_result(f"{key}.{dict_count}", list_val)
                    dict_count += 1
                else:
                    new_list.append(str(list_val))
            if len(new_list) > 0:
                result[key] = new_list
        elif isinstance(val, dict):
            put_dict_to_result(key, val)
        else:
            result[key] = str(val)

    return result


def _parse(obj: Any) -> Union[Dict[str, Any], List[Any], Any]:
    if hasattr(obj, "to_json"):
        return obj.to_json()

    if isinstance(obj, dict):
        return {key: _parse(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [_parse(item) for item in obj]

    return obj


def _parse_input(raw_input: Any) -> Any:
    if not raw_input:
        return None

    if not isinstance(raw_input, dict):
        return _parse(raw_input)

    return (
        raw_input.get("input")
        or raw_input.get("inputs")
        or raw_input.get("question")
        or raw_input.get("query")
        or _parse(raw_input)
    )


def _parse_output(raw_output: dict) -> Any:
    if not raw_output:
        return None

    if not isinstance(raw_output, dict):
        return _parse(raw_output)

    return (
        raw_output.get("text")
        or raw_output.get("output")
        or raw_output.get("output_text")
        or raw_output.get("answer")
        or raw_output.get("result")
        or _parse(raw_output)
    )


def _parse_generation(gen: Generation) -> Dict[str, Any]:
    if not gen:
        return None

    info = gen.generation_info or {}
    return {
        "text": gen.text,
        # TODO(yuanbohan): the following is OpenAI only?
        "finish_reason": info.get("finish_reason"),
        "log_probability": info.get("logprobs"),
    }


def _parse_generations(gens: List[Generation]) -> List[Dict[str, Any]]:
    """
    parse LLMResult.generations[0] to structured fields
    """
    if gens and len(gens) > 0:
        return [_parse_generation(gen) for gen in gens]

    return None


class _TraceTable:
    def __init__(self):
        "maintain the OTel span object of run_id"
        self._traces: Dict[str, Span] = {}

    def put_span(self, run_id: str, span: Span):
        self._traces[run_id] = span

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
