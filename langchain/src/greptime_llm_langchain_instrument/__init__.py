import time
from typing import Any, Dict, List, Optional, Tuple, Union

from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.trace import Span

from langchain.schema import ChatGeneration, Generation
from langchain.schema.document import Document
from langchain.schema.messages import BaseMessage

_SPAN_NAME_CHAIN = "chain"
_SPAN_NAME_AGENT = "agent"
_SPAN_NAME_LLM = "llm"
_SPAN_NAME_TOOL = "tool"
_SPAN_NAME_RETRIEVER = "retriever"


def _get_user_id(metadata: Dict[str, Any]) -> str:
    """
    get user id from metadata
    """
    user_id = metadata.get("user_id")
    return user_id if user_id else ""


def _get_serialized_id(serialized: Dict[str, Any]) -> Optional[str]:
    """
    get id if exist
    """
    ids = serialized.get("id")
    if ids and isinstance(ids, list):
        return ids[len(ids) - 1]
    return None


def _is_valid_otel_attributes_value_type(val: Any) -> bool:
    """
    check if value is valid type for OTel attributes
    """
    return isinstance(val, (bool, str, int, float, bytes))


def _sanitate_attributes(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """
    prepare attributes value to any of ['bool', 'str', 'bytes', 'int', 'float']
    or a sequence of these types
    """
    result = {}

    def _sanitate_list(lst: list) -> list:
        result = []
        for item in lst:
            if _is_valid_otel_attributes_value_type(item):
                result.append(item)
            else:
                result.append(str(item))
        return result

    for key, val in attrs.items():
        if _is_valid_otel_attributes_value_type(val):
            result[key] = val
        elif isinstance(val, list):
            result[key] = _sanitate_list(val)
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
    """
    Generation, or ChatGeneration (which contains message field)
    """
    if not gen:
        return None

    info = gen.generation_info or {}
    attrs = {
        "text": gen.text,
        # TODO(yuanbohan): the following is OpenAI only?
        "finish_reason": info.get("finish_reason"),
        "log_probability": info.get("logprobs"),
    }

    if isinstance(gen, ChatGeneration):
        message: BaseMessage = gen.message
        attrs["additional_kwargs"] = message.additional_kwargs
        attrs["type"] = message.type

    return attrs


def _parse_generations(gens: List[Generation]) -> List[Dict[str, Any]]:
    """
    parse LLMResult.generations[0] to structured fields
    """
    if gens and len(gens) > 0:
        return [_parse_generation(gen) for gen in gens]

    return None


def _parse_documents(docs: List[Document]) -> List[Dict[str, Any]]:
    """
    parse LLMResult.generations[0] to structured fields
    """

    def _parse_doc(doc: Document) -> Dict[str, Any]:
        return {
            "content": doc.page_content,
            "metadata": doc.metadata,
        }

    if docs and len(docs) > 0:
        return [_parse_doc(doc) for doc in docs]

    return None


class _TraceTable:
    def __init__(self):
        """
        maintain the OTel span object of run_id.
        Pay Attention: different name may have same run_id.
        """
        self._traces: Dict[str, List[Tuple(str, Span)]] = {}

    def put_span(self, name: str, run_id: str, span: Span):
        """
        Pay Attention: different name may have same run_id.
        """
        span_list: List[Tuple(str, Span)] = self._traces.get(run_id, [])
        span_list.append((name, span))
        self._traces[run_id] = span_list

    def get_name_span(self, name: str, run_id: str) -> Span:
        """
        first get dict by id, then get span by name
        """
        span_list = self._traces.get(run_id, [])
        for span_name, span in span_list:
            if span_name == name:
                return span
        return None

    def get_id_span(self, run_id: str) -> Span:
        """
        get last span if the matched span list of run_id exist
        """
        span_list = self._traces.get(run_id, [])
        size = len(span_list)
        if size > 0:
            tpl: Tuple(str, Span) = span_list[size - 1]  # get the last span
            return tpl[1]
        return None

    def pop_span(self, name: str, run_id: str) -> Span:
        """
        if there is only one span matched this run_id, then run_id key will be removed
        """
        span_list: List[Tuple(str, Span)] = self._traces.get(run_id)
        if not span_list:
            return None

        target_span = None
        rest_list = []
        for span_name, span in span_list:
            if span_name == name:
                target_span = span
            else:
                rest_list.append((span_name, span))

        if len(rest_list) == 0:
            self._traces.pop(run_id, None)
        else:
            self._traces[run_id] = rest_list

        return target_span


class _TimeTable:
    def __init__(self):
        "track multiple timer specified by key"
        self.__tables = {}

    def _key(self, name: str, run_id: str) -> str:
        return f"{name}.{run_id}"

    def set(self, name: str, run_id: str):
        """
        set start time of named run_id. for different name may have same run_id
        """
        key = self._key(name, run_id)
        self.__tables[key] = time.time()

    def latency_in_ms(self, name: str, run_id: str) -> Optional[float]:
        """
        return latency in milli second if key exist, None if not exist.
        """
        key = self._key(name, run_id)
        now = time.time()
        start = self.__tables.pop(key, None)
        if start:
            return 1000 * (now - start)
        return None


class _Observation:
    """
    # FIXME(yuanbohan): concurrent conflict. refer to Mutex or other solutions
    """

    def __init__(self, name: str = ""):
        self._name = name
        self._cost = {}

    def _reset(self):
        self._cost = {}

    def _dict_to_tuple(self, attrs: Dict) -> Optional[Tuple]:
        """
        sort keys first
        """
        if attrs is None or len(attrs) == 0:
            return None

        lst = [(key, attrs[key]) for key in sorted(attrs.keys())]
        return tuple(lst)

    def put(self, val: float, attrs: Dict):
        """
        if attrs is None or empty, nothing will happen
        """
        if attrs is None or len(attrs) == 0:
            print(f"None key for { attrs }")
            return

        tuple_key = self._dict_to_tuple(attrs)
        self._cost.setdefault(tuple_key, 0)
        self._cost[tuple_key] += val

    def observation_callback(self):
        """
        _cost will be reset each time being called
        """

        def callback(_: CallbackOptions):
            lst = [
                Observation(val, dict(tuple_key))
                for tuple_key, val in self._cost.items()
            ]
            self._reset()
            return lst

        return callback
