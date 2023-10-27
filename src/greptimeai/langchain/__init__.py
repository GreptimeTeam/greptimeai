import json
import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from uuid import UUID

from langchain.schema import ChatGeneration, Generation
from langchain.schema.document import Document
from langchain.schema.messages import BaseMessage
from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.trace import Span

_SPAN_NAME_CHAIN = "chain"
_SPAN_NAME_AGENT = "agent"
_SPAN_NAME_LLM = "llm"
_SPAN_NAME_TOOL = "tool"
_SPAN_NAME_RETRIEVER = "retriever"

_ERROR_TYPE_LABEL = "type"
_CLASS_TYPE_LABEL = "type"
_SPAN_TYPE_LABEL = "type"
_MODEL_NAME_LABEL = "model"

_INSTRUMENT_LIB_NAME = "greptime-llm"

_GREPTIME_HOST_ENV_NAME = "GREPTIMEAI_HOST"
_GREPTIME_DATABASE_ENV_NAME = "GREPTIMEAI_DATABASE"
_GREPTIME_USERNAME_ENV_NAME = "GREPTIMEAI_USERNAME"
_GREPTIME_PASSWORD_ENV_NAME = "GREPTIMEAI_PASSWORD"


def _check_non_null_or_empty(name: str, env_name: str, val: Optional[str]):
    if val is None or val.strip == "":
        raise ValueError(
            f"{name} MUST BE provided either by passing arguments or setup environment variable {env_name}"
        )


def _get_user_id(metadata: Optional[Dict[str, Any]]) -> str:
    """
    get user id from metadata
    """
    user_id = (metadata or {}).get("user_id")
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


def _sanitate_attributes(attrs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    prepare attributes value to any of ['bool', 'str', 'bytes', 'int', 'float']
    or a sequence of these types
    """
    result = {}
    if not attrs:
        return result

    def _sanitate_list(lst: list) -> list:
        result = []
        for item in lst:
            if _is_valid_otel_attributes_value_type(item):
                result.append(item)
            else:
                result.append(json.dumps(item))
        return result

    for key, val in attrs.items():
        if _is_valid_otel_attributes_value_type(val):
            result[key] = val
        elif isinstance(val, list):
            result[key] = _sanitate_list(val)
        else:
            result[key] = json.dumps(val)

    return result


def _parse(obj: Any) -> Union[Dict[str, Any], Sequence[Any], Any]:
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


def _parse_generation(gen: Generation) -> Optional[Dict[str, Any]]:
    """
    Generation, or ChatGeneration (which contains message field)
    """
    if not gen:
        return None

    info = gen.generation_info or {}
    attrs = {
        "text": gen.text,
        # the following is OpenAI only?
        "finish_reason": info.get("finish_reason"),
        "log_probability": info.get("logprobs"),
    }

    if isinstance(gen, ChatGeneration):
        message: BaseMessage = gen.message
        attrs["additional_kwargs"] = message.additional_kwargs
        attrs["type"] = message.type

    return attrs


def _parse_generations(
    gens: Sequence[Generation],
) -> Optional[Iterable[Dict[str, Any]]]:
    """
    parse LLMResult.generations[0] to structured fields
    """
    if gens and len(gens) > 0:
        return list(filter(None, [_parse_generation(gen) for gen in gens if gen]))

    return None


def _parse_documents(docs: Sequence[Document]) -> Optional[Sequence[Dict[str, Any]]]:
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
        self._traces: Dict[str, List[Tuple[str, Span]]] = {}

    def put_span(self, name: str, run_id: UUID, span: Span):
        """
        Pay Attention: different name may have same run_id.
        """
        str_run_id = str(run_id)
        span_list = self._traces.get(str_run_id, [])
        span_list.append((name, span))
        self._traces[str_run_id] = span_list

    def get_name_span(self, name: str, run_id: UUID) -> Optional[Span]:
        """
        first get dict by id, then get span by name
        """
        span_list = self._traces.get(str(run_id), [])
        for span_name, span in span_list:
            if span_name == name:
                return span
        return None

    def get_id_span(self, run_id: UUID) -> Optional[Span]:
        """
        get last span if the matched span list of run_id exist
        """
        span_list = self._traces.get(str(run_id), [])
        size = len(span_list)
        if size > 0:
            tpl: Tuple[str, Span] = span_list[size - 1]  # get the last span
            return tpl[1]
        return None

    def pop_span(self, name: str, run_id: UUID) -> Optional[Span]:
        """
        if there is only one span matched this run_id, then run_id key will be removed
        """
        str_run_id = str(run_id)
        span_list = self._traces.get(str_run_id)
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
            self._traces.pop(str_run_id, None)
        else:
            self._traces[str_run_id] = rest_list

        return target_span


class _TimeTable:
    def __init__(self):
        "track multiple timer specified by key"
        self.__tables = {}

    def _key(self, name: str, run_id: UUID) -> str:
        return f"{name}.{run_id}"

    def set(self, name: str, run_id: UUID):
        """
        set start time of named run_id. for different name may have same run_id
        """
        key = self._key(name, run_id)
        self.__tables[key] = time.time()

    def latency_in_ms(self, name: str, run_id: UUID) -> Optional[float]:
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
            logging.info(f"None key for { attrs }")
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
