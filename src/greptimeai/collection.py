import base64
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

from opentelemetry import metrics, trace
from opentelemetry.context.context import Context
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode, set_span_in_context

from . import logger
from .scope import _NAME, _VERSION

_JSON_KEYS_IN_OTLP_ATTRIBUTES = "otlp_json_keys"

_GREPTIME_HOST_ENV_NAME = "GREPTIMEAI_HOST"
_GREPTIME_DATABASE_ENV_NAME = "GREPTIMEAI_DATABASE"
_GREPTIME_TOKEN_ENV_NAME = "GREPTIMEAI_TOKEN"


def _extract_token(token: Optional[str]) -> Tuple[str, str]:
    """
    if token is invalid or empty, then invalid auth header will be included
    """
    if token is None or token.strip() == "":
        return "", ""

    lst = token.split(":", 2)
    if len(lst) != 2:
        return "", ""

    return lst[0], lst[1]


def _check_with_env(
    name: str, value: Optional[str], env_name: str, required: bool = True
) -> Optional[str]:
    if value and value.strip() != "":
        return value

    value = os.getenv(env_name)
    if value:
        return value

    if required:
        raise ValueError(
            f"{name} MUST BE provided either by passing arguments or setup environment variable {env_name}"
        )


def _is_valid_otel_attributes_value_type(val: Any) -> bool:
    """
    check if value is valid type for OTel attributes
    """
    return isinstance(val, (bool, str, int, float, bytes))


def _sanitate_attributes(attrs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    prepare attributes value to any of ['bool', 'str', 'bytes', 'int', 'float']
    or a sequence of these types.

    Other types will be sanitated to json string, and
    append this key in `_JSON_KEYS_IN_OTLP_ATTRIBUTES`
    """
    result = {}
    if not attrs:
        return result

    def _sanitate_list(lst: list) -> Union[list, str]:
        contains_any_invalid_value_type = False
        for item in lst:
            if not _is_valid_otel_attributes_value_type(item):
                contains_any_invalid_value_type = True
                break
        if contains_any_invalid_value_type:
            return json.dumps(lst)
        else:
            return lst

    json_keys = []
    for key, val in attrs.items():
        if val is None:
            continue
        elif _is_valid_otel_attributes_value_type(val):
            result[key] = val
        elif isinstance(val, list):
            sanitated_lst = _sanitate_list(val)
            result[key] = sanitated_lst

            if isinstance(sanitated_lst, str):
                json_keys.append(key)
        else:
            result[key] = json.dumps(val)
            json_keys.append(key)

    if len(json_keys) > 0:
        result[_JSON_KEYS_IN_OTLP_ATTRIBUTES] = json_keys
    return result


class _TraceContext:
    """
    ease context management for OTLP Context and LangChain run_id
    """

    def __init__(self, name: str, model: str, span: Span):
        self.name = name
        self.model = model
        self.span = span

    def set_self_as_current(self) -> Context:
        """
        call `get_current_span` from the returning context to retrieve the span
        """
        return set_span_in_context(self.span, Context({}))

    def __repr__(self) -> str:
        span_context = self.span.get_span_context()
        return (
            f"{self.name}.{self.model}.{span_context.trace_id}.{span_context.span_id}"
        )


class _TraceTable:
    """
    NOTE: different span_name may have same run_id.
    """

    def __init__(self):
        self._traces: Dict[str, List[_TraceContext]] = {}

    def put_trace_context(
        self,
        run_id: Union[UUID, str],
        context: _TraceContext,
    ):
        """
        Pay Attention: different span_name may have same run_id in LangChain.
        """
        str_run_id = str(run_id)
        context_list = self._traces.get(str_run_id, [])
        context_list.append(context)
        self._traces[str_run_id] = context_list

    def get_trace_context(
        self, run_id: Union[UUID, str], span_name: Optional[str] = None
    ) -> Optional[_TraceContext]:
        """
        Args:

            span_name: if is None or empty, the last context will be returned
        """
        context_list = self._traces.get(str(run_id), [])
        if len(context_list) == 0:
            return None

        if not span_name:
            return context_list[-1]

        for context in context_list:
            if span_name == context.name:
                return context

        return None

    def pop_trace_context(
        self, run_id: Union[UUID, str], span_name: Optional[str] = None
    ) -> Optional[_TraceContext]:
        """
        if there is only one span matched this run_id, then run_id key will be removed

        Args:

            span_name: if is None or empty, the last context will be returned
        """
        str_run_id = str(run_id)
        context_list = self._traces.get(str_run_id, [])

        if len(context_list) == 0:
            return None

        target_context = None
        rest_list = []

        if not span_name:
            target_context = context_list.pop()
            rest_list = context_list
        else:
            for context in context_list:
                if span_name == context.name:
                    target_context = context
                else:
                    rest_list.append(context)

        if len(rest_list) == 0:
            self._traces.pop(str_run_id, None)
        else:
            self._traces[str_run_id] = rest_list

        return target_context


class _DurationTable:
    def __init__(self):
        self._tables: Dict[str, float] = {}

    def _key(self, run_id: Union[UUID, str], name: Optional[str]) -> str:
        if name:
            return f"{name}.{run_id}"
        else:
            return str(run_id)

    def set(self, run_id: Union[UUID, str], name: Optional[str] = None):
        """
        set start time of run_id.

        Args:

            name: same span may have different name, this is to diffirentiate them.
        """
        key = self._key(run_id, name)
        self._tables[key] = time.time()

    def latency_in_ms(
        self, run_id: Union[UUID, str], name: Optional[str] = None
    ) -> Optional[float]:
        """
        return latency in milli second if key exist, None if not exist.
        """
        key = self._key(run_id, name)
        start = self._tables.pop(key, None)
        if start:
            now = time.time()
            return 1000 * (now - start)
        return None


class _Observation:
    """
    # FIXME(yuanbohan): concurrent conflict. refer to Mutex or other solutions
    """

    def __init__(self, name: str):
        self._name = name
        self._value: Dict[Tuple, float] = {}

    def _reset(self):
        self._value = {}

    def _dict_to_tuple(self, attrs: Dict) -> Tuple:
        """
        sort keys first
        """
        if attrs is None or len(attrs) == 0:
            return ()

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
        self._value.setdefault(tuple_key, 0)
        self._value[tuple_key] += val

    def observation_callback(self):
        """
        _cost will be reset each time being called
        """

        def callback(_: CallbackOptions):
            lst = [
                Observation(val, dict(tuple_key))
                for tuple_key, val in self._value.items()
            ]
            self._reset()
            return lst

        return callback


class Collector:
    """
    collect metrics and traces.
    """

    def __init__(
        self,
        host: str = "",
        database: str = "",
        token: str = "",
    ):
        self.host = _check_with_env("host", host, _GREPTIME_HOST_ENV_NAME, True)
        self.database = _check_with_env(
            "database", database, _GREPTIME_DATABASE_ENV_NAME, True
        )
        self.token = _check_with_env("token", token, _GREPTIME_TOKEN_ENV_NAME, False)

        self._duration_tables = _DurationTable()
        self._prompt_cost = _Observation("prompt_cost")
        self._completion_cost = _Observation("completion_cost")
        self._trace_tables = _TraceTable()

        self._setup_otel_exporter()
        self._setup_otel_metrics()

    def _setup_otel_exporter(self):
        resource = Resource.create({SERVICE_NAME: "greptimeai-langchain"})
        metrics_endpoint = f"{self.host}/v1/otlp/v1/metrics"
        trace_endpoint = f"{self.host}/v1/otlp/v1/traces"

        username, password = _extract_token(self.token)
        auth = f"{username}:{password}"
        b64_auth = base64.b64encode(auth.encode()).decode("ascii")
        greptime_headers = {
            "Authorization": f"Basic {b64_auth}",
            "x-greptime-db-name": self.database,
        }

        metrics_exporter = OTLPMetricExporter(
            endpoint=metrics_endpoint,
            headers=greptime_headers,
            timeout=5,
        )
        metric_reader = PeriodicExportingMetricReader(
            metrics_exporter, export_interval_millis=15000
        )
        metre_provider = MeterProvider(
            resource=resource, metric_readers=[metric_reader]
        )
        metrics.set_meter_provider(metre_provider)

        trace_provider = TracerProvider(resource=resource)
        span_processor = BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=trace_endpoint,
                headers=greptime_headers,
                timeout=5,
            )
        )
        trace_provider.add_span_processor(span_processor)
        trace.set_tracer_provider(trace_provider)

    def _setup_otel_metrics(self):
        """
        setup opentelemetry, and raise Error if something wrong
        """
        self._tracer = trace.get_tracer(
            instrumenting_module_name=_NAME,
            instrumenting_library_version=_VERSION,
        )

        meter = metrics.get_meter(name=_NAME, version=_VERSION)

        self._prompt_tokens_count = meter.create_counter(
            "llm_prompt_tokens",
            description="counts the amount of llm prompt token",
        )

        self._completion_tokens_count = meter.create_counter(
            "llm_completion_tokens",
            description="counts the amount of llm completion token",
        )

        self._llm_error_count = meter.create_counter(
            "llm_errors",
            description="counts the amount of llm errors",
        )

        self._requests_duration_histogram = meter.create_histogram(
            name="llm_request_duration_ms",
            description="duration of requests of llm in milli seconds",
            unit="ms",
        )

        meter.create_observable_gauge(
            callbacks=[self._prompt_cost.observation_callback()],
            name="llm_prompt_tokens_cost",
            description="prompt token cost in US Dollar",
        )

        meter.create_observable_gauge(
            callbacks=[self._completion_cost.observation_callback()],
            name="llm_completion_tokens_cost",
            description="completion token cost in US Dollar",
        )

    def start_span(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        span_name: str,
        event_name: str,
        span_attrs: Dict[str, Any] = {},  # model may exist in span attrs
        event_attrs: Dict[str, Any] = {},
    ):
        """
        this is mainly focused on LangChain.
        """
        span_attrs = _sanitate_attributes(span_attrs)
        event_attrs = _sanitate_attributes(event_attrs)

        def _do_start_span(ctx: Optional[Context] = None):
            span = self._tracer.start_span(
                span_name, context=ctx, attributes=span_attrs
            )
            span.add_event(event_name, attributes=event_attrs)

            trace_context = _TraceContext(
                name=span_name, model=span_attrs.get("model", ""), span=span
            )
            self._trace_tables.put_trace_context(run_id, trace_context)

        if not run_id:
            logger.error("unexpected behavior of start_span. run_id MUST NOT be empty.")
            return

        if parent_run_id:
            trace_context = self._trace_tables.get_trace_context(parent_run_id)
            if trace_context:
                _do_start_span(trace_context.set_self_as_current())
            else:
                logging.error(
                    f"unexpected behavior of start_span. parent span of { parent_run_id } not found."
                )
        else:
            # trace_context may exist for the same run_id in LangChain. For Example:
            # different Chain triggered in the same trace may have the same run_id.
            trace_context = self._trace_tables.get_trace_context(run_id)
            if trace_context:
                _do_start_span(trace_context.set_self_as_current())
            else:
                _do_start_span()

    def add_span_event(
        self, run_id: UUID, event_name: str, event_attrs: Dict[str, Any]
    ):
        """
        this is mainly focused on LangChain.
        """
        event_attrs = _sanitate_attributes(event_attrs)
        context = self._trace_tables.get_trace_context(run_id)
        if context:
            context.span.add_event(event_name, attributes=event_attrs)
        else:
            logging.error(f"{run_id} span not found for {event_name}")

    def end_span(
        self,
        run_id: UUID,
        span_name: str,
        span_attrs: Dict[str, Any],
        event_name: str,
        event_attrs: Dict[str, Any],
        ex: Optional[Exception] = None,
    ):
        """
        this is mainly focused on LangChain.
        """
        span_attrs = _sanitate_attributes(span_attrs)
        event_attrs = _sanitate_attributes(event_attrs)

        context = self._trace_tables.pop_trace_context(run_id, span_name)
        if context:
            span = context.span
            if ex:
                span.record_exception(ex)
            if span_attrs:
                span.set_attributes(attributes=span_attrs)
            code = StatusCode.ERROR if ex else StatusCode.OK
            span.set_status(Status(code))
            span.add_event(event_name, attributes=event_attrs)
            span.end()
        else:
            logging.error(
                f"unexpected behavior of end_span. context of { run_id } and { span_name } not found."
            )

    def collect_metrics(
        self,
        model_name: str,
        prompt_tokens: int,
        prompt_cost: float,
        completion_tokens: int,
        completion_cost: float,
    ):
        attrs = {
            "model": model_name,
        }

        if prompt_tokens:
            self._prompt_tokens_count.add(prompt_tokens, attrs)

        if prompt_cost:
            self._prompt_cost.put(prompt_cost, attrs)

        if completion_tokens:
            self._completion_tokens_count.add(completion_tokens, attrs)

        if completion_cost:
            self._completion_cost.put(completion_cost, attrs)

    def start_latency(self, run_id: Union[UUID, str], span_name: Optional[str]):
        self._duration_tables.set(run_id, span_name)

    def end_latency(
        self,
        run_id: Union[UUID, str],
        span_name: Optional[str],
        attributes: Dict[str, Any],
    ):
        latency = self._duration_tables.latency_in_ms(run_id, span_name)
        if latency:
            self._requests_duration_histogram.record(latency, attributes)

    def get_model_in_context(self, run_id: Union[UUID, str]) -> Optional[str]:
        context = self._trace_tables.get_trace_context(run_id)

        if context:
            return context.model

        return None
