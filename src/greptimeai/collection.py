import base64
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
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

from .scope import _NAME, _VERSION

_GREPTIME_HOST_ENV_NAME = "GREPTIMEAI_HOST"
_GREPTIME_DATABASE_ENV_NAME = "GREPTIMEAI_DATABASE"
_GREPTIME_USERNAME_ENV_NAME = "GREPTIMEAI_USERNAME"
_GREPTIME_PASSWORD_ENV_NAME = "GREPTIMEAI_PASSWORD"
_GREPTIME_INSECURE_ENV_NAME = "GREPTIMEAI_INSECURE"


def _check_non_null_or_empty(name: str, env_name: str, val: Optional[str]):
    if val is None or val.strip == "":
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


class _TraceTable:
    """
    NOTE: different name may have same run_id.
    """

    def __init__(self):
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
        self._tables: Dict[str, float] = {}

    def _key(self, name: str, run_id: UUID) -> str:
        return f"{name}.{run_id}"

    def set(self, name: str, run_id: UUID):
        """
        set start time of named run_id. for different name may have same run_id
        """
        key = self._key(name, run_id)
        self._tables[key] = time.time()

    def latency_in_ms(self, name: str, run_id: UUID) -> Optional[float]:
        """
        return latency in milli second if key exist, None if not exist.
        """
        key = self._key(name, run_id)
        now = time.time()
        start = self._tables.pop(key, None)
        if start:
            return 1000 * (now - start)
        return None


class _Observation:
    """
    # FIXME(yuanbohan): concurrent conflict. refer to Mutex or other solutions
    """

    def __init__(self, name: str):
        self._name = name
        self._cost: Dict[Tuple, float] = {}

    def _reset(self):
        self._cost = {}

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


class Collector:
    """
    collect metrics and traces
    """

    def __init__(
        self,
        host: Optional[str] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        insecure: bool = False,
    ):
        self.host = host or os.getenv(_GREPTIME_HOST_ENV_NAME)
        self.database = database or os.getenv(_GREPTIME_DATABASE_ENV_NAME)
        self.username = username or os.getenv(_GREPTIME_USERNAME_ENV_NAME)
        self.password = password or os.getenv(_GREPTIME_PASSWORD_ENV_NAME)
        self.insecure = insecure or os.getenv(_GREPTIME_INSECURE_ENV_NAME)

        _check_non_null_or_empty(
            _GREPTIME_HOST_ENV_NAME.lower(), _GREPTIME_HOST_ENV_NAME, self.host
        )
        _check_non_null_or_empty(
            _GREPTIME_DATABASE_ENV_NAME.lower(),
            _GREPTIME_DATABASE_ENV_NAME,
            self.database,
        )
        _check_non_null_or_empty(
            _GREPTIME_USERNAME_ENV_NAME.lower(),
            _GREPTIME_USERNAME_ENV_NAME,
            self.username,
        )
        _check_non_null_or_empty(
            _GREPTIME_PASSWORD_ENV_NAME.lower(),
            _GREPTIME_PASSWORD_ENV_NAME,
            self.password,
        )

        self._time_tables = _TimeTable()
        self._prompt_cost = _Observation("prompt_cost")
        self._completion_cost = _Observation("completion_cost")
        self._trace_tables = _TraceTable()

        self._setup_otel_exporter()
        self._setup_otel_metrics()

    def _setup_otel_exporter(self):
        resource = Resource.create({SERVICE_NAME: "greptimeai-langchain"})
        scheme = "http" if self.insecure else "https"
        metrics_endpoint = f"{scheme}://{self.host}/v1/otlp/v1/metrics"
        trace_endpoint = f"{scheme}://{self.host}/v1/otlp/v1/traces"

        auth = f"{self.username}:{self.password}"
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
        span_attrs: Dict[str, Any] = {},
        event_attrs: Dict[str, Any] = {},
    ):
        span_attrs = _sanitate_attributes(span_attrs)
        event_attrs = _sanitate_attributes(event_attrs)

        def _do_start_span(ctx: Optional[Context] = None):
            span = self._tracer.start_span(
                span_name, context=ctx, attributes=span_attrs
            )
            span.add_event(event_name, attributes=event_attrs)
            self._trace_tables.put_span(span_name, run_id, span)

        if not run_id:
            return

        if parent_run_id:
            parent_span = self._trace_tables.get_id_span(parent_run_id)
            if parent_span:
                context = set_span_in_context(parent_span)
                _do_start_span(context)
            else:
                logging.error(
                    f"unexpected behavior. parent span of { parent_run_id } not found."
                )
        else:
            id_span = self._trace_tables.get_id_span(run_id)
            if id_span:
                context = set_span_in_context(id_span)
                _do_start_span(context)
            else:
                _do_start_span()

    def add_span_event(
        self, run_id: UUID, event_name: str, event_attrs: Dict[str, Any]
    ):
        event_attrs = _sanitate_attributes(event_attrs)
        span = self._trace_tables.get_id_span(run_id)
        if span:
            span.add_event(event_name, attributes=event_attrs)
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
        span_attrs = _sanitate_attributes(span_attrs)
        event_attrs = _sanitate_attributes(event_attrs)

        span = self._trace_tables.pop_span(span_name, run_id)
        if span:
            if ex:
                span.record_exception(ex)
            if span_attrs:
                span.set_attributes(attributes=span_attrs)
            code = StatusCode.ERROR if ex else StatusCode.OK
            span.set_status(Status(code))
            span.add_event(event_name, attributes=event_attrs)
            span.end()
        else:
            logging.error(f"unexpected behavior. span of { run_id } not found.")

    def collect_llm_metrics(
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

        self._prompt_tokens_count.add(prompt_tokens, attrs)
        self._completion_tokens_count.add(completion_tokens, attrs)
        self._completion_cost.put(completion_cost, attrs)
        self._prompt_cost.put(prompt_cost, attrs)

    def start_latency(self, name: str, run_id: UUID):
        self._time_tables.set(name, run_id)

    def end_latency(self, span_name: str, run_id: UUID, attributes: Dict[str, Any]):
        latency = self._time_tables.latency_in_ms(span_name, run_id)
        if not latency:
            return
        self._requests_duration_histogram.record(latency, attributes)
