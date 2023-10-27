import base64
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.openai_info import (
    MODEL_COST_PER_1K_TOKENS,
    get_openai_token_cost_for_model,
    standardize_model_name,
)
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.schema.document import Document
from langchain.schema.messages import BaseMessage, get_buffer_string
from langchain.schema.output import ChatGenerationChunk, GenerationChunk, LLMResult
from opentelemetry import metrics, trace
from opentelemetry.context.context import Context
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode, set_span_in_context
from tenacity import RetryCallState

from greptimeai.langchain import (
    _CLASS_TYPE_LABEL,
    _ERROR_TYPE_LABEL,
    _GREPTIME_DATABASE_ENV_NAME,
    _GREPTIME_HOST_ENV_NAME,
    _GREPTIME_PASSWORD_ENV_NAME,
    _GREPTIME_USERNAME_ENV_NAME,
    _INSTRUMENT_LIB_NAME,
    _MODEL_NAME_LABEL,
    _SPAN_NAME_AGENT,
    _SPAN_NAME_CHAIN,
    _SPAN_NAME_LLM,
    _SPAN_NAME_RETRIEVER,
    _SPAN_NAME_TOOL,
    _SPAN_TYPE_LABEL,
    _check_non_null_or_empty,
    _get_serialized_id,
    _get_user_id,
    _Observation,
    _parse_documents,
    _parse_generations,
    _parse_input,
    _parse_output,
    _sanitate_attributes,
    _TimeTable,
    _TraceTable,
)
from greptimeai.version import __version__


class _Collector:
    """
    collect metrics and traces
    """

    def __init__(
        self,
        skip_otel_init=False,
        resource_name: Optional[str] = None,
        greptimeai_host: Optional[str] = None,
        greptimeai_database: Optional[str] = None,
        greptimeai_username: Optional[str] = None,
        greptimeai_password: Optional[str] = None,
        insecure: bool = False,
        verbose=True,
    ):
        """
        If skip_otel_init is True, then OpenTelemetry Exporter setup will be skipped,
        thus no data will be exported to GreptimeCloud.

        If verbose is False, then inputs, outputs, generations, etc. won't be collected to GreptimeCloud.
        """

        self._skip_otel_init = skip_otel_init
        self._verbose = verbose

        self._time_tables = _TimeTable()
        self._prompt_cost = _Observation("prompt_cost")
        self._completion_cost = _Observation("completion_cost")
        self._trace_tables = _TraceTable()

        if not skip_otel_init:
            self._setup_greptime_otel_exporter(
                resource_name,
                greptimeai_host,
                greptimeai_database,
                greptimeai_username,
                greptimeai_password,
                insecure,
            )
        self._setup_otel_metrics()

    def _setup_greptime_otel_exporter(
        self,
        resource_name: Optional[str] = None,
        host: Optional[str] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        insecure: bool = False,
    ):
        resource = Resource.create(
            {SERVICE_NAME: resource_name or "greptimeai-langchain"}
        )

        host = host or os.getenv(_GREPTIME_HOST_ENV_NAME)
        database = database or os.getenv(_GREPTIME_DATABASE_ENV_NAME)
        username = username or os.getenv(_GREPTIME_USERNAME_ENV_NAME)
        password = password or os.getenv(_GREPTIME_PASSWORD_ENV_NAME)
        scheme = "http" if insecure else "https"

        _check_non_null_or_empty(
            _GREPTIME_HOST_ENV_NAME.lower(), _GREPTIME_HOST_ENV_NAME, host
        )
        _check_non_null_or_empty(
            _GREPTIME_DATABASE_ENV_NAME.lower(), _GREPTIME_DATABASE_ENV_NAME, database
        )
        _check_non_null_or_empty(
            _GREPTIME_USERNAME_ENV_NAME.lower(), _GREPTIME_USERNAME_ENV_NAME, username
        )
        _check_non_null_or_empty(
            _GREPTIME_PASSWORD_ENV_NAME.lower(), _GREPTIME_PASSWORD_ENV_NAME, password
        )

        metrics_endpoint = f"{scheme}://{host}/v1/otlp/v1/metrics"
        trace_endpoint = f"{scheme}://{host}/v1/otlp/v1/traces"

        auth = f"{username}:{password}"
        b64_auth = base64.b64encode(auth.encode()).decode("ascii")
        greptime_headers = {
            "Authorization": f"Basic {b64_auth}",
            "x-greptime-db-name": database,
        }

        metrics_exporter = OTLPMetricExporter(
            endpoint=metrics_endpoint,
            headers=greptime_headers,
            timeout=5,
        )
        metric_reader = PeriodicExportingMetricReader(metrics_exporter, 5000)
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
            instrumenting_module_name=_INSTRUMENT_LIB_NAME,
            instrumenting_library_version=__version__,
        )

        meter = metrics.get_meter(name=_INSTRUMENT_LIB_NAME, version=__version__)

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

    def _start_span(
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

    def _add_span_event(
        self, run_id: UUID, event_name: str, event_attrs: Dict[str, Any]
    ):
        event_attrs = _sanitate_attributes(event_attrs)
        span = self._trace_tables.get_id_span(run_id)
        if span:
            span.add_event(event_name, attributes=event_attrs)
        else:
            logging.error(f"{run_id} span not found for {event_name}")

    def _end_span(
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

    def _collect_llm_metrics(
        self,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> Tuple[float, float]:
        if model_name is None or model_name == "":
            return (0, 0)

        attrs = {
            _MODEL_NAME_LABEL: model_name,
        }

        self._prompt_tokens_count.add(prompt_tokens, attrs)
        self._completion_tokens_count.add(completion_tokens, attrs)

        # only cost of OpenAI model will be calculated and collected
        if model_name not in MODEL_COST_PER_1K_TOKENS:
            return (0, 0)

        completion_cost = get_openai_token_cost_for_model(
            model_name, completion_tokens, is_completion=True
        )
        prompt_cost = get_openai_token_cost_for_model(model_name, prompt_tokens)
        self._completion_cost.put(completion_cost, attrs)
        self._prompt_cost.put(prompt_cost, attrs)
        return (prompt_cost, completion_cost)

    def _start_latency(self, name: str, run_id: UUID):
        self._time_tables.set(name, run_id)

    def _end_latency(self, span_name: str, run_id: UUID):
        latency = self._time_tables.latency_in_ms(span_name, run_id)
        if not latency:
            return

        attributes = {
            _SPAN_TYPE_LABEL: span_name,
        }

        self._requests_duration_histogram.record(latency, attributes)


class GreptimeCallbackHandler(_Collector, BaseCallbackHandler):
    """
    Greptime LangChain callback handler to collect metrics and traces.
    """

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Any:
        logging.debug(f"on_chain_start. { run_id =} { parent_run_id =} { kwargs = }")

        span_attrs = {
            "user_id": _get_user_id(metadata),
        }

        event_attrs = {
            "serialized": serialized,  # this will be removed when lib is stable
            _CLASS_TYPE_LABEL: _get_serialized_id(serialized),
            "metadata": metadata,
            "tags": tags,
        }
        if self._verbose:
            event_attrs["inputs"] = _parse_input(inputs)

        self._start_span(
            run_id=run_id,
            parent_run_id=parent_run_id,
            span_name=_SPAN_NAME_CHAIN,
            event_name="chain_start",
            span_attrs=span_attrs,
            event_attrs=event_attrs,
        )

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        logging.debug(f"on_chain_end. { run_id =} { parent_run_id =} { kwargs = }")
        event_attrs = {}
        if self._verbose:
            event_attrs["outputs"] = _parse_output(outputs)

        self._end_span(
            run_id=run_id,
            span_name=_SPAN_NAME_CHAIN,
            span_attrs={},
            event_name="chain_end",
            event_attrs=event_attrs,
        )

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        logging.debug(f"on_chain_error. { run_id =} { parent_run_id =} { kwargs = }")
        event_attrs = {
            _ERROR_TYPE_LABEL: error.__class__.__name__,
        }

        self._end_span(
            run_id=run_id,
            span_name=_SPAN_NAME_CHAIN,
            span_attrs={},
            event_name="chain_error",
            event_attrs=event_attrs,
            ex=error,  # type: ignore
        )
        self._llm_error_count.add(1, event_attrs)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, Any], None] = None,
        invocation_params: Union[Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Any:
        logging.debug(f"on_llm_start. { run_id =} { parent_run_id =} { kwargs = }")

        span_attrs = {
            "user_id": _get_user_id(metadata),
        }

        event_attrs = {
            "serialized": serialized,  # this will be removed when lib is stable
            _CLASS_TYPE_LABEL: _get_serialized_id(serialized),
            "metadata": metadata,
            "tags": tags,
            "params": invocation_params,
        }
        if self._verbose:
            event_attrs["prompts"] = prompts

        self._start_latency(_SPAN_NAME_LLM, run_id)
        self._start_span(
            run_id=run_id,
            parent_run_id=parent_run_id,
            span_name=_SPAN_NAME_LLM,
            event_name="llm_start",
            span_attrs=span_attrs,
            event_attrs=event_attrs,
        )

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, Any], None] = None,
        invocation_params: Union[Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Any:
        logging.debug(
            f"on_chat_model_start. { run_id =} { parent_run_id =} { kwargs = }"
        )

        span_attrs = {
            "user_id": _get_user_id(metadata),
        }

        event_attrs = {
            "serialized": serialized,  # this will be removed when lib is stable
            _CLASS_TYPE_LABEL: _get_serialized_id(serialized),
            "metadata": metadata,
            "tags": tags,
            "params": invocation_params,
        }
        if self._verbose:
            event_attrs["messages"] = get_buffer_string(messages[0])

        self._start_latency(_SPAN_NAME_LLM, run_id)
        self._start_span(
            run_id=run_id,
            parent_run_id=parent_run_id,
            span_name=_SPAN_NAME_LLM,
            event_name="chat_model_start",
            span_attrs=span_attrs,
            event_attrs=event_attrs,
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        logging.debug(
            f"on_llm_end. { run_id =} { parent_run_id =} { kwargs = } { response = }"
        )
        output = response.llm_output or {}
        token_usage = output.get("token_usage", {})
        completion_tokens = token_usage.get("completion_tokens", 0)
        prompt_tokens = token_usage.get("prompt_tokens", 0)

        model_name = output.get("model_name", "unknown")
        model_name = standardize_model_name(model_name)

        prompt_cost, completion_cost = self._collect_llm_metrics(
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        attrs = {
            _MODEL_NAME_LABEL: model_name,
            "prompt_tokens": prompt_tokens,
            "prompt_cost": prompt_cost,
            "completion_tokens": completion_tokens,
            "completion_cost": completion_cost,
        }

        event_attrs = attrs.copy()
        if self._verbose:
            event_attrs["outputs"] = _parse_generations(response.generations[0])

        self._end_latency(_SPAN_NAME_LLM, run_id)
        self._end_span(
            run_id=run_id,
            span_name=_SPAN_NAME_LLM,
            span_attrs=attrs,
            event_name="llm_end",
            event_attrs=event_attrs,
        )

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        logging.debug(f"on_llm_error. { run_id =} { parent_run_id =} { kwargs = }")
        event_attrs = {
            _ERROR_TYPE_LABEL: error.__class__.__name__,
        }

        self._end_latency(_SPAN_NAME_LLM, run_id)
        self._end_span(
            run_id=run_id,
            span_name=_SPAN_NAME_LLM,
            span_attrs={},
            event_name="llm_error",
            event_attrs=event_attrs,
            ex=error,  # type: ignore
        )
        self._llm_error_count.add(1, event_attrs)

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """
        on_llm_start, or on_chat_model_start has already started this span.
        """
        logging.debug(
            f"on_llm_new_token. { run_id = } { parent_run_id = } { kwargs = } { token = } { chunk = }"
        )

        event_attrs = {}
        if self._verbose:
            event_attrs["token"] = token

        self._add_span_event(
            run_id=run_id, event_name="streaming", event_attrs=event_attrs
        )

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Any:
        logging.debug(f"on_tool_start. { run_id = } { parent_run_id = } { kwargs = }")

        span_attrs = {
            "user_id": _get_user_id(metadata),
        }
        event_attrs = {
            "serialized": serialized,  # this will be removed when lib is stable
            _CLASS_TYPE_LABEL: _get_serialized_id(serialized),
            "tool_name": serialized.get("name"),
            "tags": tags,
            "metadata": metadata,
        }
        if self._verbose:
            event_attrs["input"] = input_str

        self._start_latency(_SPAN_NAME_TOOL, run_id)
        self._start_span(
            run_id=run_id,
            parent_run_id=parent_run_id,
            span_name=_SPAN_NAME_TOOL,
            event_name="tool_start",
            span_attrs=span_attrs,
            event_attrs=event_attrs,
        )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        logging.debug(f"on_tool_end. { run_id =} { parent_run_id =} { kwargs = }")
        event_attrs = {}
        if self._verbose:
            event_attrs["output"] = output

        self._end_latency(_SPAN_NAME_TOOL, run_id)
        self._end_span(
            run_id=run_id,
            span_name=_SPAN_NAME_TOOL,
            span_attrs={},
            event_name="tool_end",
            event_attrs=event_attrs,
        )

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        logging.debug(f"on_tool_error. { run_id =} { parent_run_id =} { kwargs = }")
        event_attrs = {
            _ERROR_TYPE_LABEL: error.__class__.__name__,
        }

        self._end_latency(_SPAN_NAME_TOOL, run_id)
        self._end_span(
            run_id=run_id,
            span_name=_SPAN_NAME_TOOL,
            span_attrs={},
            event_name="tool_error",
            event_attrs=event_attrs,
            ex=error,  # type: ignore
        )
        self._llm_error_count.add(1, event_attrs)

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Any:
        logging.debug(f"on_agent_action. { run_id =} { parent_run_id =} { kwargs = }")

        span_attrs = {
            "user_id": _get_user_id(metadata),
        }

        event_attrs = {
            "tool": action.tool,
            _CLASS_TYPE_LABEL: action.__class__.__name__,
            "log": action.log,
            "tags": tags,
            "metadata": metadata,
        }
        if self._verbose:
            event_attrs["input"] = _parse_input(action.tool_input)

        self._start_latency(_SPAN_NAME_AGENT, run_id)
        self._start_span(
            run_id=run_id,
            parent_run_id=parent_run_id,
            span_name=_SPAN_NAME_AGENT,
            event_name="agent_action",
            span_attrs=span_attrs,
            event_attrs=event_attrs,
        )

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        logging.debug(f"on_agent_finish. { run_id =} { parent_run_id =} { kwargs = }")
        event_attrs = {
            _CLASS_TYPE_LABEL: finish.__class__.__name__,
            "log": finish.log,
        }
        if self._verbose:
            event_attrs["output"] = _parse_output(finish.return_values)

        self._end_latency(_SPAN_NAME_AGENT, run_id)
        self._end_span(
            run_id=run_id,
            span_name=_SPAN_NAME_AGENT,
            span_attrs={},
            event_name="agent_finish",
            event_attrs=event_attrs,
        )

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        logging.debug(f"on_retriever_start. {run_id=} {parent_run_id=} {kwargs=}")

        span_attrs = {
            "user_id": _get_user_id(metadata),
        }
        event_attrs = {
            "serialized": serialized,  # this will be removed when lib is stable
            _CLASS_TYPE_LABEL: _get_serialized_id(serialized),
            "tags": tags,
            "metadata": metadata,
        }
        if self._verbose:
            event_attrs["query"] = query

        self._start_latency(_SPAN_NAME_RETRIEVER, run_id)
        self._start_span(
            run_id=run_id,
            parent_run_id=parent_run_id,
            span_name=_SPAN_NAME_RETRIEVER,
            event_name="retriever_start",
            span_attrs=span_attrs,
            event_attrs=event_attrs,
        )

    def on_retriever_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        logging.debug(f"on_retriever_error. {run_id=} {parent_run_id=} {kwargs=}")
        event_attrs = {
            _ERROR_TYPE_LABEL: error.__class__.__name__,
        }
        self._end_latency(_SPAN_NAME_RETRIEVER, run_id)
        self._end_span(
            run_id=run_id,
            span_name=_SPAN_NAME_RETRIEVER,
            span_attrs={},
            event_name="retriever_error",
            event_attrs=event_attrs,
            ex=error,  # type: ignore
        )

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        logging.debug(f"on_retriever_end. {run_id=} {parent_run_id=} {kwargs=}")
        event_attrs: Dict[str, Any] = {
            "tags": tags,
        }
        if self._verbose:
            event_attrs["docs"] = _parse_documents(documents)

        self._end_latency(_SPAN_NAME_RETRIEVER, run_id)
        self._end_span(
            run_id=run_id,
            span_name=_SPAN_NAME_RETRIEVER,
            span_attrs={},
            event_name="retriever_end",
            event_attrs=event_attrs,
        )

    def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        logging.debug(f"on_retriever_end. {run_id=} {parent_run_id=} {kwargs=}")
        event_attrs = {
            "retry_state": f"{retry_state}",
        }
        self._add_span_event(run_id=run_id, event_name="retry", event_attrs=event_attrs)


__all__ = ["GreptimeCallbackHandler"]
