from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID

from opentelemetry import metrics, trace
from opentelemetry.context.context import Context
from opentelemetry.trace import Status, StatusCode, set_span_in_context
from tenacity import RetryCallState

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

from . import (
    _CLASS_TYPE_LABEL,
    _ERROR_TYPE_LABEL,
    _INSTRUMENT_LIB_NAME,
    _INSTRUMENT_LIB_VERSION,
    _MODEL_NAME_LABEL,
    _SPAN_NAME_AGENT,
    _SPAN_NAME_CHAIN,
    _SPAN_NAME_LLM,
    _SPAN_NAME_RETRIEVER,
    _SPAN_NAME_TOOL,
    _SPAN_TYPE_LABEL,
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


class _Collector:
    """
    collect metrics and traces
    """

    def __init__(self, skip_otel_init=False, verbose=True):
        """
        TODO(yuanbohan): support skip_otel_init parameters
        """

        self._skip_otel_init = skip_otel_init
        self._verbose = verbose

        self._time_tables = _TimeTable()
        self._prompt_cost = _Observation("prompt_cost")
        self._completion_cost = _Observation("completion_cost")
        self._trace_tables = _TraceTable()

        self._setup_otel()

    def _setup_otel(self):
        """
        setup opentelemetry, and raise Error if something wrong
        """
        self._tracer = trace.get_tracer(
            instrumenting_module_name=_INSTRUMENT_LIB_NAME,
            instrumenting_library_version=_INSTRUMENT_LIB_VERSION,
        )

        meter = metrics.get_meter(
            name=_INSTRUMENT_LIB_NAME, version=_INSTRUMENT_LIB_VERSION
        )

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
        run_id: str,
        parent_run_id: str,
        span_name: str,
        event_name: str,
        span_attrs: Dict[str, Any] = None,
        event_attrs: Dict[str, Any] = None,
    ):
        span_attrs = _sanitate_attributes(span_attrs)
        event_attrs = _sanitate_attributes(event_attrs)

        def _do_start_span(ctx: Context = None):
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
                print(
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
        self, run_id: str, event_name: str, event_attrs: Dict[str, Any]
    ):
        event_attrs = _sanitate_attributes(event_attrs)
        span = self._trace_tables.get_id_span(run_id)
        if span:
            span.add_event(event_name, attributes=event_attrs)
        else:
            print(f"{run_id} span not found for {event_name}")

    def _end_span(
        self,
        run_id: str,
        span_name: str,
        event_name: str,
        event_attrs: Dict[str, Any],
        ex: Exception = None,
    ):
        event_attrs = _sanitate_attributes(event_attrs)
        span = self._trace_tables.pop_span(span_name, run_id)
        if span:
            if ex:
                span.record_exception(ex)
            code = StatusCode.ERROR if ex else StatusCode.OK
            span.set_status(Status(code))
            span.add_event(event_name, attributes=event_attrs)
            span.end()
        else:
            print(f"unexpected behavior. span of { run_id } not found.")

    def _collect_llm_metrics(
        self,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
    ):
        if model_name is None or model_name == "":
            return

        attrs = {
            _MODEL_NAME_LABEL: model_name,
        }

        self._prompt_tokens_count.add(prompt_tokens, attrs)
        self._completion_tokens_count.add(completion_tokens, attrs)

        # only cost of OpenAI model will be calculated and collected
        if model_name in MODEL_COST_PER_1K_TOKENS:
            completion_cost = get_openai_token_cost_for_model(
                model_name, completion_tokens, is_completion=True
            )
            prompt_cost = get_openai_token_cost_for_model(model_name, prompt_tokens)
            self._completion_cost.put(completion_cost, attrs)
            self._prompt_cost.put(prompt_cost, attrs)

    def _start_latency(self, name: str, run_id: str):
        self._time_tables.set(name, run_id)

    def _end_latency(self, span_name: str, run_id: str):
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
        # TODO(yuanbohan): remove this print in the near future
        print(f"on_chain_start. { run_id =} { parent_run_id =} { kwargs = }")

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
        # TODO(yuanbohan): remove this print in the near future
        print(f"on_chain_end. { run_id =} { parent_run_id =} { kwargs = }")
        event_attrs = {}
        if self._verbose:
            event_attrs["outputs"] = _parse_output(outputs)

        self._end_span(
            run_id=run_id,
            span_name=_SPAN_NAME_CHAIN,
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
        # TODO(yuanbohan): remove this print in the near future
        print(f"on_chain_error. { run_id =} { parent_run_id =} { kwargs = }")
        event_attrs = {
            _ERROR_TYPE_LABEL: error.__class__.__name__,
        }

        self._end_span(
            run_id=run_id,
            span_name=_SPAN_NAME_CHAIN,
            event_name="chain_error",
            event_attrs=event_attrs,
            ex=error,
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
        # TODO(yuanbohan): remove this print in the near future
        print(f"on_llm_start. { run_id =} { parent_run_id =} { kwargs = }")

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
        # TODO(yuanbohan): remove this print in the near future
        print(f"on_chat_model_start. { run_id =} { parent_run_id =} { kwargs = }")

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
        # TODO(yuanbohan): remove this print in the near future
        print(f"on_llm_end. { run_id =} { parent_run_id =} { kwargs = } { response = }")
        output = response.llm_output or {}
        token_usage = output.get("token_usage", {})
        completion_tokens = token_usage.get("completion_tokens", 0)
        prompt_tokens = token_usage.get("prompt_tokens", 0)

        model_name = output.get("model_name", "unknown")
        model_name = standardize_model_name(model_name)

        self._collect_llm_metrics(
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        event_attrs = {
            _MODEL_NAME_LABEL: model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        if self._verbose:
            event_attrs["outputs"] = _parse_generations(response.generations[0])

        self._end_latency(_SPAN_NAME_LLM, run_id)
        self._end_span(
            run_id=run_id,
            span_name=_SPAN_NAME_LLM,
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
        # TODO(yuanbohan): remove this print in the near future
        print(f"on_llm_error. { run_id =} { parent_run_id =} { kwargs = }")
        event_attrs = {
            _ERROR_TYPE_LABEL: error.__class__.__name__,
        }

        self._end_latency(_SPAN_NAME_LLM, run_id)
        self._end_span(
            run_id=run_id,
            span_name=_SPAN_NAME_LLM,
            event_name="llm_error",
            event_attrs=event_attrs,
            ex=error,
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
        on_llm_start, or on_chat_model_start has already startet this span.
        """
        if not self._verbose:
            return

        event_attrs = {
            "token": token,
        }

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
        # TODO(yuanbohan): remove this print in the near future
        print(f"on_tool_start. { run_id = } { parent_run_id = } { kwargs = }")

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
        # TODO(yuanbohan): remove this print in the near future
        print(f"on_tool_end. { run_id =} { parent_run_id =} { kwargs = }")
        event_attrs = {}
        if self._verbose:
            event_attrs["output"] = output

        self._end_latency(_SPAN_NAME_TOOL, run_id)
        self._end_span(
            run_id=run_id,
            span_name=_SPAN_NAME_TOOL,
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
        # TODO(yuanbohan): remove this print in the near future
        print(f"on_tool_error. { run_id =} { parent_run_id =} { kwargs = }")
        event_attrs = {
            _ERROR_TYPE_LABEL: error.__class__.__name__,
        }

        self._end_latency(_SPAN_NAME_TOOL, run_id)
        self._end_span(
            run_id=run_id,
            span_name=_SPAN_NAME_TOOL,
            event_name="tool_error",
            event_attrs=event_attrs,
            ex=error,
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
        # TODO(yuanbohan): remove this print in the near future
        print(f"on_agent_action. { run_id =} { parent_run_id =} { kwargs = }")

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
        # TODO(yuanbohan): remove this print in the near future
        print(f"on_agent_finish. { run_id =} { parent_run_id =} { kwargs = }")
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
        print(f"on_retriever_start. {run_id=} {parent_run_id=} {kwargs=}")

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
        print(f"on_retriever_error. {run_id=} {parent_run_id=} {kwargs=}")
        event_attrs = {
            _ERROR_TYPE_LABEL: error.__class__.__name__,
        }
        self._end_latency(_SPAN_NAME_RETRIEVER, run_id)
        self._end_span(
            run_id=run_id,
            span_name=_SPAN_NAME_RETRIEVER,
            event_name="retriever_error",
            event_attrs=event_attrs,
            ex=error,
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
        print(f"on_retriever_end. {run_id=} {parent_run_id=} {kwargs=}")
        event_attrs = {
            "tags": tags,
        }
        if self._verbose:
            event_attrs["docs"] = _parse_documents(documents)

        self._end_latency(_SPAN_NAME_RETRIEVER, run_id)
        self._end_span(
            run_id=run_id,
            span_name=_SPAN_NAME_RETRIEVER,
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
        print(f"on_retriever_end. {run_id=} {parent_run_id=} {kwargs=}")
        event_attrs = {
            "retry_state": f"{retry_state}",
        }
        self._add_span_event(run_id=run_id, event_name="retry", event_attrs=event_attrs)


__all__ = ["GreptimeCallbackHandler"]
