from typing import Dict, List, Any, Union
from uuid import UUID

from opentelemetry import metrics, trace
from opentelemetry.context.context import Context
from opentelemetry.trace import set_span_in_context, Status, StatusCode
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.schema.messages import BaseMessage, get_buffer_string
from langchain.schema.output import LLMResult
from langchain.callbacks.openai_info import (
    get_openai_token_cost_for_model,
    standardize_model_name,
)

from . import (
    _TimeTable,
    _Observation,
    _TraceTable,
    _parse_input,
    _parse_output,
    _parse_generations,
    _sanitate_attributes,
    _SPAN_NAME_AGENT,
    _SPAN_NAME_LLM,
    _SPAN_NAME_TOOL,
    _SPAN_NAME_CHAIN,
)


class GreptimeCallbackHandler(BaseCallbackHandler):
    """
    Greptime LangChain callback handler to collect metrics and traces.
    """

    def __init__(self, skip_otel_init=False, verbose=True) -> None:
        """
        TODO(yuanbohan): support skip_otel_init, verbose parameters
        """
        super().__init__()

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
        self._tracer = trace.get_tracer(__name__)

        meter = metrics.get_meter(__name__)

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
        span_name: str,
        event: str,
        run_id: str,
        parent_run_id: str,
        attrs: Dict[str, Any],
    ):
        attrs = _sanitate_attributes(attrs)

        def start_span(ctx: Context = None):
            span = self._tracer.start_span(span_name, context=ctx)
            span.add_event(event, attributes=attrs)
            self._trace_tables.put_span(span_name, run_id, span)

        if not run_id:
            return

        if parent_run_id:
            parent_span = self._trace_tables.get_id_span(parent_run_id)
            if parent_span:
                context = set_span_in_context(parent_span)
                start_span(context)
            else:
                print(
                    f"unexpected behavior. parent span of { parent_run_id } not found."
                )
        else:
            id_span = self._trace_tables.get_id_span(run_id)
            if id_span:
                context = set_span_in_context(id_span)
                start_span(context)
            else:
                start_span()

    def _end_span(
        self,
        span_name: str,
        event: str,
        run_id: str,
        attrs: Dict[str, Any],
        ex: Exception = None,
    ):
        attrs = _sanitate_attributes(attrs)
        span = self._trace_tables.pop_span(span_name, run_id)
        if span:
            if ex:
                span.record_exception(ex)
            code = StatusCode.ERROR if ex else StatusCode.OK
            span.set_status(Status(code))
            span.add_event(event, attributes=attrs)
            span.end()
        else:
            print(f"unexpected behavior. span of { run_id } not found.")

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain starts running."""
        attrs = {
            "metadata": metadata,
            "tags": tags,
            "kwargs": kwargs,
        }
        if self._verbose:
            attrs["inputs"] = _parse_input(inputs)

        self._start_span(_SPAN_NAME_CHAIN, "chain_start", run_id, parent_run_id, attrs)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain ends running."""
        attrs = {
            "kwargs": kwargs,
        }
        if self._verbose:
            attrs["outputs"] = _parse_output(outputs)
        self._end_span(_SPAN_NAME_CHAIN, "chain_end", run_id=run_id, attrs=attrs)

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain errors."""
        attrs = {
            "error": error.__class__.__name__,
            "kwargs": kwargs,
        }
        self._end_span(
            _SPAN_NAME_CHAIN, "chain_error", run_id=run_id, attrs=attrs, ex=error
        )

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, Any], None] = None,
        invocation_params: Union[Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM starts running."""
        self._time_tables.set(run_id)

        attrs = {
            "metadata": metadata,
            "tags": tags,
            "kwargs": kwargs,
            "params": invocation_params,
        }
        if self._verbose:
            attrs["prompts"] = prompts

        self._start_span(_SPAN_NAME_LLM, "llm_start", run_id, parent_run_id, attrs)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, Any], None] = None,
        invocation_params: Union[Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Chat Model starts running."""
        self._time_tables.set(run_id)

        attrs = {
            "metadata": metadata,
            "tags": tags,
            "kwargs": kwargs,
            "params": invocation_params,
        }
        if self._verbose:
            attrs["messages"] = get_buffer_string(messages[0])

        self._start_span(
            _SPAN_NAME_LLM, "chat_model_start", run_id, parent_run_id, attrs
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        latency = self._time_tables.latency_in_ms(run_id)

        output = response.llm_output or {}
        token_usage = output.get("token_usage", {})
        completion_tokens = token_usage.get("completion_tokens", 0)
        prompt_tokens = token_usage.get("prompt_tokens", 0)

        model_name = output.get("model_name", None)
        model_name = standardize_model_name(model_name)

        self._collect_llm_metrics(
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency=latency,
        )

        attrs = {
            "model": model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "kwargs": kwargs,
        }
        if self._verbose:
            attrs["outputs"] = _parse_generations(response.generations[0])

        self._end_span(_SPAN_NAME_LLM, "llm_end", run_id=run_id, attrs=attrs)

    def _collect_llm_metrics(
        self,
        model_name: str,
        latency: float,
        prompt_tokens: int,
        completion_tokens: int,
    ):
        attrs = {
            "model": model_name,
        }
        try:
            self._prompt_tokens_count.add(prompt_tokens, attrs)
            self._completion_tokens_count.add(completion_tokens, attrs)

            completion_cost = get_openai_token_cost_for_model(
                model_name, completion_tokens, is_completion=True
            )
            prompt_cost = get_openai_token_cost_for_model(model_name, prompt_tokens)
            self._completion_cost.put(completion_cost, attrs)
            self._prompt_cost.put(prompt_cost, attrs)

            if latency:
                self._requests_duration_histogram.record(latency, attrs)
        except Exception as ex:
            attrs["error"] = ex.__class__.__name__
            self._llm_error_count.add(1, attrs)

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        """
        TODO(yuanbohan): get model name
        """
        attrs = {
            "error": error.__class__.__name__,
        }
        self._llm_error_count.add(1, attrs)
        self._end_span(
            _SPAN_NAME_LLM, "llm_error", run_id=run_id, attrs=attrs, ex=error
        )

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """
        Run on new LLM token. Only available when streaming is enabled.
        TODO(yuanbohan): support stream metrics, traces
        """

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Any:
        attrs = {
            "name": serialized.get("name"),
            "tags": tags,
            "metadata": metadata,
            "kwargs": kwargs,
        }
        if self._verbose:
            attrs["input"] = input_str

        self._start_span(_SPAN_NAME_TOOL, "tool_start", run_id, parent_run_id, attrs)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        attrs = {
            "kwargs": kwargs,
        }
        if self._verbose:
            attrs["output"] = output

        self._end_span(_SPAN_NAME_TOOL, "tool_end", run_id=run_id, attrs=attrs)

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        attrs = {
            "error": error.__class__.__name__,
            "kwargs": kwargs,
        }
        self._end_span(
            _SPAN_NAME_TOOL, "tool_error", run_id=run_id, attrs=attrs, ex=error
        )

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Any:
        attrs = {
            "kwargs": kwargs,
            "type": action.__class__.__name__,
            "tool": action.tool,
            "log": action.log,
            "tags": tags,
            "metadata": metadata,
        }
        if self._verbose:
            attrs["input"] = _parse_input(action.tool_input)
        self._start_span(_SPAN_NAME_AGENT, "agent_action", run_id, parent_run_id, attrs)

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        attrs = {
            "kwargs": kwargs,
            "type": finish.__class__.__name__,
            "log": finish.log,
        }
        if self._verbose:
            attrs["output"] = _parse_output(finish.return_values)
        self._end_span(_SPAN_NAME_AGENT, "agent_finish", run_id=run_id, attrs=attrs)


__all__ = ["GreptimeCallbackHandler"]
