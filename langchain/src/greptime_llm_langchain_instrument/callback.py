from typing import Dict, List, Any, Union
from uuid import UUID

from opentelemetry import metrics, trace
from opentelemetry.trace import set_span_in_context, Status, StatusCode
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.schema.messages import BaseMessage
from langchain.schema.output import LLMResult
from langchain.callbacks.openai_info import (
    get_openai_token_cost_for_model,
    standardize_model_name,
)

from . import _TimeTable, _Observation, _SOURCE, _TraceTable


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
        name: str,
        event: str,
        run_id: str,
        parent_run_id: str,
        attrs: Dict[str, object],
    ):
        if not run_id:
            print(f"{ name } run_id is empty. This is illegal event.")
            return

        if parent_run_id:
            parent_span = self._trace_tables.get_span(parent_run_id)
            if parent_span:
                context = set_span_in_context(parent_span)
                span = self._tracer.start_span(name, context=context)
                span.set_attributes(attrs)
                span.add_event(event)
                self._trace_tables.put_span(run_id, span)
                return

        # parent_run_id is None, or parent span not found
        span = self._tracer.start_span(name)
        span.set_attributes(attrs)
        span.add_event(event)
        self._trace_tables.put_span(run_id, span)

    def _end_span(
        self, event: str, run_id: str, attrs: Dict[str, object], ex: Exception
    ):
        span = self._trace_tables.pop_span(run_id)
        if span:
            code = StatusCode.ERROR if ex else StatusCode.OK
            span.set_status(Status(code))
            span.add_event(event, attrs)
            if ex:
                span.record_exception(ex)
            span.end()
        else:
            print(f"unable to find { run_id } span context of { event }")

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
        print(f"on_chain_start. { run_id = }")
        attrs = metadata or {}
        if tags:
            attrs["tags"] = tags
        if self._verbose:
            attrs["inputs"] = str(inputs)

        self._start_span("chain", "on_chain_start", run_id, parent_run_id, attrs)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain ends running."""
        print(f"on_chain_end. { run_id = }")
        attrs = {}
        if self._verbose:
            attrs["outputs"] = str(outputs)
        self._end_span("on_chain_end", run_id, attrs, None)

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain errors."""
        print(f"on_chain_error. { run_id = }")
        self._end_span("on_chain_error", run_id, None, error)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM starts running."""
        print(f"on_llm_start. { run_id = } { parent_run_id = }")
        self._time_tables.set(run_id)

        attrs = metadata or {}
        if tags:
            attrs["tags"] = tags
        if self._verbose:
            attrs["prompts"] = prompts

        self._start_span("llm", "on_llm_start", run_id, parent_run_id, attrs)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Chat Model starts running."""
        print(f"on_chat_model_start. { run_id = } { parent_run_id = }")
        self._time_tables.set(run_id)

        attrs = metadata or {}
        if tags:
            attrs["tags"] = tags
        if self._verbose:
            attrs["messages"] = str(messages)

        self._start_span(
            "chat_model", "on_chat_model_start", run_id, parent_run_id, attrs
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        print(f"on_llm_end. { run_id = } { parent_run_id = }")
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
        }
        if self._verbose:
            attrs["response"] = str(response)

        self._end_span("on_llm_end", run_id, attrs, None)

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
            print(f"on_llm_end exception: {ex}")
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
        Run when LLM errors.
        """
        print(f"on_llm_error. { run_id = } { parent_run_id = }")
        error_name = error.__class__.__name__
        attrs = {
            "source": _SOURCE,
            "error": error_name,
        }
        self._llm_error_count.add(1, attrs)
        self._end_span("on_llm_error", run_id, None, error)

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
        **kwargs: Any,
    ) -> Any:
        """Run when tool starts running."""
        print(
            f"on_tool_start. { run_id = } { parent_run_id = } { kwargs = } { input_str = }"
        )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool ends running."""
        print(
            f"on_tool_end. { run_id = } { parent_run_id = } { kwargs = } { output = }"
        )

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool errors."""
        print(
            f"on_tool_error. { run_id = } { parent_run_id = } { kwargs = } { error = }"
        )

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent action."""
        print(
            f"on_agent_action. { action = } { run_id = } { parent_run_id = } { kwargs = }"
        )

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent end."""
        print(
            f"on_agent_finish. { finish = } { run_id = } { parent_run_id = } { kwargs = }"
        )

    def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on arbitrary text."""
        print(f"on_text. { run_id = } { parent_run_id = } { kwargs = } { text = }")


__all__ = ["GreptimeCallbackHandler"]
