from typing import Dict, List, Any, Union
from uuid import UUID
import re

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.schema.messages import BaseMessage
from langchain.schema.output import LLMResult
from langchain.callbacks.openai_info import get_openai_token_cost_for_model, \
    standardize_model_name
import openai.error as openai_error
from prometheus_client import start_http_server
from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

from . import _TimeTable, _Observation


class GreptimeCallbackHandler(BaseCallbackHandler):

    def __init__(self, port=8008, verbose=False) -> None:
        super().__init__()
        self._time_tables = _TimeTable()
        self._prompt_cost = _Observation("prompt_cost")
        self._completion_cost = _Observation("completion_cost")

        start_http_server(port=port)

        resource = Resource(
            attributes={SERVICE_NAME: "greptime-langchain-observability"})
        reader = PrometheusMetricReader()
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(provider)
        meter = metrics.get_meter("com.greptime.observability.langchain")

        self._prompt_tokens_count = meter.create_counter(
            "langchain_prompt_tokens",
            description="counts the amount of prompt token",
        )

        self._completion_tokens_count = meter.create_counter(
            "langchain_completion_tokens",
            description="counts the amount of completion token",
        )

        self._llm_error_count = meter.create_counter(
            "langchain_llm_errors",
            description="counts the amount of llm errors",
        )

        self._requests_duration_histogram = meter.create_histogram(
            name="langchain_llm_request_duration_milliseconds",
            description="duration of requests of llm",
        )

        meter.create_observable_gauge(
            callbacks=[self._prompt_cost.observation_callback()],
            name="langchain_prompt_tokens_cost",
            description="prompt token cost in US Dollar",
        )

        meter.create_observable_gauge(
            callbacks=[self._completion_cost.observation_callback()],
            name="langchain_completion_tokens_cost",
            description="completion token cost in US Dollar",
        )

    def __get_llm_repr(self, output: Any) -> str:
        """
        TODO(yuanbohan): support more llm model
        """
        if re.search("openai", repr(output), re.IGNORECASE):
            return "openai"
        else:
            return "unknown"

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain starts running."""
        ...

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain ends running."""
        ...

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain errors."""
        ...

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM starts running."""
        self._time_tables.set(run_id)
        ...

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Chat Model starts running."""
        self._time_tables.set(run_id)
        ...

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Collecting token usage, and performance.

        # Metric Table

        langchain_prompt_tokens_count{llm,model}           # count
        langchain_prompt_tokens_cost{llm,model}            # gauge
        langchain_completion_tokens_count{llm,model}       # count
        langchain_completion_tokens_cost{llm,model}        # gauge
        langchain_llm_request_duration_seconds{llm,model}  # histogram

        # Trace

        run_id, parent_run_id, model, llm, latency, event(on_llm_end)

        if verbose, including:

        message

        """
        latency = self._time_tables.latency_in_millisecond(run_id)

        output = (response.llm_output or {})
        token_usage = output.get("token_usage", {})
        completion_tokens = token_usage.get("completion_tokens", 0)
        prompt_tokens = token_usage.get("prompt_tokens", 0)

        prompt_cost, completion_cost = 0, 0
        model_name = output.get("model_name", None)
        llm = self.__get_llm_repr(output)
        if model_name is not None and llm == "openai":
            model_name = standardize_model_name(model_name)
            try:
                attrs = {
                    "llm": llm,
                    "model": model_name,
                }
                self._prompt_tokens_count.add(prompt_tokens, attrs)
                self._completion_tokens_count.add(completion_tokens, attrs)

                completion_cost = get_openai_token_cost_for_model(
                    model_name, completion_tokens, is_completion=True)
                prompt_cost = get_openai_token_cost_for_model(
                    model_name, prompt_tokens)

                self._completion_cost.put(completion_cost, attrs)
                self._prompt_cost.put(prompt_cost, attrs)

                self._requests_duration_histogram.record(latency, attrs)
            except Exception as ex:
                print(f"on_llm_end exception: {ex}")
                pass

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

        TODO(yuanbohan): more error info for tracing

        # Trace

        run_id, parent_run_id, event(on_llm_end), error
        """
        error_name = error.__class__.__name__
        llm = "unknown"

        try:
            raise error
        except (openai_error.APIError, openai_error.APIConnectionError,
                openai_error.AuthenticationError, openai_error.InvalidAPIType,
                openai_error.InvalidRequestError, openai_error.OpenAIError,
                openai_error.PermissionError, openai_error.RateLimitError,
                openai_error.ServiceUnavailableError,
                openai_error.SignatureVerificationError, openai_error.Timeout,
                openai_error.TryAgain):
            llm = "openai"
        except Exception as ex:
            print(f"on_llm_error. unknown exception: { ex = }")
        finally:
            attrs = {
                "name": error_name,
                "llm": llm,
            }
            print(f"on_llm_error. { attrs = }")
            self._llm_error_count.add(1, attrs)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        ...

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str,
                      **kwargs: Any) -> Any:
        """Run when tool starts running."""
        ...

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        ...

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt],
                      **kwargs: Any) -> Any:
        """Run when tool errors."""
        ...

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""
        ...

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        ...

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        ...
