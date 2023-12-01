from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.schema.document import Document
from langchain.schema.messages import BaseMessage, get_buffer_string
from langchain.schema.output import (
    ChatGenerationChunk,
    Generation,
    GenerationChunk,
    LLMResult,
)
from tenacity import RetryCallState

from greptimeai import logger
from greptimeai.collector import Collector
from greptimeai.labels import (
    _CLASS_TYPE_LABEL,
    _COMPLETION_COST_LABEL,
    _COMPLETION_TOKENS_LABEL,
    _ERROR_TYPE_LABEL,
    _MODEL_LABEL,
    _SPAN_NAME_LABEL,
    _USER_ID_LABEL,
    _PROMPT_COST_LABEl,
    _PROMPT_TOKENS_LABEl,
)
from greptimeai.utils.openai.token import (
    get_openai_token_cost_for_model,
    num_tokens_from_messages,
    standardize_model_name,
)

from . import (
    _SPAN_NAME_AGENT,
    _SPAN_NAME_CHAIN,
    _SPAN_NAME_LLM,
    _SPAN_NAME_RETRIEVER,
    _SPAN_NAME_TOOL,
    _get_serialized_id,
    _get_serialized_streaming,
    _get_user_id,
    _parse_documents,
    _parse_generations,
    _parse_input,
    _parse_output,
)


class GreptimeCallbackHandler(Collector, BaseCallbackHandler):
    """
    Greptime LangChain callback handler to collect metrics and traces.
    """

    def __init__(
        self,
        host: str = "",
        database: str = "",
        token: str = "",
        verbose: bool = True,
    ):
        super().__init__(
            service_name="langchain", host=host, database=database, token=token
        )
        self._verbose = verbose

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
        logger.debug(
            f"on_chain_start. { run_id =} { parent_run_id =} { kwargs = } { serialized = }"
        )

        span_attrs = {
            _USER_ID_LABEL: _get_user_id(metadata),
        }

        event_attrs = {
            _CLASS_TYPE_LABEL: _get_serialized_id(serialized),
            "metadata": metadata,
            "tags": tags,
        }
        if self._verbose:
            event_attrs["inputs"] = _parse_input(inputs)

        self._collector.start_span(
            span_id=run_id,
            parent_id=parent_run_id,
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
        logger.debug(f"on_chain_end. { run_id =} { parent_run_id =} { kwargs = }")
        event_attrs = {}
        if self._verbose:
            event_attrs["outputs"] = _parse_output(outputs)

        self._collector.end_span(
            span_id=run_id,
            span_name=_SPAN_NAME_CHAIN,
            span_attrs={},
            event_name="chain_end",
            event_attrs=event_attrs,
        )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        logger.debug(f"on_chain_error. { run_id =} { parent_run_id =} { kwargs = }")
        event_attrs = {
            _ERROR_TYPE_LABEL: error.__class__.__name__,
        }

        self._collector.end_span(
            span_id=run_id,
            span_name=_SPAN_NAME_CHAIN,
            span_attrs={},
            event_name="chain_error",
            event_attrs=event_attrs,
            ex=error,
        )
        self._collector.collect_error_count(
            {
                _SPAN_NAME_LABEL: _SPAN_NAME_CHAIN,
                **event_attrs,
            },
        )

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
        logger.debug(
            f"on_llm_start. { run_id =} { parent_run_id =} { kwargs = } { serialized = }"
        )

        span_attrs: Dict[str, Any] = {
            _USER_ID_LABEL: _get_user_id(metadata),
        }

        streaming = _get_serialized_streaming(serialized)
        if streaming and invocation_params:
            str_messages = " ".join(prompts)
            model_name: str = invocation_params.get("model_name", "")
            prompt_tokens = num_tokens_from_messages(str_messages, model_name)
            prompt_cost = get_openai_token_cost_for_model(model_name, prompt_tokens)
            span_attrs[_MODEL_LABEL] = model_name
            span_attrs[_PROMPT_TOKENS_LABEl] = prompt_tokens
            span_attrs[_PROMPT_COST_LABEl] = prompt_cost

        event_attrs = {
            _CLASS_TYPE_LABEL: _get_serialized_id(serialized),
            "metadata": metadata,
            "tags": tags,
            "params": invocation_params,
        }
        if self._verbose:
            event_attrs["prompts"] = prompts

        self._collector.start_latency(run_id, _SPAN_NAME_LLM)
        self._collector.start_span(
            span_id=run_id,
            parent_id=parent_run_id,
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
        logger.debug(
            f"on_chat_model_start. { run_id =} { parent_run_id =} { kwargs = } { serialized = }"
        )

        str_messages = get_buffer_string(messages[0])
        span_attrs: Dict[str, Any] = {
            _USER_ID_LABEL: _get_user_id(metadata),
        }

        streaming = _get_serialized_streaming(serialized)
        if streaming and invocation_params:
            model_name: str = invocation_params.get("model_name", "")
            prompt_tokens = num_tokens_from_messages(str_messages, model_name)
            prompt_cost = get_openai_token_cost_for_model(model_name, prompt_tokens)
            span_attrs[_MODEL_LABEL] = model_name
            span_attrs[_PROMPT_TOKENS_LABEl] = prompt_tokens
            span_attrs[_PROMPT_COST_LABEl] = prompt_cost

        event_attrs = {
            _CLASS_TYPE_LABEL: _get_serialized_id(serialized),
            "metadata": metadata,
            "tags": tags,
            "params": invocation_params,
        }
        if self._verbose:
            event_attrs["messages"] = str_messages

        self._collector.start_latency(run_id, _SPAN_NAME_LLM)
        self._collector.start_span(
            span_id=run_id,
            parent_id=parent_run_id,
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
        logger.debug(
            f"on_llm_end. { run_id =} { parent_run_id =} { kwargs = } { response = }"
        )
        generations: List[Generation] = (
            response.generations[0]
            if response and len(response.generations) > 0
            else []
        )
        output = response.llm_output or {}
        model_name: Optional[str] = output.get("model_name")
        if model_name is None:
            model_name = self._collector.get_model_in_context(run_id) or ""
        model_name = standardize_model_name(model_name)

        token_usage = output.get("token_usage", {})

        # NOTE: only cost of OpenAI model will be calculated and collected so far
        prompt_tokens, prompt_cost, completion_tokens, completion_cost = 0, 0.0, 0, 0.0
        if model_name.strip() != "":
            if len(token_usage) > 0:
                prompt_tokens = token_usage.get("prompt_tokens", 0)
                completion_tokens = token_usage.get("completion_tokens", 0)
                prompt_cost = get_openai_token_cost_for_model(model_name, prompt_tokens)
                completion_cost = get_openai_token_cost_for_model(
                    model_name, completion_tokens, is_completion=True
                )
            else:  # streaming
                texts = [generation.text for generation in generations]
                str_messages = " ".join(texts)
                completion_tokens = num_tokens_from_messages(str_messages, model_name)
                completion_cost = get_openai_token_cost_for_model(
                    model_name, prompt_tokens, is_completion=True
                )

        self._collector.collect_metrics(
            prompt_tokens=prompt_tokens,
            prompt_cost=prompt_cost,
            completion_tokens=completion_tokens,
            completion_cost=completion_cost,
            attrs={_MODEL_LABEL: model_name, _SPAN_NAME_LABEL: _SPAN_NAME_LLM},
        )

        attrs: Dict[str, Any] = {}

        if model_name.strip() != "":
            attrs[_MODEL_LABEL] = model_name
        if prompt_tokens > 0:
            attrs[_PROMPT_TOKENS_LABEl] = prompt_tokens
        if prompt_cost > 0:
            attrs[_PROMPT_COST_LABEl] = prompt_cost
        if completion_tokens > 0:
            attrs[_COMPLETION_TOKENS_LABEL] = completion_tokens
        if completion_cost > 0:
            attrs[_COMPLETION_COST_LABEL] = completion_cost

        event_attrs = attrs.copy()
        if self._verbose:
            event_attrs["outputs"] = _parse_generations(generations)

        self._collector.end_latency(
            run_id, _SPAN_NAME_LLM, attributes={_SPAN_NAME_LABEL: _SPAN_NAME_LLM}
        )
        self._collector.end_span(
            span_id=run_id,
            span_name=_SPAN_NAME_LLM,
            span_attrs=attrs,
            event_name="llm_end",
            event_attrs=event_attrs,
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        logger.debug(f"on_llm_error. { run_id =} { parent_run_id =} { kwargs = }")
        event_attrs = {
            _ERROR_TYPE_LABEL: error.__class__.__name__,
        }

        self._collector.end_latency(
            run_id, _SPAN_NAME_LLM, attributes={_SPAN_NAME_LABEL: _SPAN_NAME_LLM}
        )
        self._collector.end_span(
            span_id=run_id,
            span_name=_SPAN_NAME_LLM,
            span_attrs={},
            event_name="llm_error",
            event_attrs=event_attrs,
            ex=error,
        )
        self._collector.collect_error_count(
            {
                _SPAN_NAME_LABEL: _SPAN_NAME_LLM,
                **event_attrs,
            },
        )

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
        logger.debug(
            f"on_llm_new_token. { run_id = } { parent_run_id = } { kwargs = } { token = } { chunk = }"
        )

        event_attrs = {}
        if self._verbose:
            event_attrs["token"] = token

        self._collector.add_span_event(
            span_id=run_id, event_name="streaming", event_attrs=event_attrs
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
        logger.debug(
            f"on_tool_start. { run_id = } { parent_run_id = } { kwargs = } { serialized = }"
        )

        span_attrs = {
            _USER_ID_LABEL: _get_user_id(metadata),
        }
        event_attrs = {
            _CLASS_TYPE_LABEL: _get_serialized_id(serialized),
            "tool_name": serialized.get("name"),
            "tags": tags,
            "metadata": metadata,
        }
        if self._verbose:
            event_attrs["input"] = input_str

        self._collector.start_latency(run_id, _SPAN_NAME_TOOL)
        self._collector.start_span(
            span_id=run_id,
            parent_id=parent_run_id,
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
        logger.debug(f"on_tool_end. { run_id =} { parent_run_id =} { kwargs = }")
        event_attrs = {}
        if self._verbose:
            event_attrs["output"] = output

        self._collector.end_latency(
            run_id, _SPAN_NAME_TOOL, attributes={_SPAN_NAME_LABEL: _SPAN_NAME_TOOL}
        )
        self._collector.end_span(
            span_id=run_id,
            span_name=_SPAN_NAME_TOOL,
            span_attrs={},
            event_name="tool_end",
            event_attrs=event_attrs,
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        logger.debug(f"on_tool_error. { run_id =} { parent_run_id =} { kwargs = }")
        event_attrs = {
            _ERROR_TYPE_LABEL: error.__class__.__name__,
        }

        self._collector.end_latency(
            run_id, _SPAN_NAME_TOOL, attributes={_SPAN_NAME_LABEL: _SPAN_NAME_TOOL}
        )
        self._collector.end_span(
            span_id=run_id,
            span_name=_SPAN_NAME_TOOL,
            span_attrs={},
            event_name="tool_error",
            event_attrs=event_attrs,
            ex=error,
        )
        self._collector.collect_error_count(
            {
                _SPAN_NAME_LABEL: _SPAN_NAME_TOOL,
                **event_attrs,
            },
        )

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
        logger.debug(f"on_agent_action. { run_id =} { parent_run_id =} { kwargs = }")

        span_attrs = {
            _USER_ID_LABEL: _get_user_id(metadata),
        }

        event_attrs = {
            "tool": action.tool,
            "tool_input": action.tool_input,
            _CLASS_TYPE_LABEL: action.__class__.__name__,
            "log": action.log,
            "tags": tags,
            "metadata": metadata,
        }
        if self._verbose:
            event_attrs["input"] = _parse_input(action.tool_input)

        self._collector.start_latency(run_id, _SPAN_NAME_AGENT)
        self._collector.start_span(
            span_id=run_id,
            parent_id=parent_run_id,
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
        logger.debug(f"on_agent_finish. { run_id =} { parent_run_id =} { kwargs = }")
        event_attrs = {
            _CLASS_TYPE_LABEL: finish.__class__.__name__,
            "log": finish.log,
        }
        if self._verbose:
            event_attrs["output"] = _parse_output(finish.return_values)

        self._collector.end_latency(
            run_id, _SPAN_NAME_AGENT, attributes={_SPAN_NAME_LABEL: _SPAN_NAME_AGENT}
        )
        self._collector.end_span(
            span_id=run_id,
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
        logger.debug(
            f"on_retriever_start. { run_id = } { parent_run_id = } { kwargs = } { serialized = }"
        )

        span_attrs = {
            _USER_ID_LABEL: _get_user_id(metadata),
        }
        event_attrs = {
            _CLASS_TYPE_LABEL: _get_serialized_id(serialized),
            "tags": tags,
            "metadata": metadata,
        }
        if self._verbose:
            event_attrs["query"] = query

        self._collector.start_latency(run_id, _SPAN_NAME_RETRIEVER)
        self._collector.start_span(
            span_id=run_id,
            parent_id=parent_run_id,
            span_name=_SPAN_NAME_RETRIEVER,
            event_name="retriever_start",
            span_attrs=span_attrs,
            event_attrs=event_attrs,
        )

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        logger.debug(f"on_retriever_error. {run_id=} {parent_run_id=} {kwargs=}")
        event_attrs = {
            _ERROR_TYPE_LABEL: error.__class__.__name__,
        }
        self._collector.end_latency(
            run_id,
            _SPAN_NAME_RETRIEVER,
            attributes={_SPAN_NAME_LABEL: _SPAN_NAME_RETRIEVER},
        )
        self._collector.end_span(
            span_id=run_id,
            span_name=_SPAN_NAME_RETRIEVER,
            span_attrs={},
            event_name="retriever_error",
            event_attrs=event_attrs,
            ex=error,
        )
        self._collector.collect_error_count(
            {
                _SPAN_NAME_LABEL: _SPAN_NAME_RETRIEVER,
                **event_attrs,
            },
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
        logger.debug(f"on_retriever_end. {run_id=} {parent_run_id=} {kwargs=}")
        event_attrs: Dict[str, Any] = {
            "tags": tags,
        }
        if self._verbose:
            event_attrs["docs"] = _parse_documents(documents)

        self._collector.end_latency(
            run_id,
            _SPAN_NAME_RETRIEVER,
            attributes={_SPAN_NAME_LABEL: _SPAN_NAME_RETRIEVER},
        )
        self._collector.end_span(
            span_id=run_id,
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
        logger.debug(f"on_retriever_end. {run_id=} {parent_run_id=} {kwargs=}")
        event_attrs = {
            "retry_state": f"{retry_state}",
        }
        self._collector.add_span_event(
            span_id=run_id, event_name="retry", event_attrs=event_attrs
        )


__all__ = ["GreptimeCallbackHandler"]
