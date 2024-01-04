from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from uuid import UUID

import langchain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.schema.document import Document
from langchain.schema.messages import BaseMessage, get_buffer_string
from langchain.schema.output import (
    ChatGeneration,
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
    _INPUT_DISPLAY_LABEL,
    _MODEL_LABEL,
    _OUTPUT_DISPLAY_LABEL,
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

_SPAN_NAME_CHAIN = "langchain_chain"
_SPAN_NAME_AGENT = "langchain_agent"
_SPAN_NAME_LLM = "langchain_llm"
_SPAN_NAME_TOOL = "langchain_tool"
_SPAN_NAME_RETRIEVER = "langchain_retriever"


def _get_user_id(metadata: Optional[Dict[str, Any]]) -> str:
    """
    get user id from metadata
    """
    return (metadata or {}).get("user_id", "")


def _get_serialized_id(serialized: Dict[str, Any]) -> Optional[str]:
    """
    get id if exist
    """
    ids = serialized.get("id")
    if ids and isinstance(ids, list) and len(ids) > 0:
        return ids[-1]
    return None


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


def _str_generations(gens: Sequence[Generation]) -> str:
    def _str_generation(gen: Generation) -> Optional[str]:
        """
        Generation, or ChatGeneration (which contains message field)
        """
        if not gen:
            return None

        info = gen.generation_info or {}
        reason = info.get("finish_reason")
        if reason in ["function_call", "tool_calls"] and isinstance(
            gen, ChatGeneration
        ):
            kwargs = gen.message.additional_kwargs
            return f"{reason}: kwargs={kwargs}"
        else:
            return gen.text

    texts = list(filter(None, [_str_generation(gen) for gen in gens]))
    return "\n".join(texts)


def _parse_generations(
    gens: Sequence[Generation],
) -> Optional[Iterable[Dict[str, Any]]]:
    """
    parse LLMResult.generations[0] to structured fields
    """

    def _parse_generation(gen: Generation) -> Optional[Dict[str, Any]]:
        """
        Generation, or ChatGeneration (which contains message field)
        """
        if not gen:
            return None

        gen.to_json()

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

    if gens and len(gens) > 0:
        return list(filter(None, [_parse_generation(gen) for gen in gens]))

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


class GreptimeCallbackHandler(BaseCallbackHandler):
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
        self.collector = Collector(
            host=host,
            database=database,
            token=token,
            source="langchain",
            source_version=langchain.__version__,
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

        event_attrs: Dict[str, Any] = {
            _CLASS_TYPE_LABEL: _get_serialized_id(serialized),
        }
        if metadata:
            event_attrs["metadata"] = metadata
        if tags:
            event_attrs["tags"] = tags

        if self._verbose:
            event_attrs["inputs"] = _parse_input(inputs)

        self.collector.start_span(
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

        self.collector.end_span(
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

        self.collector.end_span(
            span_id=run_id,
            span_name=_SPAN_NAME_CHAIN,
            span_attrs={},
            event_name="chain_error",
            event_attrs=event_attrs,
            ex=error,
        )

    def collect_llm(
        self,
        origin_inputs: Dict[str, Any],  # prompts or messages
        inputs: str,
        serialized: Dict[str, Any],
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, Any], None] = None,
        invocation_params: Union[Dict[str, Any], None] = None,
    ):
        class_type = _get_serialized_id(serialized)

        span_attrs: Dict[str, Any] = {
            _USER_ID_LABEL: _get_user_id(metadata),
        }
        if self._verbose:
            span_attrs[_INPUT_DISPLAY_LABEL] = inputs

        if invocation_params:
            model_name: str = invocation_params.get("model_name", "")
            prompt_tokens = num_tokens_from_messages(inputs, model_name)
            prompt_cost = get_openai_token_cost_for_model(model_name, prompt_tokens)
            span_attrs[_MODEL_LABEL] = model_name
            span_attrs[_PROMPT_TOKENS_LABEl] = prompt_tokens
            span_attrs[_PROMPT_COST_LABEl] = prompt_cost

        event_attrs: Dict[str, Any] = {
            "params": invocation_params,
        }
        if class_type:
            event_attrs[_CLASS_TYPE_LABEL] = class_type
        if tags:
            event_attrs["tags"] = tags
        if metadata:
            event_attrs["metadata"] = metadata
        if self._verbose:
            event_attrs.update(origin_inputs)

        self.collector.start_latency(run_id, _SPAN_NAME_LLM)
        self.collector.start_span(
            span_id=run_id,
            parent_id=parent_run_id,
            span_name=_SPAN_NAME_LLM,
            event_name="llm_start",
            span_attrs=span_attrs,
            event_attrs=event_attrs,
        )

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        invocation_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        logger.debug(
            f"on_llm_start. { run_id =} { parent_run_id =} { kwargs = } { serialized = } {invocation_params=}"
        )
        origin_inputs = {"prompts": prompts}
        inputs = " ".join(prompts)

        self.collect_llm(
            origin_inputs=origin_inputs,
            inputs=inputs,
            serialized=serialized,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            invocation_params=invocation_params,
        )

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        invocation_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        logger.debug(
            f"on_chat_model_start. { run_id =} { parent_run_id =} { kwargs = } { serialized = } {invocation_params=}"
        )
        inputs = "\n".join([get_buffer_string(message) for message in messages])
        origin_inputs = {"messages": inputs}  # BaseMessage can't be json dumped

        self.collect_llm(
            origin_inputs=origin_inputs,
            inputs=inputs,
            serialized=serialized,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            invocation_params=invocation_params,
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
        outputs = _str_generations(generations)

        output = response.llm_output or {}
        model_name: Optional[str] = output.get("model_name")
        if model_name is None:
            model_name = self.collector.get_model_in_context(run_id) or ""
        model_name = standardize_model_name(model_name)

        token_usage = output.get("token_usage", {})

        # NOTE: only cost of OpenAI model will be calculated and collected so far
        prompt_tokens, prompt_cost, completion_tokens, completion_cost = 0, 0.0, 0, 0.0
        if model_name:
            if len(token_usage) > 0:
                prompt_tokens = token_usage.get("prompt_tokens", 0)
                completion_tokens = token_usage.get("completion_tokens", 0)
                prompt_cost = get_openai_token_cost_for_model(model_name, prompt_tokens)
                completion_cost = get_openai_token_cost_for_model(
                    model_name, completion_tokens, is_completion=True
                )
            else:  # streaming
                completion_tokens = num_tokens_from_messages(outputs, model_name)
                completion_cost = get_openai_token_cost_for_model(
                    model_name, completion_tokens, is_completion=True
                )

        attributes = {_MODEL_LABEL: model_name, _SPAN_NAME_LABEL: _SPAN_NAME_LLM}
        self.collector._collect_metrics(
            prompt_tokens=prompt_tokens,
            prompt_cost=prompt_cost,
            completion_tokens=completion_tokens,
            completion_cost=completion_cost,
            attrs=attributes,
        )

        common_attrs: Dict[str, Any] = {}

        if model_name:
            common_attrs[_MODEL_LABEL] = model_name
        if prompt_tokens > 0:
            common_attrs[_PROMPT_TOKENS_LABEl] = prompt_tokens
        if prompt_cost > 0:
            common_attrs[_PROMPT_COST_LABEl] = prompt_cost
        if completion_tokens > 0:
            common_attrs[_COMPLETION_TOKENS_LABEL] = completion_tokens
        if completion_cost > 0:
            common_attrs[_COMPLETION_COST_LABEL] = completion_cost

        span_attrs = {
            _OUTPUT_DISPLAY_LABEL: outputs,
            **common_attrs,
        }

        event_attrs = common_attrs.copy()
        if self._verbose:
            event_attrs["outputs"] = _parse_generations(generations)

        self.collector.end_latency(run_id, _SPAN_NAME_LLM, attributes=attributes)
        self.collector.end_span(
            span_id=run_id,
            span_name=_SPAN_NAME_LLM,
            span_attrs=span_attrs,
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
        logger.debug(f"on_llm_error. { run_id =} { parent_run_id =} { kwargs =}")
        event_attrs = {
            _ERROR_TYPE_LABEL: error.__class__.__name__,
        }

        self.collector.end_span(
            span_id=run_id,
            span_name=_SPAN_NAME_LLM,
            span_attrs={},
            event_name="llm_error",
            event_attrs=event_attrs,
            ex=error,
        )
        self.collector.discard_latency(run_id, _SPAN_NAME_LLM)

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

        self.collector.add_span_event(
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

        self.collector.start_span(
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

        self.collector.end_span(
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

        self.collector.end_span(
            span_id=run_id,
            span_name=_SPAN_NAME_TOOL,
            span_attrs={},
            event_name="tool_error",
            event_attrs=event_attrs,
            ex=error,
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

        self.collector.start_span(
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

        self.collector.end_span(
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

        self.collector.start_span(
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
        self.collector.end_span(
            span_id=run_id,
            span_name=_SPAN_NAME_RETRIEVER,
            span_attrs={},
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
        logger.debug(f"on_retriever_end. {run_id=} {parent_run_id=} {kwargs=}")
        event_attrs: Dict[str, Any] = {
            "tags": tags,
        }
        if self._verbose:
            event_attrs["docs"] = _parse_documents(documents)

        self.collector.end_span(
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
        self.collector.add_span_event(
            span_id=run_id, event_name="retry", event_attrs=event_attrs
        )


__all__ = ["GreptimeCallbackHandler"]
