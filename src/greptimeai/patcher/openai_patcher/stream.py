import time
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Tuple, Union

from openai import AsyncStream, Stream
from openai.types.chat import ChatCompletionChunk
from openai.types.completion import Completion

from greptimeai import logger
from greptimeai.collector import Collector
from greptimeai.labels import (
    _COMPLETION_COST_LABEL,
    _COMPLETION_TOKENS_LABEL,
    _MODEL_LABEL,
    _SPAN_NAME_LABEL,
)
from greptimeai.utils.openai.token import (
    get_openai_token_cost_for_model,
    num_tokens_from_messages,
)


def _extract_resp(resp: Any) -> Dict[str, Any]:
    try:
        if hasattr(resp, "model_dump"):
            return resp.model_dump()
        else:
            logger.warning(f"Unknown response stream type: {type(resp)}")
            return {}
    except Exception as e:
        logger.error(f"Failed to extract response: {e}")
        return {}


def _extract_chat_completion_chunk_tokens(chunk: ChatCompletionChunk) -> str:
    if not chunk:
        return ""

    tokens = ""
    try:
        for choice in chunk.choices:
            if choice.delta.content:
                tokens += "\n" + choice.delta.content
    except Exception as e:
        logger.error(f"Failed to extract chat completion chunk tokens: {e}")

    return tokens


def _extract_completion_tokens(completion: Completion) -> str:
    if not completion:
        return ""

    tokens = ""

    try:
        for choice in completion.choices:
            tokens += "\n" + choice.text
    except Exception as e:
        logger.error(f"Failed to extract completion tokens: {e}")

    return tokens


def _extract_tokens(resp: Any) -> str:
    if isinstance(resp, ChatCompletionChunk):
        return _extract_chat_completion_chunk_tokens(resp)
    elif isinstance(resp, Completion):
        return _extract_completion_tokens(resp)
    else:
        logger.warning(f"Unsupported response stream type: {type(resp)}")
        return ""


def _collect_resp(
    resp: Any, collector: Collector, span_id: str, event_name: str
) -> Tuple[str, str]:
    event_attrs = _extract_resp(resp)
    model_name = event_attrs.get("model", "")
    tokens = _extract_tokens(resp)
    collector._collector.add_span_event(
        span_id=span_id,
        event_name=event_name,
        event_attrs=event_attrs,
    )

    return tokens, model_name


def _end_collect(
    collector: Collector,
    span_id: str,
    span_name: str,
    model_name: str,
    start: float,
    completion_tokens: int,
    completion_cost: float,
):
    span_attrs = {
        _COMPLETION_TOKENS_LABEL: completion_tokens,
        _COMPLETION_COST_LABEL: completion_cost,
    }

    attrs: Dict[str, Union[str, bool]] = {
        _SPAN_NAME_LABEL: f"{span_name}[stream]",
        _MODEL_LABEL: model_name,
    }

    latency = 1000 * (time.time() - start)

    collector.collect_metrics(span_attrs=span_attrs, attrs=attrs)

    collector._collector.record_latency(latency, attributes=attrs)

    collector.end_span(
        span_id=span_id,
        span_name=span_name,
        span_attrs=span_attrs,
        event_attrs={},
    )


class Stream_(Stream):
    def __init__(
        self,
        stream: Stream,
        collector: Collector,
        span_id: str,
        span_name: str,
        start: float,
        model_name: Optional[str],
    ):
        self.stream = stream
        self.collector = collector
        self.span_id = span_id
        self.span_name = span_name
        self.start = start
        self.model_name = model_name or ""

    def __iter__(self) -> Iterator[Any]:
        completion_tokens = ""

        for item in self.stream:
            yield item

            tokens, model_name = _collect_resp(
                item, self.collector, self.span_id, "stream"
            )
            completion_tokens += tokens
            if model_name:
                self.model_name = model_name

        num = num_tokens_from_messages(completion_tokens)
        cost = get_openai_token_cost_for_model(self.model_name, num)
        _end_collect(
            collector=self.collector,
            span_id=self.span_id,
            span_name=self.span_name,
            model_name=self.model_name,
            start=self.start,
            completion_tokens=num,
            completion_cost=cost,
        )


class AsyncStream_(AsyncStream):
    def __init__(
        self,
        astream: AsyncStream,
        collector: Collector,
        span_id: str,
        span_name: str,
        start: float,
        model_name: Optional[str],
    ):
        self.astream = astream
        self.collector = collector
        self.span_id = span_id
        self.span_name = span_name
        self.start = start
        self.model_name = model_name or ""

    async def __aiter__(self) -> AsyncIterator[Any]:
        completion_tokens = ""

        async for item in self.astream:
            yield item

            tokens, model_name = _collect_resp(
                item, self.collector, self.span_id, "stream"
            )
            completion_tokens += tokens
            if model_name:
                self.model_name = model_name

        num = num_tokens_from_messages(completion_tokens)
        cost = get_openai_token_cost_for_model(self.model_name, num)
        _end_collect(
            collector=self.collector,
            span_id=self.span_id,
            span_name=self.span_name,
            model_name=self.model_name,
            start=self.start,
            completion_tokens=num,
            completion_cost=cost,
        )
