from typing import Any, AsyncIterator, Dict, Iterator, Tuple

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
    for choice in chunk.choices:
        if choice.delta.content:
            tokens = f"{tokens}\n{choice.delta.content}"
    return tokens


def _extract_completion_tokens(completion: Completion) -> str:
    if not completion:
        return ""

    tokens = ""
    for choice in completion.choices:
        tokens = f"{tokens}\n{choice.text}"
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
    resp: Any, collector: Collector, span_id: str, span_name: str, event_name: str
) -> Tuple[int, float]:
    event_attrs = _extract_resp(resp)
    model_name = event_attrs.get("model", "")

    tokens = _extract_tokens(resp)
    num = num_tokens_from_messages(tokens, model_name)
    cost = get_openai_token_cost_for_model(model_name, num, True)
    span_attrs = {
        _COMPLETION_TOKENS_LABEL: tokens,
        _COMPLETION_COST_LABEL: cost,
    }
    attrs = {
        _SPAN_NAME_LABEL: span_name,
        _MODEL_LABEL: model_name,
    }
    collector.collect_metrics(span_attrs=span_attrs, attrs=attrs)
    collector._collector.add_span_event(
        span_id=span_id,
        event_name=event_name,
        event_attrs=event_attrs,
    )

    return num, cost


def _end_collect(
    collector: Collector,
    span_id: str,
    span_name: str,
    completion_tokens: int,
    completion_cost: float,
):
    span_attrs = {
        _COMPLETION_TOKENS_LABEL: completion_tokens,
        _COMPLETION_COST_LABEL: completion_cost,
    }
    collector.end_span(
        span_id=span_id,
        span_name=span_name,
        span_attrs=span_attrs,
        event_attrs={},
    )


def patch_stream_iter(
    stream: Stream, collector: Collector, span_id: str, span_name: str
):
    def _iter(obj) -> Iterator[Any]:
        completion_tokens, completion_cost = 0, 0.0
        for item in obj._iterator:
            yield item

            num, cost = _collect_resp(item, collector, span_id, span_name, "stream")
            completion_tokens += num
            completion_cost += cost

        _end_collect(
            collector=collector,
            span_id=span_id,
            span_name=span_name,
            completion_tokens=completion_tokens,
            completion_cost=completion_cost,
        )

    setattr(stream, "__iter__", _iter)


def patch_astream_aiter(
    stream: AsyncStream, collector: Collector, span_id: str, span_name: str
):
    async def _aiter(obj) -> AsyncIterator[Any]:
        completion_tokens, completion_cost = 0, 0.0
        async for item in obj._iterator:
            yield item

            num, cost = _collect_resp(
                item, collector, span_id, span_name, "async_stream"
            )
            completion_tokens += num
            completion_cost += cost

        _end_collect(
            collector=collector,
            span_id=span_id,
            span_name=span_name,
            completion_tokens=completion_tokens,
            completion_cost=completion_cost,
        )

    setattr(stream, "__aiter__", _aiter)
