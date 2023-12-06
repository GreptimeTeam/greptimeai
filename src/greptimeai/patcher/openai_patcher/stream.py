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
            tokens += choice.delta.content
    return tokens


def _extract_completion_tokens(completion: Completion) -> str:
    if not completion:
        return ""

    tokens = ""
    for choice in completion.choices:
        tokens += choice.text
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
    completion_tokens: int,
    completion_cost: float,
):
    span_attrs = {
        _COMPLETION_TOKENS_LABEL: completion_tokens,
        _COMPLETION_COST_LABEL: completion_cost,
    }

    attrs = {
        _SPAN_NAME_LABEL: span_name,
        _MODEL_LABEL: model_name,
    }

    collector.collect_metrics(span_attrs=span_attrs, attrs=attrs)
    collector.end_span(
        span_id=span_id,
        span_name=span_name,
        span_attrs=span_attrs,
        event_attrs={},
    )


def patch_stream_iter(
    stream: Stream, collector: Collector, span_id: str, span_name: str
):
    logger.info(f"stream: {stream} {dir(stream)}")

    def _iter(obj) -> Iterator[Any]:
        completion_tokens = ""
        model_name = ""

        logger.info(f"before iterator: obj: {obj} {dir(obj)}")
        for item in obj._iterator:
            logger.info(f"item: {item} {dir(item)}")
            yield item

            tokens, model_name = _collect_resp(item, collector, span_id, "stream")
            completion_tokens += tokens

        logger.info(f"after iterator: obj: {obj} {dir(obj)}")
        num = num_tokens_from_messages(completion_tokens)
        cost = get_openai_token_cost_for_model(model_name, num)
        _end_collect(
            collector=collector,
            span_id=span_id,
            span_name=span_name,
            model_name=model_name,
            completion_tokens=num,
            completion_cost=cost,
        )

    stream.__iter__ = _iter  # type: ignore

    # setattr(stream, "__iter__", _iter)


def patch_astream_aiter(
    stream: AsyncStream, collector: Collector, span_id: str, span_name: str
):
    async def _aiter(obj) -> AsyncIterator[Any]:
        completion_tokens = ""
        model_name = ""
        async for item in obj._iterator:
            yield item

            tokens, model_name = _collect_resp(item, collector, span_id, "stream")
            completion_tokens += tokens

        num = num_tokens_from_messages(completion_tokens)
        cost = get_openai_token_cost_for_model(model_name, num)

        _end_collect(
            collector=collector,
            span_id=span_id,
            span_name=span_name,
            model_name=model_name,
            completion_tokens=num,
            completion_cost=cost,
        )

    stream.__aiter__ = _aiter  # type: ignore

    # setattr(stream, "__aiter__", _aiter)
