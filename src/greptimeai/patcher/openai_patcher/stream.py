import time
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Tuple, Union

from openai import AsyncStream, Stream
from pydantic import BaseModel

from greptimeai import logger
from greptimeai.collector import Collector
from greptimeai.labels import (
    _COMPLETION_COST_LABEL,
    _COMPLETION_TOKENS_LABEL,
    _MODEL_LABEL,
    _OUTPUT_DISPLAY_LABEL,
    _SPAN_NAME_LABEL,
)
from greptimeai.utils.openai.token import (
    get_openai_token_cost_for_model,
    num_tokens_from_messages,
)


def _extract_tokens(chunk: BaseModel) -> str:
    if not chunk:
        return ""

    tokens = ""
    try:
        dict_ = chunk.model_dump()
        choices = dict_.get("choices", [])
        for choice in choices:
            content = choice.get("delta", {}).get("content")
            if content:  # chat completion
                tokens += content
            elif choice.get("text"):  # completion
                tokens += choice.get("text")
    except Exception as e:
        logger.error(f"Failed to extract chunk tokens: {e}")

    return tokens


def _collect_stream_item(
    item: Any, collector: Collector, span_id: str, event_name: str
) -> Tuple[str, str]:
    event_attrs = {}
    if hasattr(item, "model_dump"):
        event_attrs = item.model_dump()
    else:
        logger.warning(f"Unknown response stream type: {type(item)}")

    model_name = event_attrs.get("model", "")
    tokens = _extract_tokens(item)
    collector.add_span_event(
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
    tokens: str,
):
    num = num_tokens_from_messages(tokens)
    cost = get_openai_token_cost_for_model(model_name, num)

    span_attrs = {
        _COMPLETION_TOKENS_LABEL: num,
        _COMPLETION_COST_LABEL: cost,
        _MODEL_LABEL: model_name,
        _OUTPUT_DISPLAY_LABEL: tokens,
    }

    attrs: Dict[str, Union[str, bool]] = {
        _SPAN_NAME_LABEL: f"{span_name}[stream]",
        _MODEL_LABEL: model_name,
    }

    latency = 1000 * (time.time() - start)

    collector.collect_metrics(span_attrs=span_attrs, attrs=attrs)

    collector.record_latency(latency, attrs=attrs)

    collector.end_span(
        span_id=span_id,
        span_name=span_name,
        event_name="end",
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

            tokens, model_name = _collect_stream_item(
                item, self.collector, self.span_id, "stream"
            )
            completion_tokens += tokens
            if model_name:
                self.model_name = model_name

        _end_collect(
            collector=self.collector,
            span_id=self.span_id,
            span_name=self.span_name,
            model_name=self.model_name,
            start=self.start,
            tokens=completion_tokens,
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

            tokens, model_name = _collect_stream_item(
                item, self.collector, self.span_id, "stream"
            )
            completion_tokens += tokens
            if model_name:
                self.model_name = model_name

        _end_collect(
            collector=self.collector,
            span_id=self.span_id,
            span_name=self.span_name,
            model_name=self.model_name,
            start=self.start,
            tokens=completion_tokens,
        )
