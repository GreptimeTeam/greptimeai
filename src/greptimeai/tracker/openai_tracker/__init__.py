import functools
import time
from typing import Any, Dict, Optional, Union

from openai import OpenAI
from opentelemetry.util.types import Attributes

from greptimeai import (
    _COMPLETION_COST_LABEL,
    _COMPLETION_TOKENS_LABEL,
    _ERROR_TYPE_LABEL,
    _MODEL_LABEL,
    _SPAN_NAME_LABEL,
    _PROMPT_COST_LABEl,
    _PROMPT_TOKENS_LABEl,
    logger,
)
from greptimeai.tracker import _GREPTIMEAI_WRAPPED, BaseTracker
from greptimeai.tracker.openai_tracker.chat_completion import ChatCompletionExtractor
from greptimeai.tracker.openai_tracker.completion import CompletionExtractor
from greptimeai.tracker.openai_tracker.embedding import EmbeddingExtractor

__all__ = ["Extractor"]
Extractor = Union[
    ChatCompletionExtractor,
    CompletionExtractor,
    EmbeddingExtractor,
]


def setup(
    host: str = "",
    database: str = "",
    token: str = "",
    client: Optional[OpenAI] = None,
):
    """
    patch openai main functions automatically.
    host, database and token is to setup the place to store the data, and the authority.
    They MUST BE set explicitly by passing parameters or system environment variables.
    Args:
        host: if None or empty string, GREPTIMEAI_HOST environment variable will be used.
        database: if None or empty string, GREPTIMEAI_DATABASE environment variable will be used.
        token: if None or empty string, GREPTIMEAI_TOKEN environment variable will be used.
        client: if None, then openai module-level client will be patched.
                Important: We highly recommend instantiating client instances
                instead of relying on the global client.
    """
    tracker = OpenaiTracker(host, database, token)
    tracker.setup(client)


class OpenaiTracker(BaseTracker):
    def __init__(
        self,
        host: str = "",
        database: str = "",
        token: str = "",
        verbose: bool = True,
    ):
        super().__init__(
            service_name="openai", host=host, database=database, token=token
        )
        self._verbose = verbose

    def setup(self, client: Optional[OpenAI] = None):
        self._patch(ChatCompletionExtractor(self._verbose, client))
        self._patch(CompletionExtractor(self._verbose, client))
        self._patch(EmbeddingExtractor(self._verbose, client))

    def _patch(
        self,
        extractor: Extractor,
    ):
        """
        wrap the method name of this obj with trace and metrics collection logic.

        if this obj does not contain this method name, patch will do nothing.
        if this method has already been patched, then it won't be patched multiple times.

        NOTE:
        pre_extractor and post_extractor should return two tuple of (span_attrs, event_attrs),
        and _MODEL_LABEL is required in span_attrs.

        Args:
            obj: OpenAI client, or module level client
            method_name: the method name of the object
            span_name: identify different span name
            pre_extractor: extract span attributes and event attributes from args, kwargs
            post_extractor: extract span attributes and event attributes from response
        """
        if not hasattr(extractor.obj, extractor.method_name):
            logger.warning(
                f"'{extractor.method_name}' attribute not found from the object."
            )
            return

        func = getattr(extractor.obj, extractor.method_name)

        if hasattr(func, _GREPTIMEAI_WRAPPED):
            logger.warning(f"no need to patch {extractor.method_name} multiple times.")
            return

        # TODO(yuanbohan): to support: stream, async
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ex, resp = None, None
            req_span_attrs, req_event_attrs = extractor.pre_extractor(args, **kwargs)
            span_id = self._collector.start_span(
                span_id=None,
                parent_id=None,
                span_name=extractor.span_name,
                event_name="start",
                span_attrs=req_span_attrs,
                event_attrs=req_event_attrs,
            )
            common_attrs = {_SPAN_NAME_LABEL: extractor.span_name}
            start = time.time()
            try:
                resp = func(*args, **kwargs)
            except Exception as e:
                ex = e
                self._collector._llm_error_count.add(
                    1,
                    {
                        _MODEL_LABEL: req_span_attrs.get(_MODEL_LABEL, ""),
                        _ERROR_TYPE_LABEL: ex.__class__.__name__,
                        **common_attrs,
                    },
                )
                raise ex
            finally:
                latency = 1000 * (time.time() - start)
                resp_span_attrs, resp_event_attrs = extractor.post_extractor(resp)
                attrs = {
                    _MODEL_LABEL: resp_span_attrs.get(_MODEL_LABEL, ""),
                    **common_attrs,
                }
                self._collector.record_latency(latency, attributes=attrs)

                self._collector.end_span(
                    span_id=span_id,  # type: ignore
                    span_name=extractor.span_name,
                    event_name="end",
                    span_attrs=resp_span_attrs,
                    event_attrs=resp_event_attrs,
                    ex=ex,
                )

                self._collect_metrics(attrs=attrs, span_attrs=resp_span_attrs)
            return resp

        setattr(wrapper, _GREPTIMEAI_WRAPPED, True)
        setattr(extractor.obj, extractor.method_name, wrapper)

    def _collect_metrics(self, attrs: Optional[Attributes], span_attrs: Dict[str, Any]):
        """
        Args:

            attrs: for OTLP collection
            span_attrs: attributes to get useful metrics
        """
        prompt_tokens = span_attrs.get(_PROMPT_TOKENS_LABEl, 0)
        prompt_cost = span_attrs.get(_PROMPT_COST_LABEl, 0)
        completion_tokens = span_attrs.get(_COMPLETION_TOKENS_LABEL, 0)
        completion_cost = span_attrs.get(_COMPLETION_COST_LABEL, 0)
        if not (prompt_tokens or prompt_cost or completion_tokens or completion_cost):
            logger.warning(f"no need to collect empty metrics for openai {attrs}")
            return

        self._collector.collect_metrics(
            prompt_tokens=prompt_tokens,
            prompt_cost=prompt_cost,
            completion_tokens=completion_tokens,
            completion_cost=completion_cost,
            attrs=attrs,
        )
