from typing import Any, Dict, Optional, cast

from opentelemetry.util.types import Attributes

from greptimeai import (
    _COMPLETION_COST_LABEL,
    _COMPLETION_TOKENS_LABEL,
    _ERROR_TYPE_LABEL,
    _MODEL_LABEL,
    _SPAN_NAME_LABEL,
    _PROMPT_COST_LABEl,
    _PROMPT_TOKENS_LABEl,
)
from greptimeai.collection import Collector
from greptimeai.extractor import Extraction


class BaseTracker:
    """
    base tracker to collect metrics and traces
    """

    def __init__(
        self,
        service_name: str,
        host: str = "",
        database: str = "",
        token: str = "",
    ):
        self._collector = Collector(
            service_name=service_name, host=host, database=database, token=token
        )

    def start_span(self, span_name: str, extraction: Extraction) -> str:
        span_id = self._collector.start_span(
            span_id=None,
            parent_id=None,
            span_name=span_name,
            event_name="start",
            span_attrs=extraction.span_attributes,
            event_attrs=extraction.event_attributes,
        )
        return cast(str, span_id)

    def end_span(
        self,
        span_id: str,
        span_name: str,
        extraction: Extraction,
        ex: Optional[Exception] = None,
    ):
        self._collector.end_span(
            span_id=span_id,
            span_name=span_name,
            event_name="end",
            span_attrs=extraction.span_attributes,
            event_attrs=extraction.event_attributes,
            ex=ex,
        )

    def collect_error_count(
        self,
        model_name: Optional[str],
        span_name: str,
        ex: Exception,
    ):
        attributes = {
            _ERROR_TYPE_LABEL: ex.__class__.__name__,
            _SPAN_NAME_LABEL: span_name,
        }
        if model_name:
            attributes[_MODEL_LABEL] = model_name

        self._collector.collect_error_count(attributes=attributes)

    def collect_metrics(
        self,
        span_attrs: Dict[str, Any],
        attrs: Optional[Attributes],
    ):
        """
        Collects metrics for the given extraction and attributes.

        Args:
            extraction (Extraction): The extraction object.
            attrs (Optional[Attributes]): Optional attributes.

        Returns:
            None
        """
        prompt_tokens = span_attrs.get(_PROMPT_TOKENS_LABEl, 0)
        prompt_cost = span_attrs.get(_PROMPT_COST_LABEl, 0)
        completion_tokens = span_attrs.get(_COMPLETION_TOKENS_LABEL, 0)
        completion_cost = span_attrs.get(_COMPLETION_COST_LABEL, 0)

        self._collector.collect_metrics(
            prompt_tokens=prompt_tokens,
            prompt_cost=prompt_cost,
            completion_tokens=completion_tokens,
            completion_cost=completion_cost,
            attrs=attrs,
        )

    def record_latency(self, latency: float, attributes: Optional[Attributes] = None):
        self._collector.record_latency(latency, attributes=attributes)
