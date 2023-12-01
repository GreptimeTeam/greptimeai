from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from uuid import UUID

from opentelemetry.util.types import Attributes

from greptimeai import (
    _COMPLETION_COST_LABEL,
    _COMPLETION_TOKENS_LABEL,
    _ERROR_TYPE_LABEL,
    _PROMPT_COST_LABEl,
    _PROMPT_TOKENS_LABEl,
)
from greptimeai.collection import Collector
from greptimeai.extractor import Extraction

_GREPTIMEAI_WRAPPED = "__GREPTIMEAI_WRAPPED__"


class BaseTracker(ABC):
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

    @abstractmethod
    def setup(self, _client: Optional[Any] = None):
        pass

    def start_span(
        self, span_name: str, extraction: Extraction
    ) -> Union[UUID, str, None]:
        return self._collector.start_span(
            span_id=None,
            parent_id=None,
            span_name=span_name,
            event_name="start",
            span_attrs=extraction.span_attributes,
            event_attrs=extraction.event_attributes,
        )

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

    def add_span_event(
        self,
        span_id: str,
        event_name: str,
        event_attr: Dict[str, Any],
    ):
        self._collector.add_span_event(
            span_id=span_id,
            event_name=event_name,
            event_attrs=event_attr,
        )

    def collect_error_count(self, ex: Exception, attrs: Dict[str, Any]):
        """
        Collects error count for a given extraction and exception.

        Args:
            ex (Exception): The exception object.
            attrs (Dict[str, Any]): Additional attributes.

        Returns:
            None
        """
        attributes = {
            _ERROR_TYPE_LABEL: ex.__class__.__name__,
            **attrs,
        }
        self._collector.collect_error_count(attributes=attributes)

    def collect_metrics(self, extraction: Extraction, attrs: Optional[Attributes]):
        """
        Collects metrics for the given extraction and attributes.

        Args:
            extraction (Extraction): The extraction object.
            attrs (Optional[Attributes]): Optional attributes.

        Returns:
            None
        """
        span_attrs = extraction.span_attributes
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
