from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from greptimeai.labels import _MODEL_LABEL


class Extraction:
    def __init__(
        self,
        span_attributes: Dict[str, Any] = {},
        event_attributes: Dict[str, Any] = {},
    ):
        self.span_attributes = span_attributes
        self.event_attributes = event_attributes

    def update_span_attributes(self, attrs: Dict[str, Any]):
        self.span_attributes.update(attrs)

    def update_event_attributes(self, attrs: Dict[str, Any]):
        self.event_attributes.update(attrs)

    def get_model_name(self) -> Optional[str]:
        return self.span_attributes.get(
            _MODEL_LABEL, None
        ) or self.event_attributes.get(_MODEL_LABEL, None)


class BaseExtractor(ABC):
    @abstractmethod
    def pre_extract(self, *args, **kwargs) -> Extraction:
        pass

    @abstractmethod
    def post_extract(self, resp: Any) -> Extraction:
        pass
