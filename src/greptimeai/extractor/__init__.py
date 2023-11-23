from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional


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


class BaseExtractor(ABC):
    @abstractmethod
    def pre_extract(self, *args, **kwargs) -> Extraction:
        pass

    @abstractmethod
    def post_extract(self, resp: Dict[str, Any]) -> Extraction:
        pass

    @abstractmethod
    def get_span_name(self) -> str:
        pass

    @abstractmethod
    def get_func_name(self) -> str:
        pass

    @abstractmethod
    def get_func(self) -> Optional[Callable]:
        pass

    @abstractmethod
    def set_func(self, func: Callable):
        pass
