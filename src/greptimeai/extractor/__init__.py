from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from greptimeai import logger
from greptimeai.tracker import _GREPTIMEAI_WRAPPED


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

    def hide_field_in_event_attributes(self, field: str, verbose: bool = True):
        if not verbose and field in self.event_attributes:
            self.event_attributes[field] = "..."


class BaseExtractor(ABC):
    @abstractmethod
    def pre_extract(self, *args, **kwargs) -> Extraction:
        pass

    @abstractmethod
    def post_extract(self, resp: Any) -> Extraction:
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

    def get_unwrapped_func(self) -> Optional[Callable]:
        func = self.get_func()
        if not func:
            logger.warning(f"function '{self.get_func_name()}' not found.")
            return None

        if hasattr(func, _GREPTIMEAI_WRAPPED):
            logger.warning(
                f"the function '{self.get_func_name()}' has already been patched."
            )
            return None
        return func

    @abstractmethod
    def set_func(self, func: Callable):
        pass
