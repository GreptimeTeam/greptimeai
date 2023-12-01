from typing import Any, Callable, Optional

from greptimeai import logger

_GREPTIMEAI_WRAPPED = "__GREPTIMEAI_WRAPPED__"


class Patchee:
    def __init__(self, obj: Any, method_name: str, span_name: str):
        self.obj = obj
        self.method_name = method_name
        self.span_name = span_name

    def __repr__(self):
        return self.span_name

    def get_func_name(self) -> str:
        return self.method_name

    def get_span_name(self) -> str:
        return self.span_name

    def get_unwrapped_func(self) -> Optional[Callable]:
        func = getattr(self.obj, self.method_name, None)
        if not func:
            logger.warning(f"function '{self.get_func_name()}' not found.")
            return None

        if hasattr(func, _GREPTIMEAI_WRAPPED):
            logger.warning(
                f"the function '{self.get_func_name()}' has already been patched."
            )
            return None
        return func

    def wrap_func(self, func: Callable):
        setattr(func, _GREPTIMEAI_WRAPPED, True)
        setattr(self.obj, self.method_name, func)
        logger.debug(f"patched '{self.span_name}'")
