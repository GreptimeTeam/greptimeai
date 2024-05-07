from typing import Any, Callable, Optional

from greptimeai import logger

_GREPTIMEAI_WRAPPED = "__GREPTIMEAI_WRAPPED__"


class Patchee:
    def __init__(self, obj: Any, method_name: str, span_name: str, event_name: str):
        self.obj = obj
        self.method_name = method_name
        self.span_name = span_name
        self.event_name = event_name

    def __repr__(self):
        return f"<{self.event_name}>"

    def get_func_name(self) -> str:
        return self.method_name

    def _get_func(self) -> Optional[Callable]:
        if self.obj is None:
            return None

        return getattr(self.obj, self.method_name)

    def get_unwrapped_func(self) -> Optional[Callable]:
        func = self._get_func()
        if not func:
            logger.warning(f"'{self.obj}' has no function '{self.get_func_name()}'.")
            return None

        if hasattr(func, _GREPTIMEAI_WRAPPED):
            logger.warning(
                f"the function '{self.get_func_name()}' has already been patched."
            )
            return None
        return func

    def wrap_func(self, func: Callable):
        setattr(func, _GREPTIMEAI_WRAPPED, True)

        if self.obj:
            setattr(self.obj, self.method_name, func)
