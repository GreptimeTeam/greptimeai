from typing import Any, Callable, Dict, Optional, Tuple, Union

from greptimeai import (
    _MODEL_LABEL,
    _USER_ID_LABEL,
)


def pre_extractor(
    req_call_func: Callable,
    verbose: bool,
) -> Callable:
    param, exec_fun = req_call_func()

    def _pre_extractor(
        args,
        *,
        model: str,
        user: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        span_attrs = {
            _MODEL_LABEL: model,
            _USER_ID_LABEL: user,
        }

        event_attrs = {
            _MODEL_LABEL: model,
            **kwargs,
        }
        if verbose:
            if param and kwargs[param]:
                event_attrs[param] = exec_fun(kwargs[param])

        if args and len(args) > 0:
            event_attrs["args"] = args

        return (span_attrs, event_attrs)

    return _pre_extractor


def post_extractor(
    res_call_func: Callable,
) -> Callable:
    def _post_extractor(
        resp: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        attrs = res_call_func(resp)
        usage = {}
        if "usage" in attrs:
            usage = attrs["usage"]

        span_attrs = {
            _MODEL_LABEL: resp.model,
            **usage,
        }

        event_attrs = resp.model_dump()
        for key in attrs:
            event_attrs[key] = attrs[key]

        return (span_attrs, event_attrs)

    return _post_extractor
