from typing import Any, Callable, Dict, Optional

from greptimeai import (
    _COMPLETION_COST_LABEL,
    _COMPLETION_TOKENS_LABEL,
    _MODEL_LABEL,
    _USER_ID_LABEL,
    _PROMPT_COST_LABEl,
    _PROMPT_TOKENS_LABEl,
)
from greptimeai.extractor import BaseExtractor, Extraction
from greptimeai.utils.openai.token import get_openai_token_cost_for_model

_X_USER_ID = "x-user-id"


class Extractor(BaseExtractor):
    def __init__(self, obj: object, method_name: str, span_name: str):
        self.obj = obj
        self.span_name = span_name
        self.method_name = method_name

    @staticmethod
    def extract_usage(
        model: Optional[str], usage: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        res = {}

        if not usage or not model:
            return res

        prompt_tokens = usage.get("prompt_tokens", 0)
        res[_PROMPT_TOKENS_LABEl] = prompt_tokens
        res[_PROMPT_COST_LABEl] = get_openai_token_cost_for_model(
            model, prompt_tokens, False
        )

        completion_tokens = usage.get("completion_tokens", 0)
        res[_COMPLETION_TOKENS_LABEL] = completion_tokens
        res[_COMPLETION_COST_LABEL] = get_openai_token_cost_for_model(
            model, completion_tokens, True
        )
        return res

    def pre_extract(self, *args, **kwargs) -> Extraction:
        """
        extract _MODEL_LABEL, _USER_ID_LABEL for span attributes
        merge kwargs and args into event attributes
        Args:

            kwargs:
                extra_headers in kwargs: _X_USER_ID is a custom header that is used to identify the user,
                which has higher priority than the 'user' in the kwargs
        """
        user_id = kwargs.get("user", None)
        extra_headers = kwargs.get("extra_headers", None)
        if extra_headers and _X_USER_ID in extra_headers:
            user_id = extra_headers[_X_USER_ID]

        span_attrs = {
            _MODEL_LABEL: kwargs.get("model", None),
            _USER_ID_LABEL: user_id,
        }

        event_attrs = {**kwargs}
        if len(args) > 0:
            event_attrs["args"] = args

        return Extraction(span_attributes=span_attrs, event_attributes=event_attrs)

    def post_extract(self, resp: Dict[str, Any]) -> Extraction:
        """
        extract for span attributes:
                _MODEL_LABEL
                _COMPLETION_COST_LABEL
                _COMPLETION_TOKENS_LABEL
                _PROMPT_COST_LABEl
                _PROMPT_TOKENS_LABEl

        merge usage into resp as event attributes

        Args:

            resp: response from openai api, which is the result of calling model_dump()
        """
        usage = resp.get("usage", {})
        model = resp.get("model")
        if usage and model:
            usage = Extractor.extract_usage(model, usage)

        span_attrs = {
            _MODEL_LABEL: model,
            **usage,
        }

        event_attrs = resp.copy()
        event_attrs["usage"] = usage
        return Extraction(span_attributes=span_attrs, event_attributes=event_attrs)

    def get_span_name(self) -> str:
        return self.span_name

    def get_func(self) -> Optional[Callable]:
        return getattr(self.obj, self.method_name, None)

    def set_func(self, func: Callable):
        setattr(self.obj, self.method_name, func)
