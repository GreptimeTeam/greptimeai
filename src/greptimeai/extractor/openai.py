from typing import Any, Dict, Optional, Union

from openai._response import APIResponse
from pydantic import BaseModel
from typing_extensions import override

from greptimeai import (
    _COMPLETION_COST_LABEL,
    _COMPLETION_TOKENS_LABEL,
    _MODEL_LABEL,
    _USER_ID_LABEL,
    _PROMPT_COST_LABEl,
    _PROMPT_TOKENS_LABEl,
    logger,
)
from greptimeai.extractor import BaseExtractor, Extraction
from greptimeai.utils.openai.token import get_openai_token_cost_for_model

_X_USER_ID = "x-user-id"
# TODO(yuanbohan): support more x headers in extra_headers. e.g. x-trace-id, x-span-id, x-span-context


class OpenaiExtractor(BaseExtractor):
    @staticmethod
    def parse_raw_response(resp: APIResponse) -> Dict[str, Any]:
        def _http_info() -> Dict[str, Any]:
            headers = {}
            for k, v in resp.headers.items():
                headers[k] = v

            return {
                "headers": headers,
                "status_code": resp.status_code,
                "url": str(resp.url),
                "method": resp.method,
            }

        result: Dict[str, Any] = {}
        try:
            obj: BaseModel = resp.parse()
            result = obj.model_dump()
            result["http_info"] = _http_info()
        except Exception as e:
            logger.error(f"Failed to parse response, {e}")

        return result

    @staticmethod
    def get_user_id(**kwargs) -> Optional[str]:
        user_id = kwargs.get("user", None)
        extra_headers = kwargs.get("extra_headers", None)
        if extra_headers and _X_USER_ID in extra_headers:
            user_id = extra_headers[_X_USER_ID]
        return user_id

    @staticmethod
    def extract_usage(
        model: Optional[str], usage: Optional[Dict[str, int]]
    ) -> Dict[str, Union[float, int]]:
        res: Dict[str, Union[float, int]] = {}

        if not usage or not model:
            return res

        prompt_tokens: int = usage.get("prompt_tokens", 0)
        if prompt_tokens > 0:
            res[_PROMPT_TOKENS_LABEl] = prompt_tokens
            res[_PROMPT_COST_LABEl] = get_openai_token_cost_for_model(
                model, prompt_tokens, False
            )

        completion_tokens = usage.get("completion_tokens", 0)
        if completion_tokens > 0:
            res[_COMPLETION_TOKENS_LABEL] = completion_tokens
            res[_COMPLETION_COST_LABEL] = get_openai_token_cost_for_model(
                model, completion_tokens, True
            )
        return res

    @override
    def pre_extract(self, *args, **kwargs) -> Extraction:
        """
        extract _MODEL_LABEL, _USER_ID_LABEL for span attributes
        merge kwargs and args into event attributes
        Args:

            kwargs:
                extra_headers in kwargs: _X_USER_ID is a custom header that is used to identify the user,
                which has higher priority than the 'user' in the kwargs
        """
        span_attrs = {}
        if "model" in kwargs:
            span_attrs[_MODEL_LABEL] = kwargs["model"]

        user_id = OpenaiExtractor.get_user_id(**kwargs)
        if user_id:
            span_attrs[_USER_ID_LABEL] = user_id

        event_attrs = {**kwargs}
        if len(args) > 0:
            event_attrs["args"] = args

        return Extraction(span_attributes=span_attrs, event_attributes=event_attrs)

    @override
    def post_extract(self, resp: Any) -> Extraction:
        """
        extract for span attributes:
          - _MODEL_LABEL
          - _COMPLETION_COST_LABEL
          - _COMPLETION_TOKENS_LABEL
          - _PROMPT_COST_LABEl
          - _PROMPT_TOKENS_LABEl

        merge usage into resp as event attributes

        Args:

            resp: inherit from the BaseModel class, or instance of APIResponse class
        """
        try:
            dict: Dict[str, Any] = {}
            if isinstance(resp, APIResponse):
                dict = OpenaiExtractor.parse_raw_response(resp)
                logger.debug(f"after parse_raw_response: {dict=}")
            else:
                dict = resp.model_dump()
        except Exception as e:
            logger.error(f"Failed to extract response {resp}: {e}")
            dict = {}

        span_attrs = {}

        model = dict.get("model", None)
        if model:
            span_attrs[_MODEL_LABEL] = model

            usage = dict.get("usage", {})
            if usage:
                usage = OpenaiExtractor.extract_usage(model, usage)
                span_attrs.update(usage)
                dict["usage"] = usage

        return Extraction(span_attributes=span_attrs, event_attributes=dict)