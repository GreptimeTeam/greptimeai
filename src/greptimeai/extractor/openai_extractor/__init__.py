from typing import Any, Dict, Optional, Tuple, Union

from pydantic import BaseModel
from typing_extensions import override

from greptimeai import logger
from greptimeai.extractor import BaseExtractor, Extraction
from greptimeai.labels import (
    _COMPLETION_COST_LABEL,
    _COMPLETION_TOKENS_LABEL,
    _INPUT_DISPLAY_LABEL,
    _MODEL_LABEL,
    _OUTPUT_DISPLAY_LABEL,
    _USER_ID_LABEL,
    _PROMPT_COST_LABEl,
    _PROMPT_TOKENS_LABEl,
)
from greptimeai.utils.openai.token import (
    extract_chat_inputs,
    extract_chat_outputs,
    get_openai_token_cost_for_model,
    num_tokens_from_messages,
)

_EXTRA_HEADERS_KEY = "extra_headers"  # this is the original openai parameter
_EXTRA_HEADERS_X_USER_ID_KEY = "x-user-id"
_EXTRA_HEADERS_X_TRACE_ID_KEY = "x-trace-id"
_EXTRA_HEADERS_X_SPAN_ID_KEY = "x-span-id"

_GREPTIMEAI_USER_KEY = "user_id"
_GREPTIMEAI_TRACE_KEY = "trace_id"
_GREPTIMEAI_SPAN_KEY = "span_id"

_OPENAI_USER_KEY = "user"  # this is the original openai parameter


class OpenaiExtractor(BaseExtractor):
    def __init__(self, tokens_calculation_needed: bool):
        self.tokens_calculation_needed = tokens_calculation_needed

    @staticmethod
    def is_stream(**kwargs) -> bool:
        stream = kwargs.get("stream")
        if isinstance(stream, bool):
            return stream
        elif isinstance(stream, str):
            return stream.lower() == "true"
        else:
            return False

    @staticmethod
    def extract_req_tokens(**kwargs) -> Optional[Union[str, list]]:
        """
        NOTE: only for completion and chat completion so far.
        TODO(ynanbohan): better way to extract req tokens
        """
        if kwargs.get("prompt"):
            prompt = kwargs["prompt"]
            if isinstance(prompt, str):
                return prompt
            elif isinstance(prompt, list) and all(isinstance(p, str) for p in prompt):
                return " ".join(prompt)
            else:
                logger.warning(f"Failed to extract req tokens from {prompt=}")
                return None
        elif kwargs.get("messages"):
            return kwargs["messages"]
        else:
            logger.warning(f"Failed to extract req tokens from {kwargs=}")
            return None

    @staticmethod
    def pop_out_keyword_args(kwargs: Dict[str, Any]):
        """
        pop out the keyword args which are not supported by openai
        """
        kwargs.pop(_GREPTIMEAI_USER_KEY, None)
        kwargs.pop(_GREPTIMEAI_TRACE_KEY, None)
        kwargs.pop(_GREPTIMEAI_SPAN_KEY, None)

    @staticmethod
    def update_trace_info(kwargs: Dict[str, Any], trace_id: str, span_id: str):
        attrs = {
            _EXTRA_HEADERS_X_TRACE_ID_KEY: trace_id,
            _EXTRA_HEADERS_X_SPAN_ID_KEY: span_id,
        }

        extra_headers: Dict[str, Any] = kwargs.get(_EXTRA_HEADERS_KEY, {})
        extra_headers.update(attrs)

        kwargs[_EXTRA_HEADERS_KEY] = extra_headers

    @staticmethod
    def get_trace_info(**kwargs) -> Optional[Tuple[str, str]]:
        """
        NOTE: key in headers has higher priority than the key in kwargs
        """
        trace_id = kwargs.get(_GREPTIMEAI_TRACE_KEY, "")
        span_id = kwargs.get(_GREPTIMEAI_SPAN_KEY, "")

        extra_headers = kwargs.get(_EXTRA_HEADERS_KEY)
        if extra_headers:
            trace_id = extra_headers.get(_EXTRA_HEADERS_X_TRACE_ID_KEY, "")
            span_id = extra_headers.get(_EXTRA_HEADERS_X_SPAN_ID_KEY, "")

        if trace_id or span_id:
            return trace_id, span_id

        return None

    @staticmethod
    def parse_raw_response(resp) -> Dict[str, Any]:
        def _http_info() -> Dict[str, Any]:
            headers = {k: v for k, v in resp.headers.items()}

            return {
                "headers": headers,
                "status_code": resp.status_code,
                "url": str(resp.url),
                "method": resp.method,
            }

        result: Dict[str, Any] = {}
        try:
            obj: BaseModel = resp.parse()
            result = obj.model_dump(exclude_unset=True)
            result["http_info"] = _http_info()
        except Exception as e:
            logger.error(f"Failed to parse response, {e}")

        return result

    @staticmethod
    def get_user_id(**kwargs) -> Optional[str]:
        """
        the upper has higher priority:
        - _X_USER_ID_KEY in extra_headers
        - _GREPTIMEAI_USER_KEY in kwargs
        - _OPENAI_USER_KEY in kwargs
        """
        user_id = kwargs.get(_GREPTIMEAI_USER_KEY) or kwargs.get(_OPENAI_USER_KEY)
        extra_headers = kwargs.get(_EXTRA_HEADERS_KEY)
        if extra_headers and _EXTRA_HEADERS_X_USER_ID_KEY in extra_headers:
            user_id = extra_headers[_EXTRA_HEADERS_X_USER_ID_KEY]
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
        extract fields for span attributes:
          - _MODEL_LABEL
          - _USER_ID_LABEL
          - _PROMPT_TOKENS_LABEl
          - _PROMPT_COST_LABEl
          - _INPUT_DISPLAY_LABEL

        put kwargs and args into event attributes
        """
        event_attrs = {**kwargs}
        if len(args) > 0:
            event_attrs["args"] = args

        span_attrs = {}
        if "model" in kwargs:
            model_name = kwargs["model"]
            span_attrs[_MODEL_LABEL] = model_name

            if self.tokens_calculation_needed:
                tokens = OpenaiExtractor.extract_req_tokens(**kwargs)
                num = num_tokens_from_messages(tokens or "")
                cost = get_openai_token_cost_for_model(model_name, num, False)

                span_attrs[_PROMPT_TOKENS_LABEl] = num
                span_attrs[_PROMPT_COST_LABEl] = cost

        user_id = OpenaiExtractor.get_user_id(**kwargs)
        if user_id:
            span_attrs[_USER_ID_LABEL] = user_id

        if "messages" in kwargs:
            span_attrs[_INPUT_DISPLAY_LABEL] = extract_chat_inputs(kwargs["messages"])

        return Extraction(span_attributes=span_attrs, event_attributes=event_attrs)

    @override
    def post_extract(self, resp: Any) -> Extraction:
        """
        extract fields for span attributes:
          - _MODEL_LABEL
          - _COMPLETION_COST_LABEL
          - _COMPLETION_TOKENS_LABEL
          - _PROMPT_COST_LABEl (this will update the value from pre_extract)
          - _PROMPT_TOKENS_LABEl (this will update the value from pre_extract)
          - _OUTPUT_DISPLAY_LABEL

        merge usage into resp as event attributes

        Args:

            resp: inherit from the BaseModel class, or instance of APIResponse class
        """
        data: Dict[str, Any] = {}
        try:
            if hasattr(resp, "parse"):
                data = OpenaiExtractor.parse_raw_response(resp)
                logger.debug(f"after parse_raw_response: {data=}")
            elif hasattr(resp, "model_dump"):
                data = resp.model_dump(exclude_unset=True)
            elif hasattr(resp, "text"):
                data = {"text": resp.text}
            else:
                logger.error(f"Failed to post_extract, {resp=}")
        except Exception as e:
            logger.error(f"Failed to extract response {resp}: {e}")

        span_attrs = {}

        model = data.get("model")
        if model:
            span_attrs[_MODEL_LABEL] = model

            usage = data.get("usage", {})
            if usage:
                usage = OpenaiExtractor.extract_usage(model, usage)
                span_attrs.update(usage)
                data["usage"] = usage

        outputs = extract_chat_outputs(data)
        if outputs:
            span_attrs[_OUTPUT_DISPLAY_LABEL] = outputs

        return Extraction(span_attributes=span_attrs, event_attributes=data)
