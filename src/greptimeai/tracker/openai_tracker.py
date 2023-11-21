import functools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import openai
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from greptimeai import (
    _COMPLETION_COST_LABEL,
    _COMPLETION_TOKENS_LABEL,
    _ERROR_TYPE_LABEL,
    _MODEL_LABEL,
    _SPAN_NAME_LABEL,
    _USER_ID_LABEL,
    _PROMPT_COST_LABEl,
    _PROMPT_TOKENS_LABEl,
    logger,
)
from greptimeai.utils.openai.parser import (
    parse_chat_completion_message_params,
    parse_choices,
)
from greptimeai.utils.openai.token import get_openai_token_cost_for_model

from . import _GREPTIMEAI_WRAPPED, BaseTracker


def setup(
    host: str = "",
    database: str = "",
    token: str = "",
    client: Optional[OpenAI] = None,
):
    """
    patch openai main functions automatically.
    host, database and token is to setup the place to store the data, and the authority.
    They MUST BE set explicitly by passing parameters or system environment variables.
    Args:
        host: if None or empty string, GREPTIMEAI_HOST environment variable will be used.
        database: if None or empty string, GREPTIMEAI_DATABASE environment variable will be used.
        token: if None or empty string, GREPTIMEAI_TOKEN environment variable will be used.
        client: if None, then openai module-level client will be patched.
                Important: We highly recommend instantiating client instances
                instead of relying on the global client.
    """
    tracker = OpenaiTracker(host, database, token)
    tracker.setup(client)


class OpenaiTracker(BaseTracker):
    def __init__(
        self,
        host: str = "",
        database: str = "",
        token: str = "",
        verbose: bool = True,
    ):
        super().__init__(
            service_name="openai", host=host, database=database, token=token
        )
        self._verbose = verbose

    def setup(self, client: Optional[OpenAI] = None):
        self._patch_chat_completion(client)

    def _patch_chat_completion(self, client: Optional[OpenAI] = None):
        span_name = "chat.completions.create"
        obj = client.chat.completions if client else openai.chat.completions
        self._patch(
            obj,
            "create",
            span_name,
            self._pre_chat_completion_extractor,
            self._post_chat_completion_extractor,
        )

    def _patch(
        self,
        obj: object,
        method_name: str,
        span_name: str,
        pre_extractor: Callable,
        post_extractor: Callable,
    ):
        """
        wrap the method name of this obj with trace and metrics collection logic.

        if this obj does not contain this method name, patch will do nothing.
        if this method has already been patched, then it won't be patched multiple times.

        Args:
            obj: OpenAI client, or module level client
            method_name: the method name of the object
            span_name: identify different span name
            _pre_extractor: extract span attributes and event attributes from args, kwargs
            _post_extractor: extract span attributes and event attributes from response
        """
        if not hasattr(obj, method_name):
            logger.warning(f"'{method_name}' attribute not found from the object.")
            return

        func = getattr(obj, method_name)

        if hasattr(func, _GREPTIMEAI_WRAPPED):
            logger.warning(f"no need to patch {method_name} multiple times.")
            return

        # TODO(yuanbohan): to support: stream, async
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ex, resp = None, None
            req_span_attrs, req_event_attrs = pre_extractor(args, **kwargs)
            span_id = self._collector.start_span(
                span_id=None,
                parent_id=None,
                span_name=span_name,
                event_name="start",
                span_attrs=req_span_attrs,
                event_attrs=req_event_attrs,
            )
            start = time.time()
            try:
                resp = func(*args, **kwargs)
            except Exception as e:
                ex = e
                self._collector._llm_error_count.add(
                    1,
                    {
                        _SPAN_NAME_LABEL: span_name,
                        _ERROR_TYPE_LABEL: ex.__class__.__name__,
                    },
                )
                raise ex
            finally:
                latency = 1000 * (time.time() - start)
                self._collector.record_latency(latency)
                resp_span_attrs, resp_event_attrs = post_extractor(resp)
                self._collector.end_span(
                    span_id=span_id,  # type: ignore
                    span_name=span_name,
                    event_name="end",
                    span_attrs=resp_span_attrs,
                    event_attrs=resp_event_attrs,
                    ex=ex,
                )
                self._collect_metrics(span_name, resp_span_attrs)
            return resp

        setattr(wrapper, _GREPTIMEAI_WRAPPED, True)
        setattr(obj, method_name, wrapper)

    def _pre_chat_completion_extractor(
        self,
        args,
        *,
        messages: List[ChatCompletionMessageParam],
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
        if self._verbose:
            event_attrs["messages"] = parse_chat_completion_message_params(messages)

        if args and len(args) > 0:
            event_attrs["args"] = args

        return (span_attrs, event_attrs)

    def _post_chat_completion_extractor(
        self,
        resp: ChatCompletion,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        usage = {}
        if resp.usage:
            usage[_PROMPT_TOKENS_LABEl] = resp.usage.prompt_tokens
            usage[_PROMPT_COST_LABEl] = get_openai_token_cost_for_model(
                resp.model, resp.usage.prompt_tokens, False
            )
            usage[_COMPLETION_TOKENS_LABEL] = resp.usage.completion_tokens
            usage[_COMPLETION_COST_LABEL] = get_openai_token_cost_for_model(
                resp.model, resp.usage.completion_tokens, True
            )

        span_attrs = {
            _MODEL_LABEL: resp.model,
            **usage,
        }

        event_attrs = resp.model_dump()
        event_attrs["usage"] = usage
        event_attrs["choices"] = parse_choices(resp.choices, self._verbose)

        return (span_attrs, event_attrs)

    def _collect_metrics(self, span_name: str, attributes: Dict[str, Any]):
        prompt_tokens = attributes.get(_PROMPT_TOKENS_LABEl, 0)
        prompt_cost = attributes.get(_PROMPT_COST_LABEl, 0)
        completion_tokens = attributes.get(_COMPLETION_TOKENS_LABEL, 0)
        completion_cost = attributes.get(_COMPLETION_COST_LABEL, 0)
        if not (prompt_tokens or prompt_cost or completion_tokens or completion_cost):
            logger.warning(f"no need to collect empty metrics for openai {span_name}")
            return

        model = attributes.get(_MODEL_LABEL, "")
        self._collector.collect_metrics(
            model_name=model,
            prompt_tokens=prompt_tokens,
            prompt_cost=prompt_cost,
            completion_tokens=completion_tokens,
            completion_cost=completion_cost,
        )
