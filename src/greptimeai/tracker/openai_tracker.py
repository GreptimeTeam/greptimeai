import functools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import openai
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from greptimeai import logger
from greptimeai.utils.openai.parser import (
    _parse_chat_completion_message_params,
    _parse_choices,
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
    They MUST BE set explicitely by passing parameters or system environment variables.
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
    ):
        super().__init__(host, database, token)

    def setup(self, client: Optional[OpenAI] = None):
        self._patch_chat_completion(client)

    def _patch_chat_completion(self, client: Optional[OpenAI] = None):
        span_name = "chat.completions.create"
        obj = client.chat.completions if client else openai.chat.completions
        self._patch(
            obj,
            "create",
            span_name,
            self._pre_chat_completion,
            self._post_chat_completion,
        )

    def _patch(
        self,
        obj: object,
        method_name: str,
        span_name: str,
        _pre_action: Callable,
        _post_action: Callable,
    ):
        if not hasattr(obj, method_name):
            logger.warning(f"{repr(obj)} has no '{method_name}' attribute.")
            return

        func = getattr(obj, method_name)

        if hasattr(func, _GREPTIMEAI_WRAPPED):
            logger.warning(f"no need to patch {method_name} multiple times.")
            return

        # TODO(yuanbohan): to support: stream, async
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ex, resp = None, None
            span_attrs, event_attrs = _pre_action(args, **kwargs)
            span_id = self._collector.start_span(
                None, None, span_name, "start", span_attrs, event_attrs
            )
            start = time.time()
            try:
                resp = func(*args, **kwargs)
            except Exception as e:
                ex = e
            finally:
                latency = 1000 * (time.time() - start)
                self._collector.record_latency(latency)
                span_attrs, event_attrs = _post_action(resp)  # type: ignore
                self._collector.end_span(
                    span_id, span_name, span_attrs, "end", event_attrs, ex  # type: ignore
                )
            if ex:
                raise ex
            return resp

        setattr(wrapper, _GREPTIMEAI_WRAPPED, True)
        setattr(obj, method_name, wrapper)

    def _pre_chat_completion(
        self,
        args,
        *,
        messages: List[ChatCompletionMessageParam],
        model: str,
        user: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        span_attrs = {
            "model": model,
            "user_id": user,
        }

        event_attrs = {
            "messages": _parse_chat_completion_message_params(messages),
            **kwargs,
        }
        if args and len(args) > 0:
            event_attrs["args"] = args

        return (span_attrs, event_attrs)

    def _post_chat_completion(
        self,
        resp: ChatCompletion,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        model = resp.model
        usage = {}
        if resp.usage:
            prompt_tokens = resp.usage.prompt_tokens
            prompt_cost = get_openai_token_cost_for_model(model, prompt_tokens, False)

            completion_tokens = resp.usage.completion_tokens
            completion_cost = get_openai_token_cost_for_model(
                model, completion_tokens, True
            )

            usage["prompt_tokens"] = prompt_tokens
            usage["prompt_cost"] = prompt_cost
            usage["completion_tokens"] = completion_tokens
            usage["completion_cost"] = completion_cost

            self._collector.collect_metrics(
                model_name=model,
                prompt_tokens=prompt_tokens,
                prompt_cost=prompt_cost,
                completion_tokens=completion_tokens,
                completion_cost=completion_cost,
            )

        span_attrs = {
            "model": model,
            **usage,
        }

        event_attrs = {
            "id": resp.id,
            "choices": _parse_choices(resp.choices),
            "created": resp.created,
            "object": resp.object,
            "system_fingerprint": resp.system_fingerprint,
            **usage,
        }

        return (span_attrs, event_attrs)
