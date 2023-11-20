import functools
import time
from typing import Any, Dict, Optional

import openai
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from greptimeai import logger

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
        if client:
            self._patch(client.chat.completions, "create", span_name)
        else:
            self._patch(openai.chat.completions, "create", span_name)

    def _patch(self, obj: object, method_name: str, span_name: str):
        if not hasattr(obj, method_name):
            logger.warning(f"{repr(obj)} has no '{method_name}' attribute.")
            return

        func = getattr(obj, method_name)

        if hasattr(func, _GREPTIMEAI_WRAPPED):
            logger.warning(f"no need to patch {method_name} multiple times.")
            return

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            ex, resp = None, None
            try:
                resp = func(*args, **kwargs)
            except Exception as e:
                ex = e
                logger.error(f"{ ex = }")
            finally:
                latency = 1000 * (time.time() - start)
                self.collector.record_latency(latency)
            if ex:
                raise ex
            return resp

        setattr(wrapper, _GREPTIMEAI_WRAPPED, True)
        setattr(obj, method_name, wrapper)


def _prepare_req_attributes_for_chat_completion(*args, **kwargs) -> Dict[str, Any]:
    attributes = {}
    return attributes


def _prepare_resp_attributes_for_chat_completion(
    resp: ChatCompletion,
    ex: Exception,
) -> Dict[str, Any]:
    attributes = {}
    return attributes
