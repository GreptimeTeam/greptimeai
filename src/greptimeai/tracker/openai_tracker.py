import functools
import time
from typing import Optional

import openai

from greptimeai import logger

from . import _GREPTIMEAI_WRAPPED, BaseTracker


def setup(
    host: str = "",
    database: str = "",
    token: str = "",
    client: Optional[openai.OpenAI] = None,
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

    def setup(self, client: Optional[openai.OpenAI] = None):
        self._patch_chat_completion(client)

    def _patch_chat_completion(self, client: Optional[openai.OpenAI] = None):
        if client:
            self._patch(client.chat.completions, "create")
        else:
            self._patch(openai.chat.completions, "create")

    def _patch(self, obj: object, func_name: str):
        if not hasattr(obj, func_name):
            logger.warning(f"{repr(obj)} has no '{func_name}' attribute.")
            return

        func = getattr(obj, func_name)
        logger.debug(f"{repr(func) = }")

        if hasattr(func, _GREPTIMEAI_WRAPPED):
            logger.info(f"no need to patch {func_name} multiple times.")
            return

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"{repr(obj)} calling {func_name}")
            logger.debug(f"{ args = } { kwargs = }")
            start = time.time()
            ex, resp = None, None
            try:
                resp = func(*args, **kwargs)
            except Exception as e:
                ex = e
                logger.error(f"{ ex = }")
            finally:
                latency = time.time() - start
                logger.debug(f"{ latency = }")
                logger.debug(f"{ resp = }")

            if ex:
                raise ex
            return resp

        setattr(wrapper, _GREPTIMEAI_WRAPPED, True)
        setattr(obj, func_name, wrapper)
