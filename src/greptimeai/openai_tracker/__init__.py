import functools
import time
from typing import Callable, Optional

import openai

from greptimeai.logger import logger
from greptimeai.tracker import BaseTracker


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
        self._patch_chat_completion()

    def _patch_chat_completion(self, client: Optional[openai.OpenAI] = None):
        if client:
            self._patch(client.chat.completions.create, print)
        else:
            self._patch(openai.chat.completions.create, print)

    def _patch(self, func: Callable, resp_processor: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"{ args = } { kwargs = }")
            start = time.time()
            resp = None
            try:
                resp = func(*args, **kwargs)
            except Exception as ex:
                logger.error(f"{ ex = }")
            finally:
                latency = time.time() - start
                logger.debug(f"{ func = } { latency = }")
                logger.debug(f"{ resp = }")
                resp_processor(resp)
            return resp

        func = wrapper
