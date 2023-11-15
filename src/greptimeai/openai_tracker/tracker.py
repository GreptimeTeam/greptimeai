import functools
import time
from typing import Callable

from greptimeai.tracker import BaseTracker


class OpenaiTracker(BaseTracker):
    def __init__(
        self,
        host: str = "",
        database: str = "",
        token: str = "",
    ):
        super().__init__(host, database, token)

    def setup(self):
        self._patch_chat_completion()

    def _patch_chat_completion(self):
        pass

    def _patch(self, func: Callable, resp_processor: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            resp = None
            exception = ""
            try:
                resp = func(*args, **kwargs)
            except Exception as ex:
                exception = ex.__class__.__name__
                raise ex
            finally:
                latency = time.time() - start
                resp_processor(resp)

            return resp
