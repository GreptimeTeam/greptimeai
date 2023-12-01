import functools
from typing import Union

from openai import AsyncOpenAI, OpenAI
from typing_extensions import override

from greptimeai import logger
from greptimeai.collector import Collector
from greptimeai.patchee import Patchee
from greptimeai.patchee.openai.retry import RetryPatchees

from .base import _OpenaiPatcher


class _RetryPatcher(_OpenaiPatcher):
    def __init__(
        self,
        collector: Collector,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = RetryPatchees(client=client)
        super().__init__(collector=collector, patchees=patchees, client=client)

    @override
    def patch_one(self, patchee: Patchee):
        func = patchee.get_unwrapped_func()
        if not func:
            return

        if self.is_async:
            pass
        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    logger.info(f"in retry_patcher { args = }")
                    logger.info(f"in retry_patcher { kwargs = }")
                    resp = func(*args, **kwargs)
                except Exception as e:
                    raise e
                return resp

            patchee.wrap_func(wrapper)
