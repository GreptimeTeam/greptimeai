import functools
from typing import Callable, Optional, Union

from openai import AsyncOpenAI, OpenAI
from openai._models import FinalRequestOptions
from typing_extensions import override

from greptimeai import logger
from greptimeai.collector import Collector
from greptimeai.extractor.openai_extractor import _X_SPAN_ID_KEY
from greptimeai.patchee.openai_patchee.retry import RetryPatchees

from .base import _OpenaiPatcher


class _RetryPatcher(_OpenaiPatcher):
    def __init__(
        self,
        collector: Collector,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        self._is_async = False
        if isinstance(client, AsyncOpenAI):
            self._is_async = True
        patchees = RetryPatchees(client=client)
        super().__init__(collector=collector, patchees=patchees, client=client)

    def _add_retry_event(self, *args):
        if len(args) > 0 and isinstance(args[0], FinalRequestOptions):
            dict = args[0].model_dump(exclude_unset=True)
            span_id = dict.get("headers", {}).get(_X_SPAN_ID_KEY)
            if span_id:
                logger.debug(f"in retry_patcher {span_id=}")
                self.collector._collector.add_span_event(
                    span_id=span_id,
                    event_name="retry",
                    event_attrs=dict,
                )

    @override
    def patch(self):
        for patchee in self.patchees.get_patchees():
            func: Optional[Callable] = patchee.get_unwrapped_func()
            if func is None:
                return

            if self._is_async:

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    try:
                        self._add_retry_event(*args)
                        resp = await func(*args, **kwargs)
                    except Exception as e:
                        raise e
                    return resp

                patchee.wrap_func(async_wrapper)
                logger.debug("patched 'retry[async]'")
            else:

                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    try:
                        self._add_retry_event(*args)
                        resp = func(*args, **kwargs)
                    except Exception as e:
                        raise e
                    return resp

                patchee.wrap_func(wrapper)
                logger.debug("patched 'retry'")
