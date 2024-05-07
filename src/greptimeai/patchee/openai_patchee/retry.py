from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI

from greptimeai.patchee import Patchee

from . import OpenaiPatchees


class RetryPatchees(OpenaiPatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        obj = client
        if obj is None and hasattr(openai, "_client"):
            obj = openai._client
        retry = Patchee(
            obj=obj,
            method_name="_retry_request",
            span_name="",  # retry is only event, so span_name won't be used
            event_name="retry_request",
        )

        super().__init__(patchees=[retry], client=client)
