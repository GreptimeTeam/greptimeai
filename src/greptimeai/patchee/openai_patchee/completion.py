from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI

from greptimeai.patchee import Patchee

from . import OpenaiPatchees


class CompletionPatchees(OpenaiPatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        completion_create = Patchee(
            obj=client.completions if client else openai.completions,
            method_name="create",
            span_name="completions.create",
        )

        completion_raw_create = Patchee(
            obj=client.completions.with_raw_response
            if client
            else openai.completions.with_raw_response,
            method_name="create",
            span_name="completions.with_raw_response.create",
        )

        patchees = [
            completion_create,
            completion_raw_create,
        ]

        if client:
            raw_completion_create = Patchee(
                obj=client.with_raw_response.completions,
                method_name="create",
                span_name="with_raw_response.completions.create",
            )
            patchees.append(raw_completion_create)

        super().__init__(patchees=patchees, client=client)
