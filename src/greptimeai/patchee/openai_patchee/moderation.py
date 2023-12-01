from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI

from greptimeai.patchee import Patchee

from . import OpenaiPatchees


class ModerationPatchees(OpenaiPatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        moderations_create = Patchee(
            obj=client.moderations if client else openai.moderations,
            method_name="create",
            span_name="moderations.create",
        )

        moderations_raw_create = Patchee(
            obj=client.moderations.with_raw_response
            if client
            else openai.moderations.with_raw_response,
            method_name="create",
            span_name="moderations.with_raw_response.create",
        )

        patchees = [moderations_create, moderations_raw_create]

        if client:
            raw_moderations_create = Patchee(
                obj=client.with_raw_response.moderations,
                method_name="create",
                span_name="with_raw_response.moderations.create",
            )
            patchees.append(raw_moderations_create)

        super().__init__(patchees=patchees, client=client)
