from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI

from greptimeai.patchee import Patchee
from greptimeai.utils.attr import get_attr, get_optional_attr

from . import _SPAN_NAME_COMPLETION, OpenaiPatchees


class CompletionPatchees(OpenaiPatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        completion_create = Patchee(
            obj=get_optional_attr([client, openai], ["completions"]),
            method_name="create",
            span_name=_SPAN_NAME_COMPLETION,
            event_name="completions.create",
        )

        completion_raw_create = Patchee(
            obj=get_optional_attr(
                [client, openai], ["completions", "with_raw_response"]
            ),
            method_name="create",
            span_name=_SPAN_NAME_COMPLETION,
            event_name="completions.with_raw_response.create",
        )

        patchees = [
            completion_create,
            completion_raw_create,
        ]

        if client:
            raw_completion_create = Patchee(
                obj=get_attr(client, ["with_raw_response", "completions"]),
                method_name="create",
                span_name=_SPAN_NAME_COMPLETION,
                event_name="with_raw_response.completions.create",
            )
            patchees.append(raw_completion_create)

        super().__init__(patchees=patchees, client=client)
