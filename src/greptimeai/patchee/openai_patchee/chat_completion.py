from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI

from greptimeai.patchee import Patchee
from greptimeai.utils.attr import get_attr, get_optional_attr

from . import _SPAN_NAME_COMPLETION, OpenaiPatchees


class ChatCompletionPatchees(OpenaiPatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        chat_completion_create = Patchee(
            obj=get_optional_attr([client, openai], ["chat", "completions"]),
            method_name="create",
            span_name=_SPAN_NAME_COMPLETION,
            event_name="chat.completions.create",
        )

        chat_raw_completion_create = Patchee(
            obj=get_optional_attr(
                [client, openai], ["chat", "with_raw_response", "completions"]
            ),
            method_name="create",
            span_name=_SPAN_NAME_COMPLETION,
            event_name="chat.with_raw_response.completions.create",
        )

        chat_completion_raw_create = Patchee(
            obj=get_optional_attr(
                [client, openai], ["chat", "completions", "with_raw_response"]
            ),
            method_name="create",
            span_name=_SPAN_NAME_COMPLETION,
            event_name="chat.completions.with_raw_response.create",
        )

        patchees = [
            chat_completion_create,
            chat_raw_completion_create,
            chat_completion_raw_create,
        ]

        if client:
            raw_chat_completion_create = Patchee(
                obj=get_attr(client, ["with_raw_response", "chat", "completions"]),
                method_name="create",
                span_name=_SPAN_NAME_COMPLETION,
                event_name="with_raw_response.chat.completions.create",
            )
            patchees.append(raw_chat_completion_create)

        super().__init__(patchees=patchees, client=client)
