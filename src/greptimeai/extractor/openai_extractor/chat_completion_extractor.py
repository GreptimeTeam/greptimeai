from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI
from tracker import Trackee

from greptimeai.extractor.openai_extractor import OpenaiExtractor


class ChatCompletionExtractor(OpenaiExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        chat_completion_create = Trackee(
            obj=client.chat.completions if client else openai.chat.completions,
            method_name="create",
            span_name="chat.completions.create",
        )

        chat_raw_completion_create = Trackee(
            obj=client.chat.with_raw_response.completions
            if client
            else openai.chat.with_raw_response.completions,
            method_name="create",
            span_name="chat.with_raw_response.completions.create",
        )

        chat_completion_raw_create = Trackee(
            obj=client.chat.completions.with_raw_response
            if client
            else openai.chat.completions.with_raw_response,
            method_name="create",
            span_name="chat.completions.with_raw_response.create",
        )

        trackees = [
            chat_completion_create,
            chat_raw_completion_create,
            chat_completion_raw_create,
        ]

        if client:
            raw_chat_completion_create = Trackee(
                obj=client.with_raw_response.chat.completions,
                method_name="create",
                span_name="with_raw_response.chat.completions.create",
            )
            trackees.append(raw_chat_completion_create)

        super().__init__(client=client, trackees=trackees)
