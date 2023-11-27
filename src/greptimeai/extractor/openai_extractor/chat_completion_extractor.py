from typing import Optional, Union

import openai
from openai import OpenAI, AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from typing_extensions import override

from greptimeai.extractor import Extraction
from greptimeai.extractor.openai_extractor import OpenaiExtractor
from greptimeai.utils.openai.parser import parse_choices, parse_message_params


class ChatCompletionExtractor(OpenaiExtractor):
    def __init__(
        self,
        client: Optional[Union[OpenAI, AsyncOpenAI]] = None,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.client = client

        obj = client.chat.completions if client else openai.chat.completions
        method_name = "create"
        span_name = "chat.completions.create"
        if isinstance(client, AsyncOpenAI):
            span_name = "chat.completions.create[async]"

        super().__init__(obj=obj, method_name=method_name, span_name=span_name)

    @override
    def pre_extract(self, *args, **kwargs) -> Extraction:
        extraction = super().pre_extract(*args, **kwargs)
        extraction.hide_field_in_event_attributes("messages", self.verbose)

        messages = kwargs.get("messages", None)
        if messages and self.verbose:
            extraction.update_event_attributes(
                {"messages": parse_message_params(messages)}
            )

        return extraction

    @override
    def post_extract(self, resp: ChatCompletion) -> Extraction:
        extraction = super().post_extract(resp)
        if resp.choices:
            extraction.update_event_attributes(
                {"choices": parse_choices(resp.choices, self.verbose)}
            )

        return extraction

    @override
    def is_async(self) -> bool:
        return isinstance(self.client, AsyncOpenAI)
