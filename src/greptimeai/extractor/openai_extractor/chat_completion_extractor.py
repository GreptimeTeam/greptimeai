from typing import Optional

import openai
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from greptimeai.extractor import Extraction
from greptimeai.extractor.openai_extractor import Extractor
from greptimeai.utils.openai.parser import parse_choices, parse_message_params


class ChatCompletionExtractor(Extractor):
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        verbose: bool = True,
    ):
        obj = client.chat.completions if client else openai.chat.completions
        method_name = "create"
        span_name = "chat.completions.create"

        super().__init__(obj=obj, method_name=method_name, span_name=span_name)
        self.verbose = verbose

    def pre_extract(self, *args, **kwargs) -> Extraction:
        extraction = super().pre_extract(*args, **kwargs)

        messages = kwargs.get("messages", None)
        if messages:
            if self.verbose:
                messages = parse_message_params(messages)
                extraction.update_event_attributes({"messages": messages})
            else:
                extraction.update_event_attributes({"messages": "..."})

        return extraction

    def post_extract(self, resp: ChatCompletion) -> Extraction:
        dict = resp.model_dump()
        extraction = super().post_extract(dict)

        if "choices" in dict:
            choices = parse_choices(dict["choices"], self.verbose)
            extraction.update_event_attributes({"choices": choices})

        return extraction
