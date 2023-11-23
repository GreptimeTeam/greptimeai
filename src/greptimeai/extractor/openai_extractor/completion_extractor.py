from typing import Optional

import openai
from openai import OpenAI
from openai.types import Completion

from greptimeai.extractor import Extraction
from greptimeai.extractor.openai_extractor import OpenaiExtractor
from greptimeai.utils.openai.parser import parse_choices


class CompletionExtractor(OpenaiExtractor):
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        verbose: bool = True,
    ):
        obj = client.completions if client else openai.completions
        method_name = "create"
        span_name = "completions.create"

        super().__init__(obj=obj, method_name=method_name, span_name=span_name)
        self.verbose = verbose

    def pre_extract(self, *args, **kwargs) -> Extraction:
        extraction = super().pre_extract(*args, **kwargs)

        prompt = kwargs.get("prompt", None)
        if prompt and not self.verbose:
            extraction.update_event_attributes({"prompt": "..."})

        return extraction

    def post_extract(self, resp: Completion) -> Extraction:
        extraction = super().post_extract(resp)
        if "choices" in extraction.event_attributes:
            choices = parse_choices(
                extraction.event_attributes["choices"], self.verbose
            )
            extraction.update_event_attributes({"choices": choices})

        return extraction
