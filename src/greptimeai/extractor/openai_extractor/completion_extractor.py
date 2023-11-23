from typing import Optional

import openai
from openai import OpenAI
from openai.types import Completion

from greptimeai.extractor import Extraction
from greptimeai.extractor.openai_extractor import Extractor
from greptimeai.utils.openai.parser import parse_choices


class CompletionExtractor(Extractor):
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
        dict = resp.model_dump()
        extraction = super().post_extract(dict)

        if "choices" in dict:
            choices = parse_choices(dict["choices"], self.verbose)
            extraction.update_event_attributes({"choices": choices})

        return extraction
