from typing import Optional

import openai
from openai import OpenAI
from openai.types import Completion
from typing_extensions import override

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

    @override
    def pre_extract(self, *args, **kwargs) -> Extraction:
        extraction = super().pre_extract(*args, **kwargs)
        extraction.hide_field_in_event_attributes("prompt", self.verbose)
        return extraction

    @override
    def post_extract(self, resp: Completion) -> Extraction:
        extraction = super().post_extract(resp)

        choices = extraction.event_attributes.get("choices", None)
        if choices:
            extraction.update_event_attributes(
                {"choices": parse_choices(choices, self.verbose)}
            )

        return extraction
