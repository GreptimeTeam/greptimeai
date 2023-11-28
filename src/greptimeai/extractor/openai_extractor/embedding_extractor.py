from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI
from typing_extensions import override

from greptimeai.extractor import Extraction
from greptimeai.extractor.openai_extractor import OpenaiExtractor


class EmbeddingExtractor(OpenaiExtractor):
    def __init__(
        self,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
        verbose: bool = True,
    ):
        obj = client.embeddings if client else openai.embeddings
        method_name = "create"
        span_name = "embeddings.create"

        super().__init__(
            obj=obj,
            method_name=method_name,
            span_name=span_name,
            client=client,
        )
        self.verbose = verbose

    @override
    def pre_extract(self, *args, **kwargs) -> Extraction:
        extraction = super().pre_extract(*args, **kwargs)
        extraction.hide_field_in_event_attributes("input", self.verbose)

        return extraction
