from typing import Optional

import openai
from openai import OpenAI

from greptimeai.extractor import Extraction
from greptimeai.extractor.openai_extractor import Extractor


class EmbeddingExtractor(Extractor):
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        verbose: bool = True,
    ):
        obj = client.embeddings if client else openai.embeddings
        method_name = "create"
        span_name = "embeddings.create"

        super().__init__(obj=obj, method_name=method_name, span_name=span_name)
        self.verbose = verbose

    def pre_extract(self, *args, **kwargs) -> Extraction:
        extraction = super().pre_extract(*args, **kwargs)

        input = kwargs.get("input", None)
        if input and not self.verbose:
            extraction.update_event_attributes({"input": "..."})

        return extraction
