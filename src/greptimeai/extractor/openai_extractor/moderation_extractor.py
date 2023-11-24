from typing import Optional

import openai
from openai import OpenAI

from greptimeai.extractor.openai_extractor import OpenaiExtractor


class ModerationExtractor(OpenaiExtractor):
    def __init__(self, client: Optional[OpenAI] = None):
        obj = client.moderations if client else openai.moderations
        method_name = "create"
        span_name = "moderations.create"

        super().__init__(obj=obj, method_name=method_name, span_name=span_name)
