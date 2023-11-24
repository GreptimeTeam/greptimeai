from typing import Optional

import openai
from openai import OpenAI

from greptimeai.extractor.openai_extractor import OpenaiExtractor


class _ImageExtractor(OpenaiExtractor):
    def __init__(self, client: Optional[OpenAI], method_name: str):
        obj = client.images if client else openai.images
        span_name = f"images.{method_name}"

        super().__init__(obj=obj, method_name=method_name, span_name=span_name)


class ImageGenerateExtractor(_ImageExtractor):
    def __init__(self, client: Optional[OpenAI] = None):
        super().__init__(client=client, method_name="generate")


class ImageEditExtractor(_ImageExtractor):
    def __init__(self, client: Optional[OpenAI] = None):
        super().__init__(client=client, method_name="edit")


class ImageVariationExtractor(_ImageExtractor):
    def __init__(self, client: Optional[OpenAI] = None):
        super().__init__(client=client, method_name="create_variation")
