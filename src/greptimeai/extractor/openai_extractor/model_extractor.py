from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI

from greptimeai.extractor.openai_extractor import OpenaiExtractor


class _ModelExtractor(OpenaiExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None], method_name: str):
        obj = client.models if client else openai.models
        span_name = f"models.{method_name}"

        super().__init__(
            obj=obj,
            method_name=method_name,
            span_name=span_name,
            client=client,
        )


class ModelListExtractor(_ModelExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="list")


class ModelRetrieveExtractor(_ModelExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="retrieve")


class ModelDeleteExtractor(_ModelExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="delete")
