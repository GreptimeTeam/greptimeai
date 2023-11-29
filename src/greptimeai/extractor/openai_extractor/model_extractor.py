from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI
from tracker import Trackee

from greptimeai.extractor.openai_extractor import OpenaiExtractor


class _ModelExtractor(OpenaiExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None], method_name: str):
        models = Trackee(
            obj=client.models if client else openai.models,
            method_name=method_name,
            span_name=f"models.{method_name}",
        )

        models_raw = Trackee(
            obj=client.models.with_raw_response
            if client
            else openai.models.with_raw_response,
            method_name=method_name,
            span_name=f"models.with_raw_response.{method_name}",
        )

        trackees = [
            models,
            models_raw,
        ]

        if client:
            raw_models = Trackee(
                obj=client.with_raw_response.models,
                method_name=method_name,
                span_name=f"with_raw_response.models.{method_name}",
            )
            trackees.append(raw_models)

        super().__init__(client=client, trackees=trackees)


class ModelListExtractor(_ModelExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="list")


class ModelRetrieveExtractor(_ModelExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="retrieve")


class ModelDeleteExtractor(_ModelExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="delete")
