from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI

from greptimeai.extractor.openai_extractor import OpenaiExtractor
from greptimeai.tracker import Trackee


class _ImageExtractor(OpenaiExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None], method_name: str):
        images = Trackee(
            obj=client.images if client else openai.images,
            method_name=method_name,
            span_name=f"images.{method_name}",
        )

        images_raw = Trackee(
            obj=client.images.with_raw_response
            if client
            else openai.images.with_raw_response,
            method_name=method_name,
            span_name=f"images.with_raw_response.{method_name}",
        )

        trackees = [
            images,
            images_raw,
        ]

        if client:
            raw_images = Trackee(
                obj=client.with_raw_response.images,
                method_name=method_name,
                span_name=f"with_raw_response.images.{method_name}",
            )
            trackees.append(raw_images)

        super().__init__(client=client, trackees=trackees)


class ImageGenerateExtractor(_ImageExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="generate")


class ImageEditExtractor(_ImageExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="edit")


class ImageVariationExtractor(_ImageExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="create_variation")
