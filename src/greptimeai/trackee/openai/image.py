from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI

from greptimeai.trackee import Trackee

from . import OpenaiTrackees


class _ImageTrackees:
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

        self.trackees = [
            images,
            images_raw,
        ]

        if client:
            raw_images = Trackee(
                obj=client.with_raw_response.images,
                method_name=method_name,
                span_name=f"with_raw_response.images.{method_name}",
            )
            self.trackees.append(raw_images)


class _ImageGenerateTrackees(_ImageTrackees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="generate")


class _ImageEditTrackees(_ImageTrackees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="edit")


class _ImageVariationTrackees(_ImageTrackees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="create_variation")


class ImageTrackees(OpenaiTrackees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        generate_trackees = _ImageGenerateTrackees(client=client)
        edit_trackees = _ImageEditTrackees(client=client)
        variation_trackees = _ImageVariationTrackees(client=client)

        trackees = (
            generate_trackees.trackees
            + edit_trackees.trackees
            + variation_trackees.trackees
        )

        super().__init__(trackees=trackees, client=client)
