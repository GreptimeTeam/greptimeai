from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI

from greptimeai.patchee import Patchee

from . import OpenaiPatchees


class _ImagePatchees:
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None], method_name: str):
        images = Patchee(
            obj=client.images if client else openai.images,
            method_name=method_name,
            span_name=f"images.{method_name}",
        )

        images_raw = Patchee(
            obj=client.images.with_raw_response
            if client
            else openai.images.with_raw_response,
            method_name=method_name,
            span_name=f"images.with_raw_response.{method_name}",
        )

        self.patchees = [images, images_raw]

        if client:
            raw_images = Patchee(
                obj=client.with_raw_response.images,
                method_name=method_name,
                span_name=f"with_raw_response.images.{method_name}",
            )
            self.patchees.append(raw_images)


class _ImageGeneratePatchees(_ImagePatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="generate")


class _ImageEditPatchees(_ImagePatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="edit")


class _ImageVariationPatchees(_ImagePatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="create_variation")


class ImagePatchees(OpenaiPatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        generate = _ImageGeneratePatchees(client=client)
        edit = _ImageEditPatchees(client=client)
        variation = _ImageVariationPatchees(client=client)

        patchees = generate.patchees + edit.patchees + variation.patchees

        super().__init__(patchees=patchees, client=client)
