from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI

from greptimeai.patchee import Patchee

from . import OpenaiPatchees


class EmbeddingPatchees(OpenaiPatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        embeddings_create = Patchee(
            obj=client.embeddings if client else openai.embeddings,
            method_name="create",
            span_name="embeddings.create",
        )

        embeddings_raw_create = Patchee(
            obj=client.embeddings.with_raw_response
            if client
            else openai.embeddings.with_raw_response,
            method_name="create",
            span_name="embeddings.with_raw_response.create",
        )

        patchees = [
            embeddings_create,
            embeddings_raw_create,
        ]

        if client:
            raw_embeddings_create = Patchee(
                obj=client.with_raw_response.embeddings,
                method_name="create",
                span_name="with_raw_response.embeddings.create",
            )
            patchees.append(raw_embeddings_create)

        super().__init__(patchees=patchees, client=client)
