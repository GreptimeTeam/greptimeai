from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI

from greptimeai.trackee import Trackee

from . import OpenaiTrackees


class EmbeddingTrackees(OpenaiTrackees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        embeddings_create = Trackee(
            obj=client.embeddings if client else openai.embeddings,
            method_name="create",
            span_name="embeddings.create",
        )

        embeddings_raw_create = Trackee(
            obj=client.embeddings.with_raw_response
            if client
            else openai.embeddings.with_raw_response,
            method_name="create",
            span_name="embeddings.with_raw_response.create",
        )

        trackees = [
            embeddings_create,
            embeddings_raw_create,
        ]

        if client:
            raw_embeddings_create = Trackee(
                obj=client.with_raw_response.embeddings,
                method_name="create",
                span_name="with_raw_response.embeddings.create",
            )
            trackees.append(raw_embeddings_create)

        super().__init__(trackees=trackees, client=client)
