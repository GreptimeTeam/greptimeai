from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI

from greptimeai.trackee import Trackee

from . import OpenaiTrackees


class _ModelTrackees:
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

        self.trackees = [
            models,
            models_raw,
        ]

        if client:
            raw_models = Trackee(
                obj=client.with_raw_response.models,
                method_name=method_name,
                span_name=f"with_raw_response.models.{method_name}",
            )
            self.trackees.append(raw_models)


class _ModelListTrackees(_ModelTrackees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="list")


class _ModelRetrieveTrackees(_ModelTrackees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="retrieve")


class _ModelDeleteTrackees(_ModelTrackees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="delete")


class ModelTrackees(OpenaiTrackees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        list_trackees = _ModelListTrackees(client=client)
        retrieve_trackees = _ModelRetrieveTrackees(client=client)
        delete_trackees = _ModelDeleteTrackees(client=client)

        trackees = (
            list_trackees.trackees
            + retrieve_trackees.trackees
            + delete_trackees.trackees
        )

        super().__init__(trackees=trackees, client=client)
