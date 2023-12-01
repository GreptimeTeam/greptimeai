from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI

from greptimeai.patchee import Patchee

from . import OpenaiPatchees


class _ModelPatchees:
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None], method_name: str):
        models = Patchee(
            obj=client.models if client else openai.models,
            method_name=method_name,
            span_name=f"models.{method_name}",
        )

        models_raw = Patchee(
            obj=client.models.with_raw_response
            if client
            else openai.models.with_raw_response,
            method_name=method_name,
            span_name=f"models.with_raw_response.{method_name}",
        )

        self.patchees = [models, models_raw]

        if client:
            raw_models = Patchee(
                obj=client.with_raw_response.models,
                method_name=method_name,
                span_name=f"with_raw_response.models.{method_name}",
            )
            self.patchees.append(raw_models)


class _ModelListPatchees(_ModelPatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="list")


class _ModelRetrievePatchees(_ModelPatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="retrieve")


class _ModelDeletePatchees(_ModelPatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="delete")


class ModelPatchees(OpenaiPatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        list = _ModelListPatchees(client=client)
        retrieve = _ModelRetrievePatchees(client=client)
        delete = _ModelDeletePatchees(client=client)

        patchees = list.patchees + retrieve.patchees + delete.patchees

        super().__init__(patchees=patchees, client=client)
