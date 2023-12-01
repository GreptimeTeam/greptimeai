from typing import Sequence, Union

from openai import AsyncOpenAI, OpenAI

from greptimeai.patchee import Patchee


class OpenaiPatchees:
    patchees: Sequence[Patchee] = []
    client: Union[OpenAI, AsyncOpenAI, None]

    def __init__(
        self,
        patchees: Sequence[Patchee],
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        self.patchees = patchees
        self.client = client

        self.is_async = isinstance(client, AsyncOpenAI)

    def get_patchees(self) -> Sequence[Patchee]:
        return self.patchees
