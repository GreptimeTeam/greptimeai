from typing import Sequence, Union

from openai import AsyncOpenAI, OpenAI

from greptimeai.trackee import Trackee, Trackees


class OpenaiTrackees(Trackees):
    trackees: Sequence[Trackee] = []
    client: Union[OpenAI, AsyncOpenAI, None]

    def __init__(
        self,
        trackees: Sequence[Trackee],
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        self.trackees = trackees
        self.client = client

        self.is_async = isinstance(client, AsyncOpenAI)

    def get_trackees(self) -> Sequence[Trackee]:
        return self.trackees
