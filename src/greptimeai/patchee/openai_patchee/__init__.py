from typing import Sequence, Union

from openai import AsyncOpenAI, OpenAI

from greptimeai.patchee import Patchee

_SPAN_NAME_SPEECH = "openai_speech"
_SPAN_NAME_TRANSLATION = "openai_translation"
_SPAN_NAME_TRANSCRIPTION = "openai_transcription"
_SPAN_NAME_COMPLETION = "openai_completion"
_SPAN_NAME_EMBEDDING = "openai_embedding"
_SPAN_NAME_FILE = "openai_file"
_SPAN_NAME_FINE_TUNNING = "openai_fine_tunning"
_SPAN_NAME_IMAGE = "openai_image"
_SPAN_NAME_MODEL = "openai_model"
_SPAN_NAME_MODERATION = "openai_moderation"


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

    def get_patchees(self) -> Sequence[Patchee]:
        return self.patchees
