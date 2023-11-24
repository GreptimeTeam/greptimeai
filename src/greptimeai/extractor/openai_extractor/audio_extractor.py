from typing import Optional

import openai
from openai import OpenAI

from greptimeai import _MODEL_LABEL, _PROMPT_COST_LABEl, _PROMPT_TOKENS_LABEl
from greptimeai.extractor import Extraction
from greptimeai.extractor.openai_extractor import OpenaiExtractor
from greptimeai.utils.openai.token import (
    get_openai_audio_cost_for_tts,
    num_characters_for_audio,
)


class SpeechExtractor(OpenaiExtractor):
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        verbose: bool = True,
    ):
        obj = client.audio.speech if client else openai.audio.speech
        method_name = "create"
        span_name = "audio.speech.create"

        super().__init__(obj=obj, method_name=method_name, span_name=span_name)
        self.verbose = verbose

    def pre_extract(self, *args, **kwargs) -> Extraction:
        extraction = super().pre_extract(*args, **kwargs)
        extraction.hide_field_in_event_attributes("input", self.verbose)

        input = extraction.event_attributes.get("input", None)
        if input:
            num_chars = num_characters_for_audio(input)
            extraction.update_span_attributes({_PROMPT_TOKENS_LABEl: num_chars})

            model = extraction.event_attributes.get("model")
            if model:
                cost = get_openai_audio_cost_for_tts(model, num_chars)

                extraction.update_span_attributes(
                    {
                        _MODEL_LABEL: model,
                        _PROMPT_COST_LABEl: cost,
                    }
                )

        return extraction


class TranscriptionExtractor(OpenaiExtractor):
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        verbose: bool = True,
    ):
        obj = client.audio.transcriptions if client else openai.audio.transcriptions
        method_name = "create"
        span_name = "audio.transcriptions.create"

        super().__init__(obj=obj, method_name=method_name, span_name=span_name)
        self.verbose = verbose

    def pre_extract(self, *args, **kwargs) -> Extraction:
        extraction = super().pre_extract(*args, **kwargs)
        extraction.hide_field_in_event_attributes("prompt", self.verbose)
        return extraction


class TranslationExtractor(OpenaiExtractor):
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        verbose: bool = True,
    ):
        obj = client.audio.translations if client else openai.audio.translations
        method_name = "create"
        span_name = "audio.translations.create"

        super().__init__(obj=obj, method_name=method_name, span_name=span_name)
        self.verbose = verbose

    def pre_extract(self, *args, **kwargs) -> Extraction:
        extraction = super().pre_extract(*args, **kwargs)
        extraction.hide_field_in_event_attributes("prompt", self.verbose)
        return extraction
