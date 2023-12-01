from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI

from greptimeai.patchee import Patchee

from . import OpenaiPatchees


class _SpeechPatchees:
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        audio_speech_create = Patchee(
            obj=client.audio.speech if client else openai.audio.speech,
            method_name="create",
            span_name="audio.speech.create",
        )

        audio_raw_speech_create = Patchee(
            obj=client.audio.with_raw_response.speech
            if client
            else openai.audio.with_raw_response.speech,
            method_name="create",
            span_name="audio.with_raw_response.speech.create",
        )

        audio_speech_raw_create = Patchee(
            obj=client.audio.speech.with_raw_response
            if client
            else openai.audio.speech.with_raw_response,
            method_name="create",
            span_name="audio.speech.with_raw_response.create",
        )

        self.patchees = [
            audio_speech_create,
            audio_raw_speech_create,
            audio_speech_raw_create,
        ]

        if client:
            raw_audio_speech_create = Patchee(
                obj=client.with_raw_response.audio.speech,
                method_name="create",
                span_name="with_raw_response.audio.speech.create",
            )
            self.patchees.append(raw_audio_speech_create)


class _TranscriptionPatchees:
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        audio_transcriptions_create = Patchee(
            obj=client.audio.transcriptions if client else openai.audio.transcriptions,
            method_name="create",
            span_name="audio.transcriptions.create",
        )

        audio_raw_transcriptions_create = Patchee(
            obj=client.audio.with_raw_response.transcriptions
            if client
            else openai.audio.with_raw_response.transcriptions,
            method_name="create",
            span_name="audio.with_raw_response.transcriptions.create",
        )

        audio_transcriptions_raw_create = Patchee(
            obj=client.audio.transcriptions.with_raw_response
            if client
            else openai.audio.transcriptions.with_raw_response,
            method_name="create",
            span_name="audio.transcriptions.with_raw_response.create",
        )

        self.patchees = [
            audio_transcriptions_create,
            audio_raw_transcriptions_create,
            audio_transcriptions_raw_create,
        ]

        if client:
            raw_audio_transcriptions_create = Patchee(
                obj=client.with_raw_response.audio.transcriptions,
                method_name="create",
                span_name="with_raw_response.audio.transcriptions.create",
            )
            self.patchees.append(raw_audio_transcriptions_create)


class _TranslationPatchees:
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        audio_translations_create = Patchee(
            obj=client.audio.translations if client else openai.audio.translations,
            method_name="create",
            span_name="audio.translations.create",
        )

        audio_raw_translations_create = Patchee(
            obj=client.audio.with_raw_response.translations
            if client
            else openai.audio.with_raw_response.translations,
            method_name="create",
            span_name="audio.with_raw_response.translations.create",
        )

        audio_translations_raw_create = Patchee(
            obj=client.audio.translations.with_raw_response
            if client
            else openai.audio.translations.with_raw_response,
            method_name="create",
            span_name="audio.translations.with_raw_response.create",
        )

        self.patchees = [
            audio_translations_create,
            audio_raw_translations_create,
            audio_translations_raw_create,
        ]

        if client:
            raw_audio_translations_create = Patchee(
                obj=client.with_raw_response.audio.translations,
                method_name="create",
                span_name="with_raw_response.audio.translations.create",
            )
            self.patchees.append(raw_audio_translations_create)


class AudioPatchees(OpenaiPatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        speech = _SpeechPatchees(client)
        transcription = _TranscriptionPatchees(client)
        translation = _TranslationPatchees(client)

        patchees = speech.patchees + transcription.patchees + translation.patchees

        super().__init__(patchees=patchees, client=client)
