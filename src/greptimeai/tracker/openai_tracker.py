import functools
import time
from typing import Optional, Union

from openai import OpenAI, AsyncOpenAI
from typing_extensions import override

from greptimeai import _MODEL_LABEL, _SPAN_NAME_LABEL
from greptimeai.extractor import BaseExtractor
from greptimeai.extractor.openai_extractor import (
    audio_extractor,
    chat_completion_extractor,
    completion_extractor,
    embedding_extractor,
    file_extractor,
    fine_tuning_extractor,
    image_extractor,
    model_extractor,
    moderation_extractor,
)
from greptimeai.tracker import _GREPTIMEAI_WRAPPED, BaseTracker


def setup(
    host: str = "",
    database: str = "",
    token: str = "",
    client: Optional[Union[OpenAI, AsyncOpenAI]] = None,
):
    """
    patch openai functions automatically.

    host, database and token is to setup the place to store the data, and the authority.
    They MUST BE set explicitly by passing parameters or system environment variables.

    Args:
        host: if None or empty string, GREPTIMEAI_HOST environment variable will be used.
        database: if None or empty string, GREPTIMEAI_DATABASE environment variable will be used.
        token: if None or empty string, GREPTIMEAI_TOKEN environment variable will be used.
        client: if None, then openai module-level client will be patched.
    """
    tracker = OpenaiTracker(host, database, token)
    tracker.setup(client)


class OpenaiTracker(BaseTracker):
    def __init__(
        self,
        host: str = "",
        database: str = "",
        token: str = "",
        verbose: bool = True,
    ):
        super().__init__(
            service_name="openai", host=host, database=database, token=token
        )
        self._verbose = verbose

    @override
    def setup(
        self,
        client: Optional[Union[OpenAI, AsyncOpenAI]] = None,
    ):
        if isinstance(client, AsyncOpenAI):
            self._patch(
                chat_completion_extractor.ChatCompletionExtractor(client, self._verbose)
            )
            return

        extractors = [
            chat_completion_extractor.ChatCompletionExtractor(client, self._verbose),
            completion_extractor.CompletionExtractor(client, self._verbose),
            embedding_extractor.EmbeddingExtractor(client, self._verbose),
            file_extractor.FileListExtractor(client),
            file_extractor.FileCreateExtractor(client),
            file_extractor.FileDeleteExtractor(client),
            file_extractor.FileRetrieveExtractor(client),
            file_extractor.FileContentExtractor(client),
            audio_extractor.SpeechExtractor(client),
            audio_extractor.TranscriptionExtractor(client),
            audio_extractor.TranslationExtractor(client),
            image_extractor.ImageEditExtractor(client),
            image_extractor.ImageGenerateExtractor(client),
            image_extractor.ImageVariationExtractor(client),
            model_extractor.ModelListExtractor(client),
            model_extractor.ModelRetrieveExtractor(client),
            model_extractor.ModelDeleteExtractor(client),
            moderation_extractor.ModerationExtractor(client),
            fine_tuning_extractor.FineTuningListEventsExtractor(client),
            fine_tuning_extractor.FineTuningCreateExtractor(client),
            fine_tuning_extractor.FineTuningCancelExtractor(client),
            fine_tuning_extractor.FineTuningRetrieveExtractor(client),
            fine_tuning_extractor.FineTuningListExtractor(client),
        ]

        for extractor in extractors:
            self._patch(extractor)

    def _patch(self, extractor: BaseExtractor):
        """
        TODO(yuanbohan): to support:
          - stream
          - async
          - with_raw_response
          - error, timeout, retry
          - trace headers of request and response

        Args:
            extractor: extractor helps to extract useful information from request and response
        """
        func = extractor.get_unwrapped_func()
        if not func:
            return

        if hasattr(extractor, "is_async") and extractor.is_async():

            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                req_extraction = extractor.pre_extract(*args, **kwargs)
                span_id = self.start_span(extractor.get_span_name(), req_extraction)
                common_attrs = {_SPAN_NAME_LABEL: extractor.get_span_name()}
                start = time.time()
                ex = None
                try:
                    resp = await func(*args, **kwargs)
                except Exception as e:
                    self.collect_error_count(req_extraction, e, common_attrs)
                    ex = e
                    raise e
                finally:
                    latency = 1000 * (time.time() - start)
                    resp_extraction = extractor.post_extract(resp)
                    attrs = {
                        _MODEL_LABEL: resp_extraction.span_attributes.get(
                            _MODEL_LABEL, ""
                        ),
                        **common_attrs,
                    }
                    self._collector.record_latency(latency, attributes=attrs)
                    self.end_span(span_id, extractor.get_span_name(), resp_extraction, ex)  # type: ignore
                    self.collect_metrics(extraction=resp_extraction, attrs=attrs)
                return resp

            setattr(wrapper, _GREPTIMEAI_WRAPPED, True)
            extractor.set_func(wrapper)
            return

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            req_extraction = extractor.pre_extract(*args, **kwargs)
            span_id = self.start_span(extractor.get_span_name(), req_extraction)
            common_attrs = {_SPAN_NAME_LABEL: extractor.get_span_name()}
            start = time.time()
            ex = None
            try:
                resp = func(*args, **kwargs)
            except Exception as e:
                self.collect_error_count(req_extraction, e, common_attrs)
                ex = e
                raise e
            finally:
                latency = 1000 * (time.time() - start)
                resp_extraction = extractor.post_extract(resp)
                attrs = {
                    _MODEL_LABEL: resp_extraction.span_attributes.get(_MODEL_LABEL, ""),
                    **common_attrs,
                }
                self._collector.record_latency(latency, attributes=attrs)
                self.end_span(span_id, extractor.get_span_name(), resp_extraction, ex)  # type: ignore
                self.collect_metrics(extraction=resp_extraction, attrs=attrs)
            return resp

        setattr(wrapper, _GREPTIMEAI_WRAPPED, True)
        extractor.set_func(wrapper)
