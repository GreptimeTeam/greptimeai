import functools
import time
from typing import Callable, List, Optional, Tuple, Union

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from typing_extensions import override

from greptimeai import _MODEL_LABEL, _SPAN_NAME_LABEL, logger
from greptimeai.extractor.openai_extractor import (
    OpenaiExtractor,
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
from greptimeai.tracker import BaseTracker, Extraction, Trackee


def setup(
    host: str = "",
    database: str = "",
    token: str = "",
    client: Union[OpenAI, AsyncOpenAI, None] = None,
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
    ):
        super().__init__(
            service_name="openai", host=host, database=database, token=token
        )

    @override
    def setup(
        self,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        extractors: List[OpenaiExtractor] = [
            chat_completion_extractor.ChatCompletionExtractor(client),
            completion_extractor.CompletionExtractor(client),
            embedding_extractor.EmbeddingExtractor(client),
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
            # fine_tuning_extractor.FineTuningListEventsExtractor(client),
            # fine_tuning_extractor.FineTuningCreateExtractor(client),
            # fine_tuning_extractor.FineTuningCancelExtractor(client),
            # fine_tuning_extractor.FineTuningRetrieveExtractor(client),
            # fine_tuning_extractor.FineTuningListExtractor(client),
        ]

        for extractor in extractors:
            for trackee in extractor.trackees:
                func = trackee.get_unwrapped_func()
                if func:
                    if extractor.is_async:
                        self._patch_async(func, extractor, trackee)
                    else:
                        self._patch(func, extractor, trackee)

        logger.info("greptimeai is ready to track openai metrics and traces.")

    def _pre_patch(
        self, extractor: OpenaiExtractor, span_name: str, *args, **kwargs
    ) -> Tuple[Extraction, str, float]:
        extraction = extractor.pre_extract(*args, **kwargs)
        span_id: str = self.start_span(span_name, extraction)  # type: ignore
        start = time.time()
        return (extraction, span_id, start)

    def _post_patch(
        self,
        span_id: str,
        start: float,
        extractor: OpenaiExtractor,
        span_name: str,
        resp: BaseModel,
        ex: Optional[Exception] = None,
    ):
        latency = 1000 * (time.time() - start)
        extraction = extractor.post_extract(resp)
        attrs = {
            _SPAN_NAME_LABEL: span_name
        }
        model = extraction.get_model_name()
        if model:
            attrs[_MODEL_LABEL] = model

        self._collector.record_latency(latency, attributes=attrs)
        self.end_span(span_id, span_name, extraction, ex)
        self.collect_metrics(extraction=extraction, attrs=attrs)

    def _patch(self, func: Callable, extractor: OpenaiExtractor, trackee: Trackee):
        """
        TODO(yuanbohan): to support:
          - [ ] stream
          - [x] async
          - [ ] with_raw_response
          - [ ] error, timeout, retry
          - [ ] trace headers of request and response

        Args:
            extractor: extractor helps to extract useful information from request and response
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            extraction, span_id, start = self._pre_patch(extractor, trackee.get_span_name(), *args, **kwargs)
            resp, ex = None, None
            try:
                resp = func(*args, **kwargs)
            except Exception as e:
                self.collect_error_count(extraction, trackee, e)
                ex = e
                raise e
            finally:
                self._post_patch(span_id=span_id, start=start, extractor=extractor, trackee=trackee resp=resp, ex=ex,)  # type: ignore
            return resp
        trackee.set_func_with_wrapped_attr(wrapper)

    def _patch_async(self, func: Callable, extractor: OpenaiExtractor, trackee: Trackee):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            extraction, span_id, start = self._pre_patch(extractor, trackee.get_span_name(), *args, **kwargs)
            resp, ex = None, None
            try:
                resp = await func(*args, **kwargs)
            except Exception as e:
                self.collect_error_count(extraction, trackee, e)
                ex = e
                raise e
            finally:
                self._post_patch(span_id=span_id, start=start, extractor=extractor, trackee=trackee resp=resp, ex=ex)  # type: ignore
            return resp
        trackee.set_func_with_wrapped_attr(wrapper)
