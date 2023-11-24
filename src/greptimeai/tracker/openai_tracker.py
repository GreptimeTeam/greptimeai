import functools
import time
from typing import Optional

from openai import OpenAI

from greptimeai import _MODEL_LABEL, _SPAN_NAME_LABEL
from greptimeai.extractor import BaseExtractor
from greptimeai.extractor.openai_extractor.audio_extractor import (
    SpeechExtractor,
    TranscriptionExtractor,
    TranslationExtractor,
)
from greptimeai.extractor.openai_extractor.chat_completion_extractor import (
    ChatCompletionExtractor,
)
from greptimeai.extractor.openai_extractor.completion_extractor import (
    CompletionExtractor,
)
from greptimeai.extractor.openai_extractor.embedding_extractor import EmbeddingExtractor
from greptimeai.extractor.openai_extractor.file_extractor import (
    FileContentExtractor,
    FileCreateExtractor,
    FileDeleteExtractor,
    FileListExtractor,
    FileRetrieveExtractor,
)
from greptimeai.extractor.openai_extractor.image_extractor import (
    ImageEditExtractor,
    ImageGenerateExtractor,
    ImageVariationExtractor,
)
from greptimeai.tracker import _GREPTIMEAI_WRAPPED, BaseTracker


def setup(
    host: str = "",
    database: str = "",
    token: str = "",
    client: Optional[OpenAI] = None,
):
    """
    patch openai main functions automatically.
    host, database and token is to setup the place to store the data, and the authority.
    They MUST BE set explicitly by passing parameters or system environment variables.
    Args:
        host: if None or empty string, GREPTIMEAI_HOST environment variable will be used.
        database: if None or empty string, GREPTIMEAI_DATABASE environment variable will be used.
        token: if None or empty string, GREPTIMEAI_TOKEN environment variable will be used.
        client: if None, then openai module-level client will be patched.
                Important: We highly recommend instantiating client instances
                instead of relying on the global client.
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

    def setup(self, client: Optional[OpenAI] = None):
        extractors = [
            ChatCompletionExtractor(client, self._verbose),
            CompletionExtractor(client, self._verbose),
            EmbeddingExtractor(client, self._verbose),
            FileListExtractor(client),
            FileCreateExtractor(client),
            FileDeleteExtractor(client),
            FileRetrieveExtractor(client),
            FileContentExtractor(client),
            SpeechExtractor(client),
            TranscriptionExtractor(client),
            TranslationExtractor(client),
            ImageEditExtractor(client),
            ImageGenerateExtractor(client),
            ImageVariationExtractor(client),
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
