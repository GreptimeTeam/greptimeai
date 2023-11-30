import functools
import time
from typing import Any, List, Optional, Tuple, Union

from openai import AsyncOpenAI, OpenAI

from greptimeai import _MODEL_LABEL, _SPAN_NAME_LABEL, logger
from greptimeai.extractor.openai import OpenaiExtractor
from greptimeai.trackee import Trackee
from greptimeai.trackee.openai import OpenaiTrackees
from greptimeai.trackee.openai.audio import AudioTrackees
from greptimeai.trackee.openai.chat_completion import ChatCompletionTrackees
from greptimeai.trackee.openai.completion import CompletionTrackees
from greptimeai.trackee.openai.file import FileTrackees
from greptimeai.trackee.openai.fine_tuning import FineTuningTrackees
from greptimeai.trackee.openai.image import ImageTrackees
from greptimeai.trackee.openai.model import ModelTrackees
from greptimeai.trackee.openai.moderation import ModerationTrackees
from greptimeai.tracker import BaseTracker, Extraction


class _OpenaiPatcher:
    def __init__(
        self,
        trackees: OpenaiTrackees,  # specify what methods to be patched
        tracker: BaseTracker,  # collect metrics and traces
        extractor: Optional[OpenaiExtractor] = None,  # extract info from req and resp
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        self.tracker = tracker
        self.extractor = extractor or OpenaiExtractor()
        self.is_async = isinstance(client, AsyncOpenAI)

        prefix = "client" if client else "openai"
        for trackee in trackees.get_trackees():
            trackee.span_name = f"{prefix}.{trackee.span_name}"

            if self.is_async:
                trackee.span_name = f"async_{trackee.span_name}"

        self.trackees = trackees

    def _pre_patch(
        self,
        span_name: str,
        *args,
        **kwargs,
    ) -> Tuple[Extraction, str, float]:
        extraction = self.extractor.pre_extract(*args, **kwargs)
        span_id: str = self.tracker.start_span(span_name, extraction)
        start = time.time()
        return (extraction, span_id, start)

    def _post_patch(
        self,
        span_id: str,
        start: float,
        span_name: str,
        resp: Any,
        ex: Optional[Exception] = None,
    ):
        latency = 1000 * (time.time() - start)
        extraction = self.extractor.post_extract(resp)
        attrs = {_SPAN_NAME_LABEL: span_name}
        model = extraction.get_model_name()
        if model:
            attrs[_MODEL_LABEL] = model

        self.tracker.record_latency(latency, attributes=attrs)
        self.tracker.end_span(span_id, span_name, extraction, ex)
        self.tracker.collect_metrics(span_attrs=extraction.span_attributes, attrs=attrs)

    def _patch_one(self, trackee: Trackee):
        """
        TODO(yuanbohan): to support:
          - [ ] stream
          - [x] async
          - [x] with_raw_response
          - [ ] retry
        """

        func = trackee.get_unwrapped_func()
        if not func:
            return

        span_name = trackee.get_span_name()
        if self.is_async:

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                extraction, span_id, start = self._pre_patch(span_name, *args, **kwargs)
                resp, ex = None, None
                try:
                    resp = await func(*args, **kwargs)
                except Exception as e:
                    self.tracker.collect_error_count(
                        extraction.get_model_name(), span_name, e
                    )
                    ex = e
                    raise e
                finally:
                    self._post_patch(
                        span_id=span_id,
                        start=start,
                        span_name=span_name,
                        resp=resp,
                        ex=ex,
                    )
                return resp

            trackee.wrap_func(async_wrapper)
        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                extraction, span_id, start = self._pre_patch(span_name, *args, **kwargs)
                resp, ex = None, None
                try:
                    resp = func(*args, **kwargs)
                except Exception as e:
                    self.tracker.collect_error_count(
                        extraction.get_model_name(), span_name, e
                    )
                    ex = e
                    raise e
                finally:
                    self._post_patch(
                        span_id=span_id,
                        start=start,
                        span_name=span_name,
                        resp=resp,
                        ex=ex,
                    )
                return resp

            trackee.wrap_func(wrapper)

    def patch(self):
        for trackee in self.trackees.get_trackees():
            self._patch_one(trackee)


class _AudioPatcher(_OpenaiPatcher):
    def __init__(
        self,
        tracker: BaseTracker,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        trackees = AudioTrackees(client=client)
        super().__init__(tracker=tracker, trackees=trackees, client=client)


class _ChatCompletionPatcher(_OpenaiPatcher):
    def __init__(
        self,
        tracker: BaseTracker,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        trackees = ChatCompletionTrackees(client=client)
        super().__init__(tracker=tracker, trackees=trackees, client=client)


class _CompletionPatcher(_OpenaiPatcher):
    def __init__(
        self,
        tracker: BaseTracker,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        trackees = CompletionTrackees(client=client)
        super().__init__(tracker=tracker, trackees=trackees, client=client)


class _FilePatcher(_OpenaiPatcher):
    def __init__(
        self,
        tracker: BaseTracker,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        trackees = FileTrackees(client=client)
        super().__init__(tracker=tracker, trackees=trackees, client=client)


class _FineTuningPatcher(_OpenaiPatcher):
    def __init__(
        self,
        tracker: BaseTracker,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        trackees = FineTuningTrackees(client=client)
        super().__init__(tracker=tracker, trackees=trackees, client=client)


class _ImagePatcher(_OpenaiPatcher):
    def __init__(
        self,
        tracker: BaseTracker,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        trackees = ImageTrackees(client=client)
        super().__init__(tracker=tracker, trackees=trackees, client=client)


class _ModelPatcher(_OpenaiPatcher):
    def __init__(
        self,
        tracker: BaseTracker,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        trackees = ModelTrackees(client=client)
        super().__init__(tracker=tracker, trackees=trackees, client=client)


class _ModerationPatcher(_OpenaiPatcher):
    def __init__(
        self,
        tracker: BaseTracker,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        trackees = ModerationTrackees(client=client)
        super().__init__(tracker=tracker, trackees=trackees, client=client)


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
    tracker = BaseTracker(host, database, token)
    patchers: List[_OpenaiPatcher] = [
        _AudioPatcher(tracker=tracker, client=client),
        _ChatCompletionPatcher(tracker=tracker, client=client),
        _CompletionPatcher(tracker=tracker, client=client),
        _FilePatcher(tracker=tracker, client=client),
        _FineTuningPatcher(tracker=tracker, client=client),
        _ImagePatcher(tracker=tracker, client=client),
        _ModelPatcher(tracker=tracker, client=client),
        _ModerationPatcher(tracker=tracker, client=client),
    ]

    for patcher in patchers:
        patcher.patch()

    logger.info("ready to track openai metrics and traces.")
