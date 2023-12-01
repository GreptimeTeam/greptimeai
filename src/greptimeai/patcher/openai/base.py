import functools
import time
from typing import Any, Dict, Optional, Tuple, Union

from openai import AsyncOpenAI, OpenAI
from typing_extensions import override

from greptimeai import _MODEL_LABEL, _SPAN_NAME_LABEL
from greptimeai.extractor import Extraction
from greptimeai.extractor.openai import OpenaiExtractor
from greptimeai.patchee import Patchee
from greptimeai.patchee.openai import OpenaiPatchees
from greptimeai.patchee.openai.audio import AudioPatchees
from greptimeai.patchee.openai.chat_completion import ChatCompletionPatchees
from greptimeai.patchee.openai.completion import CompletionPatchees
from greptimeai.patchee.openai.file import FilePatchees
from greptimeai.patchee.openai.fine_tuning import FineTuningPatchees
from greptimeai.patchee.openai.image import ImagePatchees
from greptimeai.patchee.openai.model import ModelPatchees
from greptimeai.patchee.openai.moderation import ModerationPatchees
from greptimeai.patcher import Patcher
from greptimeai.tracker import Tracker


class _OpenaiPatcher(Patcher):
    def __init__(
        self,
        patchees: OpenaiPatchees,  # specify what methods to be patched
        tracker: Tracker,  # collect metrics and traces
        extractor: Optional[OpenaiExtractor] = None,  # extract info from req and resp
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        self.tracker = tracker
        self.extractor = extractor or OpenaiExtractor()
        self.is_async = isinstance(client, AsyncOpenAI)

        prefix = "client" if client else "openai"
        for patchee in patchees.get_patchees():
            patchee.span_name = f"{prefix}.{patchee.span_name}"

            if self.is_async:
                patchee.span_name = f"async_{patchee.span_name}"

        self.patchees = patchees

    def _pre_patch(
        self,
        span_name: str,
        *args,
        **kwargs,
    ) -> Tuple[Extraction, str, float, Dict[str, Any]]:
        extraction = self.extractor.pre_extract(*args, **kwargs)
        trace_id, span_id = self.tracker.start_span(span_name, extraction)
        OpenaiExtractor.update_trace_info(kwargs, trace_id, span_id)
        start = time.time()
        return (extraction, span_id, start, kwargs)

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

    def patch_one(self, patchee: Patchee):
        """
        TODO(yuanbohan): to support:
          - [ ] stream
          - [x] async
          - [x] with_raw_response
          - [x] retry
          - [ ] page
        """

        func = patchee.get_unwrapped_func()
        if not func:
            return

        span_name = patchee.get_span_name()
        if self.is_async:

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                extraction, span_id, start, kwargs = self._pre_patch(
                    span_name, *args, **kwargs
                )
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

            patchee.wrap_func(async_wrapper)
        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                extraction, span_id, start, kwargs = self._pre_patch(
                    span_name, *args, **kwargs
                )
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

            patchee.wrap_func(wrapper)

    @override
    def patch(self):
        for patchee in self.patchees.get_patchees():
            self.patch_one(patchee)


class _AudioPatcher(_OpenaiPatcher):
    def __init__(
        self,
        tracker: Tracker,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = AudioPatchees(client=client)
        super().__init__(tracker=tracker, patchees=patchees, client=client)


class _ChatCompletionPatcher(_OpenaiPatcher):
    def __init__(
        self,
        tracker: Tracker,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = ChatCompletionPatchees(client=client)
        super().__init__(tracker=tracker, patchees=patchees, client=client)


class _CompletionPatcher(_OpenaiPatcher):
    def __init__(
        self,
        tracker: Tracker,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = CompletionPatchees(client=client)
        super().__init__(tracker=tracker, patchees=patchees, client=client)


class _FilePatcher(_OpenaiPatcher):
    def __init__(
        self,
        tracker: Tracker,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = FilePatchees(client=client)
        super().__init__(tracker=tracker, patchees=patchees, client=client)


class _FineTuningPatcher(_OpenaiPatcher):
    def __init__(
        self,
        tracker: Tracker,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = FineTuningPatchees(client=client)
        super().__init__(tracker=tracker, patchees=patchees, client=client)


class _ImagePatcher(_OpenaiPatcher):
    def __init__(
        self,
        tracker: Tracker,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = ImagePatchees(client=client)
        super().__init__(tracker=tracker, patchees=patchees, client=client)


class _ModelPatcher(_OpenaiPatcher):
    def __init__(
        self,
        tracker: Tracker,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = ModelPatchees(client=client)
        super().__init__(tracker=tracker, patchees=patchees, client=client)


class _ModerationPatcher(_OpenaiPatcher):
    def __init__(
        self,
        tracker: Tracker,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = ModerationPatchees(client=client)
        super().__init__(tracker=tracker, patchees=patchees, client=client)
