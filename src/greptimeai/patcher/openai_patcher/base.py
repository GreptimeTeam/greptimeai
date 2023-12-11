import functools
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from typing_extensions import override

from greptimeai import logger
from greptimeai.collector import Collector
from greptimeai.extractor import Extraction
from greptimeai.extractor.openai_extractor import OpenaiExtractor
from greptimeai.labels import (
    _MODEL_LABEL,
    _SPAN_NAME_LABEL,
    _PROMPT_COST_LABEl,
    _PROMPT_TOKENS_LABEl,
)
from greptimeai.patchee import Patchee
from greptimeai.patchee.openai_patchee import OpenaiPatchees
from greptimeai.patchee.openai_patchee.audio import AudioPatchees
from greptimeai.patchee.openai_patchee.chat_completion import ChatCompletionPatchees
from greptimeai.patchee.openai_patchee.completion import CompletionPatchees
from greptimeai.patchee.openai_patchee.embedding import EmbeddingPatchees
from greptimeai.patchee.openai_patchee.file import FilePatchees
from greptimeai.patchee.openai_patchee.fine_tuning import FineTuningPatchees
from greptimeai.patchee.openai_patchee.image import ImagePatchees
from greptimeai.patchee.openai_patchee.model import ModelPatchees
from greptimeai.patchee.openai_patchee.moderation import ModerationPatchees
from greptimeai.patcher import Patcher
from greptimeai.patcher.openai_patcher.stream import AsyncStream_, Stream_
from greptimeai.utils.openai.token import (
    get_openai_token_cost_for_model,
    num_tokens_from_messages,
)


class _OpenaiPatcher(Patcher):
    def __init__(
        self,
        patchees: OpenaiPatchees,  # specify what methods to be patched
        collector: Collector,  # collect metrics and traces
        extractor: Optional[OpenaiExtractor] = None,  # extract info from req and resp
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        self.collector = collector
        self.extractor = extractor or OpenaiExtractor()
        self._is_async = False
        if isinstance(client, AsyncOpenAI):
            self._is_async = True

        prefix = "client" if client else "openai"
        suffix = "[async]" if self._is_async else ""
        for patchee in patchees.get_patchees():
            patchee.event_name = f"{prefix}.{patchee.event_name}{suffix}"

        self.patchees = patchees

    def _collect_req_metrics_for_stream(
        self, model_name: Optional[str], span_name: str, tokens: Optional[str]
    ):
        model_name = model_name or ""
        attrs = {
            _SPAN_NAME_LABEL: f"{span_name}[stream]",
            _MODEL_LABEL: model_name,
        }

        num = num_tokens_from_messages(tokens or "")
        cost = get_openai_token_cost_for_model(model_name, num)

        span_attrs = {
            _PROMPT_TOKENS_LABEl: num,
            _PROMPT_COST_LABEl: cost,
        }

        self.collector.collect_metrics(span_attrs=span_attrs, attrs=attrs)

    def _pre_patch(
        self,
        patchee: Patchee,
        *args,
        **kwargs,
    ) -> Tuple[Extraction, str, float, Dict[str, Any]]:
        extraction = self.extractor.pre_extract(*args, **kwargs)
        trace_id, span_id = self.collector.start_span(
            span_name=patchee.span_name,
            event_name=patchee.event_name,
            span_attrs=extraction.span_attributes,
            event_attrs=extraction.event_attributes,
        )
        OpenaiExtractor.update_trace_info(kwargs, trace_id, span_id)

        # if stream, the usage won't be included in the resp,
        # so we need to extract and collect it from req for best.
        if OpenaiExtractor.is_stream(**kwargs):
            tokens = OpenaiExtractor.extract_req_tokens(**kwargs)
            self._collect_req_metrics_for_stream(
                model_name=extraction.get_model_name(),
                span_name=patchee.span_name,
                tokens=tokens,
            )

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

        self.collector._collector.record_latency(latency, attributes=attrs)
        self.collector.end_span(
            span_id=span_id,
            span_name=span_name,
            span_attrs=extraction.span_attributes,
            event_attrs=extraction.event_attributes,
            ex=ex,
        )
        self.collector.collect_metrics(
            span_attrs=extraction.span_attributes, attrs=attrs
        )

    def _patch_sync(self, func: Callable, patchee: Patchee):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            extraction, span_id, start, kwargs = self._pre_patch(
                patchee, *args, **kwargs
            )
            resp, ex = None, None
            try:
                resp = func(*args, **kwargs)
            except Exception as e:
                self.collector.collect_error_count(
                    extraction.get_model_name(), patchee.span_name, e
                )
                ex = e
                raise e
            finally:
                if isinstance(resp, Stream):
                    resp = Stream_(
                        stream=resp,
                        collector=self.collector,
                        span_id=span_id,
                        span_name=patchee.span_name,
                        start=start,
                        model_name=extraction.get_model_name(),
                    )
                else:
                    self._post_patch(
                        span_id=span_id,
                        start=start,
                        span_name=patchee.span_name,
                        resp=resp,
                        ex=ex,
                    )
            return resp

        patchee.wrap_func(wrapper)
        logger.debug(f"patched '{patchee}'")

    def _patch_async(self, func: Callable, patchee: Patchee):
        """
        NOTE: this is exactly the same as _patch_sync, but with async/await
        """

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            extraction, span_id, start, kwargs = self._pre_patch(
                patchee, *args, **kwargs
            )
            resp, ex = None, None
            try:
                resp = await func(*args, **kwargs)
            except Exception as e:
                self.collector.collect_error_count(
                    extraction.get_model_name(), patchee.span_name, e
                )
                ex = e
                raise e
            finally:
                if isinstance(resp, AsyncStream):
                    resp = AsyncStream_(
                        astream=resp,
                        collector=self.collector,
                        span_id=span_id,
                        span_name=patchee.span_name,
                        start=start,
                        model_name=extraction.get_model_name(),
                    )
                else:
                    self._post_patch(
                        span_id=span_id,
                        start=start,
                        span_name=patchee.span_name,
                        resp=resp,
                        ex=ex,
                    )
            return resp

        patchee.wrap_func(async_wrapper)
        logger.debug(f"patched '{patchee}'")

    @override
    def patch(self):
        for patchee in self.patchees.get_patchees():
            func = patchee.get_unwrapped_func()
            if not func:
                return

            if self._is_async:
                self._patch_async(func, patchee)
            else:
                self._patch_sync(func, patchee)


class _AudioPatcher(_OpenaiPatcher):
    def __init__(
        self,
        collector: Collector,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = AudioPatchees(client=client)
        super().__init__(collector=collector, patchees=patchees, client=client)


class _ChatCompletionPatcher(_OpenaiPatcher):
    def __init__(
        self,
        collector: Collector,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = ChatCompletionPatchees(client=client)
        super().__init__(collector=collector, patchees=patchees, client=client)


class _CompletionPatcher(_OpenaiPatcher):
    def __init__(
        self,
        collector: Collector,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = CompletionPatchees(client=client)
        super().__init__(collector=collector, patchees=patchees, client=client)


class _EmbeddingPatcher(_OpenaiPatcher):
    def __init__(
        self,
        collector: Collector,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = EmbeddingPatchees(client=client)
        super().__init__(collector=collector, patchees=patchees, client=client)


class _FilePatcher(_OpenaiPatcher):
    def __init__(
        self,
        collector: Collector,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = FilePatchees(client=client)
        super().__init__(collector=collector, patchees=patchees, client=client)


class _FineTuningPatcher(_OpenaiPatcher):
    def __init__(
        self,
        collector: Collector,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = FineTuningPatchees(client=client)
        super().__init__(collector=collector, patchees=patchees, client=client)


class _ImagePatcher(_OpenaiPatcher):
    def __init__(
        self,
        collector: Collector,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = ImagePatchees(client=client)
        super().__init__(collector=collector, patchees=patchees, client=client)


class _ModelPatcher(_OpenaiPatcher):
    def __init__(
        self,
        collector: Collector,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = ModelPatchees(client=client)
        super().__init__(collector=collector, patchees=patchees, client=client)


class _ModerationPatcher(_OpenaiPatcher):
    def __init__(
        self,
        collector: Collector,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = ModerationPatchees(client=client)
        super().__init__(collector=collector, patchees=patchees, client=client)
