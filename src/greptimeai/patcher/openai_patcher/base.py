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


class _OpenaiPatcher(Patcher):
    def __init__(
        self,
        tokens_calculation_needed: bool,
        patchees: OpenaiPatchees,  # specify what methods to be patched
        collector: Collector,  # collect metrics and traces
        extractor: Optional[OpenaiExtractor] = None,  # extract info from req and resp
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        self.tokens_calculation_needed = tokens_calculation_needed
        self.collector = collector
        self.extractor = extractor or OpenaiExtractor(tokens_calculation_needed)
        self._is_async = False
        if isinstance(client, AsyncOpenAI):
            self._is_async = True

        prefix = "client" if client else "openai"
        suffix = "[async]" if self._is_async else ""
        for patchee in patchees.get_patchees():
            patchee.event_name = f"{prefix}.{patchee.event_name}{suffix}"

        self.patchees = patchees

    def _collect_req_metrics_for_stream(
        self, model_name: str, span_name: str, tokens_num: int, cost: float
    ):
        attrs = {
            _SPAN_NAME_LABEL: f"{span_name}[stream]",
            _MODEL_LABEL: model_name,
        }

        span_attrs = {
            _PROMPT_TOKENS_LABEl: tokens_num,
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

        # if stream, the usage won't be included in the resp,
        # so we need to extract and collect it from req for best.
        if self.tokens_calculation_needed and OpenaiExtractor.is_stream(**kwargs):
            model_name = extraction.get_model_name() or ""
            num = extraction.span_attributes.get(_PROMPT_TOKENS_LABEl, 0)
            cost = extraction.span_attributes.get(_PROMPT_COST_LABEl, 0.0)

            self._collect_req_metrics_for_stream(
                model_name=model_name,
                span_name=patchee.span_name,
                tokens_num=num,
                cost=cost,
            )

        trace_id, span_id = self.collector.start_span(
            span_id=None,
            parent_id=None,
            span_name=patchee.span_name,
            event_name=patchee.event_name,
            span_attrs=extraction.span_attributes,
            event_attrs=extraction.event_attributes,
        )

        OpenaiExtractor.update_trace_info(kwargs, trace_id, span_id)
        OpenaiExtractor.pop_out_keyword_args(kwargs)

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
        logger.debug(f"_post_patch:\n{ex=}\n{resp=}")
        latency = 1000 * (time.time() - start)
        extraction = self.extractor.post_extract(resp)
        attrs = {_SPAN_NAME_LABEL: span_name}
        model = extraction.get_model_name()
        if model:
            attrs[_MODEL_LABEL] = model

        self.collector.record_latency(latency, attrs=attrs)
        self.collector.end_span(
            span_id=span_id,
            span_name=span_name,
            event_name="end",
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
        super().__init__(
            tokens_calculation_needed=False,
            collector=collector,
            patchees=patchees,
            client=client,
        )


class _ChatCompletionPatcher(_OpenaiPatcher):
    def __init__(
        self,
        collector: Collector,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = ChatCompletionPatchees(client=client)
        super().__init__(
            tokens_calculation_needed=True,
            collector=collector,
            patchees=patchees,
            client=client,
        )


class _CompletionPatcher(_OpenaiPatcher):
    def __init__(
        self,
        collector: Collector,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = CompletionPatchees(client=client)
        super().__init__(
            tokens_calculation_needed=True,
            collector=collector,
            patchees=patchees,
            client=client,
        )


class _EmbeddingPatcher(_OpenaiPatcher):
    def __init__(
        self,
        collector: Collector,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = EmbeddingPatchees(client=client)
        super().__init__(
            tokens_calculation_needed=False,
            collector=collector,
            patchees=patchees,
            client=client,
        )


class _FilePatcher(_OpenaiPatcher):
    def __init__(
        self,
        collector: Collector,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = FilePatchees(client=client)
        super().__init__(
            tokens_calculation_needed=False,
            collector=collector,
            patchees=patchees,
            client=client,
        )


class _FineTuningPatcher(_OpenaiPatcher):
    def __init__(
        self,
        collector: Collector,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = FineTuningPatchees(client=client)
        super().__init__(
            tokens_calculation_needed=False,
            collector=collector,
            patchees=patchees,
            client=client,
        )


class _ImagePatcher(_OpenaiPatcher):
    def __init__(
        self,
        collector: Collector,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = ImagePatchees(client=client)
        super().__init__(
            tokens_calculation_needed=False,
            collector=collector,
            patchees=patchees,
            client=client,
        )


class _ModelPatcher(_OpenaiPatcher):
    def __init__(
        self,
        collector: Collector,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = ModelPatchees(client=client)
        super().__init__(
            tokens_calculation_needed=False,
            collector=collector,
            patchees=patchees,
            client=client,
        )


class _ModerationPatcher(_OpenaiPatcher):
    def __init__(
        self,
        collector: Collector,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
        patchees = ModerationPatchees(client=client)
        super().__init__(
            tokens_calculation_needed=False,
            collector=collector,
            patchees=patchees,
            client=client,
        )
