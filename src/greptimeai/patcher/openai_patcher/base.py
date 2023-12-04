import functools
import time
from typing import Any, Dict, Optional, Tuple, Union

from openai import AsyncOpenAI, OpenAI
from typing_extensions import override

from greptimeai.collector import Collector
from greptimeai.extractor import Extraction
from greptimeai.extractor.openai_extractor import OpenaiExtractor
from greptimeai.labels import (
    _MODEL_LABEL,
    _SPAN_NAME_LABEL,
    _COMPLETION_TOKENS_LABEL,
    _COMPLETION_COST_LABEL,
    _PROMPT_TOKENS_LABEl,
    _PROMPT_COST_LABEl,
)
from greptimeai.patchee import Patchee
from greptimeai.patchee.openai_patchee import OpenaiPatchees
from greptimeai.patchee.openai_patchee.audio import AudioPatchees
from greptimeai.patchee.openai_patchee.chat_completion import ChatCompletionPatchees
from greptimeai.patchee.openai_patchee.completion import CompletionPatchees
from greptimeai.patchee.openai_patchee.file import FilePatchees
from greptimeai.patchee.openai_patchee.fine_tuning import FineTuningPatchees
from greptimeai.patchee.openai_patchee.image import ImagePatchees
from greptimeai.patchee.openai_patchee.model import ModelPatchees
from greptimeai.patchee.openai_patchee.moderation import ModerationPatchees
from greptimeai.patcher import Patcher
from greptimeai.utils.openai.token import get_openai_token_cost_for_model, count_tokens
from greptimeai.utils.openai.stream import StreamUtil


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
        self.is_async = isinstance(client, AsyncOpenAI)
        self.prompt_tokens = None
        self.prompt_cost = None

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
        prompt_tokens = extraction.span_attributes.get(_PROMPT_TOKENS_LABEl, None)
        prompt_cost = extraction.span_attributes.get(_PROMPT_COST_LABEl, None)
        if prompt_tokens:
            self.prompt_tokens = prompt_tokens
        if prompt_cost:
            self.prompt_cost = prompt_cost
        trace_id, span_id = self.collector.start_span(
            span_name=span_name,
            span_attrs=extraction.span_attributes,
            event_attrs=extraction.event_attributes,
        )
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
                    self.collector.collect_error_count(
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
                    self.collector.collect_error_count(
                        extraction.get_model_name(), span_name, e
                    )
                    ex = e
                    raise e
                finally:
                    if StreamUtil.is_stream(resp):
                        resp = self._trace_stream(span_id, span_name, start, resp, ex)

                    else:
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

    def _supplement_prompt(self, extraction: Extraction) -> Extraction:
        if self.prompt_tokens:
            prompt_tokens_span = extraction.span_attributes.get(
                _PROMPT_TOKENS_LABEl, None
            )
            usage = extraction.event_attributes.get("usage", None)
            if not prompt_tokens_span:
                extraction.span_attributes[_PROMPT_TOKENS_LABEl] = self.prompt_tokens
            if usage and _PROMPT_TOKENS_LABEl not in usage:
                extraction.event_attributes["usage"][
                    _PROMPT_TOKENS_LABEl
                ] = self.prompt_tokens

        if self.prompt_cost:
            prompt_cost_span = extraction.span_attributes.get(_PROMPT_COST_LABEl, None)
            usage = extraction.event_attributes.get("usage", None)
            if not prompt_cost_span:
                extraction.span_attributes[_PROMPT_COST_LABEl] = self.prompt_cost
            if usage and _PROMPT_COST_LABEl not in usage:
                extraction.event_attributes["usage"][
                    _PROMPT_COST_LABEl
                ] = self.prompt_cost

        return extraction

    def _trace_stream(self, span_id, span_name, start, resp, ex):
        finish_reason_stop = 0
        finish_reason_length = 0
        completion_tokens = 0
        model_str = ""
        text = ""
        latency = 1000 * (time.time() - start)

        for item in resp:
            yield item
            if hasattr(item, "model_dump"):
                item_dump = item.model_dump()
                event_name = "streaming"
                self.collector.add_span_event(span_id, event_name, item_dump)
                if item_dump and "choices" in item_dump:
                    if "model" in item_dump:
                        model_str = item_dump["model"]
                    for choice in item_dump["choices"]:
                        if "text" in choice:
                            text += choice["text"]
                        elif "delta" in choice and "content" in choice["delta"]:
                            if choice["delta"]["content"]:
                                text += choice["delta"]["content"]

                        if "finish_reason" in choice:
                            if choice["finish_reason"] == "stop":
                                finish_reason_stop += 1
                                completion_tokens = count_tokens(model_str, text)

                            elif choice["finish_reason"] == "length":
                                finish_reason_length += 1
        completion_cost = get_openai_token_cost_for_model(
            model_str, completion_tokens, True
        )
        data = {
            "finish_reason_stop": finish_reason_stop,
            "finish_reason_length": finish_reason_length,
            "model": model_str,
            "text": text,
            "usage": {
                _COMPLETION_TOKENS_LABEL: completion_tokens,
                _COMPLETION_COST_LABEL: completion_cost,
            },
        }
        span_attrs = {}
        attrs = {
            _SPAN_NAME_LABEL: span_name,
        }

        if model_str:
            span_attrs: [Dict[str:Any]] = {_MODEL_LABEL: model_str}
            attrs[_MODEL_LABEL] = model_str

        usage = data.get("usage", None)
        if usage and model_str:
            usage = OpenaiExtractor.extract_usage(model_str, usage)
            span_attrs.update(usage)
            data["usage"] = usage

        extraction = Extraction(span_attributes=span_attrs, event_attributes=data)
        extraction = self._supplement_prompt(extraction)
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
