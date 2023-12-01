import functools
import time
from typing import Optional, Tuple, Union

from openai import AsyncOpenAI, OpenAI
from openai._streaming import Stream, AsyncStream
from pydantic import BaseModel
from typing_extensions import override

from greptimeai import (
    _MODEL_LABEL,
    _SPAN_NAME_LABEL,
    _PROMPT_TOKENS_LABEl,
    _PROMPT_COST_LABEl,
    _COMPLETION_TOKENS_LABEL,
    _COMPLETION_COST_LABEL,
)
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
from greptimeai.tracker import _GREPTIMEAI_WRAPPED, BaseTracker, Extraction
from greptimeai.utils.openai.token import get_openai_token_cost_for_model, _count_tokens


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
        verbose: bool = True,
    ):
        super().__init__(
            service_name="openai", host=host, database=database, token=token
        )
        self._verbose = verbose
        self.prompt_tokens = None
        self.prompt_cost = None

    @override
    def setup(
        self,
        client: Union[OpenAI, AsyncOpenAI, None] = None,
    ):
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

    def _pre_patch(
        self, extractor: OpenaiExtractor, *args, **kwargs
    ) -> Tuple[Extraction, str]:
        extraction = extractor.pre_extract(*args, **kwargs)
        prompt_tokens = extraction.span_attributes.get(_PROMPT_TOKENS_LABEl, None)
        prompt_cost = extraction.span_attributes.get(_PROMPT_COST_LABEl, None)
        if prompt_tokens:
            self.prompt_tokens = prompt_tokens
        if prompt_cost:
            self.prompt_cost = prompt_cost
        span_id: str = self.start_span(extractor.get_span_name(), extraction)  # type: ignore
        return (extraction, span_id)

    def _post_patch(
        self,
        span_id: str,
        start: float,
        extractor: OpenaiExtractor,
        resp: BaseModel,
        ex: Optional[Exception] = None,
    ):
        latency = 1000 * (time.time() - start)
        extraction, resp = extractor.post_extract(resp)
        extraction = self._supplement_prompt(extraction)
        attrs = {
            _SPAN_NAME_LABEL: extractor.get_span_name(),
        }
        model = extraction.get_model_name()
        if model:
            attrs[_MODEL_LABEL] = model

        self._collector.record_latency(latency, attributes=attrs)
        self.end_span(span_id, extractor.get_span_name(), extraction, ex)
        self.collect_metrics(extraction=extraction, attrs=attrs)
        return resp

    def _patch(self, extractor: OpenaiExtractor):
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

        if extractor.is_async:

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                extraction, span_id = self._pre_patch(extractor, *args, **kwargs)
                start = time.time()
                span_name = extractor.get_span_name()
                resp, ex = None, None
                try:
                    resp = await func(*args, **kwargs)
                except Exception as e:
                    attrs = {
                        _SPAN_NAME_LABEL: span_name,
                        _MODEL_LABEL: extraction.get_model_name(),
                    }
                    self.collect_error_count(e, attrs)
                    ex = e
                    raise e
                finally:
                    if is_async_stream(resp):
                        resp = self._trace_async_stream(span_id, span_name, start, resp, ex)
                    else:
                        resp = self._post_patch(span_id, start, extractor, resp, ex)  # type: ignore
                return resp

            setattr(async_wrapper, _GREPTIMEAI_WRAPPED, True)
            extractor.set_func(async_wrapper)
        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                extraction, span_id = self._pre_patch(extractor, *args, **kwargs)
                start = time.time()
                span_name = extractor.get_span_name()
                resp, ex = None, None
                try:
                    resp = func(*args, **kwargs)
                except Exception as e:
                    attrs = {
                        _SPAN_NAME_LABEL: span_name,
                        _MODEL_LABEL: extraction.get_model_name(),
                    }
                    self.collect_error_count(e, attrs)
                    ex = e
                    raise e
                finally:
                    if is_stream(resp):
                        resp = self._trace_stream(span_id, span_name, start, resp, ex)
                    else:
                        resp = self._post_patch(span_id, start, extractor, resp, ex)  # type: ignore

                return resp

            setattr(wrapper, _GREPTIMEAI_WRAPPED, True)
            extractor.set_func(wrapper)

    def _supplement_prompt(self, extraction: Extraction) -> Extraction:
        if self.prompt_tokens:
            prompt_tokens_span = extraction.span_attributes.get(
                _PROMPT_TOKENS_LABEl, None
            )
            prompt_tokens_event = extraction.event_attributes.get(
                _PROMPT_TOKENS_LABEl, None
            )
            if not prompt_tokens_span:
                extraction.span_attributes[_PROMPT_TOKENS_LABEl] = self.prompt_tokens
            if not prompt_tokens_event:
                extraction.event_attributes[_PROMPT_TOKENS_LABEl] = self.prompt_tokens

        if self.prompt_cost:
            prompt_cost_span = extraction.span_attributes.get(_PROMPT_COST_LABEl, None)
            prompt_cost_event = extraction.event_attributes.get(
                _PROMPT_COST_LABEl, None
            )
            if not prompt_cost_span:
                extraction.span_attributes[_PROMPT_COST_LABEl] = self.prompt_cost
            if not prompt_cost_event:
                extraction.event_attributes[_PROMPT_COST_LABEl] = self.prompt_cost

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
                                completion_tokens = _count_tokens(model_str, text)

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
            span_attrs[_MODEL_LABEL] = model_str
            attrs[_MODEL_LABEL] = model_str

        usage = data.get("usage", {})
        if usage and model_str:
            usage = OpenaiExtractor.extract_usage(model_str, usage)
            span_attrs.update(usage)
            data["usage"] = usage

        extraction = Extraction(span_attributes=span_attrs, event_attributes=data)
        extraction = self._supplement_prompt(extraction)

        self._collector.record_latency(latency, attributes=attrs)
        self.end_span(span_id, span_name, extraction, ex)
        self.collect_metrics(extraction=extraction, attrs=attrs)

    async def _trace_async_stream(self, span_id, span_name, start, resp, ex):
        finish_reason_stop = 0
        finish_reason_length = 0
        completion_tokens = 0
        model_str = ""
        text = ""
        latency = 1000 * (time.time() - start)

        async for item in resp:
            yield item
            if hasattr(item, "model_dump"):
                item_dump = item.model_dump()
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
                                completion_tokens = _count_tokens(model_str, text)

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
            span_attrs[_MODEL_LABEL] = model_str
            attrs[_MODEL_LABEL] = model_str

        usage = data.get("usage", {})
        if usage and model_str:
            usage = OpenaiExtractor.extract_usage(model_str, usage)
            span_attrs.update(usage)
            data["usage"] = usage

        extraction = Extraction(span_attributes=span_attrs, event_attributes=data)
        extraction = self._supplement_prompt(extraction)

        self._collector.record_latency(latency, attributes=attrs)
        self.end_span(span_id, span_name, extraction, ex)
        self.collect_metrics(extraction=extraction, attrs=attrs)

def is_stream(obj):
    return obj and isinstance(obj, Stream)


def is_async_stream(obj):
    return obj and isinstance(obj, AsyncStream)
