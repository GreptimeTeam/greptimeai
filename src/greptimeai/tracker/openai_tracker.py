import functools
import time
from typing import Any, Dict, Optional

from openai import OpenAI
from openai._base_client import HttpxBinaryResponseContent
from openai._types import Headers
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from opentelemetry.util.types import Attributes

from greptimeai import (
    _COMPLETION_COST_LABEL,
    _COMPLETION_TOKENS_LABEL,
    _ERROR_TYPE_LABEL,
    _MODEL_LABEL,
    _SPAN_NAME_LABEL,
    _PROMPT_COST_LABEl,
    _PROMPT_TOKENS_LABEl,
    logger,
)
from greptimeai.extractor import BaseExtractor
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
from greptimeai.tracker import _GREPTIMEAI_WRAPPED, BaseTracker
from greptimeai.utils.openai.token import (
    get_openai_audio_cost_for_tts,
    get_openai_token_cost_for_model,
    num_characters_for_audio,
)

from . import _GREPTIMEAI_WRAPPED, BaseTracker


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
        self._patch(ChatCompletionExtractor(client, self._verbose))
        self._patch(CompletionExtractor(client, self._verbose))
        self._patch(EmbeddingExtractor(client, self._verbose))
        self._patch(FileListExtractor(client))
        self._patch(FileCreateExtractor(client))
        self._patch(FileDeleteExtractor(client))
        self._patch(FileRetrieveExtractor(client))
        self._patch(FileContentExtractor(client))

    def _patch(self, extractor: BaseExtractor):

    def _patch_audio_speech(self, client: Optional[OpenAI] = None):
        obj = client.audio.speech if client else openai.audio.speech
        self._patch(
            obj,
            "create",
            "audio.speech.create",
            self._pre_audio_speech_extractor,
            self._post_audio_speech_extractor,
        )

    def _patch(
        self,
        obj: object,
        method_name: str,
        span_name: str,
        pre_extractor: Callable,
        post_extractor: Callable,
    ):
        """
        Args:
            extractor: extractor helps to extract useful information from request and response
        """
        func = extractor.get_func()
        if not func:
            logger.warning(f"function '{extractor.get_func_name()}' not found.")
            return

        if hasattr(func, _GREPTIMEAI_WRAPPED):
            logger.warning(
                f"no need to patch '{extractor.get_func_name()}' multiple times."
            )
            return

        # TODO(yuanbohan): to support: stream, async
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            req_extraction = extractor.pre_extract(*args, **kwargs)
            span_id = self._collector.start_span(
                span_id=None,
                parent_id=None,
                span_name=extractor.get_span_name(),
                event_name="start",
                span_attrs=req_extraction.span_attributes,
                event_attrs=req_extraction.event_attributes,
            )
            common_attrs = {_SPAN_NAME_LABEL: extractor.get_span_name()}
            start = time.time()
            try:
                resp = func(*args, **kwargs)
            except Exception as e:
                ex = e
                self._collector._llm_error_count.add(
                    1,
                    {
                        _MODEL_LABEL: req_extraction.span_attributes.get(
                            _MODEL_LABEL, ""
                        ),
                        _ERROR_TYPE_LABEL: ex.__class__.__name__,
                        **common_attrs,
                    },
                )
                raise ex
            finally:
                latency = 1000 * (time.time() - start)
                resp_extraction = extractor.post_extract(resp)
                attrs = {
                    _MODEL_LABEL: resp_extraction.span_attributes.get(_MODEL_LABEL, ""),
                    **common_attrs,
                }
                self._collector.record_latency(latency, attributes=attrs)

                self._collector.end_span(
                    span_id=span_id,  # type: ignore
                    span_name=extractor.get_span_name(),
                    event_name="end",
                    span_attrs=resp_extraction.span_attributes,
                    event_attrs=resp_extraction.event_attributes,
                    ex=ex,
                )

                self._collect_metrics(
                    attrs=attrs, span_attrs=resp_extraction.span_attributes
                )
            return resp

        setattr(wrapper, _GREPTIMEAI_WRAPPED, True)
        extractor.set_func(wrapper)

    def _collect_metrics(self, attrs: Optional[Attributes], span_attrs: Dict[str, Any]):
        """
        Args:

            attrs: for OTLP collection
            span_attrs: attributes to get useful metrics
        """
        prompt_tokens = span_attrs.get(_PROMPT_TOKENS_LABEl, 0)
        prompt_cost = span_attrs.get(_PROMPT_COST_LABEl, 0)
        completion_tokens = span_attrs.get(_COMPLETION_TOKENS_LABEL, 0)
        completion_cost = span_attrs.get(_COMPLETION_COST_LABEL, 0)

        self._collector.collect_metrics(
            prompt_tokens=prompt_tokens,
            prompt_cost=prompt_cost,
            completion_tokens=completion_tokens,
            completion_cost=completion_cost,
            attrs=attrs,
        )

    def _pre_audio_speech_extractor(
        self,
        args,
        *,
        input: str,
        model: str,
        extra_headers: Optional[Headers] = None,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        num_chars = num_characters_for_audio(input)
        span_attrs = {
            _MODEL_LABEL: model,
            _PROMPT_TOKENS_LABEl: num_chars,
            _PROMPT_COST_LABEl: get_openai_audio_cost_for_tts(model, num_chars),
        }

        if extra_headers and extra_headers.get("x-user-id"):
            span_attrs[_USER_ID_LABEL] = extra_headers.get("x-user-id")  # type: ignore

        event_attrs = {
            _MODEL_LABEL: model,
            **kwargs,
        }
        if self._verbose:
            event_attrs["input"] = input

        if args and len(args) > 0:
            event_attrs["args"] = args

        return (span_attrs, event_attrs)

    def _post_audio_speech_extractor(
        self,
        resp: HttpxBinaryResponseContent,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        try:
            event_attrs = resp.json()
        except Exception as e:
            raise e

        return ({}, event_attrs)
