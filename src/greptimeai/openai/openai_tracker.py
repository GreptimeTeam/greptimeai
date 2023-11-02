import importlib
import logging
import types
from functools import wraps
from typing import Optional

import openai
from opentelemetry.trace import Span

import greptimeai.openai as go
from greptimeai.collection import (
    Collector,
    _PROMPT_TYPE,
    _COMPLETION_TYPE,
)


class OpenaiTracker:
    def __init__(
        self,
        host: Optional[str] = None,
        database: Optional[str] = None,
        token: Optional[str] = None,
        insecure: bool = False,
        service_name: Optional[str] = None,
    ):
        self._api_base = None
        self._library = None
        self._collector: Collector = None
        self._cached_encodings = {}
        self._extra_content_tokens = {
            "gpt-3.5-turbo": 4,
            "gpt-3.5-turbo-0301": 4,
            "gpt-4": 3,
            "gpt-4-0314": 3,
        }
        self._extra_name_tokens = {
            "gpt-3.5-turbo": -1,
            "gpt-3.5-turbo-0301": 1,
            "gpt-4": 1,
            "gpt-4-0314": 1,
        }
        self._extra_reply_tokens = {
            "gpt-3.5-turbo": 3,
            "gpt-3.5-turbo-0301": 3,
            "gpt-4": 3,
            "gpt-4-0314": 3,
        }
        service_name = service_name or "greptimeai-openai"
        self._collector = Collector(
            host=host,
            database=database,
            token=token,
            service_name=service_name,
            insecure=insecure,
        )

    def setup(self):
        self._api_base = openai.api_base

        _instrument_sync(
            openai.Completion,
            "create",
            "openai.Completion.create",
            self._trace_completion_req,
            self._trace_completion_res,
        )

    def _count_tokens(self, model, text):
        if model not in self._cached_encodings:
            try:
                tiktoken = importlib.import_module("tiktoken")
                encoding = tiktoken.encoding_for_model(model)
                self._cached_encodings[model] = encoding
                if encoding:
                    logging.debug("cached encoding for model %s", model)
                else:
                    logging.debug("no encoding returned for model %s", model)
            except ModuleNotFoundError:
                self._cached_encodings[model] = None
                logging.debug(
                    "tiktoken not installed, will not count OpenAI stream tokens."
                )
            except Exception:
                self._cached_encodings[model] = None
                logging.error(
                    "failed to use tiktoken for model %s", model, exc_info=True
                )

        encoding = self._cached_encodings.get(model, None)
        if encoding:
            return len(encoding.encode(text))
        return None

    def _trace_completion_req(
        self,
        span: Span,
        args,
        kwargs,
        result,
        exception,
    ):
        params = kwargs
        model = ""

        span.set_attribute("component", "OpenAI")
        span.set_attribute("endpoint", f"{self._api_base}/completions")

        if "model" in params:
            model = params["model"]
            span.set_attribute("model", model)
        param_names = [
            "model",
            "prompt",
            "max_tokens",
            "temperature",
            "top_p",
            "n",
            "stream",
            "logprobs",
            "echo",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "best_of",
        ]
        for param_name in param_names:
            if param_name in params:
                span.set_attribute(param_name, params[param_name])

        if "stream" in params and params["stream"]:
            if "model" in params and "prompt" in params:
                prompt_usage = {"prompt_tokens": 0, "prompt_cost": 0}
                if not exception:
                    if isinstance(params["prompt"], str):
                        prompt_tokens = self._count_tokens(
                            params["model"], params["prompt"]
                        )
                        if prompt_tokens:
                            prompt_usage["prompt_tokens"] = prompt_tokens
                    elif isinstance(params["prompt"], list):
                        for prompt in params["prompt"]:
                            prompt_tokens = self._count_tokens(params["model"], prompt)
                            if prompt_tokens:
                                prompt_usage["prompt_tokens"] += prompt_tokens
                prompt_usage["prompt_cost"] = calculate_cost(
                    model, prompt_usage["prompt_tokens"], _PROMPT_TYPE
                )
                span.set_attributes(prompt_usage)
                go._openai_tracker._collector.collect_llm_metrics(
                    model,
                    prompt_usage["prompt_tokens"],
                    prompt_usage["prompt_cost"],
                    0,
                    0,
                )
            return

        if result and "model" in result:
            model = result["model"]
            span.set_attribute("model", model)

        prompt_usage = {}
        completion_usage = {
            "finish_reason_stop": 0,
            "finish_reason_length": 0,
            "text": "",
            "completion_cost": 0,
        }
        if result and "usage" in result and not exception:
            if "prompt_tokens" in result["usage"]:
                prompt_usage["prompt_tokens"] = result["usage"]["prompt_tokens"]
                prompt_usage["prompt_cost"] = calculate_cost(
                    model, result["usage"]["prompt_tokens"], _PROMPT_TYPE
                )
                go._openai_tracker._collector.collect_llm_metrics(
                    model,
                    prompt_usage["prompt_tokens"],
                prompt_usage["prompt_cost"],
                    0,
                    0,
                )

            if "completion_tokens" in result["usage"]:
                completion_usage["completion_tokens"] = result["usage"][
                    "completion_tokens"
                ]
                completion_usage["completion_cost"] = calculate_cost(
                    model, result["usage"]["completion_tokens"], _COMPLETION_TYPE
                )

                go._openai_tracker._collector.collect_llm_metrics(
                    model,
                    0,
                    0,
                    completion_usage["completion_tokens"],
                    completion_usage["completion_cost"],
                )

        if "prompt" in params:
            span.set_attributes(prompt_usage)

        if result:
            if "choices" in result:
                for choice in result["choices"]:
                    if "text" in choice:
                        completion_usage["text"] += choice["text"]
                    if "finish_reason" in choice:
                        if choice["finish_reason"] == "stop":
                            completion_usage["finish_reason_stop"] += 1
                        elif choice["finish_reason"] == "length":
                            completion_usage["finish_reason_length"] += 1
        span.set_attributes(completion_usage)

    def _trace_completion_res(self, span: Span, item, data):
        if data is None:
            data = {
                "finish_reason_stop": 0,
                "finish_reason_length": 0,
                "completion_tokens": 0,
                "model": "",
                "text": "",
                "completion_cost": 0,
            }
        if item and "choices" in item:
            if "model" in item:
                data["model"] = item["model"]

            for choice in item["choices"]:
                data["completion_tokens"] += 1
                if "text" in choice:
                    data["text"] += choice["text"]
                if "finish_reason" in choice:
                    if choice["finish_reason"] == "stop":
                        data["finish_reason_stop"] += 1
                        try:
                            tokens = self._count_tokens(data["model"], data["text"])
                            if tokens and tokens != 0:
                                data["completion_tokens"] = tokens
                        finally:
                            data["completion_cost"] = calculate_cost(
                                data["model"], data["completion_tokens"], _COMPLETION_TYPE
                            )
                            span.set_attributes(data)
                            go._openai_tracker._collector.collect_llm_metrics(
                                data["model"],
                                0,
                                0,
                                data["completion_tokens"],
                                data["completion_cost"],
                            )
                    elif choice["finish_reason"] == "length":
                        data["finish_reason_length"] += 1

            return data


def _is_generator(obj):
    return obj and isinstance(obj, types.GeneratorType)


def _trace_generator(result, trace_res_func, span):
    data = None
    for item in result:
        try:
            data = trace_res_func(span, item, data)
        except Exception:
            logging.debug("trace res failed", exc_info=True)
        yield item
    span.end()


def _instrument_sync(
    obj, func_name: str, operation: str, trace_req_func, trace_res_func
):
    """
    instrument openai by wrapping
    :param trace_res_func: collect res
    :param trace_req_func: collect req
    :param obj: openai class
    :param func_name: openai method name
    :param operation: openai class name + method name
    :return:
    """
    if not hasattr(obj, func_name):
        logging.debug("instrument %s failed", func_name)

    openai_func = getattr(obj, func_name)

    def before() -> Span:
        return go._openai_tracker._collector.get_new_span(operation)

    def after(args, kwargs, result, exception, span: Span):
        try:
            if exception is not None:
                span.record_exception(exception)
            trace_req_func(span, args, kwargs, result, exception)
            # trace request params and no-stream response data

        except Exception:
            logging.debug("trace %s failed", func_name, exc_info=True)

        if not _is_generator(result):
            span.end()
        else:
            result = _trace_generator(result, trace_res_func, span)
            # trace stream response data
        return result

    @wraps(openai_func)
    def wrapper(*args, **kwargs):
        result = None
        exception = None

        span = before()

        try:
            result = openai_func(*args, **kwargs)
        except BaseException as e:
            exception = e

        result = after(args, kwargs, result, exception, span)
        if exception:
            raise exception
        return result

    setattr(obj, func_name, wrapper)


def calculate_cost(model: str, tokens, tpe: str):
    cost = 0
    if model in ["gpt-4", "gpt-4-0314"]:
        if tpe == _PROMPT_TYPE:
            cost = tokens * 0.03 / 1000
        else:
            cost = tokens * 0.06 / 1000
    elif model in ["gpt-4-32k", "gpt-4-32k-0314"]:
        if tpe == _PROMPT_TYPE:
            cost = tokens * 0.06 / 1000
        else:
            cost = tokens * 0.12 / 1000
    elif "gpt-3.5-turbo" in model:
        cost = tokens * 0.002 / 1000
    elif "davinci" in model:
        cost = tokens * 0.02 / 1000
    elif "curie" in model:
        cost = tokens * 0.002 / 1000
    elif "babbage" in model:
        cost = tokens * 0.0005 / 1000
    elif "ada" in model:
        cost = tokens * 0.0004 / 1000
    else:
        cost = 0
    return cost
