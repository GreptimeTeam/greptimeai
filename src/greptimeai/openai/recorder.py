import importlib
import logging
import types
import openai

from functools import wraps

from opentelemetry.trace import Span

import greptimeai.openai as go


class Recorder:
    def __init__(self):
        self._api_base = None
        self._library = None
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
                prompt_usage = {"token_count": 0}
                if not exception:
                    if isinstance(params["prompt"], str):
                        prompt_tokens = self._count_tokens(
                            params["model"], params["prompt"]
                        )
                        if prompt_tokens:
                            prompt_usage["token_count"] = prompt_tokens
                    elif isinstance(params["prompt"], list):
                        for prompt in params["prompt"]:
                            prompt_tokens = self._count_tokens(params["model"], prompt)
                            if prompt_tokens:
                                prompt_usage["token_count"] += prompt_tokens
                span.set_attributes(prompt_usage)
                go._collector.prompt_tokens_count.add(
                    prompt_usage["token_count"], {"model": model}
                )
            return

        if result and "model" in result:
            model = result["model"]
            span.set_attribute("model", model)

        prompt_usage = {}
        completion_usage = {"finish_reason_stop": 0, "finish_reason_length": 0}
        if result and "usage" in result and not exception:
            if "prompt_tokens" in result["usage"]:
                prompt_usage["token_count"] = result["usage"]["prompt_tokens"]
            if "completion_tokens" in result["usage"]:
                completion_usage["token_count"] = result["usage"]["completion_tokens"]
                go._collector.completion_tokens_count.add(
                    result["usage"]["completion_tokens"], {"model": model}
                )

        if "prompt" in params:
            span.set_attributes(prompt_usage)

        if result:
            if "choices" in result:
                for choice in result["choices"]:
                    if "finish_reason" in choice:
                        if choice["finish_reason"] == "stop":
                            completion_usage["finish_reason_stop"] += 1
                        elif choice["finish_reason"] == "length":
                            completion_usage["finish_reason_length"] += 1
        span.set_attributes(completion_usage)

    def _trace_completion_res(self, span: Span, item):
        completion_usage = {
            "finish_reason_stop": 0,
            "finish_reason_length": 0,
            "token_count": 0,
        }
        if item and "choices" in item:
            for choice in item["choices"]:
                if "finish_reason" in choice:
                    if choice["finish_reason"] == "stop":
                        completion_usage["finish_reason_stop"] += 1
                    elif choice["finish_reason"] == "length":
                        completion_usage["finish_reason_length"] += 1
                completion_usage["token_count"] += 1
            span.set_attributes(completion_usage)
            go._collector.completion_tokens_count.add(completion_usage["token_count"])



def _is_generator(obj):
    return obj and isinstance(obj, types.GeneratorType)


def _trace_generator(result, trace_res_func, span):
    for item in result:
        try:
            trace_res_func(span, item)
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
        return go._collector.get_new_span(operation)

    def after(args, kwargs, result, exception, span: Span):
        try:
            if exception is not None:
                span.record_exception(exception)
            trace_req_func(span, args, kwargs, result, exception)

        except Exception:
            logging.debug("trace %s failed", func_name, exc_info=True)

        if not _is_generator(result):
            span.end()
        else:
            result = _trace_generator(result, trace_res_func, span)
        # span.end()
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
