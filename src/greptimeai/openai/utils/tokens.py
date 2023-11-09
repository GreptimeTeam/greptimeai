from inspect import modulesbyfile
from typing import List, Tuple

import tiktoken
from langchain.callbacks.openai_info import get_openai_token_cost_for_model

from greptimeai import logger


def cal_openai_token_cost_for_model(
    model_name: str, num_tokens: int, is_completion: bool = False
) -> float:
    """
    just wrapped by try/catch
    """
    try:
        return get_openai_token_cost_for_model(
            model_name=model_name, num_tokens=num_tokens, is_completion=is_completion
        )
    except Exception as e:
        logger.warn(f"calculate cost for '{model_name}' Exception: {e}")
        return 0


def num_tokens_from_messages(messages: str, model="gpt-3.5-turbo-0613") -> int:
    """
    Return the number of tokens used the messages.

    NOTE:

    - this function won't assure the exact the same result of OpenAI
    - this function is only been called by streaming scenario so far

    Refer: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

    TODO(yuanbohan): update more models to match OpenAI update in DevDay 2023.11.6
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warn("model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4
        tokens_per_name = -1
    elif "gpt-3.5-turbo" in model:
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        logger.warn(
            f"num_tokens_from_messages() is not implemented for model {model}, use gpt-3.5-turbo-0613 instead"
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")

    return len(encoding.encode(messages))
