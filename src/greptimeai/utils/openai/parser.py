from typing import Any, Dict, List

from openai._response import APIResponse
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam as Param,
)

from greptimeai import logger


def parse_choices(
    choices: List[Dict[str, Any]],
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    def _parse_choice(choice: Dict[str, Any]) -> Dict[str, Any]:
        if not verbose:
            if "message" in choice:
                choice["message"] = "..."
            if "text" in choice:
                choice["text"] = "..."
        return choice

    return list([_parse_choice(choice) for choice in choices])


def parse_message_params(messages: List[Param]) -> List[Dict[str, Any]]:
    def _parse_message_param(message: Param) -> Dict[str, Any]:
        try:
            return dict(message)
        except Exception as ex:
            logger.error(f"failed to parse parse_message_param {message}, {ex=}")
            return {}

    return list([_parse_message_param(message) for message in messages])


def parse_raw_response(resp: APIResponse) -> Dict[str, Any]:
    dict = {
        "headers": resp.headers,
        "status_code": resp.status_code,
        "url": resp.url,
        "method": resp.method,
        "cookies": resp.http_response.cookies,
    }

    try:
        dict["parsed"] = resp.parse()
    except Exception as e:
        logger.error(f"Failed to parse response, {e}")
        dict["parsed"] = {}

    try:
        dict["text"] = resp.text
    except Exception as e:
        logger.error(f"Failed to get response text, {e}")

    return dict
