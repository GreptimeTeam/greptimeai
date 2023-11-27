from typing import Any, Dict, List

from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from greptimeai import logger


def parse_choices(
    choices: List[Dict[str, Any]], verbose: bool = True
) -> List[Dict[str, Any]]:
    def _parse_choice(choice: Dict[str, Any]) -> Dict[str, Any]:
        if not verbose:
            if "message" in choice:
                choice["message"] = "..."
            if "text" in choice:
                choice["text"] = "..."
        return choice

    return list([_parse_choice(choice) for choice in choices])


def parse_message_params(
    messages: List[ChatCompletionMessageParam],
) -> List[Dict[str, Any]]:
    def _parse_chat_completion_message_param(
        message: ChatCompletionMessageParam,
    ) -> Dict[str, Any]:
        try:
            return dict(message)
        except Exception as ex:
            logger.error(
                f"failed to parse parse_chat_completion_message_param {message}, {ex=}"
            )
            return {}

    return list([_parse_chat_completion_message_param(message) for message in messages])
