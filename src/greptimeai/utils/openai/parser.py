from typing import Any, Dict, List

from openai.types import CompletionChoice
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from greptimeai import logger


def parse_choices(choices: List[Choice], verbose: bool = True) -> List[Dict[str, Any]]:
    def _parse_choice(choice: Choice) -> Dict[str, Any]:
        res = choice.model_dump()
        if not verbose:
            res.pop("message", None)
        return res

    return list([_parse_choice(choice) for choice in choices])


def parse_completion_choices(
    choices: List[CompletionChoice], verbose: bool = True
) -> List[Dict[str, Any]]:
    def _parse_choice(choice: CompletionChoice) -> Dict[str, Any]:
        res = choice.model_dump()
        if not verbose:
            res.pop("message", None)
        return res

    return list([_parse_choice(choice) for choice in choices])


def parse_chat_completion_message_params(
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
