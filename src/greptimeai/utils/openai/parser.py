from typing import Any, Dict, List, Sequence, Union

from openai.types import CompletionChoice
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from greptimeai import logger


def parse_choices(
    choices: Sequence[Union[Choice, CompletionChoice]], verbose: bool = True
) -> List[Dict[str, Any]]:
    def _parse_choice(choice: Union[Choice, CompletionChoice]) -> Dict[str, Any]:
        res = choice.model_dump()
        if not verbose:
            if "message" in res:
                res.update({"message": "..."})
            if "text" in res:
                res.update({"text": "..."})
        return res

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
