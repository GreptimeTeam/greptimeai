from typing import Any, Dict, List, Union

from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    FunctionCall,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from greptimeai import logger


def parse_choices(choices: List[Choice], verbose: bool = True) -> List[Dict[str, Any]]:
    def _parse_choice(choice: Choice) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "index": choice.index,
            "finish_reason": choice.finish_reason,
        }
        if verbose:
            res["message"] = _parse_chat_completion_message(choice.message)
        return res

    return list([_parse_choice(choice) for choice in choices])


def _parse_chat_completion_message(message: ChatCompletionMessage) -> Dict[str, Any]:
    msg = {
        "role": message.role,
        "content": message.content,
    }
    if message.function_call:
        msg["function_call"] = _parse_function(message.function_call)
    if message.tool_calls:
        tool_calls = list(
            [
                _parse_chat_completion_message_tool_call(tool)
                for tool in message.tool_calls
            ]
        )
        msg["tool_calls"] = tool_calls
    return msg


def _parse_function(func: Union[Function, FunctionCall]) -> Dict[str, str]:
    return {
        "name": func.name,
        "arguments": func.arguments,
    }


def _parse_chat_completion_message_tool_call(
    tool: ChatCompletionMessageToolCall,
) -> Dict[str, Any]:
    tool_dict: Dict[str, Any] = {
        "id": tool.id,
        "type": tool.type,
    }
    if tool.function:
        tool_dict["function"] = _parse_function(tool.function)
    return tool_dict


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
