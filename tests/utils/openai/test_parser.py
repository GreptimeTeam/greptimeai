from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    Function as FunctionParam,
)

from greptimeai.utils.openai.parser import parse_choices, parse_message_params


def test_parse_choices():
    function = Function(name="fake_func_name", arguments="{}")

    tool_call = ChatCompletionMessageToolCall(
        id="fake_id", type="function", function=function
    )

    message = ChatCompletionMessage(
        role="assistant",
        content="hello Python",
        function_call=None,
        tool_calls=[tool_call],
    )
    choice = Choice(finish_reason="tool_calls", index=0, message=message)
    choice=choice.model_dump()

    expect = [
        {
            "finish_reason": "tool_calls",
            "index": 0,
            "message": {
                "content": "hello Python",
                "role": "assistant",
                "function_call": None,
                "tool_calls": [
                    {
                        "id": "fake_id",
                        "function": {"arguments": "{}", "name": "fake_func_name"},
                        "type": "function",
                    }
                ],
            },
        }
    ]
    assert expect == parse_choices([choice])

    expect_verbose_false = [
        {"index": 0, "finish_reason": "tool_calls", "message": "..."}
    ]
    assert expect_verbose_false == parse_choices([choice], False)


def test_parse_chat_completion_message_params():
    function = FunctionParam(name="fake_func_name", arguments="{}")
    tool_call = ChatCompletionMessageToolCallParam(
        id="fake_id", type="function", function=function
    )
    param = ChatCompletionAssistantMessageParam(
        role="assistant", content="fake", tool_calls=[tool_call]
    )
    expect = [
        {
            "role": "assistant",
            "content": "fake",
            "tool_calls": [
                {
                    "id": "fake_id",
                    "type": "function",
                    "function": {"name": "fake_func_name", "arguments": "{}"},
                }
            ],
        }
    ]
    assert expect == parse_message_params([param])


test_parse_choices()