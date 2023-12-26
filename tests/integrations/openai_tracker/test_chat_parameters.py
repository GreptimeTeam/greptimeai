import uuid
from typing import Any, Dict, List

import pytest
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from greptimeai import collector

from ..database.db import get_trace_data_with_retry, truncate_tables
from . import sync_client


@pytest.fixture
def _truncate_tables():
    truncate_tables()
    yield


def test_chat_completion_n_gt_1(_truncate_tables):
    user_id = str(uuid.uuid4())
    model = "gpt-3.5-turbo"
    n = 2

    resp = sync_client.chat.completions.create(
        n=n,
        messages=[
            {
                "role": "user",
                "content": "1+1=",
            }
        ],
        model=model,
        user=user_id,
        seed=1,
    )

    assert resp.choices[0].message.content == "2"
    assert len(resp.choices) == n
    assert resp.choices[1].message.content == "2"

    collector.otel._force_flush()

    trace = get_trace_data_with_retry(user_id=user_id, retry=3)

    assert trace is not None

    assert "greptimeai" == trace.get("resource_attributes", {}).get("service.name")
    assert "openai_completion" == trace.get("span_name")
    assert "openai" == trace.get("span_attributes", {}).get("source")

    assert ["client.chat.completions.create", "end"] == [
        event.get("name") for event in trace.get("span_events", [])
    ]

    assert resp.model == trace.get("model")
    assert resp.model.startswith(model)

    assert resp.usage
    assert resp.usage.prompt_tokens == trace.get("prompt_tokens")
    assert resp.usage.completion_tokens == trace.get("completion_tokens")

    for event in trace.get("span_events", []):
        if event.get("name") == "end":
            attrs = event.get("attributes")
            assert attrs
            assert len(attrs["choices"]) == n


def test_chat_completion_max_tokens(_truncate_tables):
    user_id = str(uuid.uuid4())
    model = "gpt-3.5-turbo"
    max_tokens = 3

    resp = sync_client.chat.completions.create(
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": "who are you",
            }
        ],
        model=model,
        user=user_id,
        seed=1,
    )

    collector.otel._force_flush()

    trace = get_trace_data_with_retry(user_id=user_id, retry=3)

    assert trace is not None

    assert "greptimeai" == trace.get("resource_attributes", {}).get("service.name")
    assert "openai_completion" == trace.get("span_name")
    assert "openai" == trace.get("span_attributes", {}).get("source")

    assert ["client.chat.completions.create", "end"] == [
        event.get("name") for event in trace.get("span_events", [])
    ]

    assert resp.model == trace.get("model")
    assert resp.model.startswith(model)

    assert resp.usage
    assert resp.usage.prompt_tokens == trace.get("prompt_tokens")
    assert resp.usage.completion_tokens == trace.get("completion_tokens")
    assert resp.usage.completion_tokens == max_tokens

    for event in trace.get("span_events", []):
        if event.get("name") == "end":
            attrs = event.get("attributes")
            assert attrs
            assert attrs["choices"]
            choice = attrs["choices"][0]
            assert choice
            assert choice.get("finish_reason") == "length"


def test_chat_completion_stop(_truncate_tables):
    user_id = str(uuid.uuid4())
    model = "gpt-3.5-turbo"
    stop = "0"

    resp = sync_client.chat.completions.create(
        stop=stop,
        messages=[
            {
                "role": "user",
                "content": "1+9=",
            }
        ],
        model=model,
        user=user_id,
        seed=1,
    )

    content = resp.choices[0].message.content or stop
    assert stop not in content

    collector.otel._force_flush()

    trace = get_trace_data_with_retry(user_id=user_id, retry=3)

    assert trace is not None

    assert "greptimeai" == trace.get("resource_attributes", {}).get("service.name")
    assert "openai_completion" == trace.get("span_name")
    assert "openai" == trace.get("span_attributes", {}).get("source")

    assert ["client.chat.completions.create", "end"] == [
        event.get("name") for event in trace.get("span_events", [])
    ]

    assert resp.model == trace.get("model")
    assert resp.model.startswith(model)

    assert resp.usage
    assert resp.usage.prompt_tokens == trace.get("prompt_tokens")
    assert resp.usage.completion_tokens == trace.get("completion_tokens")

    for event in trace.get("span_events", []):
        if event.get("name") == "end":
            attrs = event.get("attributes")
            assert attrs
            assert attrs["choices"]
            choice = attrs["choices"][0]
            assert choice
            assert choice.get("finish_reason") == "stop"
            message = choice.get("message")
            assert message
            assert stop not in message.get("content")


def test_chat_completion_tool_call(_truncate_tables):
    user_id = str(uuid.uuid4())
    model = "gpt-3.5-turbo"
    messages: List[ChatCompletionMessageParam] = [
        {"role": "user", "content": "What's the weather like in Boston today?"}
    ]
    tools: List[ChatCompletionToolParam] = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state",
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    ## STEP 1
    resp = sync_client.chat.completions.create(
        messages=messages,
        model=model,
        user=user_id,
        tools=tools,
    )

    choice = resp.choices[0]
    assert "tool_calls" == choice.finish_reason

    assistant_message = choice.message
    tool_calls = assistant_message.tool_calls
    assert tool_calls
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert "get_current_weather" == tool_call.function.name

    collector.otel._force_flush()

    trace = get_trace_data_with_retry(user_id=user_id)

    assert trace is not None

    assert "greptimeai" == trace.get("resource_attributes", {}).get("service.name")
    assert "openai_completion" == trace.get("span_name")
    assert "openai" == trace.get("span_attributes", {}).get("source")

    assert ["client.chat.completions.create", "end"] == [
        event.get("name") for event in trace.get("span_events", [])
    ]

    assert resp.model == trace.get("model")
    assert resp.model.startswith(model)

    assert resp.usage
    assert resp.usage.prompt_tokens == trace.get("prompt_tokens")
    assert resp.usage.completion_tokens == trace.get("completion_tokens")

    for event in trace.get("span_events", []):
        if event.get("name") == "end":
            attrs = event.get("attributes")
            assert attrs
            assert attrs["choices"]
            choice_in_db: Dict[str, Any] = attrs["choices"][0]
            assert choice_in_db
            assert choice_in_db.get("finish_reason") == "tool_calls"

    ## STEP 2
    tool_message = {
        "content": "sunny",
        "role": "tool",
        "tool_call_id": tool_call.id,
    }
    assistant_message_dict = assistant_message.model_dump()
    assistant_message_dict.pop("function_call", None)
    messages.append(assistant_message_dict)  # type: ignore
    messages.append(tool_message)  # type: ignore
    user_id = str(uuid.uuid4())

    resp = sync_client.chat.completions.create(
        messages=messages,
        model=model,
        user=user_id,
        tools=tools,
    )

    choice = resp.choices[0]
    assert "stop" == choice.finish_reason

    content = choice.message.content
    assert content
    assert "sunny" in content
    assert "weather" in content
    assert "boston" in content.lower()

    collector.otel._force_flush()

    trace = get_trace_data_with_retry(user_id=user_id)

    assert trace is not None

    assert "greptimeai" == trace.get("resource_attributes", {}).get("service.name")
    assert "openai_completion" == trace.get("span_name")
    assert "openai" == trace.get("span_attributes", {}).get("source")

    assert ["client.chat.completions.create", "end"] == [
        event.get("name") for event in trace.get("span_events", [])
    ]

    assert resp.model == trace.get("model")
    assert resp.model.startswith(model)

    assert resp.usage
    assert resp.usage.prompt_tokens == trace.get("prompt_tokens")
    assert resp.usage.completion_tokens == trace.get("completion_tokens")

    for event in trace.get("span_events", []):
        if event.get("name") == "end":
            attrs = event.get("attributes")
            assert attrs
            assert attrs["choices"]

            db_choice: Dict[str, Any] = attrs["choices"][0]
            assert db_choice
            assert db_choice.get("finish_reason") == "stop"

            content = db_choice.get("message", {}).get("content")
            assert content
            assert "sunny" in content
            assert "weather" in content
            assert "boston" in content.lower()
