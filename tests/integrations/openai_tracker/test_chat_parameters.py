import uuid
import re

import pytest

from greptimeai import collector
from . import sync_client
from ..database.db import truncate_tables, get_trace_data_with_retry


@pytest.fixture
def _truncate_tables():
    truncate_tables()
    yield


def get_lowercase_letters(letters: str) -> str:
    letters = re.sub("[^a-zA-Z]", "", letters)
    return letters.lower()


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
    resp = sync_client.chat.completions.create(
        messages=[
            {"role": "user", "content": "GREPTIMEAI"},
            {"role": "function", "name": "get_lowercase_letters", "content": "letters"},
        ],
        model=model,
        user=user_id,
        seed=1,
        tools=[
            {
                "type": "function",
                "function": {
                    "description": "get lowercase letters",
                    "name": "get_lowercase_letters",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "letters": {
                                "type": "string",
                                "description": "uppercase letters",
                            }
                        },
                        "required": ["letters"],
                    },
                },
            }
        ],
        tool_choice="auto",
    )

    assert resp.choices[0].message.content == "greptimeai"

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
            assert choice.get("finish_reason") == "tool_calls"
            message = choice.get("message")
            assert message
            assert message.get("content") == "greptimeai"
