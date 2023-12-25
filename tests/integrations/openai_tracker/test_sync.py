import json
import uuid
from typing import List

import pytest
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from greptimeai import collector
from greptimeai.utils.openai.token import num_tokens_from_messages
from . import sync_client
from ..database.db import truncate_tables, get_trace_data_with_retry


@pytest.fixture
def _truncate_tables():
    truncate_tables()
    yield


def test_chat_completion(_truncate_tables):
    user_id = str(uuid.uuid4())
    model = "gpt-3.5-turbo"
    resp = sync_client.chat.completions.create(
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

    collector.otel._force_flush()

    trace = get_trace_data_with_retry(user_id, 3)

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


def test_chat_completion_stream(_truncate_tables):
    user_id = str(uuid.uuid4())
    msg: List[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": "1+1=",
        }
    ]
    resp = sync_client.chat.completions.create(
        messages=msg,
        model="gpt-3.5-turbo",
        user=user_id,
        seed=1,
        stream=True,
    )

    prompt_tokens_num = num_tokens_from_messages(msg or "")

    ans = ""
    model = ""
    for item in resp:
        if isinstance(item, ChatCompletionChunk):
            model = item.model
            for choice in item.choices:
                if choice.delta.content:
                    ans += choice.delta.content

    assert ans == "2"

    completion_tokens_num = num_tokens_from_messages(ans or "")

    collector.otel._force_flush()

    trace = get_trace_data_with_retry(user_id, 3)

    assert trace is not None

    assert "greptimeai" == trace.get("resource_attributes", {}).get("service.name")
    assert "openai_completion" == trace.get("span_name")
    assert "openai" == trace.get("span_attributes", {}).get("source")

    assert {"client.chat.completions.create", "stream", "end"} == {
        event.get("name") for event in trace.get("span_events", [])
    }

    assert model == trace.get("model")

    assert prompt_tokens_num == trace.get("prompt_tokens")
    assert completion_tokens_num == trace.get("completion_tokens")


def test_chat_completion_with_raw_response(_truncate_tables):
    user_id = str(uuid.uuid4())
    model = "gpt-3.5-turbo"
    resp = sync_client.with_raw_response.chat.completions.create(
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
    data = json.loads(resp.content)
    assert data["choices"][0]["message"]["content"] == "2"

    collector.otel._force_flush()

    trace = get_trace_data_with_retry(user_id, 3)

    assert trace is not None

    assert "greptimeai" == trace.get("resource_attributes", {}).get("service.name")
    assert "openai_completion" == trace.get("span_name")
    assert "openai" == trace.get("span_attributes", {}).get("source")
    assert ["client.with_raw_response.chat.completions.create", "end"] == [
        event.get("name") for event in trace.get("span_events", [])
    ]

    assert data["model"] == trace.get("model")
    assert data["model"].startswith(model)

    assert data["usage"]
    assert data["usage"]["prompt_tokens"] == trace.get("prompt_tokens")
    assert data["usage"]["completion_tokens"] == trace.get("completion_tokens")
