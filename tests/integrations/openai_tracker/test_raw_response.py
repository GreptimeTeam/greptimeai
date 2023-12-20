import json
import time
import uuid

import pytest

from greptimeai.openai_patcher import _collector

from ..database.db import get_trace_data, truncate_tables
from ..openai_tracker import client


@pytest.fixture
def _truncate_tables():
    truncate_tables()
    yield


def test_chat_completion(_truncate_tables):
    user_id = str(uuid.uuid4())
    model = "gpt-3.5-turbo"
    resp = client.with_raw_response.chat.completions.create(
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

    _collector._collector._force_flush()

    trace = get_trace_data(user_id)
    retry = 0
    while retry < 3 and not trace:
        retry += 1
        time.sleep(2)
        trace = get_trace_data(user_id)

    assert trace is not None

    assert "openai" == trace.get("resource_attributes", {}).get("service.name")
    assert "openai_completion" == trace.get("span_name")
    assert ["client.with_raw_response.chat.completions.create", "end"] == [
        event.get("name") for event in trace.get("span_events", [])
    ]

    assert data["model"] == trace.get("model")
    assert data["model"].startswith(model)

    assert data["usage"]
    assert data["usage"]["prompt_tokens"] == trace.get("prompt_tokens")
    assert data["usage"]["completion_tokens"] == trace.get("completion_tokens")
