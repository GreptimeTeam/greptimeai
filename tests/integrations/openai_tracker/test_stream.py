import time
import uuid

import pytest

from greptimeai.openai_patcher import _collector
from greptimeai.utils.openai.token import num_tokens_from_messages
from ..database.db import get_trace_data, truncate_tables
from ..openai_tracker import client


@pytest.fixture
def _truncate_tables():
    truncate_tables()
    yield


def test_chat_completion(_truncate_tables):
    user_id = str(uuid.uuid4())
    msg = [
        {
            "role": "user",
            "content": "1+1=",
        }
    ]
    resp = client.chat.completions.create(
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
        model = item.model
        for choice in item.choices:
            if choice.delta.content:
                ans += choice.delta.content

    assert ans == "2"

    completion_tokens_num = num_tokens_from_messages(ans or "")

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

    event_names = ["client.chat.completions.create", "stream", "end"]
    for event in trace.get("span_events", []):
        assert event.get("name") in event_names

    assert model == trace.get("model")

    assert prompt_tokens_num == trace.get("prompt_tokens")
    assert completion_tokens_num == trace.get("completion_tokens")
