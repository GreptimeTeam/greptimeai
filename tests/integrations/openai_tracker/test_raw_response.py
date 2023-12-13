import json
import time
import uuid

import pytest

from ..database.db import (
    get_trace_data,
    truncate_tables,
)
from ..openai_tracker import client


@pytest.fixture
def _truncate_tables():
    truncate_tables()
    yield


def test_chat_completion(_truncate_tables):
    user_id = str(uuid.uuid4())
    resp = client.with_raw_response.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "1+1=",
            }
        ],
        model="gpt-3.5-turbo",
        user=user_id,
        seed=1,
    )

    data = json.loads(resp.content)
    assert data["choices"][0]["message"]["content"] == "2"

    time.sleep(6)
    trace = get_trace_data(user_id, False)

    assert data["model"] == trace[0]

    if "usage" in data:
        assert data["usage"]["prompt_tokens"] == trace[1]
        assert data["usage"]["completion_tokens"] == trace[2]
