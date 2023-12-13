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
    resp = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "1+1=",
            }
        ],
        model="gpt-3.5-turbo",
        user=user_id,
        seed=1,
        stream=True,
    )

    ans = ""
    model = ""
    for item in resp:
        model = item.model
        for choice in item.choices:
            if choice.delta.content:
                ans += choice.delta.content

    assert ans == "2"
    time.sleep(6)

    trace = get_trace_data(user_id, True)

    assert trace[0] in model  # model
    assert trace[1] == 2  # completion tokens
