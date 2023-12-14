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
    )
    assert resp.choices[0].message.content == "2"

    _collector._collector._force_flush()
    time.sleep(2)
    trace = get_trace_data(user_id)

    assert resp.model == trace[0]

    assert resp.usage
    assert resp.usage.prompt_tokens == trace[1]
    assert resp.usage.completion_tokens == trace[2]
