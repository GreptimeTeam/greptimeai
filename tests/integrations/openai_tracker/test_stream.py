import time
import uuid

import pytest
from openai import OpenAI

from greptimeai import openai_patcher  # type: ignore
from ..database.db import (
    get_trace_data,
    get_metric_data,
    truncate_tables,
)
from ..database.model import (
    Tables,
)

client = OpenAI()
openai_patcher.setup(client=client)


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
    prompt_token = get_metric_data(Tables.llm_prompt_tokens, trace[0])
    completion_token = get_metric_data(Tables.llm_completion_tokens, model)

    assert trace[0] in model  # model
    assert prompt_token[0] == "openai"
    assert completion_token[0] == "openai"
    assert trace[1] == 2  # completion tokens
    assert completion_token[1] == 2  # completion tokens
    assert prompt_token[1] == 16  # prompt tokens
