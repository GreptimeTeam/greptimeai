import time
import uuid

import pytest
from openai import AsyncOpenAI

from greptimeai import openai_patcher  # type: ignore
from ..database.db import (
    get_trace_data,
    get_metric_data,
    truncate_tables,
)
from ..database.model import (
    Tables,
)

client = AsyncOpenAI()
openai_patcher.setup(client=client)


@pytest.fixture
def _truncate_tables():
    truncate_tables()
    yield


@pytest.mark.asyncio
async def test_chat_completion(_truncate_tables):
    user_id = str(uuid.uuid4())
    resp = await client.chat.completions.create(
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
    print("---------------------------------------", resp)
    assert resp.choices[0].message.content == "2"

    time.sleep(8)
    trace = get_trace_data(user_id, False)
    prompt_token = get_metric_data(Tables.llm_prompt_tokens, resp.model)
    completion_token = get_metric_data(Tables.llm_completion_tokens, resp.model)

    assert resp.model == trace[0]
    assert "openai" == prompt_token[0]
    assert "openai" == completion_token[0]
    if resp.usage:
        assert resp.usage.prompt_tokens == trace[1]
        assert resp.usage.completion_tokens == trace[2]
        assert resp.usage.completion_tokens == completion_token[1]
        assert resp.usage.prompt_tokens == prompt_token[1]
