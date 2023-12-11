import time
import uuid

from openai import OpenAI

from greptimeai import openai_patcher  # type: ignore
from ..database.db import db
from ..database.model import (
    LlmTrace,
    LlmPromptToken,
    LlmCompletionToken,
)

cursor = db.cursor()
client = OpenAI()
openai_patcher.setup(client=client)

trace_sql = "SELECT model,prompt_tokens,completion_tokens FROM %s WHERE user_id = '%s'"
metric_sql = "SELECT model,service_name,greptime_value FROM %s WHERE model = '%s'"


def test_chat_completion():
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

    time.sleep(6)
    cursor.execute(trace_sql % (LlmTrace.table_name, user_id))
    trace = cursor.fetchone()
    cursor.execute(metric_sql % (LlmPromptToken.table_name, resp.model))
    prompt_token = cursor.fetchone()
    cursor.execute(metric_sql % (LlmCompletionToken.table_name, resp.model))
    completion_token = cursor.fetchone()

    if trace is not None:
        assert resp.model == trace[0]
        if resp.usage:
            assert resp.usage.prompt_tokens == trace[1]
            assert resp.usage.completion_tokens == trace[2]

    if prompt_token is not None:
        if resp.usage:
            assert resp.usage.prompt_tokens == prompt_token[2]
        assert "openai" == prompt_token[1]

    if completion_token is not None:
        if resp.usage:
            assert resp.usage.completion_tokens == completion_token[2]
        assert "openai" == completion_token[1]
