import os
import time
import uuid

import pymysql
from openai import OpenAI

from greptimeai import openai_patcher


class LlmTrace(object):
    table_name = "llm_traces_preview_v01"

    trace_id: str
    span_id: str
    parent_span_id: str
    resource_attributes: str
    scope_name: str
    scope_version: str
    scope_attributes: str
    trace_state: str
    span_name: str
    span_kind: str
    span_status_code: str
    span_status_message: str
    span_attributes: str
    span_events: str
    span_links: str
    start: float
    end: float
    user_id: str
    model: str
    prompt_tokens: int
    prompt_cost: float
    completion_tokens: int
    completion_cost: float
    greptime_value: str
    greptime_timestamp: float


class LlmPromptToken(object):
    table_name = "llm_prompt_tokens"

    telemetry_sdk_language: str
    telemetry_sdk_name: str
    telemetry_sdk_version: str
    service_name: str
    span_name: str
    model: str
    greptime_value: str
    greptime_timestamp: float


class LlmPromptTokenCost(object):
    table_name = "llm_prompt_tokens_cost"

    telemetry_sdk_language: str
    telemetry_sdk_name: str
    telemetry_sdk_version: str
    service_name: str
    span_name: str
    model: str
    greptime_value: str
    greptime_timestamp: float


class LlmCompletionToken(object):
    table_name = "llm_completion_tokens"

    telemetry_sdk_language: str
    telemetry_sdk_name: str
    telemetry_sdk_version: str
    service_name: str
    span_name: str
    model: str
    greptime_value: str
    greptime_timestamp: float


class LlmCompletionTokenCost(object):
    table_name = "llm_completion_tokens_cost"

    telemetry_sdk_language: str
    telemetry_sdk_name: str
    telemetry_sdk_version: str
    service_name: str
    span_name: str
    model: str
    greptime_value: str
    greptime_timestamp: float


class Number(object):
    table_name = "number"

    number: int


db = pymysql.connect(
    host=os.getenv("GREPTIMEAI_HOST"),
    user=os.getenv("GREPTIMEAI_USERNAME"),
    passwd=os.getenv("GREPTIMEAI_PASSWORD"),
    port=4002,
    db=os.getenv("GREPTIMEAI_DATABASE"),
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

    time.sleep(5)
    cursor.execute(trace_sql % (LlmTrace.table_name, user_id))
    trace = cursor.fetchone()
    cursor.execute(metric_sql % (LlmPromptToken.table_name, resp.model))
    prompt_token = cursor.fetchone()
    cursor.execute(metric_sql % (LlmCompletionToken.table_name, resp.model))
    completion_token = cursor.fetchone()

    assert resp.model == trace[0]

    assert resp.usage.prompt_tokens == trace[1]
    assert resp.usage.prompt_tokens == prompt_token[2]

    assert resp.usage.completion_tokens == trace[2]
    assert resp.usage.completion_tokens == completion_token[2]

    assert "openai" == prompt_token[1]
    assert "openai" == completion_token[1]


def test_embedding():
    user_id = str(uuid.uuid4())
    resp = client.embeddings.create(
        input="hello world",
        model="text-embedding-ada-002",
        user=user_id,
    )

    time.sleep(5)
    cursor.execute(trace_sql % (LlmTrace.table_name, user_id))
    trace = cursor.fetchone()
    cursor.execute(metric_sql % (LlmPromptToken.table_name, resp.model))
    prompt_token = cursor.fetchone()

    assert resp.model == trace[0]

    assert resp.usage.prompt_tokens == trace[1]
    assert resp.usage.prompt_tokens == prompt_token[2]

    assert "openai" == prompt_token[1]
