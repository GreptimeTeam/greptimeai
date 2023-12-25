import uuid

import pytest
from openai._exceptions import APITimeoutError, NotFoundError, AuthenticationError


from greptimeai import collector
from . import sync_client
from ..database.db import truncate_tables, get_trace_data_with_retry


@pytest.fixture
def _truncate_tables():
    truncate_tables()
    yield


def test_chat_completion_error_timeout(_truncate_tables):
    user_id = str(uuid.uuid4())
    model = "gpt-3.5-turbo"
    try:
        sync_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "1+1=",
                }
            ],
            model=model,
            user=user_id,
            seed=1,
            timeout=0.1,
        )
    except Exception as e:
        assert isinstance(e, APITimeoutError)

    collector.otel._force_flush()

    trace = get_trace_data_with_retry(user_id=user_id, retry=3)

    assert trace is not None

    assert "greptimeai" == trace.get("resource_attributes", {}).get("service.name")
    assert "openai_completion" == trace.get("span_name")
    assert "openai" == trace.get("span_attributes", {}).get("source")

    assert {"client.chat.completions.create", "retry", "exception", "end"} == {
        event.get("name") for event in trace.get("span_events", [])
    }

    for event in trace.get("span_events", []):
        if event.get("name") == "exception":
            attrs = event.get("attributes")
            assert attrs
            assert attrs["exception.type"] == "APITimeoutError"
            assert attrs["exception.message"] == "Request timed out."


def test_chat_completion_error_not_found(_truncate_tables):
    user_id = str(uuid.uuid4())
    wrong_model = "gpt-3.5-turbo________"
    try:
        sync_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "1+1=",
                }
            ],
            model=wrong_model,
            user=user_id,
            seed=1,
        )
    except Exception as e:
        assert isinstance(e, NotFoundError)

    collector.otel._force_flush()

    trace = get_trace_data_with_retry(user_id=user_id, retry=3)

    assert trace is not None

    assert "greptimeai" == trace.get("resource_attributes", {}).get("service.name")
    assert "openai_completion" == trace.get("span_name")
    assert "openai" == trace.get("span_attributes", {}).get("source")

    assert {"client.chat.completions.create", "exception", "end"} == {
        event.get("name") for event in trace.get("span_events", [])
    }

    for event in trace.get("span_events", []):
        if event.get("name") == "exception":
            attrs = event.get("attributes")
            assert attrs
            assert attrs["exception.type"] == "NotFoundError"
            assert "invalid_request_error" in attrs["exception.message"]


def test_chat_completion_error_type(_truncate_tables):
    user_id = str(uuid.uuid4())
    try:
        sync_client.chat.completions.create(
            # no messages or model
            user=user_id,
            seed=1,
        )
    except Exception as e:
        assert isinstance(e, TypeError)

    collector.otel._force_flush()

    trace = get_trace_data_with_retry(user_id=user_id, retry=3)

    assert trace is not None

    assert "greptimeai" == trace.get("resource_attributes", {}).get("service.name")
    assert "openai_completion" == trace.get("span_name")
    assert "openai" == trace.get("span_attributes", {}).get("source")

    assert {"client.chat.completions.create", "exception", "end"} == {
        event.get("name") for event in trace.get("span_events", [])
    }

    for event in trace.get("span_events", []):
        if event.get("name") == "exception":
            attrs = event.get("attributes")
            assert attrs
            assert attrs["exception.type"] == "TypeError"
            assert "Missing required arguments" in attrs["exception.message"]


def test_chat_completion_error_authentication(_truncate_tables):
    user_id = str(uuid.uuid4())
    model = "gpt-3.5-turbo"
    # wrong OPENAI_API_KEY
    sync_client.api_key = "xxx"
    try:
        sync_client.chat.completions.create(
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
    except Exception as e:
        assert isinstance(e, AuthenticationError)

    collector.otel._force_flush()

    trace = get_trace_data_with_retry(user_id=user_id, retry=3)

    assert trace is not None

    assert "greptimeai" == trace.get("resource_attributes", {}).get("service.name")
    assert "openai_completion" == trace.get("span_name")
    assert "openai" == trace.get("span_attributes", {}).get("source")

    assert {"client.chat.completions.create", "exception", "end"} == {
        event.get("name") for event in trace.get("span_events", [])
    }

    for event in trace.get("span_events", []):
        if event.get("name") == "exception":
            attrs = event.get("attributes")
            assert attrs
            assert attrs["exception.type"] == "AuthenticationError"
            assert "invalid_api_key" in attrs["exception.message"]

    sync_client.api_key = None
