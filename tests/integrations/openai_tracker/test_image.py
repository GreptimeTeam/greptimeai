import uuid

import pytest

from greptimeai import collector
from . import sync_client
from ..database.db import truncate_tables, get_trace_data_with_retry


@pytest.fixture
def _truncate_tables():
    truncate_tables()
    yield


def test_image(_truncate_tables):
    user_id = str(uuid.uuid4())
    model = "dall-e-2"
    n = 1
    prompt = "ox"
    resp = sync_client.images.generate(
        prompt=prompt,
        size="512x512",
        n=n,
        model=model,
        user=user_id,
    )

    assert resp.data
    url = resp.data[0].url
    assert url

    collector.otel._force_flush()

    trace = get_trace_data_with_retry(user_id=user_id)

    assert trace is not None

    assert "greptimeai" == trace.get("resource_attributes", {}).get("service.name")
    assert "openai_image" == trace.get("span_name")
    assert "openai" == trace.get("span_attributes", {}).get("source")

    assert ["client.images.generate", "end"] == [
        event.get("name") for event in trace.get("span_events", [])
    ]

    for event in trace.get("span_events", []):
        if event.get("name") == "client.audio.images.generate":
            assert event["attributes"]["prompt"] == prompt
            assert event["attributes"]["model"] == model
            assert event["attributes"]["n"] == n
            assert event["attributes"]["size"] == "512x512"

        elif event.get("name") == "end":
            assert event["attributes"]["data"]
            assert event["attributes"]["data"][0]["url"] == url
