import uuid

import pytest

from greptimeai import collector
from . import sync_client
from ..database.db import truncate_tables, get_trace_data_with_retry


@pytest.fixture
def _truncate_tables():
    truncate_tables()
    yield


def test_audio_speech(_truncate_tables):
    user_id = str(uuid.uuid4())
    model = "tts-1"
    text = "hello"
    audio_file = "hello.mp3"
    speed = 0.8
    resp = sync_client.audio.speech.create(
        input=text,
        voice="alloy",
        model=model,
        speed=speed,
        user_id=user_id,  # type: ignore
    )

    assert resp
    resp.stream_to_file(audio_file)

    collector.otel._force_flush()

    trace = get_trace_data_with_retry(user_id=user_id)

    assert trace is not None

    assert "greptimeai" == trace.get("resource_attributes", {}).get("service.name")
    assert "openai_speech" == trace.get("span_name")
    assert "openai" == trace.get("span_attributes", {}).get("source")

    assert ["client.audio.speech.create", "end"] == [
        event.get("name") for event in trace.get("span_events", [])
    ]

    events = trace.get("span_events")
    assert events
    attrs = events[0].get("attributes")
    assert attrs
    assert attrs["input"] == text
    assert attrs["model"] == model
    assert attrs["voice"] == "alloy"
    assert attrs["speed"] == speed


def test_audio_transcription(_truncate_tables):
    user_id = str(uuid.uuid4())
    model = "whisper-1"
    audio_file = "hello.mp3"
    language = "en"

    resp = sync_client.audio.transcriptions.create(
        file=open(audio_file, "rb"),
        model=model,
        language=language,
        user_id=user_id,  # type: ignore
    )

    assert resp
    assert "hello" in resp.text.lower()

    collector.otel._force_flush()

    trace = get_trace_data_with_retry(user_id=user_id)

    assert trace is not None

    assert "greptimeai" == trace.get("resource_attributes", {}).get("service.name")
    assert "openai_transcription" == trace.get("span_name")
    assert "openai" == trace.get("span_attributes", {}).get("source")

    assert ["client.audio.transcriptions.create", "end"] == [
        event.get("name") for event in trace.get("span_events", [])
    ]
    for event in trace.get("span_events", []):
        if event.get("name") == "client.audio.transcriptions.create":
            assert audio_file in event["attributes"]["file"]
            assert event["attributes"]["model"] == model
            assert event["attributes"]["language"] == language

        elif event.get("name") == "end":
            assert event["attributes"]["text"] == resp.text


def test_audio_translation(_truncate_tables):
    user_id = str(uuid.uuid4())
    model = "whisper-1"
    audio_file = "hello.mp3"

    resp = sync_client.audio.translations.create(
        file=open(audio_file, "rb"),
        model=model,
        user_id=user_id,  # type: ignore
    )

    assert resp
    assert "hello" in resp.text.lower()

    collector.otel._force_flush()

    trace = get_trace_data_with_retry(user_id=user_id)

    assert trace is not None
