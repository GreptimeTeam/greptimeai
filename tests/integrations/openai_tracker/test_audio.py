import pytest

from greptimeai import collector
from . import sync_client
from ..database.db import truncate_tables, get_trace_data_with_retry


@pytest.fixture
def _truncate_tables():
    truncate_tables()
    yield


def test_audio_speech(_truncate_tables):
    model = "tts-1"
    text = "你好"
    audio_file = "hello.mp3"
    speed = 0.8
    resp = sync_client.audio.speech.create(
        input=text,
        voice="alloy",
        model=model,
        speed=speed,
    )
    with open(audio_file, "wb") as f:
        f.write(resp.content)

    collector.otel._force_flush()

    trace = get_trace_data_with_retry(user_id="", span_name="openai_speech", retry=3)

    assert trace is not None

    assert "greptimeai" == trace.get("resource_attributes", {}).get("service.name")
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
    model = "whisper-1"
    audio_file = "hello.mp3"
    language = "zh"

    resp = sync_client.audio.transcriptions.create(
        file=open(audio_file, "rb"),
        model=model,
        language=language,
    )

    assert "你好" in resp.text.lower()

    collector.otel._force_flush()

    trace = get_trace_data_with_retry(
        user_id="", span_name="openai_transcription", retry=3
    )

    assert trace is not None

    assert "greptimeai" == trace.get("resource_attributes", {}).get("service.name")
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
    model = "whisper-1"
    audio_file = "hello.mp3"

    resp = sync_client.audio.translations.create(
        file=open(audio_file, "rb"),
        model=model,
    )

    assert "ni hao" or "你好" in resp.text.lower()

    collector.otel._force_flush()

    trace = get_trace_data_with_retry(
        user_id="", span_name="openai_translation", retry=3
    )

    assert trace is not None

    assert "greptimeai" == trace.get("resource_attributes", {}).get("service.name")
    assert "openai" == trace.get("span_attributes", {}).get("source")

    assert ["client.audio.translations.create", "end"] == [
        event.get("name") for event in trace.get("span_events", [])
    ]
    for event in trace.get("span_events", []):
        if event.get("name") == "client.audio.translations.create":
            assert audio_file in event["attributes"]["file"]
            assert event["attributes"]["model"] == model

        elif event.get("name") == "end":
            assert event["attributes"]["text"] == resp.text
