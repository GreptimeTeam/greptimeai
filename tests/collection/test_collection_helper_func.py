import os
from typing import Optional
from unittest import mock


from greptimeai.collection import (
    _JSON_KEYS_IN_OTLP_ATTRIBUTES,
    _extract_token,
    _get_with_default_env,
    _is_valid_otel_attributes_value_type,
    _sanitate_attributes,
)


def test_extract_valid_token():
    username, password = "fake_username", "fake_password"
    token = f"{username}:{password}"

    expected_username, expected_password = _extract_token(token)
    assert expected_username == username
    assert expected_password == password


def test_extract_invalid_token():
    def do_test(token: Optional[str]):
        assert ("", "") == _extract_token(token)

    do_test("")
    do_test("  ")
    do_test(":")
    do_test(None)
    do_test("invalid_token")


@mock.patch.dict(os.environ, {"mock_env_variable": "mock_value"}, clear=True)
def test_get_with_default_env():
    assert _get_with_default_env(None, "not_exist_env") is None
    assert _get_with_default_env("", "not_exist_env") is None
    assert _get_with_default_env("   ", "not_exist_env") is None

    assert _get_with_default_env(None, "mock_env_variable") == "mock_value"
    assert _get_with_default_env("", "mock_env_variable") == "mock_value"
    assert _get_with_default_env("  ", "mock_env_variable") == "mock_value"

    assert _get_with_default_env("real_value", "not_exist_env") == "real_value"
    assert _get_with_default_env("real_value", "mock_env_variable") == "real_value"


def test_is_valid_otlp_value_type():
    assert _is_valid_otel_attributes_value_type("value")
    assert _is_valid_otel_attributes_value_type(True)
    assert _is_valid_otel_attributes_value_type(False)
    assert _is_valid_otel_attributes_value_type(1)
    assert _is_valid_otel_attributes_value_type(1.1)

    assert not _is_valid_otel_attributes_value_type([])
    assert not _is_valid_otel_attributes_value_type({})
    assert not _is_valid_otel_attributes_value_type(None)


def test_sanitate_attributes():
    assert _sanitate_attributes(None) == {}
    assert _sanitate_attributes({}) == {}

    common = {
        "string": "otlp",
        "bool": True,
        "int": 1,
        "float": 1.1,
    }

    attrs = {
        "list": [1, 2, 3],
        "list1": [{"key": "val"}],
        "dict1": {"key": "val"},
    }
    attrs.update(common)

    expected_attrs = {
        "list": [1, 2, 3],
        "list1": '[{"key": "val"}]',
        "dict1": '{"key": "val"}',
        _JSON_KEYS_IN_OTLP_ATTRIBUTES: ["list1", "dict1"],
    }
    expected_attrs.update(common)

    assert _sanitate_attributes(attrs) == expected_attrs
