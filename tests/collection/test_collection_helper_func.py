import os
from typing import Optional
from unittest import mock

import pytest

from greptimeai.collector import (
    _JSON_KEYS_IN_OTLP_ATTRIBUTES,
    _check_with_env,
    _extract_token,
    _is_valid_otel_attributes_value_type,
    _prefix_with_scheme_if_not_found,
    _sanitate_attributes,
)


def test_extract_token():
    # valid token
    username, password = "fake_username", "fake_password"
    token = f"{username}:{password}"

    expected_username, expected_password = _extract_token(token)
    assert expected_username == username
    assert expected_password == password

    # invalid token
    def do_test(token: Optional[str]):
        assert ("", "") == _extract_token(token)

    do_test("")
    do_test("  ")
    do_test(":")
    do_test(None)
    do_test("invalid_token")


@mock.patch.dict(os.environ, {"mock_env_variable": "mock_value"}, clear=True)
def test_check_with_env():
    assert _check_with_env("var_name", None, "not_exist_env", False) == ""
    assert _check_with_env("val_name", "", "not_exist_env", False) == ""
    assert _check_with_env("val_name", "   ", "not_exist_env", False) == ""

    assert _check_with_env("var_name", None, "mock_env_variable") == "mock_value"
    assert _check_with_env("var_name", "", "mock_env_variable") == "mock_value"
    assert _check_with_env("var_name", "  ", "mock_env_variable") == "mock_value"

    assert _check_with_env("var_name", "real_value", "not_exist_env") == "real_value"
    assert (
        _check_with_env("var_name", "real_value", "mock_env_variable") == "real_value"
    )

    def do_required(value: Optional[str]):
        with pytest.raises(ValueError):
            _check_with_env("var_name", value, "not_exist_env", True)

    do_required(None)
    do_required("")
    do_required("  ")


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
        **common,
    }

    expected_attrs = {
        "list": [1, 2, 3],
        "list1": '[{"key": "val"}]',
        "dict1": '{"key": "val"}',
        _JSON_KEYS_IN_OTLP_ATTRIBUTES: ["list1", "dict1"],
        **common,
    }

    assert _sanitate_attributes(attrs) == expected_attrs


def test_prefix_with_scheme():
    assert (
        _prefix_with_scheme_if_not_found("https://example.com") == "https://example.com"
    )
    assert (
        _prefix_with_scheme_if_not_found("http://example.com") == "http://example.com"
    )
    assert _prefix_with_scheme_if_not_found("example.com") == "https://example.com"
    assert _prefix_with_scheme_if_not_found(None) is None
    assert _prefix_with_scheme_if_not_found("") == ""
    assert _prefix_with_scheme_if_not_found(" ") == ""
