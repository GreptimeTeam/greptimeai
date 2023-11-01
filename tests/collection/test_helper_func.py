import os
from typing import Optional
from unittest import mock

import pytest

from greptimeai.collection import _extract_token, _get_with_default_env


def test_extract_valid_token():
    username, password = "fake_username", "fake_password"
    token = f"{username}:{password}"

    expected_username, expected_password = _extract_token(token)
    assert expected_username == username
    assert expected_password == password


def test_extract_invalid_token():
    def do_test(token: Optional[str]):
        with pytest.raises(ValueError):
            _extract_token(token)

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