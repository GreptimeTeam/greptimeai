from typing import Optional

import pytest

from greptimeai.collection import _extract_token


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
    do_test(":")
    do_test(None)
    do_test("invalid_token")
