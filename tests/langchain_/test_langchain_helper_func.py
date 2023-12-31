from greptimeai.callbacks.langchain_callback import _get_serialized_id, _get_user_id


def test_get_user_id():
    assert _get_user_id(None) == ""
    assert _get_user_id({}) == ""

    user_id = "fake_user_id"
    assert user_id == _get_user_id({"user_id": user_id})


def test_get_serialized_id():
    serialized = {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain", "chains", "llm", "LLMChain"],
    }

    assert _get_serialized_id({}) is None
    assert "LLMChain" == _get_serialized_id(serialized)

    serialized["id"] = ""
    assert _get_serialized_id({}) is None

    serialized.pop("id", None)
    assert _get_serialized_id({}) is None
