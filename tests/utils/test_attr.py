import openai

from greptimeai.utils.attr import get_attr


def test_get_chained_attr():
    attr = get_attr(openai, ["audio", "transcriptions", "with_raw_response"])
    assert attr is not None

    attr = get_attr(openai.Client(), ["audio", "transcriptions", "with_raw_response"])
    assert attr is not None
