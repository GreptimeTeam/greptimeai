import openai

from greptimeai.utils.attr import get_attr, get_optional_attr


def test_get_chained_attr():
    chains = ["audio", "transcriptions", "with_raw_response"]

    attr = get_attr(openai, chains)
    assert attr is not None

    attr = get_attr(openai.Client(), chains)
    assert attr is not None

    attr = get_optional_attr([openai.Client(), openai], chains)
