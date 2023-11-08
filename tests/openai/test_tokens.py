from greptimeai.openai.utils.tokens import num_tokens_from_messages


def test_num_tokens():
    cases = [
        (
            18,
            "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",
        ),
        (10, "New synergies will help drive top-line growth."),
        (8, "Things working well together will increase revenue."),
        (
            18,
            "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",
        ),
        (15, "Let's talk later when we're less busy about how to do better."),
        (
            19,
            "This late pivot means we don't have time to boil the ocean for the client deliverable.",
        ),
    ]
    for token_count, message in cases:
        assert token_count == num_tokens_from_messages(message)
