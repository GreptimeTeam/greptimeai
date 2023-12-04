from greptimeai import logger

# from https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/callbacks/openai_info.py
# 2023-11-22
MODEL_COST_PER_1K_TOKENS = {
    # GPT-4 input
    "gpt-4": 0.03,
    "gpt-4-0314": 0.03,
    "gpt-4-0613": 0.03,
    "gpt-4-32k": 0.06,
    "gpt-4-32k-0314": 0.06,
    "gpt-4-32k-0613": 0.06,
    "gpt-4-vision-preview": 0.01,
    "gpt-4-1106-preview": 0.01,
    # GPT-4 output
    "gpt-4-completion": 0.06,
    "gpt-4-0314-completion": 0.06,
    "gpt-4-0613-completion": 0.06,
    "gpt-4-32k-completion": 0.12,
    "gpt-4-32k-0314-completion": 0.12,
    "gpt-4-32k-0613-completion": 0.12,
    "gpt-4-vision-preview-completion": 0.03,
    "gpt-4-1106-preview-completion": 0.03,
    # GPT-3.5 input
    "gpt-3.5-turbo": 0.0015,
    "gpt-3.5-turbo-0301": 0.0015,
    "gpt-3.5-turbo-0613": 0.0015,
    "gpt-3.5-turbo-1106": 0.001,
    "gpt-3.5-turbo-instruct": 0.0015,
    "gpt-3.5-turbo-16k": 0.003,
    "gpt-3.5-turbo-16k-0613": 0.003,
    # GPT-3.5 output
    "gpt-3.5-turbo-completion": 0.002,
    "gpt-3.5-turbo-0301-completion": 0.002,
    "gpt-3.5-turbo-0613-completion": 0.002,
    "gpt-3.5-turbo-1106-completion": 0.002,
    "gpt-3.5-turbo-instruct-completion": 0.002,
    "gpt-3.5-turbo-16k-completion": 0.004,
    "gpt-3.5-turbo-16k-0613-completion": 0.004,
    # Azure GPT-35 input
    "gpt-35-turbo": 0.0015,  # Azure OpenAI version of ChatGPT
    "gpt-35-turbo-0301": 0.0015,  # Azure OpenAI version of ChatGPT
    "gpt-35-turbo-0613": 0.0015,
    "gpt-35-turbo-instruct": 0.0015,
    "gpt-35-turbo-16k": 0.003,
    "gpt-35-turbo-16k-0613": 0.003,
    # Azure GPT-35 output
    "gpt-35-turbo-completion": 0.002,  # Azure OpenAI version of ChatGPT
    "gpt-35-turbo-0301-completion": 0.002,  # Azure OpenAI version of ChatGPT
    "gpt-35-turbo-0613-completion": 0.002,
    "gpt-35-turbo-instruct-completion": 0.002,
    "gpt-35-turbo-16k-completion": 0.004,
    "gpt-35-turbo-16k-0613-completion": 0.004,
    # Others
    "text-ada-001": 0.0004,
    "ada": 0.0004,
    "text-babbage-001": 0.0005,
    "babbage": 0.0005,
    "text-curie-001": 0.002,
    "curie": 0.002,
    "text-davinci-003": 0.02,
    "text-davinci-002": 0.02,
    "code-davinci-002": 0.02,
    # Fine Tuned input
    "babbage-002-finetuned": 0.0016,
    "davinci-002-finetuned": 0.012,
    "gpt-3.5-turbo-0613-finetuned": 0.012,
    # Fine Tuned output
    "babbage-002-finetuned-completion": 0.0016,
    "davinci-002-finetuned-completion": 0.012,
    "gpt-3.5-turbo-0613-finetuned-completion": 0.016,
    # Azure Fine Tuned input
    "babbage-002-azure-finetuned": 0.0004,
    "davinci-002-azure-finetuned": 0.002,
    "gpt-35-turbo-0613-azure-finetuned": 0.0015,
    # Azure Fine Tuned output
    "babbage-002-azure-finetuned-completion": 0.0004,
    "davinci-002-azure-finetuned-completion": 0.002,
    "gpt-35-turbo-0613-azure-finetuned-completion": 0.002,
    # Legacy fine-tuned models
    "ada-finetuned-legacy": 0.0016,
    "babbage-finetuned-legacy": 0.0024,
    "curie-finetuned-legacy": 0.012,
    "davinci-finetuned-legacy": 0.12,
    # embedding model
    # refer: https://invertedstone.com/calculators/embedding-pricing-calculator/
    # leave embedding model *-001 cost to 0,
    "text-embedding-ada-002": 0.0004,
}


def standardize_model_name(model_name: str, is_completion: bool = False) -> str:
    """
    Standardize the model name to a format that can be used in the OpenAI API.

    Args:
        model_name: Model name to standardize.
        is_completion: Whether the model is used for completion or not.
            Defaults to False.

    Returns:
        Standardized model name.

    """
    model_name = model_name.lower()
    if ".ft-" in model_name:
        model_name = model_name.split(".ft-")[0] + "-azure-finetuned"
    if ":ft-" in model_name:
        model_name = model_name.split(":")[0] + "-finetuned-legacy"
    if "ft:" in model_name:
        model_name = model_name.split(":")[1] + "-finetuned"
    if model_name == "text-embedding-ada-002-v2":
        model_name = "text-embedding-ada-002"
    if is_completion and (
        model_name.startswith("gpt-4")
        or model_name.startswith("gpt-3.5")
        or model_name.startswith("gpt-35")
        or ("finetuned" in model_name and "legacy" not in model_name)
    ):
        return model_name + "-completion"
    else:
        return model_name


def get_openai_token_cost_for_model(
    model_name: str, num_tokens: int, is_completion: bool = False
) -> float:
    """
    Get the cost in USD for a given model and number of tokens.

    Args:
        model_name: Name of the model
        num_tokens: Number of tokens.
        is_completion: Whether the model is used for completion or not.
            Defaults to False.

    Returns:
        Cost in USD.
    """

    if not model_name:
        logger.warning("failed to get token cost for model name is none")
        return 0

    if not num_tokens:
        logger.warning("failed to get token cost for num_tokens is zero")
        return 0

    model_name = standardize_model_name(model_name, is_completion=is_completion)
    if model_name not in MODEL_COST_PER_1K_TOKENS:
        logger.warning(f"failed to get token cost for '{model_name}' is unsupported")
        return 0
    cost = MODEL_COST_PER_1K_TOKENS[model_name] * (num_tokens / 1000)
    return round(cost, 6)


def num_tokens_from_messages(messages: str, model="gpt-3.5-turbo-0613") -> int:
    """
    Return the number of tokens used the messages.

    NOTE: this function won't assure exact the same result of OpenAI

    Refer: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    import tiktoken

    if not model:
        logger.warning("failed to calculate tokens for message for model name is none")
        return 0

    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        logger.warning("model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(messages))


AUDIO_MODEL_COST_PER_1K_CHARS = {
    "tts-1": 0.015,
    "tts-1-hd": 0.03,
}


def num_characters_for_audio(input: str) -> int:
    """
    The maximum length is 4096 characters.

    Refer: https://platform.openai.com/docs/api-reference/audio/createSpeech
    """
    if not input:
        logger.warning("failed to get audio cost for input is none")
        return 0

    return min(len(input), 4096)


def get_openai_audio_cost_for_tts(model_name: str, num_chars: int) -> float:
    """
    NOTE: this function won't assure exact the same result of OpenAI

    Refer: https://platform.openai.com/docs/api-reference/audio/createSpeech
    """
    if model_name not in AUDIO_MODEL_COST_PER_1K_CHARS:
        logger.warning(f"failed to get ttl cost for '{model_name}' is unsupported")
        return 0

    cost = AUDIO_MODEL_COST_PER_1K_CHARS[model_name] * (num_chars / 1000)
    return round(cost, 6)


def get_openai_audio_cost_for_whisper(seconds: int) -> float:
    """
    cost: 0.006 per minute, rounded to the nearest second
    """
    cost = 0.006 * (seconds / 60)
    return round(cost, 6)
