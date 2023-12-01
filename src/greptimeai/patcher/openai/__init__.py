from typing import List, Union

from openai import AsyncOpenAI, OpenAI

from greptimeai import logger
from greptimeai.patcher import Patcher
from greptimeai.tracker import Tracker

from .base import (
    _AudioPatcher,
    _ChatCompletionPatcher,
    _CompletionPatcher,
    _FilePatcher,
    _FineTuningPatcher,
    _ImagePatcher,
    _ModelPatcher,
    _ModerationPatcher,
)
from .retry import _RetryPatcher


def setup(
    host: str = "",
    database: str = "",
    token: str = "",
    client: Union[OpenAI, AsyncOpenAI, None] = None,
):
    """
    patch openai functions automatically.

    host, database and token is to setup the place to store the data, and the authority.
    They MUST BE set explicitly by passing parameters or system environment variables.

    Args:
        host: if None or empty string, GREPTIMEAI_HOST environment variable will be used.
        database: if None or empty string, GREPTIMEAI_DATABASE environment variable will be used.
        token: if None or empty string, GREPTIMEAI_TOKEN environment variable will be used.
        client: if None, then openai module-level client will be patched.
    """
    tracker = Tracker(host, database, token)
    patchers: List[Patcher] = [
        _AudioPatcher(tracker=tracker, client=client),
        _ChatCompletionPatcher(tracker=tracker, client=client),
        _CompletionPatcher(tracker=tracker, client=client),
        _FilePatcher(tracker=tracker, client=client),
        _FineTuningPatcher(tracker=tracker, client=client),
        _ImagePatcher(tracker=tracker, client=client),
        _ModelPatcher(tracker=tracker, client=client),
        _ModerationPatcher(tracker=tracker, client=client),
        _RetryPatcher(tracker=tracker, client=client),
    ]

    for patcher in patchers:
        patcher.patch()

    logger.info("ready to track openai metrics and traces.")
