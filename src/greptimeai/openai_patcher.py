from typing import List, Union

from openai import AsyncOpenAI, OpenAI

from greptimeai import logger
from greptimeai.collector import Collector
from greptimeai.patcher import Patcher
from greptimeai.patcher.openai_patcher.base import (
    _AudioPatcher,
    _ChatCompletionPatcher,
    _CompletionPatcher,
    _FilePatcher,
    _FineTuningPatcher,
    _ImagePatcher,
    _ModelPatcher,
    _ModerationPatcher,
)
from greptimeai.patcher.openai_patcher.retry import _RetryPatcher


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
    collector = Collector(
        service_name="openai", host=host, database=database, token=token
    )
    patchers: List[Patcher] = [
        _AudioPatcher(collector=collector, client=client),
        _ChatCompletionPatcher(collector=collector, client=client),
        _CompletionPatcher(collector=collector, client=client),
        _FilePatcher(collector=collector, client=client),
        _FineTuningPatcher(collector=collector, client=client),
        _ImagePatcher(collector=collector, client=client),
        _ModelPatcher(collector=collector, client=client),
        _ModerationPatcher(collector=collector, client=client),
        _RetryPatcher(collector=collector, client=client),
    ]

    for patcher in patchers:
        patcher.patch()

    logger.info("ready to track openai metrics and traces.")
