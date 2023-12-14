from typing import List, Union

from openai import AsyncOpenAI, OpenAI

from greptimeai import logger
from greptimeai.collector import Collector
from greptimeai.patcher import Patcher
from greptimeai.patcher.openai_patcher.base import (
    _AudioPatcher,
    _ChatCompletionPatcher,
    _CompletionPatcher,
    _EmbeddingPatcher,
    _FilePatcher,
    _FineTuningPatcher,
    _ImagePatcher,
    _ModelPatcher,
    _ModerationPatcher,
)
from greptimeai.patcher.openai_patcher.retry import _RetryPatcher

_collector: Collector = None  # type: ignore


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
    global _collector
    _collector = Collector(
        service_name="openai", host=host, database=database, token=token
    )
    patchers: List[Patcher] = [
        _AudioPatcher(collector=_collector, client=client),
        _ChatCompletionPatcher(collector=_collector, client=client),
        _CompletionPatcher(collector=_collector, client=client),
        _EmbeddingPatcher(collector=_collector, client=client),
        _FilePatcher(collector=_collector, client=client),
        _FineTuningPatcher(collector=_collector, client=client),
        _ImagePatcher(collector=_collector, client=client),
        _ModelPatcher(collector=_collector, client=client),
        _ModerationPatcher(collector=_collector, client=client),
        _RetryPatcher(collector=_collector, client=client),
    ]

    for patcher in patchers:
        patcher.patch()

    logger.info("ready to track openai metrics and traces.")
