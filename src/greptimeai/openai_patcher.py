from typing import List, TypedDict, Union

import openai
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


class Options(TypedDict, total=False):
    file: bool
    fine_tuning: bool
    embedding: bool
    model: bool
    moderation: bool


def setup(
    host: str = "",
    database: str = "",
    token: str = "",
    client: Union[OpenAI, AsyncOpenAI, None] = None,
    options: Options = {},
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
        options: options to patch specific functions.
    """
    collector = Collector(
        host=host,
        database=database,
        token=token,
        source="openai",
        source_version=openai.__version__,
    )
    patchers: List[Patcher] = [
        _AudioPatcher(collector=collector, client=client),
        _ChatCompletionPatcher(collector=collector, client=client),
        _CompletionPatcher(collector=collector, client=client),
        _ImagePatcher(collector=collector, client=client),
        _RetryPatcher(collector=collector, client=client),
    ]

    if options.get("file", False):
        patchers.append(_FilePatcher(collector=collector, client=client))

    if options.get("fine_tuning", False):
        patchers.append(_FineTuningPatcher(collector=collector, client=client))

    if options.get("model", False):
        patchers.append(_ModelPatcher(collector=collector, client=client))

    if options.get("embedding", False):
        patchers.append(_EmbeddingPatcher(collector=collector, client=client))

    if options.get("moderation", False):
        patchers.append(_ModerationPatcher(collector=collector, client=client))

    for patcher in patchers:
        patcher.patch()

    logger.info("ready to track openai metrics and traces.")
