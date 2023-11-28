from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI

from greptimeai.extractor.openai_extractor import OpenaiExtractor


class _FileExtractor(OpenaiExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None], method_name: str):
        obj = client.files if client else openai.files
        span_name = f"files.{method_name}"

        super().__init__(
            obj=obj,
            method_name=method_name,
            span_name=span_name,
            client=client,
        )


class FileListExtractor(_FileExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="list")


class FileCreateExtractor(_FileExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="create")


class FileDeleteExtractor(_FileExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="delete")


class FileRetrieveExtractor(_FileExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="retrieve")


class FileRetrieveContentExtractor(_FileExtractor):
    """
    `.retrieve_content()` is deprecated, but it should be tracked as well.
    The `.content()` method should be used instead
    """

    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="retrieve_content")


class FileContentExtractor(_FileExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="content")
