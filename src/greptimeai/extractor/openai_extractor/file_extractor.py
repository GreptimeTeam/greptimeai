from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI

from greptimeai.extractor.openai_extractor import OpenaiExtractor
from greptimeai.tracker import Trackee


class _FileExtractor(OpenaiExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None], method_name: str):
        files = Trackee(
            obj=client.files if client else openai.files,
            method_name=method_name,
            span_name=f"files.{method_name}",
        )

        files_raw = Trackee(
            obj=client.files.with_raw_response
            if client
            else openai.files.with_raw_response,
            method_name=method_name,
            span_name=f"files.with_raw_response.{method_name}",
        )

        trackees = [
            files,
            files_raw,
        ]

        if client:
            raw_files = Trackee(
                obj=client.with_raw_response.files,
                method_name=method_name,
                span_name=f"with_raw_response.files.{method_name}",
            )
            trackees.append(raw_files)

        super().__init__(client=client, trackees=trackees)


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
