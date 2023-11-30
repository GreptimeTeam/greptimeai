from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI

from greptimeai.trackee import Trackee

from . import OpenaiTrackees


class _FileTrackees:
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

        self.trackees = [
            files,
            files_raw,
        ]

        if client:
            raw_files = Trackee(
                obj=client.with_raw_response.files,
                method_name=method_name,
                span_name=f"with_raw_response.files.{method_name}",
            )
            self.trackees.append(raw_files)


class _FileListTrackees(_FileTrackees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="list")


class _FileCreateTrackees(_FileTrackees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="create")


class _FileDeleteTrackees(_FileTrackees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="delete")


class _FileRetrieveTrackees(_FileTrackees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="retrieve")


class _FileRetrieveContentTrackees(_FileTrackees):
    """
    `.retrieve_content()` is deprecated, but it should be tracked as well.
    The `.content()` method should be used instead
    """

    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="retrieve_content")


class _FileContentTrackees(_FileTrackees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="content")


class FileTrackees(OpenaiTrackees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        list_trackees = _FileListTrackees(client=client)
        create_trackees = _FileCreateTrackees(client=client)
        delete_trackees = _FileDeleteTrackees(client=client)
        retrieve_trackees = _FileRetrieveTrackees(client=client)
        retrieve_content_trackees = _FileRetrieveContentTrackees(client=client)
        content_trackees = _FileContentTrackees(client=client)

        trackees = (
            list_trackees.trackees
            + create_trackees.trackees
            + delete_trackees.trackees
            + retrieve_trackees.trackees
            + retrieve_content_trackees.trackees
            + content_trackees.trackees
        )

        super().__init__(trackees=trackees, client=client)
