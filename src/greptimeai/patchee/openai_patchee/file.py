from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI

from greptimeai.patchee import Patchee

from . import OpenaiPatchees


class _FilePatchees:
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None], method_name: str):
        files = Patchee(
            obj=client.files if client else openai.files,
            method_name=method_name,
            span_name=f"files.{method_name}",
        )

        files_raw = Patchee(
            obj=client.files.with_raw_response
            if client
            else openai.files.with_raw_response,
            method_name=method_name,
            span_name=f"files.with_raw_response.{method_name}",
        )

        self.patchees = [files, files_raw]

        if client:
            raw_files = Patchee(
                obj=client.with_raw_response.files,
                method_name=method_name,
                span_name=f"with_raw_response.files.{method_name}",
            )
            self.patchees.append(raw_files)


class _FileListPatchees(_FilePatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="list")


class _FileCreatePatchees(_FilePatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="create")


class _FileDeletePatchees(_FilePatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="delete")


class _FileRetrievePatchees(_FilePatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="retrieve")


class _FileRetrieveContentPatchees(_FilePatchees):
    """
    `.retrieve_content()` is deprecated, but it should be patched as well.
    The `.content()` method should be used instead
    """

    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="retrieve_content")


class _FileContentPatchees(_FilePatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="content")


class FilePatchees(OpenaiPatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        list = _FileListPatchees(client=client)
        create = _FileCreatePatchees(client=client)
        delete = _FileDeletePatchees(client=client)
        retrieve = _FileRetrievePatchees(client=client)
        retrieve_content = _FileRetrieveContentPatchees(client=client)
        content = _FileContentPatchees(client=client)

        patchees = (
            list.patchees
            + create.patchees
            + delete.patchees
            + retrieve.patchees
            + retrieve_content.patchees
            + content.patchees
        )

        super().__init__(patchees=patchees, client=client)
