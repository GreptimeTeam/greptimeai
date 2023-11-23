from io import BufferedReader
from typing import Optional

import openai
from openai import OpenAI
from openai._base_client import HttpxBinaryResponseContent
from openai.pagination import SyncPage
from openai.types.file_deleted import FileDeleted
from openai.types.file_object import FileObject

from greptimeai.extractor import Extraction
from greptimeai.extractor.openai_extractor import Extractor


class FileListExtractor(Extractor):
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        verbose: bool = True,
    ):
        obj = client.files if client else openai.files
        method_name = "list"
        span_name = "files.list"

        super().__init__(obj=obj, method_name=method_name, span_name=span_name)
        self.verbose = verbose

    def pre_extract(self, *args, **kwargs) -> Extraction:
        extraction = super().pre_extract(*args, **kwargs)

        return extraction

    def post_extract(self, resp: SyncPage[FileObject]) -> Extraction:
        resp_dict = resp.model_dump()
        extraction = super().post_extract(resp_dict)

        return extraction


class FileCreateExtractor(Extractor):
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        verbose: bool = True,
    ):
        obj = client.files if client else openai.files
        method_name = "create"
        span_name = "files.create"

        super().__init__(obj=obj, method_name=method_name, span_name=span_name)
        self.verbose = verbose

    def pre_extract(self, *args, **kwargs) -> Extraction:
        extraction = super().pre_extract(*args, **kwargs)
        file = kwargs["file"]
        if isinstance(file, BufferedReader):
            extraction.event_attributes["file"] = file.read()
        return extraction

    def post_extract(self, resp: FileObject) -> Extraction:
        resp_dict = resp.model_dump()
        extraction = super().post_extract(resp_dict)
        return extraction


class FileDeleteExtractor(Extractor):
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        verbose: bool = True,
    ):
        obj = client.files if client else openai.files
        method_name = "delete"
        span_name = "files.delete"

        super().__init__(obj=obj, method_name=method_name, span_name=span_name)
        self.verbose = verbose

    def pre_extract(self, *args, **kwargs) -> Extraction:
        extraction = super().pre_extract(*args, **kwargs)
        return extraction

    def post_extract(self, resp: FileDeleted) -> Extraction:
        resp_dict = resp.model_dump()
        extraction = super().post_extract(resp_dict)
        return extraction


class FileRetrieveExtractor(Extractor):
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        verbose: bool = True,
    ):
        obj = client.files if client else openai.files
        method_name = "retrieve"
        span_name = "files.retrieve"

        super().__init__(obj=obj, method_name=method_name, span_name=span_name)
        self.verbose = verbose

    def pre_extract(self, *args, **kwargs) -> Extraction:
        extraction = super().pre_extract(*args, **kwargs)
        return extraction

    def post_extract(self, resp: FileObject) -> Extraction:
        resp_dict = resp.model_dump()
        extraction = super().post_extract(resp_dict)
        return extraction


class FileContentExtractor(Extractor):
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        verbose: bool = True,
    ):
        obj = client.files if client else openai.files
        method_name = "content"
        span_name = "files.content"

        super().__init__(obj=obj, method_name=method_name, span_name=span_name)
        self.verbose = verbose

    def pre_extract(self, *args, **kwargs) -> Extraction:
        extraction = super().pre_extract(*args, **kwargs)
        return extraction

    def post_extract(self, resp: HttpxBinaryResponseContent) -> Extraction:
        resp_dict = {"content": resp.content}
        extraction = super().post_extract(resp_dict)
        return extraction
