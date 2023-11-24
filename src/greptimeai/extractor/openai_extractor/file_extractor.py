from typing import Optional

import openai
from openai import OpenAI

from greptimeai.extractor.openai_extractor import OpenaiExtractor


# TODO(yuanbohan): verbose for sensitive fields
class FileListExtractor(OpenaiExtractor):
    def __init__(self, client: Optional[OpenAI] = None):
        obj = client.files if client else openai.files
        method_name = "list"
        span_name = "files.list"

        super().__init__(obj=obj, method_name=method_name, span_name=span_name)


class FileCreateExtractor(OpenaiExtractor):
    def __init__(self, client: Optional[OpenAI] = None):
        obj = client.files if client else openai.files
        method_name = "create"
        span_name = "files.create"

        super().__init__(obj=obj, method_name=method_name, span_name=span_name)


class FileDeleteExtractor(OpenaiExtractor):
    def __init__(self, client: Optional[OpenAI] = None):
        obj = client.files if client else openai.files
        method_name = "delete"
        span_name = "files.delete"

        super().__init__(obj=obj, method_name=method_name, span_name=span_name)


class FileRetrieveExtractor(OpenaiExtractor):
    def __init__(self, client: Optional[OpenAI] = None):
        obj = client.files if client else openai.files
        method_name = "retrieve"
        span_name = "files.retrieve"

        super().__init__(obj=obj, method_name=method_name, span_name=span_name)


class FileContentExtractor(OpenaiExtractor):
    def __init__(self, client: Optional[OpenAI] = None):
        obj = client.files if client else openai.files
        method_name = "content"
        span_name = "files.content"

        super().__init__(obj=obj, method_name=method_name, span_name=span_name)
