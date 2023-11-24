from typing import Optional

import openai
from openai import OpenAI

from greptimeai.extractor.openai_extractor import OpenaiExtractor


class _FineTuningExtractor(OpenaiExtractor):
    def __init__(self, client: Optional[OpenAI], method_name: str):
        obj = client.fine_tuning.jobs if client else openai.fine_tuning.jobs
        span_name = f"fine_tuning.jobs.{method_name}"

        super().__init__(obj=obj, method_name=method_name, span_name=span_name)


class FineTuningListExtractor(_FineTuningExtractor):
    def __init__(self, client: Optional[OpenAI] = None):
        super().__init__(client=client, method_name="list")


class FineTuningCreateExtractor(_FineTuningExtractor):
    def __init__(self, client: Optional[OpenAI] = None):
        super().__init__(client=client, method_name="create")


class FineTuningCancelExtractor(_FineTuningExtractor):
    def __init__(self, client: Optional[OpenAI] = None):
        super().__init__(client=client, method_name="cancel")


class FineTuningRetrieveExtractor(_FineTuningExtractor):
    def __init__(self, client: Optional[OpenAI] = None):
        super().__init__(client=client, method_name="retrieve")


class FineTuningListEventsExtractor(_FineTuningExtractor):
    def __init__(self, client: Optional[OpenAI] = None):
        super().__init__(client=client, method_name="list_events")
