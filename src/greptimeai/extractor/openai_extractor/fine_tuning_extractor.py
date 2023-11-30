from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI

from greptimeai.extractor.openai_extractor import OpenaiExtractor
from greptimeai.tracker import Trackee


class _FineTuningExtractor(OpenaiExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None], method_name: str):
        fine_tuning_jobs = Trackee(
            obj=client.fine_tuning.jobs if client else openai.fine_tuning.jobs,
            method_name=method_name,
            span_name=f"fine_tuning.jobs.{method_name}",
        )

        fine_tuning_raw_jobs = Trackee(
            obj=client.fine_tuning.with_raw_response.jobs
            if client
            else openai.fine_tuning.with_raw_response.jobs,
            method_name=method_name,
            span_name=f"fine_tuning.with_raw_response.jobs.{method_name}",
        )

        fine_tuning_jobs_raw = Trackee(
            obj=client.fine_tuning.jobs.with_raw_response
            if client
            else openai.fine_tuning.jobs.with_raw_response,
            method_name=method_name,
            span_name=f"fine_tuning.jobs.with_raw_response.{method_name}",
        )

        trackees = [
            fine_tuning_jobs,
            fine_tuning_raw_jobs,
            fine_tuning_jobs_raw,
        ]

        if client:
            raw_fine_tuning = Trackee(
                obj=client.with_raw_response.fine_tuning.jobs,
                method_name=method_name,
                span_name=f"with_raw_response.fine_tuning.jobs.{method_name}",
            )
            trackees.append(raw_fine_tuning)

        super().__init__(client=client, trackees=trackees)


class FineTuningListExtractor(_FineTuningExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="list")


class FineTuningCreateExtractor(_FineTuningExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="create")


class FineTuningCancelExtractor(_FineTuningExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="cancel")


class FineTuningRetrieveExtractor(_FineTuningExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="retrieve")


class FineTuningListEventsExtractor(_FineTuningExtractor):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="list_events")
