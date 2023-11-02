from typing import Optional

from greptimeai.openai.openai_tracker import OpenaiTracker

_openai_tracker: OpenaiTracker = None


def init(
    service_name: Optional[str] = None,
    greptimeai_host: Optional[str] = None,
    greptimeai_database: Optional[str] = None,
    greptimeai_token: Optional[str] = None,
    insecure: bool = False,
):
    global _openai_tracker

    service_name = service_name or "greptimeai-openai"
    _openai_tracker = OpenaiTracker(
        service_name=service_name,
        host=greptimeai_host,
        database=greptimeai_database,
        token=greptimeai_token,
        insecure=insecure,
    )

    _openai_tracker.setup()
