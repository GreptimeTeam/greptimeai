import os
from typing import Optional

from greptimeai.openai.collector import Collector



_GREPTIME_HOST_ENV_NAME = "GREPTIMEAI_HOST"
_GREPTIME_DATABASE_ENV_NAME = "GREPTIMEAI_DATABASE"
_GREPTIME_USERNAME_ENV_NAME = "GREPTIMEAI_USERNAME"
_GREPTIME_PASSWORD_ENV_NAME = "GREPTIMEAI_PASSWORD"

_collector: Collector = None


def init(
    resource_name: Optional[str] = None,
    greptimeai_host: Optional[str] = None,
    greptimeai_database: Optional[str] = None,
    greptimeai_username: Optional[str] = None,
    greptimeai_password: Optional[str] = None,
    insecure: bool = False,
    verbose=True,
):
    global _collector

    resource_name = resource_name or "greptimeai-openai"
    host = greptimeai_host or os.getenv(_GREPTIME_HOST_ENV_NAME)
    database = greptimeai_database or os.getenv(_GREPTIME_DATABASE_ENV_NAME)
    username = greptimeai_username or os.getenv(_GREPTIME_USERNAME_ENV_NAME)
    password = greptimeai_password or os.getenv(_GREPTIME_PASSWORD_ENV_NAME)
    scheme = "http" if insecure else "https"
    _collector = Collector(
        resource_name=resource_name,
        host=host,
        database=database,
        username=username,
        password=password,
        scheme=scheme,
    )
    _collector.setup()
