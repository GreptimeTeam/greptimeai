import logging
from typing import Optional

from greptimeai.collection import Collector

_VERSION = "0.1.4"
_SCOPE_NAME = "greptimeai"

logger = logging.getLogger(_SCOPE_NAME)

collector: Collector = None  # type: ignore


def setup(
    host: Optional[str] = None,
    database: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    insecure: bool = False,
):
    global collector
    collector = Collector(
        host=host,
        database=database,
        username=username,
        password=password,
        insecure=insecure,
    )
