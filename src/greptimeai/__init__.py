import logging
from typing import Optional

from .collection import Collector

logger = logging.getLogger("greptimeai")

collector: Collector = None  # type: ignore


def setup(
    host: Optional[str] = None,
    database: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    insecure: bool = False,
):
    global collector
    if collector:
        logger.info("collector has been initiated, no need to setup again.")
        return

    collector = Collector(
        host=host,
        database=database,
        username=username,
        password=password,
        insecure=insecure,
    )
