import time
from typing import Union


class _TimeTable:

    def __init__(self):
        "track multiple timer specified by key"
        self.__tables = {}

    def set(self, key: str) -> None:
        self.__tables[key] = time.time()

    def latency(self, key: str) -> Union[float, None]:
        """
        return latency in second if key exist, None if key not exist.
        """
        now = time.time()

        start = self.__tables.pop(key, None)
        if start is None:
            return None

        return now - start
