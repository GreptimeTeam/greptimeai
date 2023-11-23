from greptimeai.collection import Collector

_GREPTIMEAI_WRAPPED = "__greptimeai_wrapped__"


class BaseTracker:
    """
    base tracker to collect metrics and traces
    """

    def __init__(
        self,
        service_name: str,
        host: str = "",
        database: str = "",
        token: str = "",
    ):
        self._collector = Collector(
            service_name=service_name, host=host, database=database, token=token
        )
