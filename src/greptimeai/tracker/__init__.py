from greptimeai.collection import Collector

_GREPTIMEAI_WRAPPED = "__GREPTIMEAI_WRAPPED__"


class BaseTracker:
    """
    base tracker to collect metrics and traces
    """

    def __init__(
        self,
        host: str = "",
        database: str = "",
        token: str = "",
    ):
        self.collector = Collector(host, database, token)
