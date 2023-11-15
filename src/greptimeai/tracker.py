from .collection import Collector


class BaseTracker:
    """
    base tracer to collect metrics and traces
    """

    def __init__(
        self,
        host: str = "",
        database: str = "",
        token: str = "",
    ):
        self.collector = Collector(host, database, token)
