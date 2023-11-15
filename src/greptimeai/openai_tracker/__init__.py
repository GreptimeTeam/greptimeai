from .tracker import OpenaiTracker


def setup(
    host: str = "",
    database: str = "",
    token: str = "",
):
    tracker = OpenaiTracker(host, database, token)
    tracker.setup()
