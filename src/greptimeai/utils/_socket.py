import socket
from typing import Tuple


def get_local_hostname_and_ip() -> Tuple[str, str]:
    hostname = socket.gethostname()
    try:
        ip = socket.gethostbyname(hostname)
    except Exception:
        ip = "127.0.0.1"
    return (hostname, ip)
