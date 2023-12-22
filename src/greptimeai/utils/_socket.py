import socket
from typing import Tuple


def get_local_hostname_and_ip() -> Tuple[str, str]:
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    return (hostname, ip)
