from greptimeai.utils._socket import get_local_hostname_and_ip


def test_get_local_ip():
    hostname, ip = get_local_hostname_and_ip()
    assert hostname is not None
    assert ip is not None
