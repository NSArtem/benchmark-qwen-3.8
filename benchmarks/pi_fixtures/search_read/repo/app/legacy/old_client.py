# Deprecated v1 client kept for reference only; nothing imports it.
TIMEOUT = 30


def fetch(url, transport):
    return transport(url, timeout=TIMEOUT)
