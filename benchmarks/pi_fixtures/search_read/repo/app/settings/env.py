import os

from app.settings import defaults


def request_timeout() -> float:
    value = os.environ.get("LANTERN_TIMEOUT")
    return float(value) if value else float(defaults.REQUEST_TIMEOUT_SECONDS)


def max_retries() -> int:
    value = os.environ.get("LANTERN_RETRIES")
    return int(value) if value else defaults.MAX_RETRIES
