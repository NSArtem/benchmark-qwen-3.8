from app.settings import env


class Client:
    """Tiny HTTP client facade; transport is injected for testing."""

    def __init__(self, transport):
        self.transport = transport

    def get(self, url):
        last_error = None
        for _attempt in range(env.max_retries() + 1):
            try:
                return self.transport(url, timeout=env.request_timeout())
            except TimeoutError as exc:
                last_error = exc
        raise last_error
