import unittest

from app.net.client import Client
from app.settings import defaults


class ClientTests(unittest.TestCase):
    def test_uses_default_timeout(self):
        seen = []
        Client(lambda url, timeout: seen.append(timeout) or "ok").get("http://x")
        self.assertEqual(seen, [float(defaults.REQUEST_TIMEOUT_SECONDS)])

    def test_retries_then_raises(self):
        calls = []

        def transport(url, timeout):
            calls.append(url)
            raise TimeoutError("slow")

        with self.assertRaises(TimeoutError):
            Client(transport).get("http://x")
        self.assertEqual(len(calls), defaults.MAX_RETRIES + 1)


if __name__ == "__main__":
    unittest.main()
