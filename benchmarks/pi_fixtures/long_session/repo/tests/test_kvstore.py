import unittest

from kvstore import KeyNotFound, KVStore


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class KVStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.store = KVStore(clock=self.clock)

    def test_set_get_delete(self):
        self.store.set("a", 1)
        self.assertEqual(self.store.get("a"), 1)
        self.assertTrue(self.store.delete("a"))
        self.assertIsNone(self.store.get("a"))
        self.assertFalse(self.store.delete("a"))

    def test_namespaces(self):
        self.store.set("a", 1, namespace="jobs")
        self.store.set("b", 2)
        self.assertEqual(list(self.store.keys("jobs")), ["a"])
        self.assertEqual(list(self.store.keys()), ["b"])

    def test_require_and_updated_at(self):
        self.store.set("a", 1)
        self.clock.now = 1005.0
        self.store.set("b", 2)
        self.assertEqual(self.store.updated_at("b"), 1005.0)
        with self.assertRaises(KeyNotFound):
            self.store.require("missing")

    def test_invalid_key(self):
        with self.assertRaises(ValueError):
            self.store.set("a:b", 1)


if __name__ == "__main__":
    unittest.main()
