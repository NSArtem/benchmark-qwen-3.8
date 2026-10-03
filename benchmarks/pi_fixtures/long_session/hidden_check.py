"""Hidden evaluator for the multi-prompt kvstore session (run from the repo root)."""

import inspect
import subprocess
import sys

sys.path.insert(0, ".")
from kvstore import KVStore  # noqa: E402


class Clock:
    now = 100.0

    def __call__(self):
        return self.now


clock = Clock()
store = KVStore(clock=clock)
params = inspect.signature(store.set).parameters
if "ttl_seconds" not in params or params["ttl_seconds"].default is not None:
    raise SystemExit(f"set() lacks optional ttl_seconds: {list(params)}")

# Purge first: an implementation may also drop expired entries lazily on
# read, so reads must not happen before purge_expired() is checked.
store.set("short", 1, ttl_seconds=10)
store.set("forever", 2)
store.set("ns_short", 3, namespace="jobs", ttl_seconds=5)
clock.now = 111.0
removed = store.purge_expired()
if removed != 2 or len(store) != 1:
    raise SystemExit(f"purge_expired returned {removed}, len {len(store)}")
if store.purge_expired() != 0:
    raise SystemExit("second purge should remove nothing")
if store.get("forever") != 2:
    raise SystemExit("entry without ttl was purged")

# Expiry visibility through the read API.
clock.now = 100.0
store = KVStore(clock=clock)
store.set("short", 1, ttl_seconds=10)
store.set("forever", 2)
store.set("ns_short", 3, namespace="jobs", ttl_seconds=5)
if store.get("short") != 1 or store.get("ns_short", namespace="jobs") != 3:
    raise SystemExit("values not readable before expiry")
clock.now = 106.0
if store.get("ns_short", namespace="jobs") is not None:
    raise SystemExit("namespaced entry did not expire")
clock.now = 111.0
if store.get("short") is not None or store.get("forever") != 2:
    raise SystemExit("ttl expiry wrong")
if "short" in list(store.keys()):
    raise SystemExit("expired key still listed by keys()")
result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"], capture_output=True, text=True)
if result.returncode:
    raise SystemExit(result.stderr)
tests = open("tests/test_kvstore.py").read()
if "ttl_seconds" not in tests or "purge_expired" not in tests:
    raise SystemExit("tests for ttl/purge were not added")
print("long session ok")
