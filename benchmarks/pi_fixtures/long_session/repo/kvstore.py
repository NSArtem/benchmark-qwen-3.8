"""In-memory key/value store with namespaces, used by the job scheduler."""

import time
from typing import Any, Callable, Dict, Iterator, Optional, Tuple


class KeyNotFound(KeyError):
    pass


class KVStore:
    """A small dictionary-backed store.

    Keys are strings, optionally grouped by namespace ("ns:key").  The
    ``clock`` callable returns the current time in seconds and is injected
    so tests can control time.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._data: Dict[str, Tuple[Any, float]] = {}

    @staticmethod
    def _key(key: str, namespace: Optional[str]) -> str:
        if not key or ":" in key:
            raise ValueError(f"invalid key: {key!r}")
        return f"{namespace}:{key}" if namespace else key

    def set(self, key: str, value: Any, namespace: Optional[str] = None) -> None:
        self._data[self._key(key, namespace)] = (value, self._clock())

    def get(self, key: str, namespace: Optional[str] = None, default: Any = None) -> Any:
        entry = self._data.get(self._key(key, namespace))
        return default if entry is None else entry[0]

    def require(self, key: str, namespace: Optional[str] = None) -> Any:
        full = self._key(key, namespace)
        if full not in self._data:
            raise KeyNotFound(full)
        return self._data[full][0]

    def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        return self._data.pop(self._key(key, namespace), None) is not None

    def keys(self, namespace: Optional[str] = None) -> Iterator[str]:
        prefix = f"{namespace}:" if namespace else None
        for full in sorted(self._data):
            if prefix is None and ":" not in full:
                yield full
            elif prefix is not None and full.startswith(prefix):
                yield full[len(prefix):]

    def updated_at(self, key: str, namespace: Optional[str] = None) -> float:
        full = self._key(key, namespace)
        if full not in self._data:
            raise KeyNotFound(full)
        return self._data[full][1]

    def __len__(self) -> int:
        return len(self._data)
