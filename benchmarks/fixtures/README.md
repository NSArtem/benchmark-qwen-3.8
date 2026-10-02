# Long-agent benchmark fixtures

These are deterministic, read-only snapshot files used by
`benchmarks/long_agent.py`. They are never imported or executed, and the
benchmark never writes to the repository. The runner exposes them only
through simulated `read_file` tool results.

The expected facts are deliberately narrow so task scoring can be audited:

- `buggy_cache.py`: avoid a mutable default and allocate per-instance state.
- `retry.py`: deduplicate a repeated event before applying its mutation.
- `state.py` and `events.py`: preserve monotonic event sequence and terminal
  state, rejecting a late active transition.
- `formatter.py`: emit the required response envelope and preserve the request
  identifier.
