# MLX 0.32.1 prefill-OOM diagnostic environment

This experiment uses only `envs/mlx-0.32.1` through:

```sh
uv run --project envs/mlx-0.32.1 --no-sync ...
```

Verified before model tests:

- Python 3.14.7
- MLX 0.32.1
- mlx-lm metadata 0.31.3
- mlx-lm Git URL `https://github.com/ml-explore/mlx-lm.git`
- mlx-lm Git commit `254d153fdeb6f150edd4fc5a54f9828638481fa8`
- device: Apple M3 Pro
- `memory_size`: 19,327,352,832 bytes = 18.00 GiB
- `max_recommended_working_set_size`: 12,884,918,272 bytes = 12.00 GiB
- `max_buffer_length`: 9,663,676,416 bytes = 9.00 GiB

Available MLX memory API verified:

`get_active_memory`, `get_cache_memory`, `get_peak_memory`,
`reset_peak_memory`, `set_cache_limit`, `set_memory_limit`,
`set_wired_limit`, `clear_cache`, `device_info`, and `synchronize`.

The pinned `mlx_lm.generate.wired_limit()` helper sums model array bytes,
sets the wired limit to
`device_info()["max_recommended_working_set_size"]` while generation runs,
then synchronizes the generation stream and restores the previous value.

The model is loaded from the existing local snapshot. No model files are
stored in this repository, and no dependency or lockfile is modified.
