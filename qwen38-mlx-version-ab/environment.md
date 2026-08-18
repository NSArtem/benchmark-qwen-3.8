# Environment

This experiment uses the existing locked projects only:

- `envs/mlx-0.32.0`: Python 3.14.7, MLX 0.32.0.
- `envs/mlx-0.32.1`: Python 3.14.7, MLX 0.32.1.
- Both use mlx-lm Git source commit `254d153fdeb6f150edd4fc5a54f9828638481fa8`.

Verified with uv 0.12.5:

| Environment | Python | Executable | MLX | mlx-lm metadata | Git evidence |
| --- | --- | --- | --- | --- | --- |
| `mlx-0.32.0` | 3.14.7 | `envs/mlx-0.32.0/.venv/bin/python3` | 0.32.0 | 0.31.3 | PEP 610 `direct_url.json`: commit `254d153fdeb6f150edd4fc5a54f9828638481fa8` |
| `mlx-0.32.1` | 3.14.7 | `envs/mlx-0.32.1/.venv/bin/python3` | 0.32.1 | 0.31.3 | PEP 610 `direct_url.json`: commit `254d153fdeb6f150edd4fc5a54f9828638481fa8` |

The checkpoint is the existing local snapshot:

`/Users/artem/.cache/huggingface/hub/models--lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly/snapshots/c98bba5926f51fec1c8d8737e577221673f524d7`

No model weights are stored in this repository. No dependency versions,
lockfiles, or environments are changed by this experiment.
