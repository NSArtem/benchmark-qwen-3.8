# Environment

This experiment used only `envs/mlx-0.32.1` and the local cached checkpoint.

Initial repository status before this experiment: clean working tree on `main`.
The existing root project and the other investigation directories were not
modified.

## Runtime

| Field | Value |
|---|---|
| Python | 3.14.7 (`[Clang 17.0.0 (clang-1700.6.4.2)]`) |
| Python executable | `/Users/artem/projects/benchmark-qwen-3.8/envs/mlx-0.32.1/.venv/bin/python3` |
| MLX | 0.32.1 |
| mlx-lm metadata | 0.31.3 |
| mlx-lm source | `https://github.com/ml-explore/mlx-lm.git` |
| mlx-lm commit | `254d153fdeb6f150edd4fc5a54f9828638481fa8` |
| model | `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly` |
| model path | `/Users/artem/.cache/huggingface/hub/models--lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly/snapshots/c98bba5926f51fec1c8d8737e577221673f524d7` |

The mlx-lm Git revision was verified from installed `direct_url.json` and was
the exact requested commit. The local snapshot was used; no model download was
performed during the inference matrix.

## MLX device and memory

`mx.device_info()` reported:

```json
{
  "architecture": "applegpu_g15s",
  "device_name": "Apple M3 Pro",
  "memory_size": 19327352832,
  "max_recommended_working_set_size": 12884918272,
  "max_buffer_length": 9663676416,
  "resource_limit": 499000
}
```

That is 18.0 GiB physical unified memory, a 12.000015 GiB recommended working
set, and a 9.0 GiB maximum buffer length. The harness never called
`mx.set_wired_limit()`; this MLX build does not expose a wired-limit getter, so
the result records the setter as untouched rather than fabricating a getter
value. It also did not call `set_cache_limit()` or change any system setting.

## Fixed generation settings

- direct `mlx_lm.generate.generate_step()` only;
- `max_tokens=128`;
- default sampler, which is the installed mlx-lm argmax sampler;
- `prefill_step_size=2048`;
- no prompt cache reuse, KV quantization, activation quantization, MTP,
  speculative decoding, APC, or `max_kv_size`;
- one model load and one prompt per fresh Python process.

The prompt runner materialized the model once after load for the required
memory checkpoint, then used the exact saved chat-formatted token IDs.
