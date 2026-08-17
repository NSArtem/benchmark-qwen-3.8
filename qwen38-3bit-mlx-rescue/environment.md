# Qwen3.8 27B 3-bit MLX rescue environment

Captured 2026-08-17 on the shared Apple M3 Pro Mac, Asia/Yerevan.

## Safety and machine checks

```text
iogpu.wired_limit_mb: 0
```

The wired limit was not changed. No `sudo` was used, swap settings were not
modified, and no unrelated processes were stopped.

MLX device information from the installed public API:

```text
device_name: Apple M3 Pro
memory_size: 19327352832 bytes
max_recommended_working_set_size: 12884918272 bytes
max_buffer_length: 9663676416 bytes
architecture: applegpu_g15s
resource_limit: 499000
```

Initial public MLX counters were all zero in the fresh inspection process:

```text
mx.get_active_memory(): 0
mx.get_peak_memory(): 0
mx.get_cache_memory(): 0
```

## Installed stack

```text
uv run --no-sync python --version: Python 3.14.7
mlx: 0.32.0
mlx-lm: 0.31.3
```

`mlx.__version__` is not exposed by this installed MLX build, and importing
`mlx_lm.__version__` directly is not reliable in a headless process. Versions
above were recorded with `importlib.metadata`, using the same installed
packages that the benchmark runner imports.

## Verified public interfaces

```text
mlx_lm.load(path_or_hf_repo, tokenizer_config=None, model_config=None,
            adapter_path=None, lazy=False, return_config=False, revision=None)
mlx_lm.stream_generate(model, tokenizer, prompt, max_tokens=256,
                        draft_model=None, **kwargs)
mlx_lm.generate(model, tokenizer, prompt, verbose=False, **kwargs)
```

The installed lower-level generation step is:

```text
generate_step(prompt, model, *, max_tokens=256, sampler=None,
              logits_processors=None, max_kv_size=None, prompt_cache=None,
              prefill_step_size=2048, kv_bits=None, kv_group_size=64,
              quantized_kv_start=0, prompt_progress_callback=None,
              input_embeddings=None)
```

Therefore the public `stream_generate(..., prefill_step_size=...)` path is
available. KV8, KV group size, and quantized-KV start are also available.
The CLI exposes `--quantize-activations` / `-qa`; the CLI loads with
`model_config={"quantize_activations": args.quantize_activations}`.

The installed `mlx_lm.generate --help` also confirms:

```text
--prefill-step-size       not a generate CLI option; use public stream_generate kwargs
--quantize-activations, -qa
--kv-bits KV_BITS
--kv-group-size KV_GROUP_SIZE
--quantized-kv-start QUANTIZED_KV_START
```

The CLI parser default for `quantized_kv_start` is `5000`; the public
`generate_step` default is `0`. The rescue runner records this distinction and
uses an explicit value only for the controlled KV8 test.

The runner uses only `mlx_lm.load`, the model tokenizer's normal
`apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`,
and `mlx_lm.stream_generate`. Greedy decoding is the normal default sampler.

