# Stage 7 Status

Last updated: 2026-08-30 (Phase 3 top-8 gate stopped)

## Current phase

Phase 3 — incremental DWQ scaling is stopped at top-8. Top-4 is the best
internal validation checkpoint. Adding ranks 5–8 caused a clear train and
validation KL/NLL regression despite top-1 recovery, so top-16/top-32 are not
being executed under the phase gate.

## System memory baseline

The host-level pre-load snapshot before Stage 7 model work reported
19,327,352,832 bytes total, 21,014 free pages, 499,612 active pages, 496,024
inactive pages, 2,449 speculative pages, 105,708 wired-down pages, 14,244
pages occupied by the compressor, and system-wide free percentage 89%.
`vm.swapusage` reported 63.25 MiB used and 960.75 MiB free on a 1 GiB
encrypted swap device. The largest processes were ordinary Firefox tabs and
the UI; no MLX/Qwen workload was present. Disk space was 122 GiB available on
the data volume.

Safety: no `iogpu.wired_limit_mb` or equivalent setting was changed, and no
`mx.set_wired_limit()` call is permitted in the canonical runtime.

## Current default checkpoint

- Model: g128 affine 3-bit, group size 128
- Path: `/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX`
- Stage 6 MLX model payload: 10,930,715,648 bytes / 10.180022 GiB
- Stage 7 entry role: DEFAULT
- Reference environment: `envs/mlx-0.32.1`, mlx 0.32.1, pinned mlx-lm
  revision `254d153fdeb6f150edd4fc5a54f9828638481fa8`

## Stage 6 recovery

- Corrected heldout_v2 hash: `1749a8b47e99bb3e0e04347d75e24d7022ed3ab8d5ffa5bc56926b9cf02cbcbd`
- Stage 6 diverse-ish calibration hash used for the earlier 12-case ranking:
  `e856f39622c48f29f913d65d55f18d4ff9fb9bca16b0b23736361347472b9558`
- g128 corrected heldout_v2: 20/21 objective, 3/3 semi-objective,
  repetition flag 1
- g128 Quality Suite V2: 23/24 objective, 4/4 semi-objective
- g128 teacher-forced subset: NLL 1.369012, top-1 agreement 0.694611
- Stage 6 bounded DWQ proof: layer 60 `linear_attn.in_proj_a`, LR 1e-6,
  40 steps; local MSE 0.253556 -> 0.218637 and one-case KL
  0.091406 -> 0.090705; peak 10.681 GiB; no complete checkpoint
- MTP: 15 source `mtp.*` BF16 tensors, 849,398,784 bytes / 0.791 GiB;
  pinned text runtime strips them; no decode path validated
- Serialized agent smoke: 10 sequential 27B exchanges passed transport/cache
  checks; native structured tool-call generation was not claimed

## DWQ train/validation corpus hashes

- Train: `research/stage7/calibration/train.jsonl`, SHA-256
  `3fedf6d50c6f202f189f0d585fae418e3475eb4458e5f352af509e4f9a58db98`, 16
  cases, 8 domains, 2 cases/domain.
- Validation: `research/stage7/calibration/validation.jsonl`, SHA-256
  `1193e48e77bee8ada73d27ab5a025eb7100c8072c03051bec8fe0fb6feaa91dd`, 8
  cases, 8 domains, 1 case/domain.
- Final heldout_v2: SHA-256
  `1749a8b47e99bb3e0e04347d75e24d7022ed3ab8d5ffa5bc56926b9cf02cbcbd`.
- Quality Suite V2: SHA-256
  `f24e3983efbd898084852c3fde7b0f824604bc6a63349de5f32efeaa36ca3de0`.

The Stage 7 protocol is materially broader than Stage 5's six-domain
calibration: it adds explicit patch-reasoning and tool/agent transcripts and
uses disjoint content for validation.

## DWQ refined checkpoint status

The resumable one-module patch
`research/stage7/artifacts/dwq/reproduce-layer60.safetensors` is a research
checkpoint for `language_model.model.layers.60.linear_attn.in_proj_a`; it is
not yet a complete model directory or production candidate. The canonical
g128 model is untouched. On the Stage 7 train split, local MSE was
0.265708 -> 0.231171 (12.998% reduction), end-to-end KL was
0.156752 -> 0.155615, and top-1 was 0.917526 -> 0.918814. On validation, KL
was 0.142997 -> 0.142217 while top-1 was 0.918317 -> 0.915842. This is a
positive reproduction direction with a small validation top-1 tradeoff, not a
promotion decision.

Top-4 cumulative checkpoint:
`research/stage7/artifacts/dwq/top4.safetensors`, four modules, 31.6 KiB on
disk. Relative to the unpatched g128 baseline, train KL is 0.156752 ->
0.150978 and validation KL is 0.142997 -> 0.136866. Train top-1 is
0.917526 -> 0.916237 and validation top-1 is 0.918317 -> 0.913366; this
small top-1 regression is retained as a cautionary metric. All eight
validation-prompt sanity generations completed without exceptions; structured,
retrieval, reasoning, instruction, and prose outputs remained coherent and
matching in semantic content.

Top-8 cumulative checkpoint:
`research/stage7/artifacts/dwq/top8.safetensors`, eight modules. It is retained
as a rejected interference control. Relative to top-4, train KL regressed
0.150978 -> 0.156082 and validation KL regressed 0.136866 -> 0.142290;
validation teacher-argmax NLL regressed 0.275873 -> 0.278951. Validation top-1
recovered to 0.918317, which is not sufficient to override the KL/NLL
regression. No top-16 or top-32 run was launched.

Best internal DWQ candidate remains top-4; final heldout evaluation is still
pending and no production promotion has been made.

## Teacher/student baseline on frozen Stage 7 calibration

Reference teacher: g64 logits captured in separate train and validation
processes. The g128 baseline metrics process peaked at 10.922 GiB MLX memory.

| split | target tokens | KL | teacher-argmax NLL | actual-token NLL | top-1 agreement | logit MSE |
|---|---:|---:|---:|---:|---:|---:|
| train | 776 | 0.156752 | 0.285187 | 7.843152 | 0.917526 | 0.764543 |
| validation | 404 | 0.142997 | 0.277587 | 7.081791 | 0.918317 | 0.757290 |

Raw teacher manifests and baseline metrics are under
`research/stage7/artifacts/teacher/` and `research/stage7/artifacts/baseline/`.

## MTP architecture status

Initial Stage 6 evidence says the tensors form one `mtp.layers.0` attention/MLP
block plus `mtp.fc` and three normalization groups, but this is not yet a
Stage 7 architecture derivation. Exact graph, proposal width, and hybrid
attention/recurrent rollback semantics remain pending Phase 7.

## MTP weight precision/status

No MTP artifact. BF16 source payload is 849,398,784 bytes; 4-bit and 3-bit
figures in Stage 6 were arithmetic estimates only.

## Current runtime Pareto

| mode | context | decode | warm TTFT | peak | role |
|---|---:|---:|---:|---:|---|
| g128 normal KV | 8K | 11.137 tok/s | ~1.1 s warm reference | 13.983 GiB | baseline |
| g128 normal KV | 12K | 10.803 tok/s | ~1.1 s warm reference | 14.609 GiB | practical |
| g128 normal KV | 16K | 10.4–10.6 tok/s | ~1.2 s warm reference | 15.236 GiB | practical upper daily |
| g128 normal KV | 20K | 10.367 tok/s | 0.72 s follow-up in one run | 15.859 GiB | clean-host upper envelope |

## Agent benchmark status

Only the Stage 6 serialized transport smoke is recovered. The Stage 7
deterministic coding-agent fixture/benchmark is pending.

## Rejected hypotheses

- Generic static mixed-bit search is saturated and out of scope.
- Q1 did not beat g128; Q2/Q3 were correctly skipped by the Stage 6 gate.
- Whole-model teacher/student DWQ residency is unsafe on this 18 GB host.
- MTP cannot be implemented from tensor names alone.

## Next action

Do not scale beyond top-8. Finish the rejected top-8 sanity artifact, then run
the fixed small diverse-vs-coding-heavy-vs-reasoning-heavy calibration ablation
on the top-4-sized subset without using final heldout cases.
