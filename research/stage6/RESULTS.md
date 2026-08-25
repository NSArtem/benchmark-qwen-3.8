# Stage 6 Results

## Phase 0 — clean baseline and artifact recovery

### Measured facts

- Repository starts on `main` at `1439ec7`; the only pre-existing worktree change is the untracked user file `archive_3.8_stage5.zip`.
- `memory_pressure` reports 90% free memory, zero compressor pages, and zero swap I/O. `vm_stat` reports 221,405 free pages and 108,954 wired pages.
- A read-only escalated inspection confirmed `vm.swapusage` at zero on the clean baseline and found no stale MLX/Qwen/Python/uv workload; no Stage 6 heavy workload had started at that point.
- Disk inspection reports ~124 GiB available on the data volume.
- Q1, Q2, and Q3 are present under `/Users/artem/models/stage5-frontier/`, not `/private/tmp`, and each has 20/20 safetensors shards, `config.json`, `model.safetensors.index.json`, tokenizer files, and `chat_template.jinja`.
- Candidate payload sizes match the Stage 5 frozen manifest: Q1 11,088,002,048 B, Q2 11,245,288,448 B, Q3 11,402,574,848 B.

### Interpretation

The memory snapshot is consistent with a clean post-reboot/recovery state, but
the exact swap-used number and process absence must be confirmed before a heavy
load. Candidate preservation is complete without copying or rebuilding weights.

### Next gate

Finish the read-only baseline inspection, then repair and freeze heldout_v2
before loading Q1.

## Phase 1 — heldout_v2 correction

### Measured facts

- Raw inspection identified nine symmetric budget defects: `h5-coding-01` through `h5-coding-06` had 384-token caps, and `h5-reasoning-01`, `h5-reasoning-02`, and `h5-reasoning-04` had 256-token caps. In each case the model could reach the required content but was vulnerable to stopping before the required final marker.
- The frozen v2 suite changes only those generation budgets, to 1024 tokens. Prompts, semantic answer keys, answer regexes, and scorer logic are unchanged.
- Final `heldout_v2` SHA-256 is `1749a8b47e99bb3e0e04347d75e24d7022ed3ab8d5ffa5bc56926b9cf02cbcbd`; the machine-readable correction record is `research/stage6/heldout/heldout_v2_manifest.json`.
- Serialized final-hash reruns completed without exceptions: Q1 6/9 objective, g128 8/9, g64 7/9, and mixed34 6/9 on the nine affected cases. Repetition flags are retained as measured harness signals.
- Final-hash full merges are: Q1 17/21 objective + 2/3 semi-objective; g64 19/21 + 2/3; g128 20/21 + 3/3; mixed34 18/21 + 2/3. No exceptions or early-EOS events occurred. Q1 coding-04 and reasoning-02 are not clean semantic failures because required content / the final answer is present; coding-06, instruction-02, and prose-01 remain substantive misses.

### Interpretation

The nine known max-token/sentinel artifacts are corrected symmetrically without
using a candidate output to rewrite the answer key. The corrected suite is now
the only heldout suite used for Stage 6 decisions. Repetition flags remain
separate measured quality/harness signals and are not silently converted to
passes.

### Safety note

The serialized regression and Q1 quality runs temporarily raised swap, but the
post-regression and post-teacher-forced checks remained usable. The latest
post-teacher check reported 88% free memory and 239.69 MiB used / 784.31 MiB
free on the 1 GiB encrypted swap device.

## Phase 2 — Q1 validation so far

### Measured facts

- Q1 short smoke was coherent: 11.5796 tok/s decode, 1.8576 s TTFT, 19.9189 prompt tok/s, 10.642 GiB MLX peak, and 10.541 GiB active+cache peak.
- Q1 Quality Suite V2 completed with no exceptions: 22/24 objective and 4/4 semi-objective. Corrected g128 is 23/24 and 4/4. Q1 therefore has no quality-suite win.
- Q1 teacher-forced evaluation completed with no exceptions over 167 target tokens: weighted NLL 1.415419 nats and weighted top-1 agreement 0.682635. Existing g128 evidence is 1.369012 nats and 0.694611. Q1 is slightly worse on both measures.
- Q1 artifact paths are `research/stage6/artifacts/runtime/Q1/short/result.json`, `research/stage6/artifacts/quality-v2/Q1/model.json`, and `research/stage6/artifacts/teacher-forced/Q1/evaluation.json`.

### Interpretation

Before runtime comparison, Q1 is already behind g128 on the corrected heldout
formal score, Quality Suite V2, and teacher-forced loss. Its only remaining
credible route to the default is a meaningful speed, memory, or context win;
the larger payload makes that unlikely. Fresh 8K/12K/16K measurements are the
gate before deciding whether Q2/Q3 merit any load.

### Runtime and gate decision

- Q1 8K cold: 11.0128 tok/s decode, 106.478 s TTFT, 76.928 prompt tok/s, 15.171 GB MLX peak (14.129 GiB), coherent.
- Q1 12K cold: 10.4426 tok/s decode, 164.261 s TTFT, 74.808 prompt tok/s, 15.844 GB MLX peak (14.756 GiB), coherent.
- Q1 16K cold: 10.2601 tok/s decode, 232.916 s TTFT, 71.816 prompt tok/s, 16.517 GB MLX peak (15.382 GiB), coherent.
- The g128 fresh/Stage 5 references are 11.1369 tok/s and 13.983 GiB peak at 8K, and 10.5827 tok/s and 15.236 GiB peak at 16K. Q1's slightly higher fresh prompt throughput does not offset its slower decode and larger peak.
- Q1 five-turn exact-suffix chat passed at both envelopes. At 8K it reached logical context 7,825 with warm TTFT 0.633–0.641 s and decode 11.239–11.275 tok/s; at 16K it reached 15,731 with warm TTFT 0.695–0.701 s and decode 10.511–10.580 tok/s. Both reset cleanly to zero and had no replay flags.

Q1 is rejected as the default and as an optional quality mode. It is larger,
slightly worse on teacher-forced loss/top-1 and corrected formal quality, and
does not improve 16K decode or peak memory. Q2 and Q3 are explicitly skipped:
the Stage 6 gate does not justify loading larger candidates when Q1 supplies no
credible Pareto signal or failure mode that more promotions would address. The
fixed-weight winner entering reproducibility and refinement work is g128.

## Phase 6 — bounded DWQ on g128

### Measured facts

- The Stage 5 synthetic dry run was re-used as the safety gate: 40 direct scale/bias steps reduced local MSE 0.881265 -> 0.718006 (18.53%) with no wired-limit calls. The pinned whole-model DWQ path remains unsafe because the minimum source-plus-g128 payload is 61.927 GiB before activations and optimizer state.
- A Stage 6 one-module streamed pass selected the Stage 5 top-ranked `language_model.model.layers.60.linear_attn.in_proj_a` module in g128. It loaded one 983,040-byte BF16 source tensor, captured 148 activation rows from three calibration cases, and optimized only the module's affine 3-bit scales/biases.
- With a conservative learning rate of `1e-6`, 40 steps reduced local source-output MSE from 0.253556 to 0.218637 (13.77%). One in-memory end-to-end calibration probe reduced g64-teacher KL from 0.091406 to 0.090705 (delta 0.000701).
- Peak MLX memory was 10.681 GiB; the full teacher and full source were never resident, and wired-limit calls were zero. A first `1e-3` attempt diverged to infinity and is retained as a numerical-stability warning.
- Artifacts: `research/stage6/artifacts/dwq/g128-layer60-in_proj_a-stable.json`, `research/stage6/artifacts/dwq/g128-layer60-in_proj_a-stable-partial.safetensors`, and the unstable first attempt at `research/stage6/artifacts/dwq/g128-layer60-in_proj_a.json`.

### Interpretation

Bounded layer-local DWQ is technically feasible and can improve a local
teacher agreement objective under safe residency, but the result is not yet a
production refinement: only one module and three calibration cases were used,
the step size is sensitive, and no complete refined checkpoint was created.
Therefore no heldout or final-quality win is claimed, and g128 remains the
fixed-weight winner.

## Phase 4 — KL-ranking reproducibility

### Measured facts

- A controlled repeat measured 45 modules selected from Stage 5's top-25,
  middle-10, and low-10 rank bands against the identical 12-case calibration
  set and calibration hash. It completed in 465.289 s with 14.480 GiB peak
  and 12.302 GiB active+cache.
- Same-data comparison with the Stage 5 ranking was numerically exact:
  Spearman 1.000, Pearson 1.000, top-10 overlap 10/10, top-25 overlap 25/25,
  sign agreement 100%, and zero mean/max absolute score delta.
- A separately selected two-case subset (`cal-instruction-01` and
  `cal-reasoning-01`) completed in 101.988 s with 13.932 GiB peak. Its
  comparison with the full ranking was Spearman 0.410, Pearson 0.317,
  top-10 overlap 5/10, top-25 overlap 20/25, and sign agreement 73.3%.

### Interpretation

The end-to-end KL ranking is reproducible when the calibration data and
algorithm are held constant. Its movement across calibration subsets is real
calibration sensitivity, not evidence of numerical nondeterminism. Future
fixed-budget allocations should use a more diverse calibration set or report
subset robustness rather than treating one small subset as universal.

Artifacts: `research/stage6/artifacts/kl-repro/same-data-repeat.json`,
`alternate-2case.json`, and `analysis.json`.

## Phase 8 — final context envelope

The final fixed-weight winner, g128, completed one normal-KV cold probe at
exactly 20,480 prompt tokens. It was coherent, with 72.688 prompt tok/s,
281.756 s TTFT, 10.367 decode tok/s, 11.622 GiB active+cache after
completion, and 15.859 GiB MLX peak. The completion snapshot reported 23%
free host memory and 488.62 MiB encrypted swap used; after process exit the
host returned to 90% free memory. A follow-up run then passed three exact-suffix
turns, reached logical context 20,686, and reset to zero; warm TTFT was
0.724–0.726 s and warm decode was 10.419–10.433 tok/s with no replay flags. No
24K probe was launched: extrapolated headroom from this result did not satisfy
the strict gate. 32K was not tested.

### Interpretation

20K works as a clean, one-shot upper-envelope mode but is not the daily
recommendation on an 18 GB machine. Use 16K normal KV for the practical
long-context mode, and 12K when sustained system headroom matters more than
context. KV8 was not needed for the decision and was not launched.

Artifacts: `research/stage6/artifacts/context/g128-20k-normal/result.json` and
`research/stage6/artifacts/context/g128-20k-persistent/result.json`.

## Phase 9 — Qwen3.8 MTP investigation

Header-only inspection of the original source shard confirms 15 top-level
`mtp.*` BF16 tensors (849,398,784 bytes, 0.791 GiB). The set is one
`mtp.layers.0` attention/MLP block with `mtp.fc`, three MLP projections,
Q/K/V/O projections, norms, and pre-FC normalization weights; the exact names,
shapes, and byte counts are in `research/stage6/mtp_static_inspection.json`.

The pinned MLX-LM revision's `qwen3_5.py` intentionally strips `mtp.*` before
loading the validated text model and contains no native Qwen3.8 MTP head or
reference decode integration. Its generic speculative-draft interface is not
an MTP implementation. Ideal raw payload arithmetic suggests approximately
0.396 GiB at 8-bit, 0.198 GiB at 4-bit, and 0.148 GiB at 3-bit before scale and
metadata overhead, but no quantized MTP artifact was produced.

### Interpretation

MTP is **NOT READY / UNSUPPORTED** for today's affine g128 runtime. Acceptance
rate, token identity, decode speed, and MTP residency are therefore N/A. No
head was fabricated and no new MTP training was attempted. The Stage 5
target-only `mlx-vlm` feasibility failure remains supporting evidence that an
MTP drafter cannot be justified without a compatible target path.

## Phase 10 — serialized agent backend

The Stage 6 client completed ten sequential real 27B exchanges against g128.
The transcript included system, user, assistant, and tool roles, with two
tool-call/result injections. `/health`, `/v1/models`, and `/metrics` worked;
all ten completion requests returned successfully; cache offsets matched
logical context; replay detection remained false; and the session reset
returned zero messages and zero logical context. Ordinary warm TTFT was about
0.583–0.593 s, with longer suffix-prefill turns around 1.38 s, and decode was
about 11.9–12.0 tok/s on the short responses.

### Interpretation

The server is ready for single-user serialized OpenAI-compatible text/tool
role transport. It is not a concurrent or batching server, and the smoke does
not claim that a model will autonomously emit provider-native structured tool
calls; the server normalizes supplied tool-call payloads to text. Artifact:
`research/stage6/artifacts/agent/g128-serialized-smoke.json`.

## Final fixed-weight decision

g128 remains the only practical default. Q1 is larger and behind g128 on the
corrected heldout suite, Quality Suite V2, teacher-forced NLL/top-1, 16K
decode, and 16K peak memory. Q2 and Q3 are intact but were skipped under the
explicit Q1 gate; no evidence justified paying their larger payload. Bounded
DWQ remains a local feasibility result without a complete checkpoint. The
recommended context/runtime configuration is g128, normal KV, 12K–16K routine
use, with 20K reserved for clean-host one-shot work.
