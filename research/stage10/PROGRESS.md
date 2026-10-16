- [x] A1 common helpers
- [x] A2 token sets + data_manifest.json
- [x] A3 layerwise + check A3-loop + check A3-sanitize
- [x] A4 BF16 reference (= E00)
- [x] A5 qformat + tests (pack 2/3/4, rotation)
- [x] A6 candidate loader + test
- [x] A7 evaluate
- [x] A8 ladder
- [x] A9 gen_check
- [x] A10 summary + make_table
- [x] E00 - [x] E01 - [x] E02 - [x] E03 - [x] E04 - [x] E05 - [x] E06 - [ ] E07
- [ ] E08 - [ ] E09 - [ ] E10 - [ ] E11 - [ ] E12 - [ ] E13 - [ ] E14 - [ ] E15

Recorded checks: A3-loop PASS (4094 positions; max absolute log-probability
difference 0.0; argmax agreement 1.0). A3-sanitize PASS (layers 0 and 3,
55/55 tensors byte-identical) after a user-authorized resume of the §7 STOP.

Unit tests: 28 pass (`uv run python -m unittest discover -s tests`).
Next: Stopped on user request 2026-10-16 14:41. E00-E06 done (summaries committed). E07 (had3-gptq) build was aborted after layer 48 of 64; partial output moved to `/Users/artem/models-stage10/E07.aborted-layer48-20261016/`. On resume: `cd research/stage10 && uv run python -m stage10.run_exp --exp E07` (full rebuild, about 3.1 h + 35 min measurements), then `benchmarks/results/stage10/finish_exp.sh E07 had3-gptq OK "E08 ..." "Ladder: ..."`, then E08-E15 in order. E14: every escha-mlx generation path calls mx.set_wired_limit (mlx-lm 0.31.3 stream_generate -> wired_limit; server; BatchGenerator) -> record SKIPPED_UNSAFE with the grep output. E13/E14 tokenizers match ours on eval sequence 0 (full eval allowed). Note: `benchmarks/results/stage10/finish_exp.sh` is git-ignored; it runs summary, make_table, PROGRESS check-off, commit, and push.
