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
- [x] E00 - [x] E01 - [x] E02 - [x] E03 - [x] E04 - [x] E05 - [x] E06 - [x] E07
- [ ] E08 - [ ] E09 - [ ] E10 - [ ] E11 - [ ] E12 - [ ] E13 - [ ] E14 - [ ] E15

Recorded checks: A3-loop PASS (4094 positions; max absolute log-probability
difference 0.0; argmax agreement 1.0). A3-sanitize PASS (layers 0 and 3,
55/55 tensors byte-identical) after a user-authorized resume of the §7 STOP.

Unit tests: 31 pass (`uv run python -m unittest discover -s tests`).
Next: E08 build RUNNING (layers 0–15 complete at 2026-10-16 21:39); rotation + GPTQ, mixA layout, runtime rotate_dtype=bfloat16; then E09–E15 in order. E14 must be SKIPPED_UNSAFE with existing wired-limit grep evidence.

- [x] AM1 tools
- [x] AM1 gen-check rerun
- [x] AM1 ladder 24K/32K
