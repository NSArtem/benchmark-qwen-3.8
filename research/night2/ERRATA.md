# Erratum: prior persistent-cache warm conclusion

Date: 2026-08-19

The prior `research/FINAL_REPORT.md` and `research/STATUS.md` described the
8K persistent-cache run (`P3-WARM8K`) as `EXCELLENT`. That correctness claim is
superseded.

The preserved raw artifacts show that the turn-2 generated token sequence is
exactly the first 64 generated token IDs of turn 1, despite a different
follow-up query. Turn 1 has 128 tokens; turn 2 has 64. The compact-token
SHA256 for both the turn-2 sequence and turn 1's first 64 IDs is
`cb7ec89a619202c1a75d78f8b67140100a66b9ec3bfcd36af3ea3be66b0a32ce`, with an
exact 64/64 prefix overlap.

Corrected interpretation:

- the old 8K timing and memory observations are real measurements of that
  process and may be retained as performance observations;
- the old 8K correctness result is invalid and must not support an interactive
  recommendation;
- the old 4K run did not show the same exact replay, but it still requires the
  Night-2 exact-token and full-transcript oracles;
- g128 warm behavior was never established by the prior run.

Night 2 will preserve all historical raw files and determine whether the fault
is cache state, incremental turn serialization, the old harness, or more than
one layer. A grammatical replay is classified as `BROKEN`.
