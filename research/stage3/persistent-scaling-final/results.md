# Phase 3 exact-suffix persistent scaling

Overall outcome: **PASS**

| context | turns | warm TTFT (s) | decode (tok/s) | peak (GiB) | correctness |
|---|---:|---:|---:|---:|---|
| 8k | 5 | 0.735–1.098 | 11.104 | 13.983 | PASS |

Fresh oracle at turn 2 (pre-patch artifact, invalid transcript): coherent but
serialized 16,536 tokens due duplicated initial user content; do not use for
performance or correctness evidence.

Fresh oracle at turn 2 (corrected): semantic pass, 8,352 tokens, TTFT
116.796 s, decode 11.036 tok/s, exact output token equality (29/29).
| 12k | 5 | 0.736–1.121 | 10.803 | 14.609 | PASS |
| 16k | 5 | 0.807–1.223 | 10.409 | 15.236 | PASS |

Fresh oracle at turn 5: True (TTFT 267.58757433400024, decode 10.231362019964815).
