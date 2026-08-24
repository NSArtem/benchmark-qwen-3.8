# Phase 5 g64 vs g128 diverse quality

g128 verdict: **PERFORMANCE/LONG-CONTEXT MODE**

Objective pass rate: g64 25/38 (65.8%); g128 22/38 (57.9%).

| category | cases | g64 pass | g128 pass | delta | mean token LCP |
|---|---:|---:|---:|---:|---:|
| coding | 10 | 2 | 0 | -20.0% | 4.6 |
| instruction | 6 | 6 | 6 | +0.0% | 8.0 |
| prose | 5 | 4 | 4 | +0.0% | 6.4 |
| reasoning | 6 | 4 | 3 | -16.7% | 5.0 |
| retrieval | 5 | 3 | 3 | +0.0% | 11.6 |
| structured | 6 | 6 | 6 | +0.0% | 24.0 |

Prior teacher-forced 4K reference: g64 mean NLL 4.1387 nats; g128 5.8335 nats; delta +1.6947 nats. This is supplementary, not the sole quality verdict.
