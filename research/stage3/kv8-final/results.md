# Phase 6 g128 normal KV versus KV8 persistent 16K comparison

Decision: **SELECTABLE_LONG_CONTEXT_MODE**

Both modes used separate fresh child processes and the same 16K-class
prompt/follow-up shape under the validated exact-suffix protocol.

| mode | turns | cold TTFT (s) | warm TTFT median (s) | warm decode mean (tok/s) | peak (GiB) | active+cache peak (GiB) | correctness |
|---|---:|---:|---:|---:|---:|---:|---|
| normal | 5 | 224.005 | 1.169 | 10.513 | 15.236 | 11.418 | True |
| kv8 | 5 | 223.990 | 1.169 | 9.522 | 14.771 | 11.390 | True |

Peak saving (normal minus KV8): 0.465 GiB.
KV8 warm decode ratio: 0.9057; penalty: 9.4%.

| mode | turn | prompt | suffix | TTFT (s) | decode (tok/s) | peak (GiB) | classification | protocol |
|---|---:|---:|---:|---:|---:|---:|---|---|
| normal | 1 | 16727 | 16727 | 224.005 | 10.599 | 15.236 | COHERENT | True |
| normal | 2 | 16888 | 33 | 0.782 | 10.415 | 15.236 | COHERENT | True |
| normal | 3 | 16950 | 34 | 1.164 | 10.554 | 15.236 | COHERENT | True |
| normal | 4 | 17052 | 38 | 1.174 | 10.566 | 15.236 | COHERENT | True |
| normal | 5 | 17155 | 39 | 1.197 | 10.518 | 15.236 | COHERENT | True |
| kv8 | 1 | 16727 | 16727 | 223.990 | 9.555 | 14.771 | COHERENT | True |
| kv8 | 2 | 16888 | 33 | 0.735 | 9.493 | 14.771 | COHERENT | True |
| kv8 | 3 | 16941 | 34 | 1.164 | 9.558 | 14.771 | COHERENT | True |
| kv8 | 4 | 17043 | 38 | 1.179 | 9.544 | 14.771 | COHERENT | True |
| kv8 | 5 | 17144 | 38 | 1.174 | 9.495 | 14.771 | COHERENT | True |
