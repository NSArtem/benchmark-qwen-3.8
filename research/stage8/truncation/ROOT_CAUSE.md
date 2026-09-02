# Stage 8 truncation continuation root cause

## Verdict

The HTTP 409 was a cache-identity failure in the old session reconstruction
path, not a valid semantic conflict in the conversation.  The old server
stored the assistant response as decoded text and rendered that text through
the Qwen chat template on the next request.  That transformation is not
token-lossless at some length boundaries.

There are two independently reproducible forms:

1. The production-model case is a terminal-whitespace/template-boundary
   mismatch.  The Qwen assistant template applies `|trim` to assistant
   content.  A length-truncated response can therefore end in a cached newline
   token while the reconstructed template starts the assistant-close boundary
   immediately.
2. A length boundary can also cut through a byte-level token sequence.  Decoding
   the partial sequence can insert U+FFFD; encoding that replacement character
   produces a different token ID.

Both make `exact_suffix()` fail in the pre-fix server, which the HTTP handler
translated to `409 session_state_error`.  Neither condition makes the supplied
conversation invalid.

## Raw evidence

The complete token dumps are in [repro.json](repro.json) and
[real-targeted.json](real-targeted.json).  The lightweight oracle uses the
local production tokenizer without loading weights.  The real probe loads the
g128 27B model once and performs serialized continuations.

| case | pre-fix result | generated stream | decoded/re-encoded | first raw mismatch | repaired prefix |
|---|---:|---:|---:|---:|---:|
| real g128, max 256 | projected 409; actual continuation 200 after fix | 256 IDs, final ID `198` (`\\n`) | identical | absolute `314`, generated index `255`: cached `198`, template `248046` (`<|im_end|>`) | exact |
| tokenizer partial emoji | 409 | `[9008, 239, 102, 373]` | `"👩�"` -> `[9008, 239, 102, 5513]` | generated index `3` | exact |
| tokenizer stress, max 31 | 409 | 31 IDs, final ID `198` (`\\n`) | identical | generated index `30`: cached `198`, template `248046` | exact |
| tokenizer stress, max 128 | 409 | 128 IDs, final ID `198` (`\\n`) | identical | generated index `127`: cached `198`, template `248046` | exact |
| explicit EOS | 200 | `[14311, 248046]` | `"complete"` -> `[14311]` | no prefix mismatch | exact |

The real targeted run tested budgets 31, 128, and 256 with a newline-heavy
code-generation prompt.  All three reached `finish_reason=length`; the old
logic projected one 409 at budget 256, while all three patched continuations
returned successfully with exact cache reuse.  The real 256-token control
probe in [real-repro.json](real-repro.json) also continued successfully, which
shows that truncation alone is not sufficient: the terminal token boundary
determines whether the old reconstruction is identity-preserving.

## Hypothesis checks

### MEASURED FACT — decode/re-encode non-identity exists

The partial-byte oracle has a first mismatch inside the generated stream.  The
server therefore cannot use response text as the authoritative cache identity.
The real production-model newline case has decode/re-encode identity, so this
is a secondary independent failure mode rather than the explanation for that
specific 409.

### MEASURED FACT — the real 409-class mismatch is a chat-template boundary

At real g128 index 314, the cache contains the generated final newline token
`198`; the raw next canonical stream contains `<|im_end|>` token `248046` at
that same position.  The assistant text ends in a newline, while the template
renders `content|trim` before emitting its close token.

### MEASURED FACT — cache lookahead/accounting is not the cause

The real control recorded prompt length 44, 256 generated IDs, and cache offset
300 (`44 + 256`).  The targeted 256 case likewise recorded the generated
boundary at index 314 and no offset error.  The mismatch is in the
reconstructed prompt, before suffix reuse, not in the number of tokens the
generator committed.

### MEASURED FACT — the client/session history is an exact extension

The real probe constructs the next request from the server's own returned
assistant text and appends one user message.  The old message-prefix check
passes.  The failure occurs after that check during token reconstruction.

### INFERENCE — the old 409 is an optimization miss misclassified as conflict

The transcript is valid and the server already has enough authoritative state
to continue.  Returning 409 merely because text-rendered tokens are not an
exact cache prefix is therefore a protocol-policy bug.

### HYPOTHESIS — other tokenizers/templates may expose additional variants

The current evidence covers terminal whitespace, partial byte sequences, and
EOS.  Other special-token or client serialization variants may exist.  The
patch treats exact stored IDs as authoritative and has a safe fresh-prefill
fallback when proof of reuse is unavailable, so those variants do not need to
be guessed at the cache boundary.

## Fix implied by the evidence

Each assistant generation now retains generated IDs, decoded text, finish
reason, and prompt-token boundary.  For an exact session extension, the next
canonical token stream replaces the rendered assistant-content span with the
stored generated IDs and retains one template boundary token.  This repairs
both the `|trim` and partial-byte cases while keeping the API response text
unchanged.

If exact reuse still cannot be proven, a syntactically valid transcript clears
the hybrid cache and fresh-prefills the full canonical transcript.  HTTP 409 is
reserved for a later genuine state error, not for this cache optimization miss.
