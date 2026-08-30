# MTP speculative decoding for Qwen3.8-Flash-Next (experimental fork branch)

This branch implements MTP speculative decoding for
`RadixArk/Qwen3.8-Flash-Next-NVFP4` on top of the merged qwen4_exp support
(#257). It was written primarily by an AI assistant (Claude); the measurements
and hardware testing below were run by the branch owner. It is shared as a
reference implementation rather than a PR - cherry-picking, adopting, or
requesting changes are all welcome.

## What it does

The checkpoint ships 31 `mtp.*` tensors that main currently drops. This branch
loads them and uses the one-layer MTP predictor for speculative decoding:

1. The predictor proposes one to three draft tokens recursively from the
   target's previous pre-mixer hidden state. `FREETOKEN_QWEN4_MTP_DRAFT_P_MIN`
   gates low-confidence drafts (a failed gate shortens the chain).
2. The target verifies `[current, draft...]` in one fixed-row forward. GDN,
   the PLE conv history and the shared ngram context window save an
   accepted-prefix checkpoint after each non-final row.
3. The longest accepted draft prefix plus the target's replacement or bonus
   token is emitted. On a partial rejection the checkpoint commits the
   accepted state directly - no target replay.
4. The predictor re-runs over the emitted rows to propose the next draft.

Verify (2-4 rows) and predictor (1-4 rows) forwards replay captured CUDA
graphs with restage-per-replay buffers; eager fallbacks remain for chunked
prefill and graph-less setups. The scheduler reserves the speculative KV pages
up front and drains multi-token results synchronously.

## Enabling it

Off by default. Everything is opt-in via environment variables:

| variable | default | meaning |
|---|---|---|
| `FREETOKEN_QWEN4_MTP` | `0` | enable MTP |
| `FREETOKEN_QWEN4_MTP_MAX_DRAFTS` | `1` | draft chain depth, 1-3 |
| `FREETOKEN_QWEN4_MTP_DRAFT_P_MIN` | `0` | confidence gate (0.6 measured best) |
| `FREETOKEN_QWEN4_MTP_EXPERT_QUANT` | `bf16` | predictor expert bank: `bf16` or `fp8_block` |
| `FREETOKEN_QWEN4_MTP_MOE_CACHE_SIZE` | `64` | predictor GPU expert-cache slots |

Current restrictions: greedy sampling, one running request, TP=1, offload
family MoE backends, naive cache (forced automatically). The API only takes
the speculative path when the request resolves as greedy - with the model's
default sampling (`top_p=0.95`) that means passing `temperature=0` AND
`top_p=1`.

## Memory footprint

MTP itself adds only the predictor expert bank to the pinned load: ~4.69 GiB
in bf16, ~2.34 GiB with `FREETOKEN_QWEN4_MTP_EXPERT_QUANT=fp8_block`, plus a
~0.6 GiB GPU cache (64 slots, budgeted before the main MoE/KV pools).

On stock main the full model still needs the ~111 GiB pinned load (47.7 GiB
PLE table + 63.5 GiB expert banks), which does not fit a 128 GB machine. The
combination that does fit - and the one every number below was measured on -
is this branch stacked on PR #279 (`--ple-backend mmap`): pinned memory drops
to the expert banks plus the predictor (~66 GiB), and MTP works unchanged on
top, including inside the mmap PLE's CUDA-graph staging. So a 128 GB RAM
machine runs Qwen3.8 with MTP today via #279 + this branch; neither branch
depends on the other.

## Measurements

RTX 5090 (32 GB) + 125 GiB RAM, WSL2, CUDA 13,
`RadixArk/Qwen3.8-Flash-Next-NVFP4`, three drafts, `p_min=0.6`, FP8 predictor
bank. Because main's pinned PLE table does not fit this machine's RAM next to
the expert banks, the local runs stacked PR #279 (mmap PLE) underneath; this
branch itself does not depend on #279.

Decode-only speedup versus ordinary decode in the same load (256-512 output
tokens per arm, per-prompt warmup):

| prompt | speedup | acceptance |
|---|---:|---:|
| repetitive synthetic | ~3.3x | 100% |
| code | ~1.66x | 97-98% |
| long context (2k prompt) | ~1.61x | 100% |
| tool use | ~1.30x | 84-90% |
| prose | ~1.10x | 76-77% |

Greedy output was byte-identical to ordinary decode for every comparable
token on the measured prompts (up to 511 tokens compared). Multi-row
verification is not guaranteed bit-exact in general - near-tie top-1 flips
from batch-shape BF16 reduction order remain possible.

Known limitation: predictor prefill costs roughly 0.7-0.9 s per request on
this machine (more when cold), so short outputs are end-to-end neutral to
negative; long generations converge to the decode ratios above. Lazy or
cheaper predictor prefill is the main follow-up.

## The detokenizer fix (first commit, independent of MTP)

`fix(tokenizer): keep streamed deltas incremental when one uid repeats in a
batch` fixes a latent bug on main: `detokenize()` freezes every message's
read/surr window before decoding, so when the worker's queue drain batches
several tokens of one request together, later deltas re-send the earlier
tokens ("How", "How can", "How can I"). Reachable on main whenever the
detokenizer worker falls one step behind; speculative decoding hits it every
cycle. The added test fails on main and passes with the fix. This commit can
be cherry-picked on its own.

## Testing status

- CPU: the affected suites pass (1,092 tests; the only failures are the three
  pre-existing FlashInfer-environment failures that also fail on main).
  New tests: worker contract (12), MTP config parsing (3), detokenizer (2).
- GPU (dummy weights): zero-draft speculative output is byte-identical to
  ordinary decode; all five fixed-row graphs replay bit-identically to eager
  execution of the same staged computation.
- GPU (real weights, via #279 locally): the measurements above, plus
  server-level checks - streamed output equals non-streamed output,
  greedy determinism, stop-string trimming (including streamed and Japanese),
  EOS inside a multi-token drain, concurrent-request serialization, and
  1k-token generation stability.
