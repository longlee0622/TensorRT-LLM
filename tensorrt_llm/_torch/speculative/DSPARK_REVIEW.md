# DSpark review backlog

This file contains only findings that remain actionable on the current development
branch. Completed fixes, commit-by-commit history, benchmark logs, and machine-specific
instructions are intentionally omitted. `DSPARK_DEV.md` summarizes the current design
and validation state.

Severity: **H** blocks merging or can corrupt behavior, **M** is a material correctness
or performance issue, and **L** is cleanup or a guarded edge case.

## Status

PR #15808 is merged to main. The initial functional port, the rebase onto the
unified one-model speculative-decoding base (`SpecWorkerBase`, PR #15775), removal
of the experimental real-fp8-MLA path, and the confidence-head scaffolding cleanup
all landed there. These notes track the follow-up work that is still open on the
development branch (which also carries a small decode-perf patch and the two
rolling-window slot fixes described below on top of main).

Already resolved and therefore removed from this backlog: the target-layer-id
fail-fast validation, gating the corrected block logits behind `return_logits`,
the `with_markov` / `markov_rank` head-consistency fix, removal of the
`TLLM_DSPARK_REAL_MLA` env path, and removal of the unwired
`enable_confidence_head` / `confidence_threshold` public config fields. The mHC
fused-HC scratch is now a per-call allocation (`_alloc_fused_hc_scratch`), so the
former CUDA-stream-keyed scratch cache and its retention concern no longer apply.
Also resolved (rolling-window slots): unknown / padded request IDs now map to a
dedicated scratch row instead of aliasing slot 0; and disaggregated
generation-server requests, which never run `_seed_context_windows` (the prompt is
seeded on the context server), now get a persistent per-request slot assigned in
`DSparkSpecMetadata.prepare()` instead of all sharing one row — the batch-size >1
accept-length collapse in GitHub #16767. Both are covered by hardware-agnostic unit
tests (`test_dspark_worker.py`), and an end-to-end disaggregated run at draft length 7
confirmed the fix: LBS16 accept length recovered to the batch-1 level (~4.5, up from
the ~1.3–1.5 collapse). The full multi-draft-length sweep was not re-refreshed, but
DL7 is the representative worst case, so the collapse is considered resolved.

## Correctness and configuration

- **M — confidence-based truncation is scaffolding, not wired.** The confidence
  head is loaded but the batched worker always proposes the full block
  (`confidence_threshold=0.0`), and the head's proposed length is not reflected in
  worker output. The user-facing knobs were removed for now; wiring this requires a
  graph-safe variable proposal length threaded through the speculative
  scheduler/verifier, plus a regression test, before the config fields are re-added.

- **M — sparse KV capacity calculations do not consistently include
  `num_extra_kv_tokens`.** The context capacity and rewind configuration account for
  the extra speculative tokens, but generation warmup and typical generation
  descriptors still use the bare sequence length. Recheck max-sequence boundary
  cases and size every compressed pool consistently.

- **M — inherited dynamic-draft controls are unsupported but accepted.** DSpark does
  not participate in `support_dynamic_draft_len()`, and no drafter enforces
  `max_concurrency`. Reject these options for DSpark until they are implemented.

- **L — validate guarded edge cases.** The fallback noise token can equal
  `vocab_size`; a too-short `compress_ratios` list silently disables the draft sparse
  config; compressor bindings validate little beyond `next_n`; and
  `defer_post_mapping` is mutated during forward. These should fail early or become
  initialization-time invariants.

## Decode-path performance

- **M — reuse metadata staging buffers.** `DSparkSpecMetadata.prepare()` creates a CPU
  `arange` and a pinned slot-mapping tensor each step. Initialize the invariant batch
  indices once and reuse a preallocated pinned mapping buffer. (The dev-branch perf
  patch removed the per-step `.item()` syncs in context seeding and sparse-metadata
  offset computation, but not this allocation.) `prepare()` also now scans the
  generation-tail request IDs each step to assign disagg slots (the #16767 fix); the
  scan is pure host Python (dict lookups / `_assign_slot`) but should be folded into
  the same staging-buffer reuse when this is optimized.

- **M — remove repeated RoPE work.** The same stage-invariant RoPE phases are gathered
  once per draft stage. Compute them once in `forward_batched()` and pass them through.

- **M — replace materialized rolling-window top-k attention.** The batched path builds
  `kv_full`, a dense top-k/mask tensor, and replicated gathers for a simple structured
  pattern. A fused or compiled kernel should accept context and block KV separately and
  predicate context columns from `start_pos`. Require scalar parity, attention-sink,
  mixed-position, and CUDA-graph replay tests.

- **M — extend the compressor's multi-warp dispatch.** CR=128 with `next_n` 5–8 uses
  the single-warp path even though it is the normal DSpark range. Add and benchmark
  the corresponding multi-warp instantiations.

- **M — avoid duplicate draft-attention weights.** The default path loads the fp8 MLA
  projection modules from the checkpoint but only the dequantized bf16 attention cache
  executes (the real-MLA path that consumed the fp8 modules was removed). Drop or avoid
  materializing the unused fp8 modules for the draft stages while preserving checkpoint
  loading.

- **M — skip rolling-window backfill work when there are no interim accepted tokens.**
  The batched path still gathers captured hidden states, projects them, and prepares
  KV writes when every request has `nacc == 1`. Profile the frequency first; eager can
  short-circuit, while graph replay needs a mask-aware fused path to avoid upstream
  work rather than only masking the final write.

- **L — smaller hot-path cleanup.** The scalar attention top-k cache keys on raw
  positions and stops hitting after saturation; mHC eager execution repeats allocation
  and capture queries; a function-body import and private attention-metadata access
  remain in the worker. Treat these as profiling-driven cleanup. (Note: the
  `CUDA_GRAPH_DUMMY_REQUEST_ID` import in `_lazy_init` is a deliberate lazy import to
  break the `dspark -> cuda_graph_runner -> speculative.utils -> dspark` cycle, runs
  once, and is not a cleanup target.)

## API and ownership cleanup

- **L — tighten config consistency.** Make nested and top-level DSpark config
  resolution consistent for both the LLM API and direct model construction, and
  centralize the resolution rules. (When the confidence config fields are re-added,
  give `confidence_threshold` a declarative `[0, 1]` bound.)

- **L — document or register cached attention tensors.** `stage._dspark_attn` is a
  plain dictionary outside `state_dict()` and `.to()`. If registering the large cached
  tensors is undesirable, document that they are post-load, device-local derived
  state and define their lifecycle explicitly.

## Suggested order

1. Rolling-window slot fixes (scratch-row alias + disagg gen-request assignment) landed
   with unit tests and confirmed end-to-end at draft length 7 on disagg (GitHub #16767,
   PR #16772); ready to move the PR out of draft.
2. Make the remaining unsupported config fields fail fast (dynamic draft length,
   `max_concurrency`), and size sparse KV pools consistently with
   `num_extra_kv_tokens`.
3. Profile structured rolling-window attention and the backfill path, then pursue the
   fused-kernel and duplicate-weight changes with targeted GPU validation.
4. Pursue compressor multi-warp changes only with targeted GPU validation.
