# DSpark speculative decoding development notes

This document records the current state of the DeepSeek DSpark implementation in
the TensorRT-LLM PyTorch backend. It is maintained on the development branch and
was not part of PR #15808 (now merged to main).

## Current status

The functional port is complete and has run end to end with the
DeepSeek-V4-Pro-DSpark checkpoint on 8 GPUs. The implementation includes:

- three full DeepSeek-V4 draft stages loaded from the `mtp.*` namespace;
- target hidden-state capture from the configured target layers;
- parallel block drafting followed by a Markov head; a confidence head is loaded
  but not yet wired into proposal truncation (scaffolding for future dynamic
  drafting);
- standard target verification, so draft errors affect acceptance and
  performance but not greedy output correctness;
- batched generation that runs eagerly and is CUDA-graph capturable;
- model/config validation, weight remapping and loading, and hardware-agnostic
  unit coverage.

The branch also carries a small performance patch on top of the PR. It removes
avoidable host synchronizations in DSpark context seeding and DeepSeek-V4 sparse
metadata preparation, and right-sizes hidden-state capture buffers for CUDA graph
keys. These changes do not alter draft semantics.

The remaining correctness, configuration, and performance work is tracked in
`DSPARK_REVIEW.md`. Two rolling-window slot defects are now fixed: the scratch-slot
aliasing (unknown / padded request IDs map to a dedicated scratch row instead of slot
0), and the disaggregated-serving accept-length collapse (GitHub #16767) — on the
generation server no request runs `_seed_context_windows`, so `prepare()` now assigns
each real generation request its own rolling-window slot instead of sharing one row.
Both are covered by hardware-agnostic unit tests and confirmed end to end: a
disaggregated run at draft length 7 recovered LBS16 accept length to the batch-1 level
(~4.5, up from the ~1.3–1.5 collapse). The full multi-draft-length sweep was not
re-refreshed, but DL7 is the representative worst case. The code fix is up as a
main-targeted PR (GitHub #16772), ready to move out of draft.

## Algorithm and implementation

DSpark extends the DeepSeek MTP family by producing a block of draft tokens in one
backbone pass. The shipped checkpoint uses three draft stages and a default block
size of five.

For each generation step:

1. The target model verifies the previous draft and captures hidden states from
   the configured target layers.
2. The worker constructs a block beginning with the target-sampled bonus token,
   followed by mask/noise tokens.
3. `DSparkDraftModel` projects the captured target states and runs the three draft
   stages.
4. The Markov head refines the block autoregressively. A confidence head is
   loaded, but confidence-based truncation is not wired into the production
   worker (the batched path always proposes the full block); it is scaffolding
   for future dynamic drafting.
5. The target model verifies the proposed prefix using the normal speculative
   decoding acceptance rule.

The draft attention is a pure-PyTorch, rolling-window MLA implementation.
Persistent windows are owned by `DSparkWorker` and indexed through per-request slots.
A slot is assigned when a request first seeds its window in `_seed_context_windows`
(aggregated prefill); in disaggregated serving the prefill happens on the context
server, so the generation server instead assigns the slot in
`DSparkSpecMetadata.prepare()`. Generation uses the fixed-shape batched path, which
runs eagerly and is CUDA-graph capturable; it is the only draft-attention path.

Key source files:

- `tensorrt_llm/_torch/models/modeling_dspark.py`: draft model, stage chain,
  weight loading, and one-engine wrapper;
- `tensorrt_llm/_torch/models/dspark/`: attention, proposal, and head helpers;
- `tensorrt_llm/_torch/speculative/dspark.py`: metadata and worker orchestration;
- `tensorrt_llm/_torch/speculative/utils.py`: speculative worker construction;
- `tensorrt_llm/llmapi/llm_args.py`: `DSparkDecodingConfig` and config resolution;
- `tests/unittest/_torch/speculative/hw_agnostic/test_dspark_*.py`: unit tests.

## Validated behavior

The port was checked against the checkpoint's DeepSpec reference implementation.
The attention path, full three-stage chain, weight remapping, hidden-state capture,
and CUDA graph capture/replay were validated independently. The integration test
also exercises full model load and generation on the real checkpoint.

The most representative acceptance sweep used TP8, MoE EP8, attention DP,
`MEGAMOE_DEEPGEMM`, greedy decoding, and 12 prompts from each of GSM8K,
HumanEval, and MATH500 in chat and thinking modes. Mean accepted length includes
the guaranteed target token:

| Draft length | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean accepted length | 1.905 | 2.683 | 3.324 | 3.912 | 4.240 | 4.621 | 4.634 |

These numbers are exploratory, not a default-selection result. Acceptance depends
on dataset, prompt mode, sampling, parallel layout, MoE backend, and attention DP.
Longer drafts also cost more draft compute, so steady-state tokens/s—not accepted
length alone—should determine the production block size. A direct vLLM comparison
requires a matched dataset, sampler, source revision, and runtime configuration.

## Validation

Hardware-agnostic DSpark tests:

```bash
pytest tests/unittest/_torch/speculative/hw_agnostic/test_dspark_heads.py \
  tests/unittest/_torch/speculative/hw_agnostic/test_dspark_draft.py \
  tests/unittest/_torch/speculative/hw_agnostic/test_dspark_attention.py \
  tests/unittest/_torch/speculative/hw_agnostic/test_dspark_worker.py \
  tests/unittest/_torch/speculative/hw_agnostic/test_dspark_cuda_graph.py -q
```

The full-model test is
`tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestDeepSeekV4ProDSpark::test_gsm8k_dep8_megamoe_deepgemm`.
It requires eight suitable GPUs and `LLM_MODELS_ROOT` pointing to the model
store.

When changing user-facing DSpark config fields, regenerate and check the LLM-args
golden manifest:

```bash
python3 scripts/generate_llm_args_golden_manifest.py
python3 scripts/generate_llm_args_golden_manifest.py --check
```

## Next work

The disaggregated-serving fix is confirmed (draft length 7 recovered LBS16 accept
length to the batch-1 level), so PR #16772 can move out of draft. Address the
correctness and config-contract items in `DSPARK_REVIEW.md` before larger performance
work. Beyond that, the highest-value measurements are:

1. interleaved steady-state eager/CUDA-graph throughput at draft lengths 5–7;
2. a CUTLASS plus attention-DP control to separate the effects of attention DP and
   the MoE backend;
3. profiling of the structured rolling-window attention and Markov-logit hot paths.

An MTP-1 comparison on the same stack remains blocked by the DeepSeek-V4 sparse-MLA
backend not allocating an SWA buffer for the MTP layer. That is an upstream backend
issue rather than a DSpark algorithm issue.
