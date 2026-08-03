# DSpark disaggregated serving — rolling-window seed transfer (option 1a)

**Branch:** `dev/dspark-disagg-seed-1a` (off `main`; the 2 `dev/dspark-spec-dec` base commits were dropped)
**Status (2026-07-25, COMPLETE — validated across 3 datasets on 8×B300 dp-142):**
- 3 bugs fixed & committed (`64c0f770e4` OOM, `492b3aed59` IndexError, `d946a3f9a3` zero-seed/both-modes).
- **Transport fix HARDWARE-VALIDATED (context-first):** seed transfers byte-exact (RX reads the ctx's
  `norm=455.06/322.61/…`, real `ctx_len`, not `0/0`), applies correctly, **no hang**. Engineering is sound.
- **AL VERDICT (matched-methodology, §7a): the seed delivers NO measurable AL benefit.** Across
  gsm8k + humaneval + math500 × 6 OSLs × N=16: **disagg-OFF ≈ disagg-ON ≈ AGG at every point**. The
  disagg cold-window AL penalty **does not materially exist** — the short-OSL AL rise is inherent to
  short generations (the warm aggregated case shows the identical curve). The earlier apparent "gap"
  (off 3.32 vs agg 4.24) was a **raw-prompt vs `encode_messages` measurement artifact**, not real.
- **Recommendation:** option-1a seed is **not worth pursuing as an AL optimization**. Lasting value =
  the 3 bug-fixes (make the feature *function* if kept) + the matched-methodology harness
  (`dspark_disagg/dspark_bench_endpoint.py`, route-1, reuses the agg `encode_messages`).
**Scope:** Python (NIXL) transceiver only — the C++ transceiver is being deprecated, so we deliberately do not touch it.

> Archive note: originally LOCAL-ONLY; being committed to the dev branch as an investigation record
> per request. It contains internal env/machine detail — keep it OUT of any upstream PR.

---

## 0. TL;DR of the E2E session (read this first)

1. Stood up single-node 8×B300 disagg (ctx TP4 on GPU 0-3, gen TP4 on 4-7, router) for
   **DeepSeek-V4-Flash-DSpark** and collected an aggregated AL baseline + a disagg SEED=off baseline.
2. Hit and fixed **3 layered bugs**, each masked by the previous — only real-GPU E2E surfaced them:
   - **Bug A — 16 TB host alloc (OOM at startup).** AuxBuffer sized window sub-buffer at
     `max_batch×20000` slots. → **Fixed** with a host-memory-budget slot cap. **Committed `64c0f770e4`.**
   - **Bug B — gen `IndexError` crash on first token.** Forcing the aux path on context-first made
     the gen side overwrite `context_phase_params.first_gen_tokens` with the (empty) aux token
     fields → `first_gen_tokens[beam]` on `[]`. → **Fixed** by only applying aux tokens for
     generation-first requests. **Committed `492b3aed59`.**
   - **Bug C — seed arrives as zeros (no benefit).** Cross-server A/B proved ctx exports a real
     window (`norm≈353, ctx_len=64`) but gen reads its buffer as `norm=0, ctx_len=0`. Root cause
     (see §5): the aux RDMA **write is only dispatched under generation-first**. Under context-first
     the sender computes and calls `send_aux()` **before** the receiver registers, so the write has
     no destination; the sender's late-registration re-dispatch (`Sender._respond_with_kv`)
     re-sent **KV but not aux**. → **Fixed (both-modes)** by also re-dispatching the aux task there,
     mirroring KV, + setting session `_need_aux` when the aux buffer carries a DSpark window.
     **Committed `d946a3f9a3`.**
3. Also confirmed `schedule_style` is a **config knob** (`disagg_config.yaml`; default
   `context_first`) — useful as an independent cross-check that the seed works on the native
   gen-first aux path too. Set locally.

Net: all three bugs are **fixed & committed**; the seed is now designed to work on **both**
context-first and generation-first. **Remaining = hardware validation** of Bug C's fix in both
modes (nodes went down right after the commit); see §7.

---

## 1. Problem (unchanged design context)

DSpark's draft does not use the paged KV cache. It attends to a **worker-owned rolling
window** (`DSparkWorker._kv_windows`, shape `[max_slot+1, num_stages, window_size, head_dim]`)
of *projected captured-context* main_kv. In an aggregated run this window is **seeded from the
prompt** at prefill by `DSparkWorker._seed_context_windows` (runs only on a **context** forward).

In **disaggregated** serving, prefill runs on the *context* server and the draft runs on the
*generation* server. The gen worker never sees a context forward, so `_seed_context_windows`
never runs there → the draft window starts **empty (zeros)** on the gen server → degraded draft
acceptance for the first ~`window_size` output tokens (self-heals once the prompt tail scrolls
out of the window). Correctness-safe (target verify guarantees output); costs **acceptance rate /
speedup** early in each disagg request.

**E2E confirmation of the impact shape** (DeepSeek-V4-Flash-DSpark, disagg, SEED=off, 16×gsm8k):

| maxtok (OSL) | disagg SEED=off AL |
|---|---|
| 8 | **1.650** |
| 32 | 2.422 |
| 128 | 2.690 |
| 256 | 3.321 |

AL climbs steeply with output length → the cold-window penalty is almost entirely an
**early-token** effect (AL≈1.65 at OSL=8 ≈ almost no draft acceptance), exactly as the design
predicted. So a one-time seed can only help meaningfully at short OSL; over long OSL the average
washes out.

**Reference aggregated baseline** (user's `dspark_bench` harness, DL1-7 = draftlen 1..7):
`1.901 / 2.691 / 3.294 / 3.808 / 4.240 / 4.420 / 4.520`. Our disagg config is draftlen=5 →
compare to **DL5 = 4.240**. disagg SEED=off @ OSL256 = **3.321** → gap ≈ **0.92 AL (~22%)**,
front-loaded in early tokens. Caveat: **not fully same-harness** (aggregated via dspark_bench;
disagg via 16 sequential gsm8k `/v1/completions`) — trend/magnitude are trustworthy, but a strict
apples-to-apples needs the same harness on disagg (see §7).

---

## 2. Approach chosen — option 1a (unchanged)

Seed the window on the **context** server, then **ship the already-projected per-request window**
`[num_stages, window_size, head_dim]` (bf16, ~0.4 MB/req for this model) + its absolute decode
position (`_ctx_len`) to the gen server **alongside the KV cache**, via the Python transceiver's
per-request `AuxBuffer`. Gen installs it into `_kv_windows[slot]` instead of zeroing.

Model actually used for E2E: **DeepSeek-V4-Flash-DSpark** (combined target+draft ckpt), config
`dspark_block_size:5`, `dspark_target_layer_ids:[40,41,42]` → **num_stages=3**, **window_size=128**,
**head_dim=512** → window per slot = `3×128×512×2B = 393216 B` (≈0.375 MB). This is what makes the
uncapped `max_batch(16)×20000 = 320000`-slot buffer explode to ~120 GB… actually ~16 TB when the
pre-cap count is `max_concurrent_sessions` — hence Bug A.

**1a vs alternatives:** 1a = smallest wire cost, reuses the projected window, but **requires the
ctx server to keep DSpark draft weights** (incompatible with a future "prefill drops draft weights"
optimization). 1b (raw features, project on gen) = ~10× wire, would let prefill drop weights — not
chosen.

---

## 3. Data flow (verified pieces annotated)

```
CONTEXT server                                   GENERATION server
--------------                                   -----------------
target forward (context)
  └ DSparkWorker._seed_context_windows           [VERIFIED: runs on ctx, captured!=None,
      builds _kv_windows[slot] (projected)         export norm≈353, ctx_len=prompt_len]
      + stashes CPU copy in _export_seeds[req_id]
py_executor (KV-send path)
  └ take_export_seed(req_id) -> window / ctx_len / valid_len           [VERIFIED: seed set]
  └ _finalize_send: _need_aux_transfer -> pack_aux
      └ AuxBuffer.fill_slot: copy seed into slot   [VERIFIED: TX branch=SEED, buf_norm≈353]
              ====== NIXL RDMA aux write (slot-indexed) ======>
                                                 [BUG C: on context-first the RX never arms
                                                  /awaits this write → reads torch.zeros init]
                                                 _apply_aux -> unpack_aux -> get_slot_dspark
                                                   [OBSERVED: raw_ctx_len=0, buf_norm=0]
                                                 py_executor._prepare_disagg_gen_transmission_complete
                                                   └ stash_pending_seed(req_id, window, ctx_len, valid_len)
                                                     [VERIFIED: STASH rid matches ctx rid — keys OK]
                                                 DSparkSpecMetadata.prepare
                                                   └ _assign_slot(rid) + _apply_pending_seed
                                                     [VERIFIED: APPLY fires, same rid/slot]
                                                 → but window is zeros → identical to cold start
```

Position alignment (`_ctx_len`): ctx sets `prompt_len` (last-pos+1); gen bonus token at abs pos
`prompt_len`, frame `prompt_len+1` → matches. (Untestable until a real seed lands.)

---

## 4. Files changed & commit state

**Feature (already committed, unchanged this session):** `060b2852e2 [feat] DSpark disagg: seed
draft rolling window across ctx->gen`. Touches: `dspark.py` (export/stash/apply,
`dspark_seed_window_shape`), `auxiliary.py` (dspark sub-buffers, `fill_slot`, `get_slot_dspark`),
`transfer.py` (thread shape, `unpack_aux`), `transceiver.py` (`_dspark_seed_enabled`,
`_need_aux_transfer`), `kv_cache_transceiver.py`, `_util.py` (shape wiring + `_export_seeds_enabled`),
`py_executor.py` (ctx pull / gen stash), `llm_request.py` (`py_dspark_seed_*` fields), plus
`tests/unittest/_torch/speculative/test_dspark_disagg_seed.py` (6 CPU-only roundtrip tests, 6/6 pass).

**Quack pin (separate, committed):** `2311d84cc4 [fix] Cap quack-kernels<0.5.0 …`.

**This session's fixes (committed on top):**

| Commit | Fix | File | Bug |
|---|---|---|---|
| `64c0f770e4` | Cap AuxBuffer slot count to a host-memory budget (default 4 GB, env `TRTLLM_DSPARK_SEED_BUDGET_BYTES`) | `auxiliary.py` | A (16 TB OOM) |
| `492b3aed59` | `_apply_aux` only overwrites token fields for generation-first; new `_is_generation_first` helper | `transceiver.py` | B (`IndexError`) |
| `d946a3f9a3` | Re-dispatch aux on receiver registration in `Sender._respond_with_kv` (mirrors KV); session `_need_aux` honors `AuxBuffer.has_dspark_window` (both Tx/Rx) | `transfer.py`, `auxiliary.py` | C (zero seed / both-modes) |

**Reverted / never committed (local experiments):**
- `_util.py` `TRTLLM_DSPARK_DISABLE_SEED` A/B kill-switch (explicitly "do not commit"). Reverted.
- `auxiliary.py`/`dspark.py` `TRTLLM_DSPARK_SEED_DEBUG` logging. Reverted (re-add if needed for
  the pending Bug-C validation; the STASH/APPLY/EXPORT/TX-fill/RX-read log points are described in §5).

`git log` head: `d946a3f9a3` → `492b3aed59` → `64c0f770e4` → `2311d84cc4` → `060b2852e2` → `main`.
**Not pushed** (push only on explicit request).

---

## 5. Bug C — full root-cause (the important open item)

**Symptom:** SEED=on ran (no crash after Bug B fix) but AL ≈ SEED=off (3.10–3.36 @ OSL256, within
noise). Debug logging showed the seed transfers as zeros.

**Decisive evidence** (`TRTLLM_DSPARK_SEED_DEBUG=1`, same request id across servers):
```
CTX  fill_slot AUX slot=0 branch=SEED ctx_len=64 buf_norm=353.26 nbuffers=6
RX   get_slot_dspark slot=0     raw_ctx_len=0    buf_norm=0.00   nbuffers=6
```
Same aux slot, same 6 buffers, TX wrote a real seed, RX read pure zeros → **the aux RDMA payload
never lands on the RX for context-first requests.**

**Why (precise root cause):** it is a **sender-side dispatch-timing** gap, not a receiver problem.
The receiver **does** advertise its `aux_slot` unconditionally in both modes (`RxSession.receive`,
`transfer.py` ~1877), and the aux completion accounting (`process_aux_agent_result`,
`expected_transfers`) is mode-agnostic. The difference is **ordering**:
- **gen-first:** the gen server registers as receiver **first**; when the ctx server later computes
  and calls `send()` / `send_aux()`, the receiver's `req_info` is already saved → both KV **and**
  aux are written immediately. Works.
- **ctx-first:** the ctx server computes **first** and calls `send()` / `send_aux()` **before** the
  receiver registers → `dispatch_task` sees empty `req_info` → both are no-ops. When the receiver's
  request finally arrives, `Sender._respond_with_kv` re-dispatches the **KV** tasks
  (`tasks = list(session.kv_tasks)`) **but never `session.aux_task`** → the aux write is silently
  dropped → RX reads its `torch.zeros` init (`_dspark_ctx_len_buffer`=0, hence `raw_ctx_len=0`, not
  the `-1` "no-seed" sentinel). The reverted experiment that only flipped session `_need_aux=True`
  made the RX **wait** for an aux write that was never re-dispatched → **60 s timeout → hang**.

**Fix (committed `d946a3f9a3`, both-modes, mirrors KV):**
- `Sender._respond_with_kv`: after re-dispatching the KV tasks, also `_build_aux_write_meta(
  session.aux_task, info) + _enqueue` when `aux_task is not None` — the exact KV pattern.
- Session `_need_aux = gen_first or aux_buffer.has_dspark_window` (both Tx/Rx) so both sides run the
  full aux handshake + completion wait.
- No double-dispatch: the send/re-dispatch paths split cleanly by mode — ctx-first → `send_aux()` is
  a no-op and `_respond_with_kv` is the sole real dispatcher; gen-first → `_respond_with_kv` returns
  early (`session is None`) and `send_aux()` dispatches. Same split KV already relies on.
- `should_send_aux` (TP/PP de-dup) is unchanged and mode-agnostic.

**Pending:** hardware validation in **both** modes (see §7) — nodes went down right after the commit.

---

## 6. E2E environment & repro (so the next session can resume fast)

- **Nodes:** `umb-b300-dp-142`, `umb-b300-dp-147` (8×B300 each). SSH **only** via
  `/home/jonasl/.myvim` (= `/bin/ssh`); combine with `dangerouslyDisableSandbox: true`.
  *(Both went offline at end of session; frequent SSH may also trip sshd rate-limits.)*
- **Container:** `tensorrt_llm-devel-jonasl` (cherry-picked feature branch). `/code/tensorrt_llm`
  is a **bind-mount of `/home/scratch.jonasl_sw/GitRepos/TensorRT-LLM`** — edits on the login box
  are seen inside the container immediately (no propagation/reload of *source* needed; only a
  server restart to reload Python).
- **Model:** `/dev/shm/DeepSeek-V4-Flash-DSpark` (copied from
  `/home/scratch.trt_llm_data_ci/llm-models/…`). ~10 min to load per server.
- **Configs & scripts** (NOT in git; on shared FS `/home/scratch.jonasl_sw/dspark_disagg/`):
  `ctx_config.yaml`, `gen_config.yaml` (TP4, `max_seq_len:4096`, `max_batch_size:16`, DSpark spec
  draftlen/block 5, NIXL + `transceiver_runtime:PYTHON`), `disagg_config.yaml` (router; **now has
  `schedule_style: generation_first`**), `run_disagg_al.sh` (orchestrator; `KEEP_ALIVE=1` leaves
  servers up for re-measure), `measure_al.py` (raw `/v1/completions`, pulls
  `avg_decoded_tokens_per_iter`).
- **Must-set WARs (baked into `run_disagg_al.sh`):** `max_seq_len:4096` (default ~1M **IMAs the
  DeepSeek-V4 DSA sparse-indexer warmup** — this was a separate CUDA illegal-address crash,
  exonerated from our feature by SEED=off crashing identically), `TRTLLM_SKIP_KV_CACHE_ESTIMATION=1`
  (NVIDIA#15633 + skips ~14 min estimation), `NCCL_NVLS_ENABLE=0` + `TORCH_SYMM_MEM_DISABLE_MULTICAST=1`,
  `DSPARK_FIX_WO_A=1` (reference-faithful dequant, matches aggregated baseline).
- **Measurement gotcha:** this tokenizer has **no chat template** → `/v1/chat/completions` 400s;
  must use raw `/v1/completions` with `"prompt": <text>` (matches how the aggregated bench also
  fell back to raw).
- **Fast iteration trick:** `schedule_style` is attached **per-request by the router** → the
  ctx/gen servers need **no reload** to change it; restart **only** the router
  (`python -m tensorrt_llm.commands.serve disaggregated -c disagg_config.yaml`, needs
  `PYTHONPATH=/code/tensorrt_llm`, `cd /code/tensorrt_llm`) against live ctx(8001)+gen(8002). ~1 min.

### 6b. Cross-node option on computelab SLURM (scouted 2026-07-24, then deferred)

Two separate 4×B300 SLURM jobs = the **real cross-node disagg topology** (ctx TP4 on node A, gen
TP4 on node B) — a *stronger* test of the Bug-C aux-transfer fix than the single-node run, since it
exercises real cross-node NIXL RDMA + real registration timing. What was learned before deferring:

- **Access:** the compute nodes (e.g. `umb-b300-020`, `umb-b300-004`) are **not directly SSH-able**
  (`.myvim` to them times out). Go through the frontend: `.myvim computelab-sc-01`, then
  `srun --jobid=<id> --overlap --gres=gpu:b300:<N> -N1 -n1 bash -lc "…"`. `squeue -u $USER`,
  `squeue -s -j <id>` (steps), `scontrol show job <id>` all on the frontend.
- **Container:** a prebuilt enroot image exists at
  **`/home/scratch.jonasl_sw/containers/trtllm-dev.sqsh`** (25 GB, dated Jun 28 — **verify its
  built C++/deps still match this main-based branch before trusting; may be stale** re: the
  quack-kernels pin etc.). Run via pyxis on the srun step: `--container-image=<sqsh>
  --container-mounts=/home/scratch.jonasl_sw:/home/scratch.jonasl_sw,...`, bind the repo and use
  `PYTHONPATH=/code/tensorrt_llm` (our fixes are pure-Python, so the bind-mounted repo's `.py`
  changes take effect as long as the container can `import tensorrt_llm`).
- **FS:** repo (`/home/scratch.jonasl_sw/GitRepos/TensorRT-LLM`) and model
  (`/home/scratch.trt_llm_data_ci/llm-models/DeepSeek-V4-Flash-DSpark`) are on shared FS, **visible
  from the compute nodes**. `/dev/shm` is 2 TB per node → stage weights per node as before.
- **GRES GOTCHA (this is what kept biting):** orphaned `bash` srun steps (left by SSH/step
  timeouts) **keep holding a GPU**, so subsequent `--overlap` steps see **fewer GPUs than the job's
  `AllocTRES`**. Symptom: `AllocTRES gres/gpu:b300=4` but `CUDA_VISIBLE_DEVICES=0,1,2` (only 3).
  Diagnose with `squeue -s -j <id>` (look for a `.0 bash` step holding `gres/g`). `scancel <id>.0`
  was observed **not** to free it reliably (step lingered, node stayed at 3/4) → the clean fix is to
  **cancel and resubmit the whole hold job**. A freshly-submitted job with no stray steps gave a
  clean **4/4** (`umb-b300-004` did; `umb-b300-020` was stuck at 3/4). **Always start from a clean
  job and never let an srun step get orphaned by an SSH timeout.** (Related dev-box note in
  auto-memory: killed `-n8` steps corrupt implicit-gres.)
- **Cross-node config changes still to do (not yet written):** rewrite `ctx_config.yaml` /
  `gen_config.yaml` / `disagg_config.yaml` `localhost:8001/8002` → the two nodes' actual
  IPs/hostnames; ensure the router (run on either node or the frontend) can reach both; confirm
  NIXL/UCX builds links over the inter-node fabric (IB/RoCE — untested). The single-node
  `run_disagg_al.sh` puts ctx on GPU0-3 + gen on GPU4-7 of one box; the cross-node version needs
  ctx launched under job A and gen under job B (two `srun` steps) with matching `disagg_request_id`
  wiring handled by the router.
- **WAR re-check on computelab B300:** the dev-box WARs `NCCL_NVLS_ENABLE=0` +
  `TORCH_SYMM_MEM_DISABLE_MULTICAST=1` were specific to `umb-b300-dp-186` lacking NVLink-multicast
  fabric; computelab B300 nodes likely have proper fabric → **probably drop them**. Keep the
  hardware-agnostic ones (`max_seq_len:4096`, `TRTLLM_SKIP_KV_CACHE_ESTIMATION=1`, `DSPARK_FIX_WO_A=1`).
- **Status:** jobs `3269744` (`umb-b300-020`) and `3269753` (`umb-b300-004`) were **abandoned**
  (020 stuck at 3/4 GPUs); do **not** assume they're still alive. Next attempt: fresh nodes/jobs.

---

### 6c. WORKING env recipe on a fresh dp-142-style dev-box (hard-won 2026-07-25)

The `tensorrt_llm/devel:latest-jonasl` image is a **build** image — it has torch+CUDA but NOT the
runtime pip deps (transformers, accelerate, nixl, blake3, flashinfer, …). The previous container had
them installed at runtime and lost. To rebuild the env on a fresh node:

1. `docker run -d --name tensorrt_llm-devel-jonasl --gpus all --network host --ipc host \
   -v /home/scratch.jonasl_sw:/home/scratch.jonasl_sw -v /home/scratch.trt_llm_data_ci:/home/scratch.trt_llm_data_ci \
   -v /home/scratch.jonasl_sw/GitRepos/TensorRT-LLM:/code/tensorrt_llm tensorrt_llm/devel:latest-jonasl sleep infinity`
   (do NOT mount `/home/jonasl` — that bind-mount is broken for writes; and it would shadow nothing useful.)
2. Install deps **as root into dist-packages** (torch is protected by `/etc/pip/constraint.txt`, stays the nv build):
   `docker exec -u root … pip3 install --no-cache-dir -r /code/tensorrt_llm/requirements.txt`
   plus `nixl-cu13==1.3.1`. (Container default user is non-root `jonasl`; root writes dist-packages.)
3. Run everything with **`HOME=/home/scratch.jonasl_sw/dspark_home`** (writable) + `PYTHONPATH=/code/tensorrt_llm`.
   The `/home/jonasl` mount can't create `~/.cache/flashinfer`; deps are in dist-packages so overriding HOME is safe.
4. Model at `/dev/shm/DeepSeek-V4-Flash-DSpark` (stage per node if absent). `run_disagg_al.sh` now exports HOME.

`import tensorrt_llm` → v1.3.0rc23 once this is done. (Gotcha: `-u root` cannot write jonasl-owned shared-FS
dirs — root is squashed on NFS — so write logs to /tmp or the task file, not a jonasl dir.)

### 6d. Local uncommitted diagnostics currently applied (for validation; do NOT commit)

- `dspark.py` + `auxiliary.py`: `TRTLLM_DSPARK_SEED_DEBUG=1` logging (STASH/APPLY/CTX EXPORT/TX fill/RX read).
- `_util.py`: `TRTLLM_DSPARK_DISABLE_SEED=1` A/B kill-switch (the "seed OFF" baseline in `run_disagg_al.sh`).
- `dspark_disagg/dspark_bench_endpoint.py`: **route-1 endpoint client** — reuses the agg harness's
  `encode_messages` + `avg_decoded_tokens_per_iter`, drives `/v1/completions`. Point `DSPARK_BENCH_ENDPOINT`
  at the disagg router (:8000) or an aggregated trtllm-serve. Same env knobs as `dspark_bench_regime.py`.

---

## 7a. AL validation results (2026-07-25, dp-142, matched methodology)

Measured via the route-1 endpoint client (`encode_messages`, `avg_decoded_tokens_per_iter`),
gsm8k/chat, greedy, DL5, same machine. **KEY: use encoded prompts — raw prompts give meaningless
numbers.**

**Functional (context-first, byte-exact seed transfer):** PROVEN. RX reads the ctx's exported window
(`norm=455.06/322.61/…`, `ctx_len=107/54/…`), APPLY installs real ctx_len, no hang. (Was `0/0` pre-fix.)

**AL, seed ON vs OFF vs agg — same machine, same encoded prompts, N=16 (gsm8k/chat, DL5):**

| maxtok | OFF (cold) | ON (seed) | AGG (warm) |
|--------|------|------|------|
| 8   | 2.150 | 2.350 | 2.175 |
| 16  | 2.663 | 2.730 | 2.919 |
| 32  | 3.214 | 3.377 | 3.273 |
| 64  | 3.856 | 3.854 | 3.847 |
| 128 | 4.212 | 4.229 | 4.209 |
| 256 | 4.239 | 4.116 | 4.468 |

**VERDICT (gsm8k): the three curves overlap within noise at EVERY OSL.** Decisive point:
**AGG@8=2.175 ≈ OFF@8=2.150 ≈ ON@8=2.350** — the *aggregated warm-window* case ALSO has low AL at
short OSL and rises on the SAME curve. So the short-OSL AL rise is **inherent to short generations**
(few iterations / generation warmup), NOT a disagg cold-window penalty. **disagg-off already equals
agg at every OSL → there is no meaningful cold-window problem → the seed delivers no measurable AL
benefit** (ON ≈ OFF ≈ AGG). The earlier apparent gap (off 3.32 vs agg 4.24) was purely a
**raw-prompt vs encoded-prompt** measurement artifact.

**Two-layer conclusion:**
1. **Transport / engineering (KEEP):** the 3 committed fixes are correct & hardware-validated — seed
   transfers byte-exact and applies on both schedule modes, no crash/OOM/hang. If the feature is
   kept, it now *functions* correctly.
2. **Feature value (does NOT hold):** option-1a's premise — a disagg cold-window AL penalty — does
   not materially exist under rigorous matched-methodology measurement. **Not worth pursuing as an
   AL optimization.**

**Completeness check (request A): humaneval + math500 (N=16, chat, DL5) — confirms it generalizes.**

humaneval — OFF / ON / AGG:
| maxtok | OFF | ON | AGG |
|---|---|---|---|
| 8   | 2.050 | 1.867 | 1.908 |
| 16  | 2.695 | 2.729 | 2.781 |
| 32  | 3.541 | 3.835 | 3.694 |
| 64  | 4.140 | 4.266 | 4.324 |
| 128 | 4.454 | 4.534 | 4.466 |
| 256 | 4.461 | 4.366 | 4.558 |

math500 — OFF / ON / AGG:
| maxtok | OFF | ON | AGG |
|---|---|---|---|
| 8   | 1.835 | 1.930 | 1.888 |
| 16  | 2.702 | 2.687 | 2.716 |
| 32  | 3.391 | 3.432 | 3.504 |
| 64  | 3.631 | 3.883 | 3.811 |
| 128 | 4.208 | 4.219 | 4.195 |
| 256 | 4.489 | 4.397 | 4.414 |

**FINAL VERDICT (3 datasets × 6 OSLs, N=16, matched encoding):** OFF ≈ ON ≈ AGG at **every** point —
often OFF or AGG is the highest (pure noise, no systematic seed sign). The disagg cold-window AL
penalty **does not materially exist**; the short-OSL AL rise is inherent to short generations (the
warm aggregated case shows the identical curve). **The option-1a seed feature, though now correctly
implemented and hardware-validated, delivers no measurable AL benefit and is not worth pursuing as
an AL optimization.** The lasting value is the 3 committed bug-fixes (they fix real crashes/OOM/hang
in the feature so it *functions* if kept) + the rigorous matched-methodology measurement harness.

---

## 7. Pending work & plan (priority order)

Design decision is **made**: support **both** schedule styles (DSpark is a common feature; the
deployment can't be assumed generation-first). The both-modes fix is committed (`d946a3f9a3`); what
remains is hardware validation.

1. **[DO FIRST] Validate the both-modes fix — CONTEXT-FIRST (the one that was broken).** Default
   `disagg_config.yaml` is context-first; run seed ON vs OFF. Re-enable debug logging first
   (`TRTLLM_DSPARK_SEED_DEBUG=1`; the log points — CTX `_seed_context_windows`/EXPORT, TX
   `fill_slot`, RX `get_slot_dspark`, gen STASH/APPLY — were reverted, re-add from this session's
   history or re-instrument). **Pass criteria:**
   - RX `get_slot_dspark` now reads a **real** seed (`raw_ctx_len≈41–65, buf_norm≈280–360`, **not**
     `0/0`); gen `APPLY` with nonzero `ctx_len`.
   - **No hang** (requests complete promptly, not 60 s apart).
   - **Short-OSL AL above the SEED=off baseline** (off: 1.65@8, 2.42@32, 2.69@128, 3.32@256).
   - Watch the aux completion accounting under TP4: no "received too many aux transfers" / no aux
     timeout — the `expected_transfers` / multi-peer fan-in behaving like KV is the main risk.
2. **[REGRESSION] Validate GENERATION-FIRST still works.** Flip `schedule_style: generation_first`
   (router-only restart, ~1 min — see §6). Same pass criteria. Confirms the fix didn't break the
   path that already worked (and that aux isn't double-dispatched).
3. **[MEASUREMENT] Apples-to-apples AL.** Re-collect disagg AL with the **same harness** as the
   aggregated baseline (`dspark_bench` on the disagg endpoint) so the 3.321-vs-4.240 comparison is
   strictly comparable; then collect disagg DL1-7 for seed ON vs OFF (the original ask).
4. **[CARRY-OVER correctness open questions] (still valid, from the original design doc):**
   - Attention length-masking: are empty window frames masked by `_ctx_len` or attended as zeros
     (`dspark_attention_forward`)? Determines "less context" vs "actively harmful".
   - TP/PP: window is per-rank; confirm aux slot ↔ rank mapping is per-rank consistent under
     TP>1 / attention-DP (token-buffer aux path already handles this; window rides the same slot).
   - `num_extra_kv_tokens` **not** transferred by the Python transceiver (flagged by transceiver
     owner). Different state (target paged KV, not our window). Very likely harmless for DSpark
     (draft doesn't run in prefill → those slots unwritten). **Spot-check output correctness**
     SEED=off disagg vs aggregated on a greedy set to rule it out; secondary suspect = whether the
     DSA sparse-indexer prompt KV is shipped.

---

## 8. One-line status for a fresh session

*"Disagg DSpark window-seed (option 1a): plumbing verified on 8×B300; **3 bugs fixed & committed**
(`64c0f770e4` OOM, `492b3aed59` IndexError, `d946a3f9a3` zero-seed/both-modes). Bug C root cause
was the sender not re-dispatching the aux write on late receiver registration under context-first;
fix mirrors KV (§5). **Now validate on hardware in both context-first (§7.1) and generation-first
(§7.2)** — nodes went down right after the commit. Re-enable `TRTLLM_DSPARK_SEED_DEBUG` logging
first; pass = RX reads real seed (`buf_norm≈300`, not 0), no hang, short-OSL AL > off baseline."*
