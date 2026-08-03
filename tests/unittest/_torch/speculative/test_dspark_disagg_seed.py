# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for DSpark disaggregated rolling-window seed transfer (option 1a).

These exercise the CPU-only, hardware-agnostic core of the feature: the
DSparkWorker export/stash/apply roundtrip, the draft-window shape helper, and
the AuxBuffer window sub-buffer. They do NOT need a GPU or a real draft model —
the end-to-end ctx->gen NIXL transfer is covered by the disagg integration run
described in DSPARK_DISAGG_SEED_HANDOFF.md.
"""

from collections import deque
from types import SimpleNamespace

import torch

from tensorrt_llm._torch.speculative.dspark import DSparkWorker, dspark_seed_window_shape


def _bare_worker(
    max_batch: int, num_stages: int, win: int, head_dim: int, *, export_enabled: bool
) -> DSparkWorker:
    """A DSparkWorker with rolling-window buffers wired by hand (CPU), bypassing
    ``__init__`` / ``_lazy_init`` so the test needs no draft model or GPU."""
    w = DSparkWorker.__new__(DSparkWorker)
    w._win_inited = True
    w._win = win
    # rows: max_batch request slots + 1 scratch row (mirrors _lazy_init)
    w._kv_windows = torch.zeros((max_batch + 1, num_stages, win, head_dim))
    w._ctx_len = torch.zeros(max_batch + 1, dtype=torch.long)
    w._valid_len = torch.zeros(max_batch + 1, dtype=torch.long)
    w._batch_to_slot = torch.zeros(max_batch, dtype=torch.long)
    w._free_slots = deque(range(max_batch))
    w._req_to_slot = {}
    w._scratch_slot = max_batch
    w._graph_dummy_id_floor = 10_000_000
    w._export_seeds_enabled = export_enabled
    w._export_seeds = {}
    w._pending_seeds = {}
    return w


def test_worker_seed_export_transfer_apply_roundtrip():
    max_batch, num_stages, win, head_dim = 4, 3, 8, 16

    # --- context server: a seeded (non-zero) window for req 42, then export ---
    ctx = _bare_worker(max_batch, num_stages, win, head_dim, export_enabled=True)
    ctx_slot = ctx._assign_slot(42, reset=True)
    ctx._kv_windows[ctx_slot].normal_()  # stand in for _seed_context_windows
    ctx._ctx_len[ctx_slot] = 137
    ctx._valid_len[ctx_slot] = 5
    # what _seed_context_windows stashes when _export_seeds_enabled:
    ctx._export_seeds[42] = (
        ctx._kv_windows[ctx_slot].detach().to("cpu").clone(),
        int(ctx._ctx_len[ctx_slot].item()),
        int(ctx._valid_len[ctx_slot].item()),
    )

    seed = ctx.take_export_seed(42)
    assert seed is not None
    assert ctx.take_export_seed(42) is None, "export must be pop-once"
    exported_window, exported_ctx_len, exported_valid_len = seed
    assert exported_ctx_len == 137
    assert exported_valid_len == 5

    # --- generation server: fresh worker, stash pending, assign slot, apply ---
    gen = _bare_worker(max_batch, num_stages, win, head_dim, export_enabled=False)
    gen.stash_pending_seed(42, exported_window, exported_ctx_len, exported_valid_len)

    gen_slot = gen._assign_slot(42, reset=False)  # zeros window (as in prepare())
    assert torch.count_nonzero(gen._kv_windows[gen_slot]) == 0

    applied = gen._apply_pending_seed(42, gen_slot)
    assert applied is True
    assert 42 not in gen._pending_seeds, "pending seed must be consumed once"

    # gen window is the ctx window (seeded, not zeroed); ctx_len propagated.
    assert torch.equal(gen._kv_windows[gen_slot], exported_window)
    assert int(gen._ctx_len[gen_slot].item()) == 137
    assert int(gen._valid_len[gen_slot].item()) == 5


def test_apply_pending_seed_noop_without_seed():
    gen = _bare_worker(4, 3, 8, 16, export_enabled=False)
    slot = gen._assign_slot(7, reset=False)
    assert gen._apply_pending_seed(7, slot) is False
    assert torch.count_nonzero(gen._kv_windows[slot]) == 0  # left zeroed


def test_export_disabled_does_not_stash():
    # A worker with export disabled (aggregated / non-python-transceiver) must not
    # accumulate export entries even if seeding runs.
    w = _bare_worker(4, 3, 8, 16, export_enabled=False)
    assert w._export_seeds_enabled is False
    assert w.take_export_seed(1) is None


def test_seed_window_shape_ducktyping():
    # DSpark-like draft model -> concrete shape.
    dspark_like = SimpleNamespace(
        num_stages=3,
        _attn_params={"window_size": 128, "head_dim": 576},
    )
    assert dspark_seed_window_shape(dspark_like) == (3, 128, 576)

    # Non-DSpark draft models -> None (feature auto-disables).
    assert dspark_seed_window_shape(SimpleNamespace()) is None
    assert dspark_seed_window_shape(SimpleNamespace(num_stages=3)) is None  # no _attn_params
    assert (
        dspark_seed_window_shape(SimpleNamespace(num_stages=3, _attn_params={"head_dim": 576}))
        is None
    )


def test_auxbuffer_window_subbuffer_roundtrip():
    from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxBuffer

    num_stages, win, head_dim = 3, 8, 16
    buf = AuxBuffer(
        max_slot_num=4,
        beam_width=1,
        max_draft_len=6,
        dspark_window_shape=(num_stages, win, head_dim),
    )

    # meta carries the 4 token sub-buffers + 3 DSpark sub-buffers.
    assert len(buf.meta.ptrs) == 7
    assert len(buf.meta.item_sizes) == 7

    slot = buf.alloc_slot().id
    # Simulate what fill_slot writes without a full LlmRequest.
    window = torch.randn(num_stages, win, head_dim, dtype=torch.bfloat16)
    buf._dspark_window_buffer[slot].copy_(window)
    buf._dspark_ctx_len_buffer[slot, 0] = 137
    buf._dspark_valid_len_buffer[slot, 0] = 5

    got = buf.get_slot_dspark(slot)
    assert got is not None
    got_window, got_ctx_len, got_valid_len = got
    assert got_ctx_len == 137
    assert got_valid_len == 5
    assert torch.equal(got_window, window)

    # ctx_len == -1 is the "no seed shipped" sentinel -> None.
    buf._dspark_ctx_len_buffer[slot, 0] = -1
    assert buf.get_slot_dspark(slot) is None


def test_auxbuffer_without_dspark_shape_has_no_window():
    from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxBuffer

    buf = AuxBuffer(max_slot_num=4, beam_width=1, max_draft_len=6)
    assert len(buf.meta.ptrs) == 4  # only the token sub-buffers
    slot = buf.alloc_slot().id
    assert buf.get_slot_dspark(slot) is None
