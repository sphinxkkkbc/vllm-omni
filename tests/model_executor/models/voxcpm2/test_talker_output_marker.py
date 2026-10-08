# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VoxCPM2 dense-mode audio outputs must carry the sparse-alignment marker.

Dense mode (`_uses_sparse_audio_outputs()` False) still yields a strict
SUBSET of the batch whenever some requests emit no audio in a step (e.g.
prefill phase). The runner routes per-request `model_outputs` lists by batch
position unless `meta.sparse_audio` declares `meta.req_id` alignment — so a
subset without the marker misroutes audio across requests (RFC #5450 C3
review follow-up).
"""

from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

torch = pytest.importorskip("torch")


@functools.lru_cache(maxsize=1)
def _talker_cls():
    """Defer talker import (pulls vLLM model_executor) until first use."""
    from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import (
        VoxCPM2TalkerForConditionalGeneration,
    )

    return VoxCPM2TalkerForConditionalGeneration


def _make_dense_talker():
    cls = _talker_cls()
    talker = cls.__new__(cls)
    # Dense mode: emit every step, no delayed copy, no batched VAE decode.
    talker._audio_emit_every = 1
    talker._vae_decode_every = 1
    talker._enable_delayed_audio_copy = False
    talker._coalesce_audio_d2h = False
    talker._sample_rate = 16000
    talker._audio_queue = []
    return talker


def test_dense_subset_audio_carries_sparse_alignment_marker():
    talker = _make_dense_talker()
    # Batch had [r0, r1, r2]; only r1/r2 produced audio this step.
    talker._audio_queue = [("r1", torch.zeros(4)), ("r2", torch.ones(4))]

    out = talker.make_omni_output(torch.zeros(1))

    mm = out.multimodal_outputs
    assert mm["meta"] == {"req_id": ["r1", "r2"], "sparse_audio": ["1"]}
    assert len(mm["model_outputs"]) == 2
    assert len(mm["sr"]) == 2
    # (The coalesce-D2H sibling branch carries the same marker; it needs
    # device-resident chunks, so its marker contract is pinned here via the
    # shared dense-subset shape rather than a GPU-only test.)


def test_dense_full_batch_audio_still_carries_marker_and_order():
    # Full coverage is the degenerate subset: the marker stays truthful and
    # `meta.req_id` order matches the emitted list order.
    talker = _make_dense_talker()
    talker._audio_queue = [("r0", torch.zeros(2)), ("r1", torch.ones(2))]

    out = talker.make_omni_output(torch.zeros(1))

    mm = out.multimodal_outputs
    assert mm["meta"]["req_id"] == ["r0", "r1"]
    assert mm["meta"]["sparse_audio"] == ["1"]


def test_mrv2_sparse_audio_materializes_by_request_id():
    talker = _make_dense_talker()
    talker.vllm_config = SimpleNamespace(model_config=SimpleNamespace(use_v2_model_runner=True))
    talker._last_audio_output_req_ids = ["r0", "r1"]
    audio = torch.arange(4, dtype=torch.float32)
    talker._audio_queue = [("r0", None), ("r1", audio)]

    mm = talker.make_omni_output(torch.zeros(2)).multimodal_outputs

    assert mm["model_outputs"][0].numel() == 0
    assert torch.equal(mm["model_outputs"][1], audio)
    assert len(mm["sr"]) == 2
    assert "meta" not in mm

    talker._audio_queue = [("r0", None), ("r1", None)]
    empty_mm = talker.make_omni_output(torch.zeros(2)).multimodal_outputs
    assert [audio.numel() for audio in empty_mm["model_outputs"]] == [0, 0]


def test_mrv2_sparse_audio_preserves_request_order():
    talker = _make_dense_talker()
    talker.vllm_config = SimpleNamespace(model_config=SimpleNamespace(use_v2_model_runner=True))
    talker._last_audio_output_req_ids = ["r1", "r0"]
    talker._audio_queue = [("r1", torch.tensor([2.0])), ("r0", torch.tensor([1.0]))]

    # The runner's batch order is authoritative even if model-local requests
    # were visited in a different order during this step.
    mm = talker.make_omni_output(torch.zeros(2), request_ids=["r0", "r1"]).multimodal_outputs

    assert mm["model_outputs"][0].item() == 1.0
    assert mm["model_outputs"][1].item() == 2.0


def test_mrv2_preprocess_uses_scheduler_req_id():
    talker = _make_dense_talker()
    assert talker._resolve_request_id({"request_id": "v1-request"}) == "v1-request"
    talker.config = SimpleNamespace(hidden_size=4)
    talker._side_dtype = torch.float32
    talker._active_states = {}
    talker._pending_requests = []

    for req_id in ("r0", "r1"):
        talker.preprocess(
            torch.tensor([0]),
            torch.zeros(1, 4),
            req_id=req_id,
            request_id="stale-additional-id",
            _omni_is_prefill=False,
        )

    assert [entry[0] for entry in talker._pending_requests] == ["r0", "r1"]
