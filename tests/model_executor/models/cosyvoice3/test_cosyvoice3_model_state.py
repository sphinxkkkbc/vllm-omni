# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-stage audio consumes only valid samples at the existing eager hook."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.cosyvoice3.model_state import CosyVoice3ModelState
from vllm_omni.model_executor.models.cosyvoice3.pipeline_adapter import InnerStageAdapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Decoder:
    def __init__(self):
        self.calls = []

    def forward_streaming(self, **kwargs):
        self.calls.append(kwargs)
        return torch.ones(1, 1, 4), None


def _run(state, batch, hidden, tokens, outputs):
    state.run_eager_mtp(batch, hidden, tokens, outputs)


def _fixture():
    state = object.__new__(CosyVoice3ModelState)
    state.adapter = InnerStageAdapter(6561, chunk_size=1, lookahead=0)
    state._max_model_len = 32
    state._req_states = SimpleNamespace(max_seq_len=np.array([32, 32]), prompt_len=SimpleNamespace(np=np.array([4, 6])))
    decoder = _Decoder()
    state.model = SimpleNamespace(
        code2wav=decoder, config=SimpleNamespace(sample_rate=24000), _maybe_enable_code2wav_trt=lambda: None
    )
    payloads = [{"sampling_params": SimpleNamespace(max_tokens=2)} for _ in range(2)]
    state.intermediate_buffer = SimpleNamespace(gather=lambda batch: payloads)
    batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping_np=np.array([0, 1]),
        req_ids=["prefill", "decode"],
        num_computed_tokens_np=np.array([0, 5]),
        num_computed_prefill_tokens_np=np.array([0, 5]),
        prefill_len_np=np.array([4, 5]),
        is_prefilling_np=np.array([True, False]),
        num_scheduled_tokens=np.array([2, 1], dtype=np.int32),
    )
    conditioning = {
        "speech_token": [torch.ones(1, 1, dtype=torch.long), torch.ones(1, 1, dtype=torch.long)],
        "speech_feat": [torch.ones(1, 2, 80), torch.ones(1, 2, 80)],
        "embedding": [torch.ones(1, 192), torch.ones(1, 192)],
    }
    return state, batch, conditioning, decoder


def test_final_prefill_creates_state_but_intermediate_prefill_does_not():
    state, batch, conditioning, decoder = _fixture()
    outputs = {"embed": conditioning}
    _run(state, batch, torch.zeros(3, 8), torch.tensor([[0], [7]]), outputs)
    assert "prefill" not in state.adapter.states
    batch.num_computed_prefill_tokens_np[0] = 2
    batch.num_computed_tokens_np[0] = 2
    batch.num_computed_tokens_np[1] += 1
    _run(state, batch, torch.zeros(3, 8), torch.tensor([[12], [8]]), {"audio": []})
    assert state.adapter.states["prefill"].token[:, :1].tolist() == [[12]]


def test_scheduler_length_limit_flushes_last_token():
    state, batch, conditioning, decoder = _fixture()
    state._req_states.max_seq_len[1] = 7
    _run(state, batch, torch.zeros(3, 8), torch.tensor([[0], [12]]), {"embed": conditioning})
    assert decoder.calls[-1]["finalize"] is True
    request = state.adapter.states["decode"]
    assert request.token[:, :1].tolist() == [[12]]


def test_resumed_prefill_uses_original_prompt_length_for_append_position():
    state, batch, conditioning, _ = _fixture()
    _run(state, batch, torch.zeros(3, 8), torch.tensor([[0], [12]]), {"embed": conditioning})
    batch.num_computed_tokens_np[1] = 6
    _run(state, batch, torch.zeros(3, 8), torch.tensor([[0], [13]]), {"audio": []})
    # Replay prefill includes the two accepted speech tokens, but original
    # prompt_len remains 6. The new sample belongs at speech-buffer offset 2.
    batch.is_prefilling_np[1] = True
    batch.prefill_len_np[1] = 8
    batch.num_computed_prefill_tokens_np[1] = 7
    batch.num_computed_tokens_np[1] = 7
    _run(state, batch, torch.zeros(3, 8), torch.tensor([[0], [14]]), {"audio": []})
    assert state.adapter.states["decode"].token[:, :3].tolist() == [[12, 13, 14]]


class _TaggedDecoder(_Decoder):
    def forward_streaming(self, **kwargs):
        self.calls.append(kwargs)
        final = kwargs["finalize"]
        size = kwargs["token"].shape[1] + (10 if final else 0)
        return torch.full((1, 1, size), 100.0 if final else 10.0), None


def test_eos_between_chunk_boundaries_flushes_without_intermediate_decodes():
    state, batch, conditioning, _ = _fixture()
    state.adapter = InnerStageAdapter(6561, chunk_size=25, lookahead=3)
    state.model.code2wav = _TaggedDecoder()
    for step, token in enumerate([12, 13, 6562]):
        batch.num_computed_tokens_np[1] = 5 + step
        outputs = {"embed": conditioning} if step == 0 else {"audio": []}
        _run(state, batch, torch.zeros(3, 8), torch.tensor([[0], [token]]), outputs)
        if step < 2:
            assert outputs["audio"][1].numel() == 0
            assert state.model.code2wav.calls == []
        else:
            assert outputs["audio"][1].tolist() == [100.0] * 12
        request = state.adapter.states["decode"]
        assert request.emitted == 0
        assert request.cache is None


def test_mixed_speech_and_empty_eos_keep_batch_order():
    state, batch, conditioning, _ = _fixture()
    batch.num_computed_tokens_np[0] = 2
    batch.num_computed_prefill_tokens_np[0] = 2
    outputs = {"embed": conditioning}
    _run(state, batch, torch.zeros(3, 8), torch.tensor([[7], [6562]]), outputs)
    assert outputs["audio"][0].tolist() == [1.0] * 4
    assert outputs["audio"][1].numel() == 0
