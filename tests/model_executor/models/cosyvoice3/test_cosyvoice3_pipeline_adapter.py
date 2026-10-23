# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The two pipeline boundaries share a tensor-only decoder contract."""

import pytest
import torch

from vllm_omni.model_executor.models.cosyvoice3.pipeline_adapter import InnerStageAdapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _conditioning(prompt_len=3):
    return {
        "speech_token": torch.ones((1, prompt_len), dtype=torch.long),
        "speech_feat": torch.ones((1, prompt_len * 2, 80)),
        "embedding": torch.ones((1, 192)),
    }


def _decode_step(
    adapter,
    req_id,
    sampled_token,
    token_id,
    conditioning,
    *,
    capacity,
    max_tokens,
    length_finished=False,
    position=0,
):
    speech = (sampled_token >= 0) & (sampled_token < adapter.vocab_size)
    adapter.append(req_id, sampled_token, conditioning, capacity=capacity, position=position, speech_mask=speech)
    is_speech = bool(speech.item())
    return adapter.prepare(req_id, position + int(is_speech), not is_speech or length_finished or max_tokens == 1)


def _prepare(adapter, req_id, value, **kwargs):
    return _decode_step(
        adapter, req_id, torch.tensor([value]), value, _conditioning(), capacity=128, max_tokens=100, **kwargs
    )


def test_inner_windows_match_connector_hop_padding_and_lookahead():
    adapter = InnerStageAdapter(6561, chunk_size=4, lookahead=3)
    items = []
    for value in range(40):
        item = _prepare(adapter, "a", value, position=value)
        if item is not None:
            # Input storage may be reused for subsequent appends; take a test snapshot.
            items.append((item["token"].tolist(), item["token_offset_tokens"], item["finalize"]))
    final = _prepare(adapter, "a", 6562, position=40)
    assert [(len(tokens[0]), offset, end) for tokens, offset, end in items] == [
        (8, 0, False),
        (16, 5, False),
        (32, 13, False),
    ]
    assert final["token"].tolist() == [list(range(40))]
    assert final["token_offset_tokens"] == 29
    assert final["finalize"] is True
    assert not any(key in final for key in ("req_id", "index", "stream_finished"))


def test_request_state_and_decoder_cache_are_isolated_and_released():
    adapter = InnerStageAdapter(6561, chunk_size=1, lookahead=0)
    a = _prepare(adapter, "a", 7)
    b = _prepare(adapter, "b", 9)
    cache = {"mel": torch.ones(1, 80, 2)}
    adapter.apply_result("a", cache)
    assert adapter.states["a"].cache is cache
    assert adapter.states["b"].cache is None
    assert a["token"].tolist() == [[7]]
    assert b["token"].tolist() == [[9]]
    adapter.finish_requests({"a"})
    assert set(adapter.states) == {"b"}
    # A reused request ID starts with fresh tokens and no vocoder state.
    a = _prepare(adapter, "a", 11)
    assert a["token"].tolist() == [[11]]
    assert a["cache_state"] is None


def test_prompt_padding_is_removed_before_window_alignment():
    adapter = InnerStageAdapter(6561, chunk_size=4, lookahead=0)
    conditioning = _conditioning(8)
    conditioning["speech_token_len"] = torch.tensor([[3]])
    for value in range(5):
        item = _decode_step(
            adapter, "a", torch.tensor([value]), value, conditioning, capacity=16, max_tokens=16, position=value
        )
    assert item is not None
    assert item["prompt_token"].shape == (1, 3)
    assert item["prompt_feat"].shape == (1, 6, 80)
    assert item["token"].shape == (1, 5)
