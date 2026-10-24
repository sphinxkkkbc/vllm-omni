# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The two pipeline boundaries share a tensor-only decoder contract."""

import pytest
import torch

from vllm_omni.model_executor.models.cosyvoice3.pipeline_adapter import SingleStageAdapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _conditioning(prompt_len=3):
    return {
        "speech_token": torch.ones((1, prompt_len), dtype=torch.long),
        "speech_feat": torch.ones((1, prompt_len * 2, 80)),
        "embedding": torch.ones((1, 192)),
    }


def test_chunk_windows_match_connector_hop_padding_and_lookahead():
    adapter = SingleStageAdapter(6561, chunk_size=4, lookahead=3)
    key = ("a", 1)
    adapter.initialize(key, _conditioning(), torch.arange(40).reshape(1, -1))
    items = []
    for length in range(1, 41):
        item = adapter.prepare(key, length, final=False)
        if item is not None:
            # Input storage may be reused for subsequent appends; take a test snapshot.
            items.append((item["token"].tolist(), item["token_offset_tokens"], item["finalize"]))
    final = adapter.prepare(key, 40, final=True)
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
    adapter = SingleStageAdapter(6561, chunk_size=1, lookahead=0)
    a_key, b_key = ("a", 1), ("b", 2)
    adapter.initialize(a_key, _conditioning(), torch.tensor([[7]]))
    adapter.initialize(b_key, _conditioning(), torch.tensor([[9]]))
    a = adapter.prepare(a_key, 1, final=False)
    b = adapter.prepare(b_key, 1, final=False)
    cache = {"mel": torch.ones(1, 80, 2)}
    adapter.apply_result(a_key, cache)
    assert adapter.states[a_key].cache is cache
    assert adapter.states[b_key].cache is None
    assert a["token"].tolist() == [[7]]
    assert b["token"].tolist() == [[9]]
    adapter.finish_requests({a_key})
    assert set(adapter.states) == {b_key}
    reused = ("a", 3)
    adapter.initialize(reused, _conditioning(), torch.tensor([[11]]))
    a = adapter.prepare(reused, 1, final=False)
    assert a["token"].tolist() == [[11]]
    assert a["cache_state"] is None


def test_prompt_padding_is_removed_before_window_alignment():
    adapter = SingleStageAdapter(6561, chunk_size=4, lookahead=0)
    conditioning = _conditioning(8)
    conditioning["speech_token_len"] = torch.tensor([[3]])
    key = ("a", 1)
    adapter.initialize(key, conditioning, torch.arange(5).reshape(1, -1))
    item = adapter.prepare(key, 5, final=False)
    assert item is not None
    assert item["prompt_token"].shape == (1, 3)
    assert item["prompt_feat"].shape == (1, 6, 80)
    assert item["token"].shape == (1, 5)
