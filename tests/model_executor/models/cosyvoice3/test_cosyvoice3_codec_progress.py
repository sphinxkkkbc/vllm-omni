# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Incremental codec dispatch retains batched Flow and request ordering."""

import pytest
import torch

from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import (
    CosyVoice3Code2Wav,
    iter_decode_streaming_batch,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _decoder(monkeypatch):
    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav as module

    monkeypatch.setattr(module, "cosyvoice3_packed_streaming_enabled", lambda: True)
    monkeypatch.setenv("COSYVOICE3_BATCH_FLOW", "1")
    model = object.__new__(CosyVoice3Code2Wav)
    torch.nn.Module.__init__(model)
    model.calls = []
    model.ratio = 1
    # Avoid loading weights: exercise actual dispatch, row slicing and yields.
    monkeypatch.setattr(CosyVoice3Code2Wav, "token_mel_ratio", property(lambda self: self.ratio))

    def flow(items, n_timesteps, *, stream_items):
        assert stream_items
        model.calls.append(("flow", [item["tag"] for item in items]))
        features = torch.stack([torch.full((1, 6), float(item["tag"])) for item in items])
        return features, [6] * len(items), [1] * len(items)

    def hift(feat, *, cache_state, finalize):
        tag = int(feat[0, 0, 0])
        model.calls.append(("hift", tag))
        return feat, None if finalize else {"tag": tag}

    model.flow_batch_mels = flow
    model._stream_hift_from_feat = hift
    return model


def _items(count, finals=()):
    return [
        {
            "tag": index,
            "token": torch.zeros(1, count - index, dtype=torch.long),
            "prompt_token": torch.zeros(1, 1, dtype=torch.long),
            "token_offset_tokens": 0,
            "finalize": index in finals,
        }
        for index in range(count)
    ]


def test_streaming_iterator_pauses_between_hift_and_flow_groups(monkeypatch):
    model = _decoder(monkeypatch)
    iterator = iter_decode_streaming_batch(model, _items(3, finals=(0,)))
    assert model.calls == []
    first = next(iterator)
    assert first[0] == 1
    assert model.calls == [("flow", [1, 2]), ("hift", 1)]
    second = next(iterator)
    assert second[0] == 2
    assert model.calls[-1] == ("hift", 2)
    assert len(model.calls) == 3  # Neither final Flow nor final HiFT ran ahead.
    last = next(iterator)
    assert last[0] == 0
    assert last[2] is None
    assert model.calls[-2:] == [("flow", [0]), ("hift", 0)]
    with pytest.raises(StopIteration):
        next(iterator)


@pytest.mark.parametrize("count", [1, 8, 11])
def test_list_api_restores_request_order_across_incremental_groups(monkeypatch, count):
    model = _decoder(monkeypatch)
    outputs = model.forward_streaming_batch(_items(count, finals=(0, 2)))
    assert [int(speech[0, 0, 0]) for speech, _ in outputs] == list(range(count))
    assert [cache is None for _, cache in outputs] == [i in (0, 2) for i in range(count)]
    assert all(len(tags) <= 8 for name, tags in model.calls if name == "flow")
