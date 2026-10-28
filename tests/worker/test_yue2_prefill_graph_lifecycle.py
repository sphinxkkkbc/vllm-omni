# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU lifecycle checks for persistent B=1 prefill KV slots."""

from unittest.mock import Mock

import pytest
import torch

from vllm_omni.model_executor.models.yue2.cuda_graph_wrapper import PrefillGraphWrapper, query_bucket
from vllm_omni.model_executor.models.yue2.yue2 import ARKVSnapshot, _SynthesisJob

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def wrapper():
    model = Mock(max_position_embeddings=128)
    model.vae2llm = torch.nn.Linear(4, 4)
    model.model.layers = [object(), object()]
    model.config.num_key_value_heads = 1
    model.config.head_dim = 4
    return PrefillGraphWrapper(model, fa_version=2, max_slots=1)


def test_slot_views_share_storage_and_reuse_across_lengths(wrapper):
    first = wrapper.acquire(17)
    assert first is not None
    keys, values = first.views(17)
    assert keys.shape == (2, 17, 1, 4)
    paged = first.slot.keys[0].view(-1, 16, 1, 4)
    assert paged.data_ptr() == keys[0].data_ptr()
    keys.fill_(3)
    assert (paged.flatten(0, 1)[:17] == 3).all()
    assert wrapper.acquire(17) is None  # Still owned by the first request.
    pointer = first.slot.keys.data_ptr()
    first.release()
    second = wrapper.acquire(29)
    assert second is not None
    assert second.slot.keys.data_ptr() == pointer
    assert second.views(29)[0].shape[1] == 29
    with pytest.raises(ValueError, match="lease"):
        first.views(17)
    second.release()
    larger = wrapper.acquire(33)
    assert larger is not None and larger.slot.capacity == 64
    assert len(wrapper.slots) == 1


def test_cancel_before_generator_start_defers_slot_reuse(wrapper):
    lease = wrapper.acquire(17)
    assert lease is not None
    keys, values = lease.views(17)
    ready = Mock(spec=torch.Event)
    ready.query.return_value = False
    job = _SynthesisJob("queued", lambda keep: iter(()))
    job.keep.append(ARKVSnapshot(keys, values, ready, lease))
    job.cancel()
    assert lease.released
    assert wrapper.acquire(17) is None  # AR-stream D2D is still in flight.
    ready.query.return_value = True
    next_lease = wrapper.acquire(17)
    assert next_lease is not None and next_lease.slot is lease.slot
    job.release()  # Idempotent cleanup cannot release the next owner's slot.
    assert next_lease.slot.leased


@pytest.mark.parametrize("length,expected", [(1, 1), (9, 16), (13, 16), (65, 96), (96, 96)])
def test_query_bucket(length, expected):
    assert query_bucket(length, 96) == expected


@pytest.mark.parametrize("length", [0, 97])
def test_query_bucket_rejects_invalid_length(length):
    with pytest.raises(ValueError):
        query_bucket(length, 96)
