# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""First-chunk KV snapshots exclude HOLD and preserve END conditioning."""

from unittest.mock import Mock

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_omni.model_executor.models.yue2 import yue2 as y
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("length", [4, 6])
@pytest.mark.parametrize("layout", ["stacked", "packed"])
def test_paged_snapshot_owns_data_and_excludes_tail(length, layout):
    runner = object.__new__(GPUARModelRunner)
    group = Mock(layer_names=["layer0", "layer1"])
    group.kv_cache_spec = FullAttentionSpec(block_size=4, num_kv_heads=2, head_size=3, dtype=torch.float32)
    runner.kv_cache_config = Mock(kv_cache_groups=[group])
    runner.cache_config = Mock(cache_dtype="auto")
    runner._kernel_block_sizes = [4]
    runner.requests = {"song": Mock(block_ids=([2, 0],))}
    contexts = {}
    expected = []
    for index, name in enumerate(group.layer_names):
        k = torch.arange(3 * 4 * 2 * 3, dtype=torch.float32).reshape(3, 4, 2, 3) + index * 100
        v = -k
        cache = torch.stack((k, v)) if layout == "stacked" else torch.cat((k, v), dim=-1).transpose(1, 2)
        contexts[name] = Mock(kv_cache=cache)
        expected.append((k[[2, 0]].flatten(0, 1)[:length], v[[2, 0]].flatten(0, 1)[:length]))
    runner.compilation_config = Mock(static_forward_context=contexts)
    keys = torch.full((2, 12, 2, 3), -999.0)
    values = torch.full_like(keys, -999.0)
    assert runner._copy_request_kv("song", group.layer_names, length, keys, values)
    for index, (k, v) in enumerate(expected):
        torch.testing.assert_close(keys[index, :length], k)
        torch.testing.assert_close(values[index, :length], v)
    assert (keys[:, length:] == -999).all()
    for context in contexts.values():
        context.kv_cache.zero_()
    torch.testing.assert_close(keys[0, :length], expected[0][0])
    runner.cache_config.cache_dtype = "fp8"
    assert not runner._copy_request_kv("song", group.layer_names, length, keys, values)


def test_end_only_prefill_matches_full_prefill(mocker):
    torch.manual_seed(7)
    layers = []
    for _ in range(3):
        layer = Mock()
        layer.input_layernorm = torch.nn.Identity()
        layer.self_attn.o_proj = torch.nn.Linear(4, 4, bias=False)
        layer.post_attention_layernorm = torch.nn.Identity()
        layer.mlp = torch.nn.Linear(4, 4, bias=False)
        layers.append(layer)
    model = Mock()
    model.model.layers = layers
    model.model.embed_tokens = torch.nn.Embedding(16, 4)
    model.config.num_key_value_heads = 1
    model.config.head_dim = 4
    model.rotary = lambda positions: (positions, positions)

    def project(layer, x, cos, sin):
        h = x.reshape(-1, 1, 4)
        return h, h * 0.7, h * 0.3

    model.ar_project = Mock(side_effect=project)

    def engine():
        obj = object.__new__(y.CachedNAR)
        obj.model = model
        obj.chunk = y.Chunk([1, 2, 3, 4, 5], torch.zeros(2, 64))
        obj.ar_length, obj.nar_length = 5, 4
        obj.dtype, obj.device, obj.fa3 = torch.float32, torch.device("cpu"), None
        return obj

    full = engine()
    full._prefill()
    keys, values = torch.empty_like(full.keys), torch.empty_like(full.values)
    keys[:, :4].copy_(full.keys[:, :4])
    values[:, :4].copy_(full.values[:, :4])
    snapshot = y.ARKVSnapshot(keys, values, Mock(spec=torch.Event))
    stream = mocker.patch.object(y.torch.cuda, "current_stream").return_value
    model.ar_project.reset_mock()
    resumed = engine()
    resumed._prefill(snapshot)
    stream.wait_event.assert_called_once_with(snapshot.ready)
    assert resumed.keys is keys and resumed.values is values
    assert all(call.args[1].shape[0] == 1 for call in model.ar_project.call_args_list)
    torch.testing.assert_close(resumed.keys[:, :5], full.keys[:, :5])
    torch.testing.assert_close(resumed.values[:, :5], full.values[:, :5])
