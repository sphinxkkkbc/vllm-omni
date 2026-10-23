# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise RAS through the upstream sampler call boundary, without GPU work."""

from functools import partial
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_ras_accepts_upstream_call_and_preserves_bias_context(monkeypatch):
    from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import _ras_mrv2_sample

    class CheckedContextError(Exception):
        pass

    bounds = torch.tensor([4], dtype=torch.int32)
    idx = torch.tensor([0], dtype=torch.int32)
    logits = torch.zeros(1, 8)
    batch = SimpleNamespace(
        expanded_idx_mapping=idx,
        idx_mapping=idx,
        idx_mapping_np=np.array([0]),
        cu_num_logits_np=np.array([0, 1]),
        expanded_local_pos=idx,
        positions=torch.tensor([3]),
        logits_indices=torch.tensor([0]),
        input_ids=torch.tensor([2]),
        seq_lens_cpu_upper_bound=bounds,
    )
    bias = object.__new__(LogitBiasState)

    def apply_bias(values, ctx):
        assert values.dtype == torch.float32
        np.testing.assert_array_equal(ctx.seq_lens_upper_bound_np, bounds.numpy())
        assert ctx.expanded_idx_mapping is idx
        assert ctx.input_ids.tolist() == [2]
        assert ctx.pos.tolist() == [3]
        raise CheckedContextError

    monkeypatch.setattr(bias, "apply", apply_bias)

    def unexpected_penalty(*args):
        pytest.fail("RAS must not apply ordinary penalties")

    sampler = SimpleNamespace(
        compute_nans=False,
        get_logprobs_dims=lambda rows: None,
        logits_processors=[SimpleNamespace(apply=unexpected_penalty), bias],
    )
    sampler.sample = MethodType(
        partial(
            _ras_mrv2_sample,
            default_top_p=0.8,
            default_top_k=25,
            win_size=10,
            tau_r=0.1,
            eps=1e-5,
        ),
        sampler,
    )
    # Execute the actual upstream positional/keyword call; stop before GPU sampling.
    with pytest.raises(CheckedContextError):
        Sampler.__call__(sampler, logits, batch)
    assert torch.equal(logits, torch.zeros_like(logits))
