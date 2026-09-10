# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.auk.auk_flow import (
    FLASH_TIMES,
    AukConditionOutput,
    AukDenoiseGraphs,
    AukInputBatch,
    AukPaddedBackbone,
    AukStepRequestState,
    roundup,
)
from vllm_omni.model_executor.models.auk.modeling_auk import AukConditionEncoder, AukLayerFusion
from vllm_omni.model_executor.models.auk.prompt_utils import target_frames

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def condition(s=5, r=3, t=17, seed=0):
    g = torch.Generator().manual_seed(seed)
    return AukConditionOutput(
        torch.randn(1, s, 16, generator=g),
        torch.ones(1, s, dtype=torch.bool),
        torch.randn(1, r, 64, generator=g),
        torch.ones(1, r, dtype=torch.bool),
        t,
        seed,
    )


def state(rid="a", **kwargs):
    return AukStepRequestState.create(rid, condition(**kwargs), "cpu")


@pytest.mark.parametrize("seconds,expected", [(0.02, 1), (2.31, 116), (2.56, 128)])
def test_duration(seconds, expected):
    assert target_frames(seconds) == expected
    assert roundup(expected, 32) % 32 == 0


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_bad_duration(seconds):
    with pytest.raises(ValueError):
        target_frames(seconds)


def test_payload_ownership():
    c = condition()
    payload = c.to_payload()
    restored = AukConditionOutput.from_payload(payload)
    torch.testing.assert_close(c.semantic_embeddings, restored.semantic_embeddings)
    payload["reference_latents"].zero_()
    assert c.reference_latents.count_nonzero() > 0


def test_fusion_excludes_embedding():
    fusion = AukLayerFusion(3)
    fusion.layer_weights.data.copy_(torch.tensor([-2.0, 0.1, 1.2]))
    fusion.layer_scale.data.fill_(2.4)
    hs = [torch.randn(1, 6, 16) for _ in range(4)]
    expected = (
        sum(w * torch.nn.functional.layer_norm(h, [16]) for w, h in zip(fusion.layer_weights.softmax(0), hs[1:])) * 2.4
    )
    torch.testing.assert_close(fusion(hs), expected)
    hs[0].fill_(999)
    torch.testing.assert_close(fusion(hs), expected)


class Thinker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, **kwargs):
        assert kwargs["use_cache"] is False and kwargs["output_hidden_states"] is True
        assert "past_key_values" not in kwargs
        self.calls += 1
        h = kwargs["input_ids"].float().unsqueeze(-1) + torch.arange(16).float()
        return SimpleNamespace(hidden_states=(h, h + 1, h * 2))


class VAE(torch.nn.Module):
    def encoding_and_normalization(self, x, sample_lengths):
        assert x.shape == (1, 1, 1000) and sample_lengths.tolist() == [960]
        return torch.full((1, 3, 64), 0.3), torch.tensor([2])


def test_one_shot_semantic_and_reference_encode():
    thinker = Thinker()
    encoder = AukConditionEncoder(thinker, None, VAE(), AukLayerFusion(2))
    c = encoder.encode_inputs(
        {"input_ids": torch.ones(1, 5), "attention_mask": torch.tensor([[1, 1, 1, 0, 0]])}, torch.ones(1, 1000), 17, 0
    )
    assert thinker.calls == 1
    assert c.semantic_embeddings.shape == (1, 3, 16)
    torch.testing.assert_close(c.reference_latents, torch.full((1, 2, 64), 0.3))


def test_different_step_batch_and_seed_isolation():
    global_rng = torch.random.get_rng_state().clone()
    a, b = state(), state("b", s=7, r=0, t=21)
    a.advance(torch.ones_like(a.x_t))
    batch = AukInputBatch.materialize([a, b], 8, 4)
    assert batch.inputs["x"].shape == (4, 24, 64)
    assert batch.inputs["ref"].shape == (4, 8, 64)
    assert batch.inputs["time"][:2].tolist() == [FLASH_TIMES[1], 0]
    assert a.x_t.shape == (1, 17, 64)
    torch.testing.assert_close(global_rng, torch.random.get_rng_state())


def test_graph_descriptor_independent_of_step_and_graph_miss():
    a = state()
    calls = []

    def denoise(**inputs):
        calls.append(inputs["time"].clone())
        return torch.ones_like(inputs["x"])

    graphs = AukDenoiseGraphs(denoise)
    keys = []
    for _ in range(4):
        batch = AukInputBatch.materialize([a])
        keys.append(graphs.key(batch))
        a.advance(graphs(batch))
    assert len(set(keys)) == 1 and len(calls) == 4 and a.finished
    with pytest.raises(RuntimeError):
        a.advance(torch.ones_like(a.x_t))


@pytest.fixture
def official_backbone():
    module = pytest.importorskip("auk.model.flux2_edit")
    torch.manual_seed(52)
    m = module.Flux2Edit(
        dim=32,
        heads=2,
        dim_head=16,
        latent_dim=64,
        text_hidden_dim=16,
        num_layers=1,
        num_single_layers=1,
        ff_mult=2,
        dropout=0,
        attn_mask_enabled=True,
    ).eval()
    # Official constructor zero-initializes the output and adaptive gates.
    # Nonzero values are essential: zero-output tests cannot detect padding bugs.
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "linear" in name or "proj_out" in name:
                p.uniform_(-0.15, 0.15)
    return m


@pytest.mark.parametrize("lengths", [((5, 3, 17), (7, 6, 21)), ((3, 0, 1), (6, 4, 19))])
def test_exact_vs_roundup_with_official_blocks(official_backbone, lengths):
    states = [state(str(i), s=s, r=r, t=t, seed=i) for i, (s, r, t) in enumerate(lengths)]
    model = AukPaddedBackbone(official_backbone)
    batch = AukInputBatch.materialize(states, 8, 4)
    with torch.no_grad():
        padded = model(**batch.inputs)
        for i, st in enumerate(states):
            exact = AukInputBatch.materialize([st], 1)
            expected = official_backbone(**exact.inputs)
            assert expected.abs().max() > 0.01
            torch.testing.assert_close(padded[i : i + 1, : st.condition.target_length], expected, atol=2e-6, rtol=2e-5)


def test_four_step_official_loop_parity(official_backbone):
    odeint = pytest.importorskip("torchdiffeq").odeint
    st = state()
    exact = AukInputBatch.materialize([st], 1)
    inputs = exact.inputs.copy()

    def fn(t, x):
        return official_backbone(**{**inputs, "time": t.expand(1), "x": x})

    with torch.no_grad():
        expected = odeint(fn, st.x_t.clone(), torch.tensor(FLASH_TIMES), method="euler")[-1]
        model = AukPaddedBackbone(official_backbone)
        for _ in range(4):
            st.advance(model(**AukInputBatch.materialize([st], 8).inputs))
    torch.testing.assert_close(st.x_t, expected, atol=3e-6, rtol=3e-5)


def test_online_offline_prompt_transport():
    from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information
    from vllm_omni.model_executor.models.auk.prompt_utils import build_auk_prompt

    prompt = build_auk_prompt("Hello", 2.31, seed=7, reference_audio=(torch.ones(1, 960), 24000))
    assert prompt["prompt_token_ids"] == [0]
    restored = deserialize_additional_information(serialize_additional_information(prompt["additional_information"]))
    assert int(restored["seed"].item()) == 7
    torch.testing.assert_close(restored["reference_waveform"], torch.ones(1, 960))


@pytest.mark.asyncio
async def test_online_adapter_matches_offline():
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.entrypoints.openai.tts_adapters.auk import AukAdapter
    from vllm_omni.model_executor.models.auk.prompt_utils import build_auk_prompt

    adapter = AukAdapter.__new__(AukAdapter)
    adapter.ctx = SimpleNamespace(server=SimpleNamespace())
    req = OpenAICreateSpeechRequest(input="Hello", duration_seconds=2.31, voice="default", seed=7)
    assert adapter.validate(req) is None
    prepared = await adapter.build(req, [], False)
    expected = build_auk_prompt("Hello", 2.31, seed=7)
    assert prepared.prompt["additional_information"]["text"] == expected["additional_information"]["text"]
    torch.testing.assert_close(
        prepared.prompt["additional_information"]["seed"], expected["additional_information"]["seed"]
    )
    assert adapter.validate(req.model_copy(update={"duration_seconds": None})) is not None
    assert adapter.validate(req.model_copy(update={"speed": 2})) is not None


def test_flow_admission_different_steps_and_retirement():
    from vllm_omni.model_executor.models.auk.auk_flow import AukFlowModel

    m = AukFlowModel.__new__(AukFlowModel)
    torch.nn.Module.__init__(m)
    m.states = {}
    m.config = SimpleNamespace(bucket_granularity=8)
    times = []

    def denoise(batch):
        times.append(batch.inputs["time"].tolist())
        return torch.ones_like(batch.inputs["x"])

    m.denoise_step = denoise
    m.post_decode = lambda state: state.x_t.reshape(-1)
    a, b = condition(), condition(s=7, r=0, t=21)
    m.forward(torch.tensor([0]), model_intermediate_buffer=[a.to_payload()], request_ids=["a"])
    out = m.forward(
        torch.tensor([0, 0]), model_intermediate_buffer=[a.to_payload(), b.to_payload()], request_ids=["a", "b"]
    )
    assert times[1] == [FLASH_TIMES[1], FLASH_TIMES[0]]
    assert all(not bool(x.item()) for x in out.multimodal_outputs["auk_step_finished"])
    for _ in range(2):
        out = m.forward(torch.tensor([0, 0]), model_intermediate_buffer=[{}, {}], request_ids=["a", "b"])
    assert out.multimodal_outputs["audio"][0].numel() > 0
    assert out.multimodal_outputs["audio"][1].numel() == 0
    m.on_requests_finished({"a"})
    assert set(m.states) == {"b"}
    m.forward(torch.tensor([0]), model_intermediate_buffer=[{}], request_ids=["b"])
    m.on_requests_finished({"b"})
    assert not m.states


def test_model_scheduler_rearms_without_tokens(monkeypatch):
    from vllm_omni.model_executor.models.auk.scheduler import AukGenerationScheduler, OmniGenerationScheduler

    scheduler = AukGenerationScheduler.__new__(AukGenerationScheduler)
    request = SimpleNamespace(num_computed_tokens=1, is_finished=lambda: False)
    scheduler.requests = {"a": request}
    output = SimpleNamespace(
        multimodal_outputs=[{"auk_step_finished": torch.tensor([False]), "audio": torch.empty(0)}],
        req_id_to_index={"a": 0},
    )
    called = []
    monkeypatch.setattr(
        OmniGenerationScheduler, "update_from_output", lambda self, scheduled, result: called.append(result)
    )
    scheduler.update_from_output(SimpleNamespace(num_scheduled_tokens={"a": 1}), output)
    assert request.num_computed_tokens == 0 and output.multimodal_outputs == [None] and len(called) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires NVIDIA CUDA graph runtime")
def test_cuda_graph_all_flash_timesteps(official_backbone):
    from vllm.config import VllmConfig

    st = AukStepRequestState.create("a", condition(), "cuda")
    model = AukPaddedBackbone(official_backbone.cuda())
    graphs = AukDenoiseGraphs(model, VllmConfig())
    batch = AukInputBatch.materialize([st], 8)
    graphs.capture(batch)
    key = graphs.key(batch)
    for _ in range(4):
        batch = AukInputBatch.materialize([st], 8)
        assert graphs.key(batch) == key
        torch.testing.assert_close(graphs(batch), model(**batch.inputs), atol=2e-5, rtol=2e-4)
        st.advance(graphs(batch))
