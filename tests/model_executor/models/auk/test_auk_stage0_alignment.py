# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from transformers import Qwen2_5OmniProcessor, WhisperFeatureExtractor
from vllm.model_executor.models.interfaces_base import is_pooling_model, is_text_generation_model
from vllm.transformers_utils import processor as processor_utils

from vllm_omni.model_executor.models.auk.auk_flow import AukFlowModel
from vllm_omni.model_executor.models.auk.configuration_auk import AukConfig
from vllm_omni.model_executor.models.auk.modeling_auk import (
    AukConditionEncoder,
    AukConditionModel,
    AukConditionOutput,
    AukDummyInputsBuilder,
    AukLayerFusion,
    AukProcessingInfo,
)
from vllm_omni.model_executor.models.auk.prompt_utils import build_auk_prompt
from vllm_omni.model_executor.stage_input_processors.auk import (
    build_stage1_inputs,
    serialize_condition_payload,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_condition_model_is_generation_not_pooling():
    assert is_text_generation_model(AukConditionModel)
    assert not is_pooling_model(AukConditionModel)


def test_flow_model_is_generation_not_pooling():
    assert is_text_generation_model(AukFlowModel)
    assert not is_pooling_model(AukFlowModel)
    assert AukFlowModel.have_multimodal_outputs is True
    assert AukFlowModel.requires_raw_input_tokens is True


def test_stage0_uses_qwen_tokenizer_without_ineffective_runner_override():
    deploy = yaml.safe_load((Path(__file__).parents[4] / "vllm_omni" / "deploy" / "auk.yaml").read_text())
    stage0 = deploy["stages"][0]
    assert "runner" not in stage0
    assert stage0["tokenizer"] == "Qwen/Qwen2.5-Omni-3B"


def test_outer_config_and_processor_source_remain_distinct():
    config = AukConfig(qwen_path="Qwen/Qwen2.5-Omni-3B")
    ctx = SimpleNamespace(get_hf_config=lambda: config)
    info = AukProcessingInfo(ctx)
    assert isinstance(config, AukConfig)
    assert info.model_id == config.qwen_path
    assert info.get_supported_mm_limits() == {"audio": 1}


def test_qwen_processor_uses_vllm_process_cache(monkeypatch):
    config = AukConfig(qwen_path="Qwen/Qwen2.5-Omni-3B")
    model_config = SimpleNamespace(revision="qwen-revision", trust_remote_code=True)
    ctx = SimpleNamespace(
        get_hf_config=lambda: config,
        get_merged_mm_kwargs=lambda kwargs: dict(kwargs),
        model_config=model_config,
        tokenizer=None,
    )
    info1 = AukProcessingInfo(ctx)
    info2 = AukProcessingInfo(ctx)
    calls = []

    def from_pretrained(model_id, **kwargs):
        calls.append((model_id, kwargs))
        processor = object.__new__(Qwen2_5OmniProcessor)
        processor.feature_extractor = object.__new__(WhisperFeatureExtractor)
        return processor

    monkeypatch.setattr(Qwen2_5OmniProcessor, "from_pretrained", from_pretrained)
    processor_utils.cached_get_processor.cache_clear()
    processor_utils.get_processor_kwargs_type.cache_clear()
    try:
        processor1 = info1.get_hf_processor()
        processor2 = info2.get_hf_processor()
        assert processor1 is processor2
        assert info1.get_feature_extractor() is info2.get_feature_extractor()
        # Upstream performs one discovery load and one final keyed load. All
        # later calls, including calls from another info instance, reuse them.
        assert len(calls) == 2

        processor3 = info1.get_hf_processor(use_fast=False)
        assert processor3 is not processor1
        assert len(calls) == 3
        assert all(call[0] == config.qwen_path for call in calls)
        assert all(call[1]["revision"] == "qwen-revision" for call in calls)
        assert all(call[1]["trust_remote_code"] is True for call in calls)
    finally:
        processor_utils.cached_get_processor.cache_clear()
        processor_utils.get_processor_kwargs_type.cache_clear()


def test_auk_dummy_builder_is_audio_only():
    assert AukDummyInputsBuilder is not None


def test_forward_has_explicit_positions_and_no_second_tokenization():
    signature = inspect.signature(AukConditionModel.forward)
    assert "positions" in signature.parameters
    source = inspect.getsource(AukConditionModel.forward)
    assert "apply_chat_template" not in source
    assert "from_pretrained" not in source


def test_model_exposes_runner_embed_multimodal_delegate():
    assert callable(AukConditionModel.embed_multimodal)
    assert callable(AukConditionModel.get_multimodal_embeddings)
    assert AukConditionModel.have_multimodal_outputs is True


def test_layer_fusion_requires_embedding_plus_all_layer_outputs():
    fusion = AukLayerFusion(3)
    hidden_states = [torch.randn(1, 2, 4) for _ in range(4)]

    assert fusion(hidden_states).shape == (1, 2, 4)
    flat_hidden_states = [hidden_state.squeeze(0) for hidden_state in hidden_states]
    assert fusion(flat_hidden_states).shape == (2, 4)
    with pytest.raises(ValueError, match="all layer outputs"):
        fusion(hidden_states[1:])

    init_source = inspect.getsource(AukConditionModel.__init__)
    assert "for layer in layers:" in init_source
    assert "layers[1:]" not in init_source


def test_hidden_state_hooks_follow_vllm_qwen_layer_contract():
    model = AukConditionModel.__new__(AukConditionModel)
    torch.nn.Module.__init__(model)
    model._layer_hidden_states = []
    positions = torch.arange(2)
    hidden_states = torch.randn(2, 4)
    residual = torch.randn(2, 4)

    model._capture_layer_input(None, (positions, hidden_states, None))
    model._capture_layer_input(None, (positions, hidden_states, residual))
    model._capture_norm_output(None, (), (hidden_states, residual))

    torch.testing.assert_close(model._layer_hidden_states[0], hidden_states)
    torch.testing.assert_close(model._layer_hidden_states[1], hidden_states + residual)
    torch.testing.assert_close(model._layer_hidden_states[2], hidden_states)
    assert all(value.is_floating_point() for value in model._layer_hidden_states)


def test_condition_encoder_accepts_serialized_reference_waveform():
    class Fusion(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states[0]

    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

        def encoding_and_normalization(self, waveform, sample_lengths):
            assert waveform.shape == (1, 1, 960)
            assert waveform.dtype == torch.bfloat16
            assert sample_lengths.tolist() == [960]
            return torch.ones(1, 2, 64), torch.tensor([2])

    thinker = torch.nn.Linear(1, 1, bias=False)
    encoder = AukConditionEncoder(thinker, Fusion())
    encoder.vae = VAE()
    condition = encoder.encode_hidden_states(
        [torch.ones(2, 4)],
        [0.0] * 960,
        [24000],
        target_length=5,
        seed=7,
    )

    assert condition.semantic_embeddings.shape == (1, 2, 4)
    assert condition.reference_latents.shape == (1, 2, 64)


def test_reference_audio_has_qwen_and_vae_paths():
    waveform = torch.randn(1, 960).tolist()
    prompt = build_auk_prompt("hello", 2.0, reference_audio=(waveform, 24000), seed=9)
    qwen_waveform, qwen_sr = prompt["multi_modal_data"]["audio"]
    metadata = prompt["model_intermediate_buffer"]
    assert isinstance(qwen_waveform, torch.Tensor) and qwen_waveform.shape == (960,)
    assert qwen_sr == 24000
    assert isinstance(metadata["reference_waveform"], list)
    torch.testing.assert_close(torch.tensor(metadata["reference_waveform"]), qwen_waveform)
    assert metadata["reference_sample_rate"] == 24000
    assert metadata["duration_seconds"] == 2.0 and metadata["seed"] == 9
    assert prompt["prompt"].count("<|AUDIO|>") == 1


def test_profiling_path_without_runtime_metadata():
    class Thinker(torch.nn.Module):
        def forward(self, input_ids, positions, inputs_embeds=None, **kwargs):
            assert input_ids is None
            return inputs_embeds

    model = AukConditionModel.__new__(AukConditionModel)
    torch.nn.Module.__init__(model)
    model.thinker = Thinker()
    model._layer_hidden_states = []
    inputs_embeds = torch.randn(2, 8)
    output = model.forward(None, positions=torch.tensor([0, 1]), inputs_embeds=inputs_embeds)
    assert output.text_hidden_states.shape == (2, 8)
    torch.testing.assert_close(output.text_hidden_states, inputs_embeds)


def test_followup_runtime_info_without_condition_metadata_is_safe():
    class Thinker(torch.nn.Module):
        def forward(self, input_ids, positions, inputs_embeds=None, **kwargs):
            return inputs_embeds

    model = AukConditionModel.__new__(AukConditionModel)
    torch.nn.Module.__init__(model)
    model.thinker = Thinker()
    model._layer_hidden_states = []
    model._pending_condition_payload = None
    output = model.forward(
        None,
        positions=torch.tensor([0]),
        runtime_additional_information=[{}],
        inputs_embeds=torch.randn(1, 8),
    )
    assert output.multimodal_outputs == {}


def test_followup_decode_reuses_prefill_condition_payload():
    class Thinker(torch.nn.Module):
        def forward(self, input_ids, positions, inputs_embeds=None, **kwargs):
            return inputs_embeds

    model = AukConditionModel.__new__(AukConditionModel)
    torch.nn.Module.__init__(model)
    model.thinker = Thinker()
    model._layer_hidden_states = []
    model._pending_condition_payload = {"semantic_embeddings": torch.ones(1, 2, 3)}
    output = model.forward(
        None,
        positions=torch.tensor([0]),
        runtime_additional_information=[{}],
        inputs_embeds=torch.randn(1, 8),
    )
    assert output.multimodal_outputs["semantic_embeddings"][0].shape == (1, 2, 3)


def test_connector_skips_empty_finalization_payload():
    with pytest.raises(RuntimeError, match="missing condition fields"):
        serialize_condition_payload(None, {}, None)


def test_connector_unwraps_runner_singleton_payload():
    payload = {
        "semantic_embeddings": [torch.ones(1, 2, 4)],
        "semantic_mask": [torch.ones(1, 2, dtype=torch.bool)],
        "reference_latents": [torch.ones(1, 3, 64)],
        "reference_mask": [torch.ones(1, 3, dtype=torch.bool)],
        "target_length": [torch.tensor([5])],
        "seed": [torch.tensor([7])],
    }
    result = serialize_condition_payload(None, payload, None)
    assert result is not None
    assert set(payload).issubset(result)
    assert result["meta"]["finished"].item() is True
    assert result["semantic_embeddings"].shape == (1, 2, 4)
    assert result["target_length"].item() == 5


def test_stage1_consumer_builds_token_only_prompt_for_connector_payload():
    source = SimpleNamespace(finished=True)
    prompts = build_stage1_inputs([source], [None], False)
    assert len(prompts) == 1
    assert prompts[0]["prompt_token_ids"] == [0]
    assert "model_intermediate_buffer" not in prompts[0]


def test_stage1_consumer_skips_nonterminal_source_output():
    assert build_stage1_inputs([SimpleNamespace(finished=False)]) == []


def test_auk_full_payload_replace_keys_do_not_concat_snapshots():
    from vllm_omni.distributed.omni_connectors.model_runner.omni_connector_payload_transport import (
        _OmniConnectorPayloadTransportMixin,
    )
    from vllm_omni.model_executor.stage_input_processors.auk import _FULL_PAYLOAD_REPLACE_KEYS

    transport = _OmniConnectorPayloadTransportMixin()
    transport._pending_full_payload_send = {}
    transport._custom_process_func = serialize_condition_payload
    transport._full_payload_replace_keys_cached = None
    transport.accumulate_full_payload_output("r1", {"semantic_embeddings": torch.ones(2, 4)}, None)
    transport.accumulate_full_payload_output("r1", {"semantic_embeddings": torch.ones(3, 4)}, None)
    materialized, _ = transport._materialize_full_payload_entry(transport._pending_full_payload_send["r1"])
    assert _FULL_PAYLOAD_REPLACE_KEYS
    assert materialized["semantic_embeddings"].shape == (3, 4)


def test_auk_condition_is_a_single_finished_payload():
    assert not getattr(AukConditionModel, "omni_payload_at_request_end", False)
    assert AukConditionModel.enable_update_additional_information is True
    assert AukConditionModel.omni_pooler_payload_include_hidden is False
    assert AukFlowModel.enable_update_additional_information is True

    condition = AukConditionOutput(
        semantic_embeddings=torch.ones(1, 2, 4),
        semantic_mask=torch.ones(1, 2, dtype=torch.bool),
        reference_latents=torch.ones(1, 3, 64),
        reference_mask=torch.ones(1, 3, dtype=torch.bool),
        target_length=5,
        seed=7,
    ).to_payload()
    serialized = serialize_condition_payload(None, condition, None, is_finished=False)
    terminal = serialize_condition_payload(
        None,
        {"hidden": torch.ones(1, 4)},
        None,
        is_finished=True,
    )

    assert set(condition).issubset(serialized)
    assert serialized["target_length"].item() == 5
    assert serialized["seed"].item() == 7
    assert serialized["meta"]["finished"].item() is True
    assert terminal is None

    with pytest.raises(RuntimeError, match="missing condition fields"):
        serialize_condition_payload(
            None,
            {"hidden": torch.ones(1, 4)},
            None,
            is_finished=False,
        )
