# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AuK model implementation for vLLM-Omni."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Qwen2_5OmniProcessor
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.transformers_utils.processor import (
    cached_get_processor_without_dynamic_kwargs,
)

from vllm_omni.model_executor.models.auk.modules.vae import BigVGANFlowVAE, BigVGANFlowVAEConfig
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerDummyInputsBuilder,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerMultiModalDataParser,
    Qwen2_5OmniThinkerMultiModalProcessor,
    Qwen2_5OmniThinkerProcessingInfo,
)

logger = init_logger(__name__)


class AukLayerFusion(nn.Module):
    """AuK layer fusion for combining Qwen thinker hidden states."""

    def __init__(self, num_layers):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        self.layer_scale = nn.Parameter(torch.ones(1))

    def forward(self, hidden_states):
        if len(hidden_states) != self.layer_weights.numel() + 1:
            raise ValueError("Thinker must expose all layer outputs, including final normalized output")

        # Match CFMEdit.encode_text exactly
        stacked = torch.stack([F.layer_norm(h, [h.shape[-1]]) for h in hidden_states[1:]])
        weights = self.layer_weights.softmax(dim=0)
        weight_shape = (weights.shape[0],) + (1,) * (stacked.ndim - 1)
        return (stacked * weights.view(weight_shape)).sum(dim=0) * self.layer_scale


class AukConditionEncoder(nn.Module):
    """Encoder for AuK conditions using vLLM-Omni's Qwen2.5-Omni thinker."""

    def __init__(self, thinker, fusion):
        super().__init__()
        self.thinker = thinker.eval().requires_grad_(False)
        self.fusion = fusion
        self.vae = None

    @torch.inference_mode()
    def encode_hidden_states(self, hidden_states, reference_waveform, reference_sample_rate, target_length, seed):
        """Build AuK conditions from the already-tokenized vLLM Thinker pass."""
        device = next(self.thinker.parameters()).device
        semantic = self.fusion(hidden_states)
        if semantic.ndim == 2:
            semantic = semantic.unsqueeze(0)
        if semantic.shape[0] != 1:
            raise ValueError("AuK Stage0 currently processes one full request at a time")
        semantic = semantic.float()

        # Process reference audio
        ref = torch.zeros(1, 0, 64, device=device)
        if reference_waveform is not None:
            if self.vae is None:
                raise RuntimeError("AuK VAE must be initialized and loaded before encoding reference audio")
            vae = self.vae
            vae_dtype = next(vae.parameters()).dtype
            reference_waveform = torch.as_tensor(reference_waveform, device=device, dtype=vae_dtype)
            reference_sample_rate = int(torch.as_tensor(reference_sample_rate).item())
            if reference_sample_rate != 24000:
                reference_waveform = torchaudio.functional.resample(reference_waveform, reference_sample_rate, 24000)
            frames = reference_waveform.shape[-1] // 480
            if frames < 1:
                raise ValueError("Reference audio must contain at least 480 samples at 24 kHz")
            samples = torch.tensor([frames * 480], device=device)
            ref, lengths = vae.encoding_and_normalization(reference_waveform.reshape(1, 1, -1), sample_lengths=samples)
            ref = ref[:, : min(frames, int(lengths[0]))].float()

        return AukConditionOutput(
            semantic,
            torch.ones(semantic.shape[:2], device=device, dtype=torch.bool),
            ref,
            torch.ones(ref.shape[:2], device=device, dtype=torch.bool),
            target_length,
            seed,
        )

    @staticmethod
    def target_frames(seconds: float) -> int:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("duration must be finite and positive")
        return math.ceil(seconds * 50)


# Reuse Qwen2.5-Omni multimodal parsing
class AukMultiModalDataParser(Qwen2_5OmniThinkerMultiModalDataParser):
    """Multimodal data parser for AuK using Qwen2.5-Omni backend."""

    def get_field_config(self) -> MultiModalFieldConfig:
        # Same as Qwen2.5-Omni
        return super().get_field_config()


class AukMultiModalProcessor(Qwen2_5OmniThinkerMultiModalProcessor):
    """Multimodal processor for AuK using Qwen2.5-Omni backend."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # AuK uses the same processor as Qwen2.5-Omni

    def get_processing_info(self):
        return super().get_processing_info()


class AukDummyInputsBuilder(Qwen2_5OmniThinkerDummyInputsBuilder):
    """Build only the audio dummy used by AuK Stage0."""

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        feature_extractor = self.info.get_feature_extractor()
        audio_options = mm_options.get("audio")
        # Qwen's generic dummy builder uses up to 30 seconds of audio. AuK
        # only needs to warm the audio tower/MM contract, so one second is
        # sufficient and avoids repeating a very expensive long resample.
        target_audio_length = min(feature_extractor.chunk_length, 1) * feature_extractor.sampling_rate
        return {
            "audio": self._get_dummy_audios(
                length=target_audio_length,
                num_audios=mm_counts.get("audio", 0),
                overrides=audio_options,
            )
        }


class AukProcessingInfo(Qwen2_5OmniThinkerProcessingInfo):
    """Resolve Qwen preprocessing assets through AukConfig.qwen_path."""

    @property
    def model_id(self) -> str:
        return self.ctx.get_hf_config().qwen_path

    def get_hf_config(self):
        return AutoConfig.from_pretrained(self.model_id, trust_remote_code=True).thinker_config

    def get_hf_processor(self, **kwargs):
        model_config = self.ctx.model_config
        use_fast = kwargs.pop("use_fast", True)
        merged_kwargs = self.ctx.get_merged_mm_kwargs(kwargs)
        merged_kwargs.pop("tokenizer", None)
        return cached_get_processor_without_dynamic_kwargs(
            self.model_id,
            revision=model_config.revision,
            trust_remote_code=model_config.trust_remote_code,
            processor_cls=Qwen2_5OmniProcessor,
            tokenizer=self.ctx.tokenizer,
            use_fast=use_fast,
            **merged_kwargs,
        )

    def get_supported_mm_limits(self):
        return {"audio": 1}


@dataclass(frozen=True)
class AukConditionOutput:
    """Stage 0 output for AuK condition encoding."""

    semantic_embeddings: torch.Tensor  # [1,S,H], unpadded
    semantic_mask: torch.Tensor  # [1,S]
    reference_latents: torch.Tensor  # [1,R,64], unpadded
    reference_mask: torch.Tensor  # [1,R]
    target_length: int
    seed: int

    def to_payload(self) -> dict[str, torch.Tensor]:
        # Convert to CPU for async transfer
        return {
            "semantic_embeddings": self.semantic_embeddings.detach().cpu().clone(),
            "semantic_mask": self.semantic_mask.detach().cpu().clone(),
            "reference_latents": self.reference_latents.detach().cpu().clone(),
            "reference_mask": self.reference_mask.detach().cpu().clone(),
            "target_length": torch.tensor([self.target_length], dtype=torch.long),
            "seed": torch.tensor([self.seed], dtype=torch.long),
        }

    @classmethod
    def from_payload(cls, payload):
        return cls(
            semantic_embeddings=payload["semantic_embeddings"],
            semantic_mask=payload["semantic_mask"],
            reference_latents=payload["reference_latents"],
            reference_mask=payload["reference_mask"],
            target_length=int(torch.as_tensor(payload["target_length"]).item()),
            seed=int(torch.as_tensor(payload["seed"]).item()),
        )


@MULTIMODAL_REGISTRY.register_processor(
    AukMultiModalProcessor,
    info=AukProcessingInfo,
    dummy_inputs=AukDummyInputsBuilder,
)
class AukConditionModel(
    nn.Module,
    SupportsMultiModal,
):
    """Stage 0: Qwen2.5-Omni Thinker for condition encoding (LLM_AR)."""

    have_multimodal_outputs = True
    enable_update_additional_information = True
    omni_pooler_payload_include_hidden = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        logger.info(f"Initializing AukConditionModel with config: {self.vllm_config}")
        qwen_config = AutoConfig.from_pretrained(
            self.config.qwen_path,
            trust_remote_code=True,
        )

        thinker_config = qwen_config.thinker_config

        thinker_vllm_config = vllm_config.with_hf_config(thinker_config)

        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration(
            vllm_config=thinker_vllm_config,
            prefix=f"{prefix}.thinker",
        )

        # Remove vision tower (AuK doesn't need image input)
        self.thinker.visual = None

        # Initialize AuK encoder
        self.encoder = AukConditionEncoder(
            self.thinker,
            AukLayerFusion(self.thinker.config.text_config.num_hidden_layers),
        )
        self.encoder.vae = self._load_vae()
        self._layer_hidden_states: list[torch.Tensor] = []
        # Stage0 is configured with max_num_seqs=1. The AR lifecycle may run a
        # final decode step after prefill; retain the condition so the finished
        # output forwarded to Stage1 still contains a complete payload.
        self._pending_condition_payload: dict[str, torch.Tensor] | None = None
        # HF output_hidden_states records layer inputs after the embedding and
        # the final normalized state. Mirror that contract without tokenizing
        # again or changing the inner vLLM Qwen implementation.
        layers = self.thinker.get_language_model().model.layers
        for layer in layers:
            layer.register_forward_pre_hook(self._capture_layer_input)
        self.thinker.get_language_model().model.norm.register_forward_hook(self._capture_norm_output)

    def _capture_layer_input(self, module, args):
        hidden_states = args[1]
        residual = args[2]
        self._layer_hidden_states.append(hidden_states + residual if residual is not None else hidden_states)

    def _capture_norm_output(self, module, args, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        self._layer_hidden_states.append(hidden_states)

    def get_language_model(self):
        return self.thinker.get_language_model()

    def get_multimodal_embeddings(self, **kwargs):
        return self.embed_multimodal(**kwargs)

    def embed_multimodal(self, **kwargs):
        """Delegate the runner's actual multimodal encoder contract."""
        return self.thinker.embed_multimodal(**kwargs)

    def get_mm_mapping(self):
        return self.thinker.get_mm_mapping()

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        return self.thinker.embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def compute_logits(self, hidden_states):
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        return self.thinker.compute_logits(hidden_states)

    def _load_vae(self):
        return BigVGANFlowVAE(BigVGANFlowVAEConfig.from_dict(self.config.vae_config))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Load AuK layer fusion weights from the weights iterator
        weight_dict = dict(weights)

        # Load fusion weights
        fusion_keys = ["layer_weights", "layer_scale"]
        fusion_state_dict = {}
        for key in fusion_keys:
            if key in weight_dict:
                fusion_state_dict[key] = weight_dict[key]

        if fusion_state_dict:
            self.encoder.fusion.load_state_dict(fusion_state_dict)

        return set(dict(self.named_parameters()))

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None = None,
        runtime_additional_information: list | None = None,
        **kwargs,
    ) -> OmniOutput:
        """Forward pass for LLM_AR stage."""
        self._layer_hidden_states.clear()
        hidden = self.thinker(input_ids=input_ids, positions=positions, **kwargs)
        if not runtime_additional_information:
            # vLLM profiling commonly supplies inputs_embeds with input_ids=None.
            # Run the real Thinker so activation/KV memory is profiled correctly,
            # but do not build a request-owned AuK condition payload.
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

        if len(runtime_additional_information) != 1:
            raise ValueError("AuK Stage0 max_num_seqs must remain 1 for full-request condition encoding")
        info = runtime_additional_information[0]
        # The AR runner may invoke subsequent decode steps with an empty or
        # partial runtime-info entry after the prefill payload was consumed.
        # AuK condition encoding is a prefill-only operation; do not attempt
        # to reconstruct duration/seed from those follow-up steps.
        if "duration_seconds" not in info or "seed" not in info:
            payload = self._pending_condition_payload
            return OmniOutput(
                text_hidden_states=hidden,
                multimodal_outputs=({key: [value] for key, value in payload.items()} if payload is not None else {}),
            )
        expected_hidden_states = self.encoder.fusion.layer_weights.numel() + 1
        if len(self._layer_hidden_states) != expected_hidden_states:
            raise RuntimeError(
                "Failed to capture all Qwen hidden states for AuK layer "
                f"fusion: expected {expected_hidden_states}, got "
                f"{len(self._layer_hidden_states)}"
            )
        duration = float(torch.as_tensor(info["duration_seconds"]).item())
        seed = int(torch.as_tensor(info["seed"]).item())
        condition = self.encoder.encode_hidden_states(
            self._layer_hidden_states,
            info.get("reference_waveform"),
            info.get("reference_sample_rate", 24000),
            self.encoder.target_frames(duration),
            seed,
        ).to_payload()
        self._pending_condition_payload = condition

        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={key: [value] for key, value in condition.items()},
        )
