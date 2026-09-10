# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AuK-Flash full-request flow inference and request-level batching."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.auk.modules.flux2_edit import Flux2Edit
from vllm_omni.model_executor.models.auk.modules.vae import BigVGANFlowVAE, BigVGANFlowVAEConfig
from vllm_omni.model_executor.models.output_templates import OmniOutput

FLASH_TIMES = (0.0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1.0)

logger = init_logger(__name__)


def stage1_checkpoint_name(param_name: str) -> str:
    """Map the vLLM Stage1 namespace to the clean AuK checkpoint namespace."""
    if param_name.startswith("backbone."):
        return "transformer." + param_name.removeprefix("backbone.")
    if param_name.startswith("vae."):
        return param_name.removeprefix("vae.")
    return param_name


def _pad(x: torch.Tensor, length: int) -> torch.Tensor:
    return F.pad(x, (0, 0, 0, length - x.shape[1])) if x.shape[1] < length else x[:, :length]


class AukInputBatch:
    """Materialize independent semantic, reference, and target dimensions."""

    def __init__(self, inputs: dict[str, torch.Tensor], target_lengths: list[int]):
        self.inputs = inputs
        self.target_lengths = target_lengths

    @classmethod
    def materialize(cls, conditions, latents, device, dtype):
        s_max = max(c.semantic_embeddings.shape[1] for c in conditions)
        r_max = max(c.reference_latents.shape[1] for c in conditions)
        t_max = max(x.shape[1] for x in latents)
        b = len(conditions)
        text = torch.zeros(b, s_max, conditions[0].semantic_embeddings.shape[-1], device=device, dtype=dtype)
        semantic_mask = torch.zeros(b, s_max, dtype=torch.bool, device=device)
        latent_dim = latents[0].shape[-1]
        ref = torch.zeros(b, r_max, latent_dim, device=device, dtype=dtype)
        ref_mask = torch.zeros(b, r_max, dtype=torch.bool, device=device)
        x = torch.zeros(b, t_max, latent_dim, device=device, dtype=dtype)
        target_mask = torch.zeros(b, t_max, dtype=torch.bool, device=device)
        times = torch.empty(b, device=device, dtype=dtype)
        for i, (condition, latent) in enumerate(zip(conditions, latents)):
            s, r, t = condition.semantic_embeddings.shape[1], condition.reference_latents.shape[1], latent.shape[1]
            text[i, :s] = condition.semantic_embeddings[0].to(device=device, dtype=dtype)
            semantic_mask[i, :s] = condition.semantic_mask[0].to(device)
            if r:
                ref[i, :r] = condition.reference_latents[0].to(device=device, dtype=dtype)
                ref_mask[i, :r] = condition.reference_mask[0].to(device)
            x[i, :t] = latent[0].to(device=device, dtype=dtype)
            target_mask[i, :t] = True
            times[i] = 0
        return cls(
            {
                "x": x,
                "time": times,
                "text": text,
                "ref": ref,
                "mask": target_mask,
                "c_mask": semantic_mask,
                "ref_mask": ref_mask,
            },
            [x.shape[1] for x in latents],
        )


class AukFlowModel(nn.Module):
    """Stage 1: one forward runs all AuK-Flash steps and decodes every request."""

    have_multimodal_outputs = True
    enable_update_additional_information = True
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.backbone = Flux2Edit(
            dim=self.config.transformer_dim,
            heads=self.config.transformer_heads,
            dim_head=self.config.transformer_head_dim,
            ff_mult=self.config.transformer_ff_mult,
            latent_dim=self.config.latent_dim,
            text_hidden_dim=self.config.text_hidden_dim,
            num_layers=self.config.transformer_num_layers,
            num_single_layers=self.config.transformer_num_single_layers,
            dropout=0.0,
            attn_backend=self.config.attn_backend,
            attn_mask_enabled=self.config.attn_mask_enabled,
        )
        self.vae = BigVGANFlowVAE(BigVGANFlowVAEConfig.from_dict(self.config.vae_config))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        incoming = dict(weights)
        expected = self.state_dict()
        loaded_param_names: set[str] = set()
        consumed_checkpoint_keys: set[str] = set()
        ignored_prefixes = ("text_encoder.",)
        ignored_keys = {"layer_weights", "layer_scale"}
        for param_name, parameter in expected.items():
            ckpt_name = stage1_checkpoint_name(param_name)
            candidates = (ckpt_name, f"ema_model.{ckpt_name}")
            checkpoint_key = next((key for key in candidates if key in incoming), None)
            if checkpoint_key is None:
                continue
            match = incoming[checkpoint_key]
            if tuple(match.shape) != tuple(parameter.shape):
                raise RuntimeError(
                    f"AuK checkpoint shape mismatch for {param_name} from {checkpoint_key}: "
                    f"{tuple(match.shape)} != {tuple(parameter.shape)}"
                )
            with torch.no_grad():
                parameter.copy_(match.to(parameter.device, parameter.dtype))
            loaded_param_names.add(param_name)
            consumed_checkpoint_keys.add(checkpoint_key)
        missing = set(expected) - loaded_param_names
        owned_keys = {key for key in incoming if key not in ignored_keys and not key.startswith(ignored_prefixes)}
        unexpected = owned_keys - consumed_checkpoint_keys
        if missing or unexpected:
            raise RuntimeError(
                f"AuK checkpoint coverage failure: missing={sorted(missing)[:8]}, unexpected={sorted(unexpected)[:8]}"
            )
        return {name for name in loaded_param_names if name in dict(self.named_parameters())}

    def embed_input_ids(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return torch.zeros(
            (input_ids.shape[0], 1),
            device=input_ids.device,
            dtype=torch.float32,
        )

    def compute_logits(self, hidden_states, sampling_metadata=None) -> None:
        return None

    def _noise(self, condition, device):
        generator = torch.Generator(device=device)
        generator.manual_seed(condition.seed)
        dtype = next(self.backbone.parameters()).dtype
        return torch.randn(
            (1, condition.target_length, self.config.latent_dim), generator=generator, device=device, dtype=dtype
        )

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        runtime_additional_information: list | None = None,
        **kwargs,
    ) -> OmniOutput:
        del positions, kwargs
        if not runtime_additional_information:
            return OmniOutput(torch.zeros(input_ids.numel(), device=input_ids.device), {})
        from .modeling_auk import AukConditionOutput

        logger.info(
            "AuK Stage1 forward received: batch=%d keys=%s",
            len(runtime_additional_information),
            [sorted(info) for info in runtime_additional_information],
        )
        conditions = [AukConditionOutput.from_payload(info) for info in runtime_additional_information]
        latents = [self._noise(c, input_ids.device) for c in conditions]
        model_dtype = next(self.backbone.parameters()).dtype
        batch = AukInputBatch.materialize(conditions, latents, input_ids.device, model_dtype)
        x = batch.inputs["x"]
        try:
            for i in range(len(self.config.flash_timesteps) - 1):
                batch.inputs["time"].fill_(self.config.flash_timesteps[i])
                velocity = self.backbone(**batch.inputs, cache=True)
                dt = self.config.flash_timesteps[i + 1] - self.config.flash_timesteps[i]
                x = x + dt * velocity
                batch.inputs["x"] = x
        finally:
            self.backbone.clear_cache()
        outputs = []
        for i, condition in enumerate(conditions):
            target = x[i : i + 1, : condition.target_length]
            # Flow boundary is [B,T,D]; the official VAE decoder accepts [B,D,T].
            vae_dtype = next(self.vae.parameters()).dtype
            decoder_latents = self.vae.denormalize(target).transpose(1, 2).to(dtype=vae_dtype)
            waveform = self.vae.inference_from_latents(decoder_latents)
            outputs.append(waveform.reshape(-1).float())
        return OmniOutput(
            torch.zeros(input_ids.numel(), device=input_ids.device),
            {
                "audio": outputs,
                "sr": [torch.tensor([self.config.sample_rate], device=input_ids.device) for _ in outputs],
            },
        )
