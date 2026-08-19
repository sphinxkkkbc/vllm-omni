# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OmniVoice Decoder (Stage 1) - Audio token to waveform conversion.

Implements the HiggsAudioV2 decode path using transformers' DacModel decoder
and a custom RVQ quantizer, compatible with transformers 4.x.

Decode path:
  audio_codes [B, 8, T]
    → RVQ codebook lookup + project_out → sum → [B, 1024, T]
    → fc2 Linear(1024, 256) → [B, 256, T]
    → DAC acoustic decoder (conv transpose upsampling) → [B, 1, T*960]
    → 24kHz waveform (25fps × 960 samples/frame)
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
from torch.cuda.graphs import CUDAGraph
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig

logger = init_logger(__name__)


class HiggsAudioVQLayer(nn.Module):
    """Single VQ layer: codebook lookup + project_out."""

    def __init__(self, codebook_size: int = 1024, codebook_dim: int = 64, hidden_size: int = 1024):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.project_out = nn.Linear(codebook_dim, hidden_size)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: [B, T] → [B, hidden_size, T]"""
        quantized = self.codebook(indices)  # [B, T, codebook_dim]
        quantized = self.project_out(quantized)  # [B, T, hidden_size]
        return quantized.permute(0, 2, 1)  # [B, hidden_size, T]


class HiggsAudioRVQ(nn.Module):
    """Residual Vector Quantizer with 8 codebook layers."""

    def __init__(
        self, num_quantizers: int = 8, codebook_size: int = 1024, codebook_dim: int = 64, hidden_size: int = 1024
    ):
        super().__init__()
        self.quantizers = nn.ModuleList(
            [HiggsAudioVQLayer(codebook_size, codebook_dim, hidden_size) for _ in range(num_quantizers)]
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: [num_quantizers, B, T] → [B, hidden_size, T]"""
        result = torch.zeros(
            codes.shape[1],
            self.quantizers[0].project_out.out_features,
            codes.shape[2],
            device=codes.device,
            dtype=torch.float32,
        )
        for i, quantizer in enumerate(self.quantizers):
            result = result + quantizer.decode(codes[i])
        return result


class DecoderGraph:
    def __init__(
        self,
        decode_fn,
        batch_sizes: tuple[int] = (1, 2),
        bucket_sizes: tuple[int] = (64, 128, 256),
        device: torch.device = torch.device("cuda"),
    ):
        self.graphs: dict[tuple[int, int], CUDAGraph] = {}
        self.capture_batch_size = batch_sizes
        self.device = device
        self.capture_bucket_size = bucket_sizes
        self.decode_fn = decode_fn
        self.static_inputs: dict[tuple[int, int], torch.Tensor] = {}
        self.static_lengths: dict[tuple[int, int], torch.Tensor] = {}
        self.static_outputs: dict[tuple[int, int], torch.Tensor] = {}
        self.captured = False

    def warmup(self):
        for batch_size in self.capture_batch_size:
            for bucket_size in self.capture_bucket_size:
                self._capture(batch_size, bucket_size)
        self.captured = bool(self.graphs)

    def _capture(self, batch_size: int, bucket_size: int):
        if torch.cuda.is_current_stream_capturing() or self.device.type != "cuda":
            return
        static_inputs = torch.zeros((batch_size, 8, bucket_size), device=self.device, dtype=torch.long)
        static_lengths = torch.full((batch_size,), bucket_size, device=self.device, dtype=torch.long)
        self.static_inputs[(batch_size, bucket_size)] = static_inputs
        self.static_lengths[(batch_size, bucket_size)] = static_lengths
        for _ in range(3):
            self.decode_fn(static_inputs, static_lengths)
        graph = CUDAGraph()
        with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
            static_outputs = self.decode_fn(static_inputs, static_lengths)
        self.graphs[(batch_size, bucket_size)] = graph
        self.static_outputs[(batch_size, bucket_size)] = static_outputs
        logger.info(f"Captured graph for batch_size={batch_size}, bucket_size={bucket_size}")

    def find_nearest_padding(self, batch, bucket):
        candidates = [
            (b, bk) for b in self.capture_batch_size for bk in self.capture_bucket_size if b >= batch and bk >= bucket
        ]
        return min(candidates, key=lambda x: x[0] * x[1]) if candidates else None

    def forward(self, codes, lengths):
        batch_size, _, bucket_size = codes.shape
        if batch_size > max(self.capture_batch_size) or bucket_size > max(self.capture_bucket_size):
            return self.decode_fn(codes, lengths)

        graph_key = self.find_nearest_padding(batch_size, bucket_size)
        if graph_key is None:
            return self.decode_fn(codes, lengths)

        if graph_key not in self.graphs:
            return self.decode_fn(codes, lengths)

        static_input = self.static_inputs[graph_key]
        static_lengths = self.static_lengths[graph_key]
        static_input.zero_()
        static_lengths.zero_()
        static_input[:batch_size, :, :bucket_size].copy_(codes)
        static_lengths[:batch_size].copy_(lengths)
        self.graphs[graph_key].replay()
        return self.static_outputs[graph_key][:batch_size].clone()


class OmniVoiceDecoder(nn.Module):
    """OmniVoice Stage 1: Token-to-audio decoder.

    Uses DAC acoustic decoder from transformers + custom HiggsAudio RVQ
    quantizer to convert 8-codebook tokens into 24kHz waveform.
    """

    def __init__(self, config: OmniVoiceConfig):
        super().__init__()
        self.config = config
        self.sample_rate = config.sample_rate
        self._loaded = False

        # These are populated by load_weights
        self.quantizer = None
        self.fc2 = None
        self.acoustic_decoder = None
        self.graphs: DecoderGraph | None = None

    @staticmethod
    def _mask_by_lengths(hidden_states: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(hidden_states.shape[-1], device=hidden_states.device)
        valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
        return hidden_states.masked_fill(~valid.unsqueeze(1), 0.0)

    def _decode_acoustic_length_aware(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Run the non-causal DAC decoder while zeroing padded activations."""
        decoder = self.acoustic_decoder
        hidden_states = self._mask_by_lengths(hidden_states, lengths)
        hidden_states = decoder.conv1(hidden_states)
        hidden_states = self._mask_by_lengths(hidden_states, lengths)

        for block in decoder.block:
            hidden_states = block.snake1(hidden_states)
            hidden_states = block.conv_t1(hidden_states)
            stride = block.conv_t1.stride[0]
            lengths = lengths * stride
            hidden_states = self._mask_by_lengths(hidden_states, lengths)

            for residual_unit in (block.res_unit1, block.res_unit2, block.res_unit3):
                hidden_states = residual_unit(hidden_states)
                hidden_states = self._mask_by_lengths(hidden_states, lengths)

        hidden_states = decoder.snake1(hidden_states)
        hidden_states = self._mask_by_lengths(hidden_states, lengths)
        hidden_states = decoder.conv2(hidden_states)
        hidden_states = self._mask_by_lengths(hidden_states, lengths)
        hidden_states = decoder.tanh(hidden_states)
        return self._mask_by_lengths(hidden_states, lengths)

    def _decode_impl(
        self,
        audio_codes: torch.Tensor,
        target_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode audio tokens to waveform.

        Args:
            audio_codes: [B, 8, T] - 8-codebook audio token IDs

        Returns:
            waveform: [B, 1, audio_samples] at 24kHz
        """
        # Transpose: [B, 8, T] → [8, B, T]
        codes = audio_codes.transpose(0, 1).long()

        # RVQ decode: sum codebook embeddings → [B, 1024, T]
        quantized = self.quantizer.decode(codes)

        # Project: [B, 1024, T] → fc2 → [B, 256, T]
        # Cast to fc2 weight dtype (may be fp16 when checkpoint stores weights as fp16),
        # then upcast back to float32 — acoustic decoder ConvTranspose1d upsampling
        # produces intermediate values that exceed the fp16 range (~65504), causing NaN.
        quantized = self.fc2(quantized.transpose(1, 2).to(self.fc2.weight.dtype)).transpose(1, 2).float()

        # Acoustic decoder: [B, 256, T] → [B, 1, T*960]
        if target_lens is None or not all(
            hasattr(self.acoustic_decoder, name) for name in ("conv1", "block", "snake1", "conv2", "tanh")
        ):
            audio = self.acoustic_decoder(quantized)
        else:
            audio = self._decode_acoustic_length_aware(quantized, target_lens)

        # Ensure [B, 1, samples]
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        return audio

    @torch.inference_mode()
    def forward(
        self,
        audio_codes: torch.Tensor,
        target_lens: list[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode audio tokens to waveform.

        Args:
            audio_codes: [B, 8, T] - 8-codebook audio token IDs

        Returns:
            waveform: [B, 1, audio_samples] at 24kHz
        """
        if not self._loaded:
            raise RuntimeError("Decoder not loaded. Call load_weights() first.")

        device = audio_codes.device
        if target_lens is None:
            lengths = torch.full(
                (audio_codes.shape[0],),
                audio_codes.shape[-1],
                dtype=torch.long,
                device=device,
            )
        elif isinstance(target_lens, torch.Tensor):
            lengths = target_lens.to(device=device, dtype=torch.long)
        else:
            lengths = torch.tensor(target_lens, device=device, dtype=torch.long)

        if lengths.shape != (audio_codes.shape[0],):
            raise ValueError(
                f"Expected one target length per request, got shape {tuple(lengths.shape)} "
                f"for batch size {audio_codes.shape[0]}."
            )
        if torch.any(lengths <= 0) or torch.any(lengths > audio_codes.shape[-1]):
            raise ValueError(f"Target lengths must be in [1, {audio_codes.shape[-1]}], got {lengths.tolist()}.")

        if self.graphs is not None and self.graphs.captured:
            return self.graphs.forward(audio_codes, lengths).to(device)
        else:
            return self._decode_impl(audio_codes, lengths).to(device)

    def _adjust_output_padding(self, decoder: nn.Module):
        """Adjust ConvTranspose1d output_padding (HiggsAudioV2 modification)."""
        for module in decoder.modules():
            if isinstance(module, nn.ConvTranspose1d):
                stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                module.output_padding = (stride % 2,)

    def load_weights(self, model_dir: str, device: torch.device) -> None:
        """Load decoder components from audio_tokenizer/model.safetensors."""
        from safetensors.torch import load_file
        from transformers import DacConfig, DacModel

        audio_tokenizer_path = os.path.join(model_dir, "audio_tokenizer")
        config_path = os.path.join(audio_tokenizer_path, "config.json")
        weights_path = os.path.join(audio_tokenizer_path, "model.safetensors")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Audio tokenizer weights not found at {weights_path}")

        with open(config_path) as f:
            tokenizer_config = json.load(f)

        state_dict = load_file(weights_path, device=str(device))

        # 1. Build RVQ quantizer
        codebook_dim = tokenizer_config.get("codebook_dim", 64)
        codebook_size = tokenizer_config.get("codebook_size", 1024)
        # Hidden size = quantizer project_out output dim
        hidden_size = state_dict["quantizer.quantizers.0.project_out.weight"].shape[0]
        num_quantizers = sum(
            1 for k in state_dict if k.startswith("quantizer.quantizers.") and k.endswith(".codebook.embed")
        )

        self.quantizer = HiggsAudioRVQ(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            hidden_size=hidden_size,
        ).to(device)

        # Load quantizer weights
        for i in range(num_quantizers):
            prefix = f"quantizer.quantizers.{i}"
            embed_key = f"{prefix}.codebook.embed"
            if embed_key in state_dict:
                self.quantizer.quantizers[i].codebook.weight.data.copy_(state_dict[embed_key])
            proj_out_w = f"{prefix}.project_out.weight"
            proj_out_b = f"{prefix}.project_out.bias"
            if proj_out_w in state_dict:
                self.quantizer.quantizers[i].project_out.weight.data.copy_(state_dict[proj_out_w])
            if proj_out_b in state_dict:
                self.quantizer.quantizers[i].project_out.bias.data.copy_(state_dict[proj_out_b])

        # 2. Build fc2 projection
        fc2_w = state_dict["fc2.weight"]
        fc2_b = state_dict["fc2.bias"]
        self.fc2 = nn.Linear(fc2_w.shape[1], fc2_w.shape[0]).to(device)
        self.fc2.weight.data.copy_(fc2_w)
        self.fc2.bias.data.copy_(fc2_b)

        # 3. Build DAC acoustic decoder
        dac_cfg = DacConfig(**tokenizer_config["acoustic_model_config"])
        dac_model = DacModel(dac_cfg)
        self.acoustic_decoder = dac_model.decoder.to(device)

        # Load acoustic decoder weights
        loaded = 0
        for name, param in self.acoustic_decoder.named_parameters():
            higgs_name = f"acoustic_decoder.{name}"
            if higgs_name in state_dict:
                param.data.copy_(state_dict[higgs_name])
                loaded += 1

        # Checkpoint weights may be fp16; force float32 so ConvTranspose1d
        # upsampling doesn't overflow (fp16 max ~65504 is too small for
        # intermediate activations in the DAC upsampling chain).
        self.acoustic_decoder.float()

        # Apply HiggsAudioV2 output padding adjustment
        self._adjust_output_padding(self.acoustic_decoder)

        # Remove tanh if present (HiggsAudioV2 uses Identity instead)
        if hasattr(self.acoustic_decoder, "tanh"):
            self.acoustic_decoder.tanh = nn.Identity()

        self.acoustic_decoder.eval()
        self._loaded = True

        self.graphs = DecoderGraph(
            self._decode_impl,
            device=device,
        )
        self.graphs.warmup()

        logger.info(
            "Loaded OmniVoice decoder: %d quantizers, fc2(%d→%d), acoustic decoder (%d weights)",
            num_quantizers,
            fc2_w.shape[1],
            fc2_w.shape[0],
            loaded,
        )
