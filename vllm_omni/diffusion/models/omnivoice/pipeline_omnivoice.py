# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OmniVoice TTS Pipeline for vLLM-Omni diffusion engine.

Single-stage pipeline that runs the full text-to-speech flow:
  text → tokenize → 32-step iterative unmasking → 8-codebook tokens → DAC decode → 24kHz audio

Uses request-mode execution (all steps in one forward() call).
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections.abc import Iterable, Sequence
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer as HFTokenizer
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.interface import SupportAudioOutput
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.model_executor.models.omnivoice.duration import RuleDurationEstimator
from vllm_omni.model_executor.models.omnivoice.omnivoice_decoder import OmniVoiceDecoder
from vllm_omni.model_executor.models.omnivoice.omnivoice_generator import (
    OmniVoiceGenerator,
    _get_time_steps,
    _gumbel_sample,
)
from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig
from vllm_omni.utils.speaker_cache import get_speaker_cache

try:
    from transformers import HiggsAudioV2TokenizerModel
except ImportError:
    HiggsAudioV2TokenizerModel = None

import torchaudio

logger = init_logger(__name__)


def get_omnivoice_post_process_func(od_config: OmniDiffusionConfig):
    """Post-processing: convert audio tensor to numpy for WAV encoding."""

    def post_process_func(audio: torch.Tensor, output_type: str = "np"):
        if output_type == "pt":
            return audio
        return audio.cpu().float().numpy()

    return post_process_func


def _combine_text(text, ref_text: str | None = None) -> str:
    # combine with reference text if not None
    if ref_text:
        full_text = ref_text.strip() + " " + text.strip()
    else:
        full_text = text.strip()

    # filter out newline / carriage-return characters
    full_text = re.sub(r"[\r\n]+", "", full_text)

    # replace Chinese parentheses with English ones
    full_text = full_text.replace("\uff08", "(").replace("\uff09", ")")

    # collapse consecutive spaces / tabs into a single space
    full_text = re.sub(r"[ \t]+", " ", full_text)

    # remove spaces around chinese characters
    chinese_range = r"[\u4e00-\u9fff]"
    pattern = rf"(?<={chinese_range})\s+|\s+(?={chinese_range})"
    full_text = re.sub(pattern, "", full_text)

    return full_text


_NONVERBAL_PATTERN = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)


def _tokenize_with_nonverbal_tags(text: str, tokenizer) -> list[int]:
    """Tokenize text containing non-verbal tags, handling each tag independently.

    Non-verbal tags are tokenized standalone to guarantee consistent token
    IDs regardless of surrounding language context (Chinese, English, etc.).

    Args:
        text: Full text string potentially containing non-verbal tags.
        tokenizer: HuggingFace text tokenizer instance.
    Returns:
        Token IDs list of length seq_len.
    """
    parts = []
    last_end = 0
    for m in _NONVERBAL_PATTERN.finditer(text):
        if m.start() > last_end:
            segment = text[last_end : m.start()]
            ids = tokenizer.encode(segment)
            if ids:
                parts.append(ids)
        tag_ids = tokenizer.encode(m.group())
        if tag_ids:
            parts.append(tag_ids)
        last_end = m.end()
    if last_end < len(text):
        segment = text[last_end:]
        ids = tokenizer.encode(segment)
        if ids:
            parts.append(ids)

    if not parts:
        return tokenizer.encode(text).ids
    else:
        combined = []
        for p in parts:
            combined.extend(p.ids)
    return combined


class OmniVoicePipeline(nn.Module, SupportAudioOutput):
    """OmniVoice text-to-speech pipeline for the diffusion engine.

    Wraps OmniVoiceGenerator (32-step iterative unmasking) and
    OmniVoiceDecoder (HiggsAudioV2 RVQ + DAC) into a single forward() call.
    """

    support_audio_output: ClassVar[bool] = True
    supports_request_batch: ClassVar[bool] = True
    supports_step_execution: ClassVar[bool] = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self.model_path = od_config.model

        # Resolve model path (HF hub ID → local cache)
        if not os.path.isdir(self.model_path):
            from huggingface_hub import snapshot_download

            self.model_path = snapshot_download(self.model_path)

        # Load OmniVoice config
        config_path = os.path.join(self.model_path, "config.json")
        with open(config_path) as f:
            hf_config = json.load(f)
        self.config = OmniVoiceConfig(**hf_config)

        # Build generator and decoder
        self.generator = OmniVoiceGenerator(self.config)
        self.decoder = OmniVoiceDecoder(self.config)

        # Tokenizer (low-level, avoids HF tokenizer extra_special_tokens issue)
        tokenizer_path = os.path.join(self.model_path, "tokenizer.json")
        self.tokenizer = HFTokenizer.from_file(tokenizer_path)

        # Audio tokenizer for voice cloning (requires transformers>=5.3)
        if HiggsAudioV2TokenizerModel is not None:
            audio_tokenizer_path = os.path.join(self.model_path, "audio_tokenizer")
            self.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
                audio_tokenizer_path, device_map=self.device
            ).eval()
            logger.info("HiggsAudioV2 tokenizer loaded for voice cloning on %s", self.device)
        else:
            self.audio_tokenizer = None
            logger.warning("Voice cloning disabled (requires transformers>=5.3.0).")

        # Duration estimator
        self.duration_estimator = RuleDurationEstimator()

        # Speaker cache for ref_audio_tokens
        self._speaker_cache = get_speaker_cache()

        # Generation parameters
        self.num_step = self.config.num_step
        self.guidance_scale = self.config.guidance_scale
        self.t_shift = self.config.t_shift
        self.layer_penalty_factor = self.config.layer_penalty_factor
        self.position_temperature = self.config.position_temperature
        self.class_temperature = self.config.class_temperature
        self.sample_rate = self.config.sample_rate

    def _encode_ref_audio(self, audio_signal: torch.Tensor, sr: int) -> torch.Tensor:
        """Encode reference audio to 8-codebook tokens for voice cloning."""
        if self.audio_tokenizer is None:
            raise RuntimeError("Audio tokenizer not available for voice cloning")
        if audio_signal.dim() == 1:
            audio_signal = audio_signal.unsqueeze(0)
        # Resample to tokenizer's expected sample rate
        target_sr = self.audio_tokenizer.config.sample_rate
        if sr != target_sr:
            audio_signal = torchaudio.functional.resample(audio_signal, sr, target_sr)
        # Ensure mono [B, 1, samples]
        if audio_signal.dim() == 2:
            audio_signal = audio_signal.unsqueeze(1)
        with torch.inference_mode():
            tokens = self.audio_tokenizer.encode(
                audio_signal.to(self.audio_tokenizer.device), return_dict=False
            )  # [B, 8, T_ref]
            tokens = tokens.squeeze(0)  # [8, T_ref]
        return tokens

    def prepare_encode(self, states: list[StepRequestState]) -> DiffusionRequestBatch:
        ref_audio = None
        ref_text = None
        lang = "None"
        instruct = "None"
        voice_name = None
        device = self.device
        num_cb = self.config.num_audio_codebook
        mask_id = self.config.audio_mask_id
        batch_target_len: list[int] = []
        batch_input_ids: list[torch.Tensor] = []
        batch_audio_mask: list[torch.Tensor] = []
        batch_attn_mask: list[torch.Tensor] = []
        if isinstance(states, StepRequestState):
            states = [states]
        for state in states:
            prompt = state.prompt if state.prompt else ""
            extra = state.sampling.extra_args or {}
            seed = extra.get("seed", None)
            if isinstance(prompt, dict):
                # Top-level keys (used by serving_speech.py /v1/audio/speech path)
                text = prompt.get("input") or prompt.get("text") or prompt.get("prompt")
                ref_audio = prompt.get("ref_audio")
                ref_text = prompt.get("ref_text")
                voice_name = prompt.get("voice_name")
                lang = prompt.get("lang")
                instruct = prompt.get("instruct")
                # OmniTextPrompt format (used by offline Omni.generate path):
                # ref_audio comes via multi_modal_data["audio"] and the rest via
                # mm_processor_kwargs. Fall back to those when top-level keys are
                # absent so both invocation styles work.
                mm_data = prompt.get("multi_modal_data") or {}
                mm_kwargs = prompt.get("mm_processor_kwargs") or {}
                if ref_audio is None:
                    audio_field = mm_data.get("audio")
                    # Standard multimodal shape allows a list of audios; OmniVoice
                    # voice cloning conditions on a single reference clip, so
                    # unwrap a length-1 list and reject multi-reference prompts up
                    # front (otherwise a list would later crash inside
                    # ``_encode_ref_audio`` when it calls ``audio.dim()``).
                    if isinstance(audio_field, list):
                        if len(audio_field) == 1:
                            audio_field = audio_field[0]
                        elif len(audio_field) > 1:
                            return DiffusionOutput(
                                error=f"OmniVoice voice cloning supports a single reference audio; got {len(audio_field)}"  # noqa: E501
                            )
                        else:
                            audio_field = None
                    if audio_field is not None:
                        if isinstance(audio_field, tuple) and len(audio_field) == 2:
                            ref_audio = audio_field
                        else:
                            sr = mm_kwargs.get("sample_rate") or self.sample_rate
                            ref_audio = (audio_field, int(sr))
                if ref_text is None:
                    ref_text = mm_kwargs.get("ref_text")
                if lang is None:
                    lang = mm_kwargs.get("lang")
                if instruct is None:
                    instruct = mm_kwargs.get("instruct")

                if not text:
                    return DiffusionOutput(error="Empty text prompt")
                lang = lang or "None"
                instruct = instruct or "None"
            else:
                text = str(prompt)
                if not text:
                    return DiffusionOutput(error="Empty text prompt")

            # Estimate target duration
            target_len = self.duration_estimator.estimate_duration(text, "Nice to meet you.", 25)
            target_len = max(1, int(target_len))
            batch_target_len.append(target_len)

            # Build text prompt with control tokens
            style_text = f"<|denoise|><|lang_start|>{lang}<|lang_end|><|instruct_start|>{instruct}<|instruct_end|>"
            full_text = _combine_text(ref_text=ref_text, text=text)
            wrapped_text = f"<|text_start|>{full_text}<|text_end|>"
            style_tokens = self.tokenizer.encode(style_text).ids
            text_tokens = _tokenize_with_nonverbal_tags(wrapped_text, self.tokenizer)
            encoding_ids = style_tokens + text_tokens
            text_tokens = torch.tensor(encoding_ids, dtype=torch.long, device=device)
            text_len = text_tokens.shape[0]

            # Encode reference audio tokens if provided (with voice caching)
            ref_audio_tokens = None
            if ref_audio is not None:
                if self.audio_tokenizer is None:
                    raise RuntimeError(
                        "Voice cloning requires transformers>=5.3.0. Try: uv pip install 'transformers>=5.3.0'"
                    )
                # Check speaker cache first
                _cache_key = None
                if voice_name:
                    _cache_key = self._speaker_cache.make_cache_key(
                        voice_name,
                        model_type="omnivoice",
                        created_at=int(prompt.get("voice_created_at") or 0),
                    )
                    cached = self._speaker_cache.get(_cache_key)
                    if cached is not None:
                        ref_audio_tokens = cached["ref_audio_tokens"].to(device)
                        _cache_key = None  # hit → don't store again
                        logger.debug("Speaker cache HIT for OmniVoice speaker '%s'", voice_name)

                if ref_audio_tokens is None:
                    audio_signal, sr = ref_audio
                    if isinstance(audio_signal, np.ndarray):
                        audio_signal = torch.from_numpy(audio_signal).float()
                    ref_audio_tokens = self._encode_ref_audio(audio_signal, int(sr)).to(device)

                    # Store in cache for next request
                    if _cache_key is not None:
                        self._speaker_cache.put(_cache_key, {"ref_audio_tokens": ref_audio_tokens.cpu()})
                        logger.debug("Speaker cache STORE for OmniVoice speaker '%s'", voice_name)

            # Build conditional + unconditional batches [2, 8, max_len]
            text_ids = text_tokens.unsqueeze(0).repeat(num_cb, 1)
            target_ids = torch.full((num_cb, target_len), mask_id, dtype=torch.long, device=device)

            if ref_audio_tokens is not None:
                cond_ids = torch.cat([text_ids, ref_audio_tokens, target_ids], dim=1)
            else:
                cond_ids = torch.cat([text_ids, target_ids], dim=1)

            cond_len = cond_ids.shape[1]
            uncond_ids = target_ids.clone()
            uncond_len = target_len
            max_len = max(cond_len, uncond_len)
            if uncond_len < max_len:
                pad = torch.full(
                    (num_cb, max_len - uncond_len),
                    mask_id,
                    dtype=torch.long,
                    device=device,
                )
                uncond_ids = torch.cat([uncond_ids, pad], dim=1)
            input_ids = torch.stack([cond_ids, uncond_ids])
            batch_input_ids.append(input_ids)

            audio_mask = torch.zeros(2, max_len, dtype=torch.bool, device=device)
            audio_mask[0, text_len:cond_len] = True
            audio_mask[1, :uncond_len] = True
            batch_audio_mask.append(audio_mask)

            attn_mask = torch.zeros(2, 1, max_len, max_len, dtype=torch.bool, device=device)
            attn_mask[0, :, :cond_len, :cond_len] = True
            attn_mask[1, :, :uncond_len, :uncond_len] = True
            batch_attn_mask.append(attn_mask)

        if len(batch_input_ids) > 1:
            max_input_ids_len = max([ids.shape[-1] for ids in batch_input_ids])
            num_padded_len = [max_input_ids_len - ids.shape[-1] for ids in batch_input_ids]
            for i, ids in enumerate(batch_input_ids):
                batch_input_ids[i] = torch.cat(
                    [
                        ids,
                        torch.full(
                            (ids.shape[0], ids.shape[1], num_padded_len[i]), mask_id, dtype=torch.long, device=device
                        ),
                    ],
                    dim=-1,
                )
            for i, mask in enumerate(batch_audio_mask):
                batch_audio_mask[i] = torch.cat(
                    [mask, torch.full((mask.shape[0], num_padded_len[i]), False, dtype=torch.bool, device=device)],
                    dim=-1,
                )
            for i, mask in enumerate(batch_attn_mask):
                n = num_padded_len[i]
                batch_attn_mask[i] = F.pad(mask, (0, n, 0, n))

        target_lens = batch_target_len
        B = len(target_lens)
        device = input_ids.device
        max_target_len = max(target_lens)
        mask_id = self.config.audio_mask_id
        num_codebooks = self.config.num_audio_codebook
        if seed is None:
            seed = random.randint(0, 2**63 - 1)
        num_step = self.num_step
        t_shift = self.t_shift

        # Initialize all target tokens as [MASK]
        positions = torch.arange(max_target_len, device=device).unsqueeze(0)
        valid_target_mask = positions < torch.tensor(target_lens, device=device).unsqueeze(1)
        tokens = torch.zeros((B, num_codebooks, max_target_len), dtype=torch.long, device=device)
        tokens.masked_fill_(valid_target_mask.unsqueeze(1), mask_id)

        timesteps = _get_time_steps(0.0, 1.0, num_step + 1, t_shift)

        # Compute unmasking schedule
        schedules = []
        for t_len in target_lens:
            total_mask = t_len * num_codebooks
            rem = total_mask
            sched = []
            for step in range(num_step):
                num = (
                    rem
                    if step == num_step - 1
                    else min(
                        math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])),
                        rem,
                    )
                )
                sched.append(int(num))
                rem -= int(num)
            schedules.append(sched)
        schedules = torch.tensor(schedules, dtype=torch.long, device=device)

        layer_ids = torch.arange(num_codebooks, device=device).view(1, -1, 1)
        generator = torch.Generator(device=device).manual_seed(seed)

        for i in range(B):
            states[i].latents = batch_input_ids[i]
            states[i].timesteps = schedules[i]
            states[i].guidance = self.guidance_scale
            states[i].extra["schedules"] = schedules
            states[i].extra["layer_ids"] = layer_ids
            states[i].extra["generator"] = generator
            states[i].extra["t_shift"] = t_shift
            states[i].extra["target_len"] = target_lens[i]
            states[i].extra["audio_mask"] = batch_audio_mask[i]
            states[i].extra["attn_mask"] = batch_attn_mask[i]
            states[i].extra["tokens"] = tokens[i]

        use_cuda_graph = self.generator._cuda_graph_fwd is not None and input_ids.is_cuda
        if not use_cuda_graph:
            # Eager-path-only constants (the cuda-graph captures its own).
            text_embeds_cached = self.text_embedding(input_ids[:, 0, :])
            audio_mask_3d = audio_mask.unsqueeze(-1)
            self._ensure_rope(input_ids.shape[-1], device)
            target_dtype = text_embeds_cached.dtype
            cos = self._rope_cos.to(device=device, dtype=target_dtype)
            sin = self._rope_sin.to(device=device, dtype=target_dtype)

            for i in range(B):
                states[i].extra["text_embeds"] = text_embeds_cached
                states[i].extra["audio_mask_3d"] = audio_mask_3d
                states[i].extra["cos"] = cos
                states[i].extra["sin"] = sin

    def denoise_step(self, input_batch: InputBatch, *, states: Sequence[StepRequestState] | None = None, **kwargs: Any):
        use_cuda_graph = self.generator._cuda_graph_fwd is not None
        input_ids = input_batch.latents
        layer_ids = states[0].extra["layer_ids"]
        generator = states[0].extra["generator"]
        schedules = states[0].extra["schedules"]

        batch_audio_mask: list[torch.Tesor] = []
        batch_attn_mask: list[torch.Tensor] = []
        batch_target_len: list[int] = []
        batch_tokens: list[torch.Tensor] = []

        steps: list[int] = []

        for state in states:
            batch_audio_mask.append(state.extra.get("audio_mask", None))
            batch_attn_mask.append(state.extra.get("attn_mask", None))
            guidance_scale = state.extra.get("guidance", self.guidance_scale)
            batch_target_len.append(state.extra["target_len"])
            batch_tokens.append(state.extra["tokens"])
            steps.append(state.step_index)
        batch_tokens = torch.stack(batch_tokens, dim=0)
        if batch_tokens.dim() == 4:
            batch_tokens = batch_tokens.squeeze(0)
        if len(batch_audio_mask) == 1:
            audio_mask = batch_audio_mask[0]
            batch_attn_mask = batch_attn_mask[0]
        else:
            audio_mask = torch.stack(batch_audio_mask, dim=0)
            batch_attn_mask = torch.stack(batch_attn_mask, dim=0)

        B = len(batch_target_len)
        target_lens = batch_target_len

        text_embeds_cached = states[0].extra.get("text_embeds", None)
        audio_mask_3d = states[0].extra.get("audio_mask_3d", None)
        cos = states[0].extra.get("cos", None)
        sin = states[0].extra.get("sin", None)

        mask_id = self.config.audio_mask_id
        position_temperature = self.position_temperature
        class_temperature = self.class_temperature
        layer_penalty_factor = self.layer_penalty_factor

        c_lens = batch_attn_mask[:B, 0, 0].sum(dim=-1).tolist()

        # Materialize the SDPA float mask once so the captured graph (and eager path) skip per-layer conversion.
        sdpa_attn_mask = torch.zeros_like(batch_attn_mask, dtype=torch.float32).masked_fill_(
            ~batch_attn_mask, float("-inf")
        )
        if use_cuda_graph:
            # Float mask skips per-layer conversion; fp32 cast deferred to the per-item slices below.
            batch_logits = self.generator._cuda_graph_fwd(input_ids, audio_mask, sdpa_attn_mask)
        else:
            # Eager fallback reuses hoisted constants (text embeds, sdpa mask, cos/sin).
            inputs_embeds = self.generator._prepare_embeddings(
                input_ids, audio_mask, text_embeds=text_embeds_cached, audio_mask_3d=audio_mask_3d
            )
            hidden_states = self.generator._transformer_forward(inputs_embeds, sdpa_attn_mask, cos=cos, sin=sin)
            # fp32 cast deferred to the per-item slices below.
            batch_logits = self.generator._get_logits(hidden_states)
        # batch_logits: [2*B, 8, S, 1025]

        for i in range(B):
            k = schedules[i][steps[i]]
            if k <= 0:
                continue

            c_len = c_lens[i]
            t_len = target_lens[i]

            # Extract logits for target region; upcast only the slices we actually consume.
            c_logits = batch_logits[i : i + 1, :, c_len - t_len : c_len, :].to(torch.float32)
            u_logits = batch_logits[B + i : B + i + 1, :, :t_len, :].to(torch.float32)

            # Classifier-free guidance. Fuse the chain: the two inner
            # log_softmax normalizers are per-position scalars that the final
            # shift-invariant log_softmax cancels, so guide on the raw logits
            # with a single softmax: log_softmax((1+s)*c - s*u). Exact.
            if guidance_scale != 0:
                log_probs = F.log_softmax(
                    (1.0 + guidance_scale) * c_logits - guidance_scale * u_logits,
                    dim=-1,
                )
            else:
                log_probs = F.log_softmax(c_logits, dim=-1)

            # Prevent predicting [MASK]
            log_probs[..., mask_id] = -float("inf")

            # Token prediction
            if class_temperature > 0.0:
                pred_tokens = _gumbel_sample(log_probs, class_temperature, generator).argmax(dim=-1)
            else:
                pred_tokens = log_probs.argmax(dim=-1)  # [1, 8, T]

            # Confidence scores
            scores = log_probs.max(dim=-1)[0]  # [1, 8, T]

            # Layer penalty (earlier codebooks get higher priority)
            scores = scores - (layer_ids * layer_penalty_factor)

            # Gumbel noise for position selection
            if position_temperature > 0.0:
                scores = _gumbel_sample(scores, position_temperature, generator)

            # Mask out already unmasked positions
            sample_tokens = batch_tokens[i : i + 1, :, :t_len]
            scores.masked_fill_(sample_tokens != mask_id, -float("inf"))

            # Select top-k positions to unmask. .flatten() on this non-contiguous view already copies.
            _, topk_idx = torch.topk(scores.flatten(), k)
            flat_tokens = sample_tokens.flatten()
            flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
            sample_tokens.copy_(flat_tokens.view_as(sample_tokens))
            states[i].extra["tokens"] = sample_tokens

            # Mirror update into both cond and uncond input_ids halves for the next step.
            input_ids[i, :, c_len - t_len : c_len] = sample_tokens.squeeze(0)
            input_ids[B + i, :, :t_len] = sample_tokens.squeeze(0)

        return input_ids

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor, **kwargs: Any):
        state.latents = noise_pred
        state.step_index += 1

    def post_decode(self, state: StepRequestState, **kwargs: Any):
        tokens = state.extra.get("tokens", None)
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        audio = self.decoder(tokens)
        return DiffusionOutput(output=audio)

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        """Generate speech audio from text, optionally with voice cloning.

        Accepts either a plain text prompt or a structured dict:
          {"text": "...", "ref_audio": (samples, sr), "ref_text": "...",
           "lang": "...", "instruct": "..."}
        """
        ref_audio = None
        ref_text = None
        lang = "None"
        instruct = "None"
        voice_name = None
        outputs: list[DiffusionOutput] = []
        device = self.device
        num_cb = self.config.num_audio_codebook
        mask_id = self.config.audio_mask_id
        batch_target_len: list[int] = []
        batch_input_ids: list[torch.Tensor] = []
        batch_audio_mask: list[torch.Tensor] = []
        batch_attn_mask: list[torch.Tensor] = []
        for request in req.requests:
            prompt = request.prompt if request.prompt else ""
            extra = request.sampling_params.extra_args or {}
            seed = extra.get("seed", None)
            if isinstance(prompt, dict):
                # Top-level keys (used by serving_speech.py /v1/audio/speech path)
                text = prompt.get("input") or prompt.get("text") or prompt.get("prompt")
                ref_audio = prompt.get("ref_audio")
                ref_text = prompt.get("ref_text")
                voice_name = prompt.get("voice_name")
                lang = prompt.get("lang")
                instruct = prompt.get("instruct")
                # OmniTextPrompt format (used by offline Omni.generate path):
                # ref_audio comes via multi_modal_data["audio"] and the rest via
                # mm_processor_kwargs. Fall back to those when top-level keys are
                # absent so both invocation styles work.
                mm_data = prompt.get("multi_modal_data") or {}
                mm_kwargs = prompt.get("mm_processor_kwargs") or {}
                if ref_audio is None:
                    audio_field = mm_data.get("audio")
                    # Standard multimodal shape allows a list of audios; OmniVoice
                    # voice cloning conditions on a single reference clip, so
                    # unwrap a length-1 list and reject multi-reference prompts up
                    # front (otherwise a list would later crash inside
                    # ``_encode_ref_audio`` when it calls ``audio.dim()``).
                    if isinstance(audio_field, list):
                        if len(audio_field) == 1:
                            audio_field = audio_field[0]
                        elif len(audio_field) > 1:
                            return DiffusionOutput(
                                error=f"OmniVoice voice cloning supports a single reference audio; got {len(audio_field)}"  # noqa: E501
                            )
                        else:
                            audio_field = None
                    if audio_field is not None:
                        if isinstance(audio_field, tuple) and len(audio_field) == 2:
                            ref_audio = audio_field
                        else:
                            sr = mm_kwargs.get("sample_rate") or self.sample_rate
                            ref_audio = (audio_field, int(sr))
                if ref_text is None:
                    ref_text = mm_kwargs.get("ref_text")
                if lang is None:
                    lang = mm_kwargs.get("lang")
                if instruct is None:
                    instruct = mm_kwargs.get("instruct")

                if not text:
                    return DiffusionOutput(error="Empty text prompt")
                lang = lang or "None"
                instruct = instruct or "None"
            else:
                text = str(prompt)
                if not text:
                    return DiffusionOutput(error="Empty text prompt")

            # Estimate target duration
            target_len = self.duration_estimator.estimate_duration(text, "Nice to meet you.", 25)
            target_len = max(1, int(target_len))
            batch_target_len.append(target_len)

            # Build text prompt with control tokens
            style_text = f"<|denoise|><|lang_start|>{lang}<|lang_end|><|instruct_start|>{instruct}<|instruct_end|>"
            full_text = _combine_text(ref_text=ref_text, text=text)
            wrapped_text = f"<|text_start|>{full_text}<|text_end|>"
            style_tokens = self.tokenizer.encode(style_text).ids
            text_tokens = _tokenize_with_nonverbal_tags(wrapped_text, self.tokenizer)
            encoding_ids = style_tokens + text_tokens
            text_tokens = torch.tensor(encoding_ids, dtype=torch.long, device=device)
            text_len = text_tokens.shape[0]

            # Encode reference audio tokens if provided (with voice caching)
            ref_audio_tokens = None
            if ref_audio is not None:
                if self.audio_tokenizer is None:
                    raise RuntimeError(
                        "Voice cloning requires transformers>=5.3.0. Try: uv pip install 'transformers>=5.3.0'"
                    )
                # Check speaker cache first
                _cache_key = None
                if voice_name:
                    _cache_key = self._speaker_cache.make_cache_key(
                        voice_name,
                        model_type="omnivoice",
                        created_at=int(prompt.get("voice_created_at") or 0),
                    )
                    cached = self._speaker_cache.get(_cache_key)
                    if cached is not None:
                        ref_audio_tokens = cached["ref_audio_tokens"].to(device)
                        _cache_key = None  # hit → don't store again
                        logger.debug("Speaker cache HIT for OmniVoice speaker '%s'", voice_name)

                if ref_audio_tokens is None:
                    audio_signal, sr = ref_audio
                    if isinstance(audio_signal, np.ndarray):
                        audio_signal = torch.from_numpy(audio_signal).float()
                    ref_audio_tokens = self._encode_ref_audio(audio_signal, int(sr)).to(device)

                    # Store in cache for next request
                    if _cache_key is not None:
                        self._speaker_cache.put(_cache_key, {"ref_audio_tokens": ref_audio_tokens.cpu()})
                        logger.debug("Speaker cache STORE for OmniVoice speaker '%s'", voice_name)

            # Build conditional + unconditional batches [2, 8, max_len]
            text_ids = text_tokens.unsqueeze(0).repeat(num_cb, 1)
            target_ids = torch.full((num_cb, target_len), mask_id, dtype=torch.long, device=device)

            if ref_audio_tokens is not None:
                cond_ids = torch.cat([text_ids, ref_audio_tokens, target_ids], dim=1)
            else:
                cond_ids = torch.cat([text_ids, target_ids], dim=1)
            cond_len = cond_ids.shape[1]
            uncond_ids = target_ids.clone()
            uncond_len = target_len
            max_len = max(cond_len, uncond_len)
            if uncond_len < max_len:
                pad = torch.full(
                    (num_cb, max_len - uncond_len),
                    mask_id,
                    dtype=torch.long,
                    device=device,
                )
                uncond_ids = torch.cat([uncond_ids, pad], dim=1)
            input_ids = torch.stack([cond_ids, uncond_ids])
            batch_input_ids.append(input_ids)

            audio_mask = torch.zeros(2, max_len, dtype=torch.bool, device=device)
            audio_mask[0, text_len:cond_len] = True
            audio_mask[1, :uncond_len] = True
            batch_audio_mask.append(audio_mask)

            attn_mask = torch.zeros(2, 1, max_len, max_len, dtype=torch.bool, device=device)
            attn_mask[0, :, :cond_len, :cond_len] = True
            attn_mask[1, :, :uncond_len, :uncond_len] = True
            batch_attn_mask.append(attn_mask)
        if len(batch_input_ids) > 1:
            max_input_ids_len = max([ids.shape[-1] for ids in batch_input_ids])
            num_padded_len = [max_input_ids_len - ids.shape[-1] for ids in batch_input_ids]
            for i, ids in enumerate(batch_input_ids):
                batch_input_ids[i] = torch.cat(
                    [
                        ids,
                        torch.full(
                            (ids.shape[0], ids.shape[1], num_padded_len[i]), mask_id, dtype=torch.long, device=device
                        ),
                    ],
                    dim=-1,
                )
            for i, mask in enumerate(batch_audio_mask):
                batch_audio_mask[i] = torch.cat(
                    [mask, torch.full((mask.shape[0], num_padded_len[i]), False, dtype=torch.bool, device=device)],
                    dim=-1,
                )
            for i, mask in enumerate(batch_attn_mask):
                n = num_padded_len[i]
                batch_attn_mask[i] = F.pad(mask, (0, n, 0, n))
        # Each per-request tensor is ordered as [cond_i, uncond_i]. The
        # generator expects the full batch to be ordered as
        # [cond_0, ..., cond_B-1, uncond_0, ..., uncond_B-1] so that request i
        # pairs logits i and B+i for classifier-free guidance.
        input_id_pairs = torch.stack(batch_input_ids, dim=0)
        audio_mask_pairs = torch.stack(batch_audio_mask, dim=0)
        attn_mask_pairs = torch.stack(batch_attn_mask, dim=0)
        batch_input_ids = torch.cat([input_id_pairs[:, 0], input_id_pairs[:, 1]], dim=0)
        batch_audio_mask = torch.cat([audio_mask_pairs[:, 0], audio_mask_pairs[:, 1]], dim=0)
        batch_attn_mask = torch.cat([attn_mask_pairs[:, 0], attn_mask_pairs[:, 1]], dim=0)
        # Run 32-step iterative unmasking
        tokens = self.generator(
            input_ids=batch_input_ids,
            audio_mask=batch_audio_mask,
            attention_mask=batch_attn_mask,
            target_lens=batch_target_len,
            num_step=self.num_step,
            guidance_scale=self.guidance_scale,
            t_shift=self.t_shift,
            layer_penalty_factor=self.layer_penalty_factor,
            position_temperature=self.position_temperature,
            class_temperature=self.class_temperature,
            seed=seed,
        )

        audio = self.decoder(tokens, batch_target_len)  # [B, 1, max_target_len * 960]
        for i in range(len(batch_target_len)):
            audio_output = audio[i : i + 1, :, : batch_target_len[i] * 960]
            outputs.append(DiffusionOutput(output=audio_output))
        return outputs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from model directory (not from the iterator).

        The diffusion model loader passes HF safetensors weights, but OmniVoice
        has custom weight names (llm.* → generator.*, audio_tokenizer.* → decoder.*).
        We load from model_path directly and return all param names to satisfy
        the loader's "all weights initialized" check.
        """
        # Consume the iterator (required by the loader contract)
        for _ in weights:
            pass

        device = self.device
        self.generator.load_weights(self.model_path, device)
        self.generator = self.generator.to(device).eval()
        self.decoder.load_weights(self.model_path, device)
        logger.info("OmniVoice pipeline loaded on %s", device)

        # Return all parameter names to indicate they're initialized
        return {name for name, _ in self.named_parameters()}
