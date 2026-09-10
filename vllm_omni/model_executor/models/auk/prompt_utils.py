# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prompt utilities for AuK model."""

import math

import torch

QWEN_AUDIO_PLACEHOLDER = "Audio 0: <|audio_bos|><|AUDIO|><|audio_eos|>"


def target_frames(seconds: float) -> int:
    """Convert duration in seconds to target frames."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("duration must be finite and positive")
    return math.ceil(seconds * 50)


def build_auk_prompt(
    text: str,
    duration_seconds: float | None,
    *,
    instructions: str | None = None,
    reference_audio: tuple[torch.Tensor, int] | None = None,
    seed: int | None = None,
) -> dict:
    """Build AuK prompt for multimodal processing.

    Args:
        text: Input text to synthesize
        duration_seconds: Target duration in seconds
        instructions: Optional instructions (for future use)
        reference_audio: Optional reference audio tensor and sample rate
        seed: Random seed for generation

    Returns:
        Formatted prompt dictionary for AuK model
    """
    # Prepare request data
    if instructions:
        instruction_text = instructions
    else:
        instruction_text = "请根据以下文本生成语音"

    prompt_text = instruction_text + "\n" + text
    metadata = {
        "duration_seconds": duration_seconds,
        "seed": seed if seed is not None else 42,
    }
    if reference_audio is not None:
        waveform, sample_rate = reference_audio
        waveform = torch.as_tensor(waveform, dtype=torch.float32)
        if waveform.ndim == 2 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
        if waveform.ndim != 1:
            raise ValueError(f"AuK reference audio must be mono, got shape {tuple(waveform.shape)}")
        # The MM processor consumes this copy through Qwen's audio tower.
        # Keep the original waveform separately for AuK VAE encoding.
        metadata["reference_waveform"] = waveform.tolist()
        metadata["reference_sample_rate"] = sample_rate
        return {
            # Qwen's MM processor replaces the <|AUDIO|> token with one token
            # per extracted audio feature. The item and placeholder counts must
            # match exactly.
            "prompt": QWEN_AUDIO_PLACEHOLDER + "\n" + prompt_text,
            "multi_modal_data": {"audio": (waveform, sample_rate)},
            "model_intermediate_buffer": metadata,
        }
    prompt_text += "|<no_prompt_audio>|"
    prompt = {"prompt": prompt_text, "model_intermediate_buffer": metadata}
    return prompt


def format_auk_request(
    text: str,
    duration_seconds: float,
    *,
    instructions: str | None = None,
    reference_audio: tuple[torch.Tensor, int] | None = None,
    seed: int | None = None,
) -> dict:
    """Format AuK request for direct model input.

    This provides the raw request format expected by AuK models.
    """
    return build_auk_prompt(
        text,
        duration_seconds,
        instructions=instructions,
        reference_audio=reference_audio,
        seed=seed,
    )
