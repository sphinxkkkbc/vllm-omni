# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline E2E coverage for StepAudioEditX."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from examples.offline_inference.text_to_speech.step_audio_editx import end2end

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MODEL = "stepfun-ai/Step-Audio-EditX"
AUDIO_TOKENIZER = "stepfun-ai/Step-Audio-Tokenizer"
STAGE_CONFIG = "vllm_omni/deploy/step_audio_editx.yaml"
REF_AUDIO = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it!"


def create_dummy_audio(sample_rate: int = 16000, duration_sec: float = 1.0) -> tuple[np.ndarray, int]:
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t), sample_rate


def _args(**overrides):
    base = dict(
        model=MODEL,
        audio_tokenizer=AUDIO_TOKENIZER,
        deploy_config=STAGE_CONFIG,
        edit_type="clone",
        edit_info=None,
        text="Please review the document before we begin.",
        ref_text=REF_TEXT,
        ref_audio=REF_AUDIO,
        output=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_inputs_clone_uses_duration_based_prompt_len() -> None:
    with patch.object(end2end, "estimate_step_audio_editx_prompt_len", return_value=123) as estimate:
        inputs = end2end._build_inputs(_args())

    assert len(inputs) == 1
    assert inputs[0]["prompt_token_ids"] == [0] * 123
    additional_information = inputs[0]["additional_information"]
    assert additional_information == {
        "edit_type": "clone",
        "ref_audio": [REF_AUDIO],
        "ref_text": [REF_TEXT],
        "text": ["Please review the document before we begin."],
    }
    estimate.assert_called_once_with(additional_information, MODEL)


def test_build_inputs_edit_includes_edit_info() -> None:
    with patch.object(end2end, "estimate_step_audio_editx_prompt_len", return_value=77):
        inputs = end2end._build_inputs(
            _args(
                edit_type="emotion",
                edit_info="angry",
                text="Please review the document before we begin.",
            )
        )

    assert inputs[0]["prompt_token_ids"] == [0] * 77
    assert inputs[0]["additional_information"]["edit_type"] == "emotion"
    assert inputs[0]["additional_information"]["edit_info"] == "angry"


def test_build_inputs_requires_ref_text_when_ref_audio_is_explicit() -> None:
    with pytest.raises(ValueError, match="ref_text must be provided"):
        end2end._build_inputs(_args(ref_text=None))


def test_create_dummy_audio_shape() -> None:
    audio, sr = create_dummy_audio(duration_sec=0.25)

    assert sr == 16000
    assert audio.dtype == np.float32
    assert audio.shape == (4000,)


@pytest.mark.advanced_model
def test_offline_step_audio_editx_clone_smoke(tmp_path) -> None:
    """Run real offline StepAudioEditX clone inference."""
    from vllm import SamplingParams

    from vllm_omni.entrypoints.omni import Omni

    output_path = tmp_path / "step_audio_editx.wav"

    args = _args(
        model=MODEL,
        audio_tokenizer=AUDIO_TOKENIZER,
        deploy_config=STAGE_CONFIG,
        output=str(output_path),
    )

    os.environ["STEP_AUDIO_TOKENIZER_PATH"] = AUDIO_TOKENIZER
    omni = Omni(model=MODEL, deploy_config=STAGE_CONFIG, trust_remote_code=True)
    try:
        inputs = end2end._build_inputs(args)
        prompt_len = len(inputs[0]["prompt_token_ids"])
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=max(1, min(2048, 8192 - prompt_len)),
            skip_special_tokens=False,
        )
        outputs = list(omni.generate(inputs, sampling_params_list=[sampling_params, sampling_params]))
    finally:
        omni.close()

    assert outputs


@pytest.mark.advanced_model
def test_offline_step_audio_editx_emotion_smoke(tmp_path) -> None:
    """Run real offline StepAudioEditX edit inference."""
    from vllm import SamplingParams

    from vllm_omni.entrypoints.omni import Omni

    output_path = tmp_path / "step_audio_editx_emotion.wav"
    args = _args(
        edit_type="emotion",
        edit_info="angry",
        model=MODEL,
        audio_tokenizer=AUDIO_TOKENIZER,
        deploy_config=STAGE_CONFIG,
        output=str(output_path),
    )

    os.environ["STEP_AUDIO_TOKENIZER_PATH"] = AUDIO_TOKENIZER
    omni = Omni(model=MODEL, deploy_config=STAGE_CONFIG, trust_remote_code=True)
    try:
        inputs = end2end._build_inputs(args)
        prompt_len = len(inputs[0]["prompt_token_ids"])
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=max(1, min(2048, 8192 - prompt_len)),
            skip_special_tokens=False,
        )
        outputs = list(omni.generate(inputs, sampling_params_list=[sampling_params, sampling_params]))
    finally:
        omni.close()

    assert outputs
