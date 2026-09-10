# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AuK-Flash offline integration. AUK_MODEL points at a prepared local release."""

import os

import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.model_executor.models.auk.prompt_utils import build_auk_prompt

MODEL = os.environ.get("AUK_MODEL")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.tts,
    pytest.mark.skipif(not MODEL, reason="Set AUK_MODEL to a prepared Flash bundle"),
]


@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize(
    "omni_runner", [(MODEL or "tencent/AuK-Flash", get_deploy_config_path("auk.yaml"))], indirect=True
)
def test_offline_different_lengths(omni_runner):
    prompts = [build_auk_prompt("Hello, this is a speech test.", duration, seed=7) for duration in (2.31, 2.55)]
    outputs = list(omni_runner.omni.generate(prompts))
    assert len(outputs) == len(prompts)
    for output in outputs:
        mm = output.multimodal_output
        assert "auk_step_finished" not in mm
        audio = mm["audio"]
        if isinstance(audio, list):
            audio = torch.cat([x.reshape(-1) for x in audio])
        assert torch.isfinite(audio).all() and audio.abs().max() > 0
        assert audio.numel() in (116 * 480, 128 * 480)
