# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AuK-Flash real HTTP integration; two isolated GPU stages."""

import io
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import requests
import soundfile as sf

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("AUK_MODEL")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.tts,
    pytest.mark.skipif(not MODEL, reason="Set AUK_MODEL to a prepared Flash bundle"),
]


@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize(
    "omni_server",
    [OmniServerParams(model=MODEL or "tencent/AuK-Flash", stage_config_path=get_deploy_config_path("auk.yaml"))],
    indirect=True,
)
def test_online_concurrent_durations(omni_server):
    def generate(duration):
        response = requests.post(
            f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech",
            json={
                "model": omni_server.model,
                "input": "Hello, this is a speech test.",
                "voice": "default",
                "duration_seconds": duration,
                "seed": 7,
                "response_format": "wav",
            },
            timeout=300,
        )
        assert response.status_code == 200, response.text
        waveform, sr = sf.read(io.BytesIO(response.content))
        assert sr == 24000 and np.isfinite(waveform).all() and np.max(np.abs(waveform)) > 0
        return waveform.shape[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(generate, (2.31, 2.55))) == [116 * 480, 128 * 480]
