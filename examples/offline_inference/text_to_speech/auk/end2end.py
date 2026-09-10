# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run the same AuK request contract as /v1/audio/speech."""

import argparse

import soundfile as sf
import torch

from vllm_omni import Omni
from vllm_omni.model_executor.models.auk.prompt_utils import build_auk_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", default="Hello, this is a speech synthesis test.")
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--ref-audio")
    parser.add_argument("--instructions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="auk.wav")
    parser.add_argument("--deploy-config", default="vllm_omni/deploy/auk.yaml")
    args = parser.parse_args()
    reference = None
    if args.ref_audio:
        waveform, sr = sf.read(args.ref_audio, always_2d=True, dtype="float32")
        reference = (waveform.T, sr)
    prompt = build_auk_prompt(
        args.text, args.duration_seconds, instructions=args.instructions, reference_audio=reference, seed=args.seed
    )
    omni = Omni(model=args.model, stage_configs_path=args.deploy_config)
    try:
        for result in omni.generate(prompt):
            audio = result.multimodal_output["audio"]
            if isinstance(audio, list):
                audio = torch.cat([x.reshape(-1) for x in audio])
            sf.write(args.output, audio.detach().cpu().numpy().reshape(-1), 24000)
    finally:
        omni.close()


if __name__ == "__main__":
    main()
