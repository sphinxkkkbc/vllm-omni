# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import argparse
import base64
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", default="Hello, this is a speech synthesis test.")
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("auk.wav"))
    args = parser.parse_args()
    payload = {
        "model": args.model,
        "input": args.text,
        "duration_seconds": args.duration_seconds,
        "voice": "default",
        "seed": args.seed,
        "response_format": "wav",
    }
    if args.ref_audio:
        payload["ref_audio"] = "data:audio/wav;base64," + base64.b64encode(args.ref_audio.read_bytes()).decode()
    response = requests.post(args.api_base.rstrip("/") + "/v1/audio/speech", json=payload, timeout=300)
    response.raise_for_status()
    args.output.write_bytes(response.content)


if __name__ == "__main__":
    main()
