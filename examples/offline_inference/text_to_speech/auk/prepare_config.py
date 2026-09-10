# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Add serving metadata to a local official Flash bundle; weights stay unchanged."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", required=True, help="Official YAML filename inside the bundle")
    parser.add_argument("--checkpoint", required=True, help="Official Flash safetensors filename inside the bundle")
    parser.add_argument("--qwen-path", default="Qwen/Qwen2.5-Omni-3B")
    args = parser.parse_args()
    for name in (args.config, args.checkpoint, "vae.safetensors"):
        if not (args.model / name).is_file():
            parser.error(f"Missing {args.model / name}")
    metadata = {
        "model_type": "auk",
        "architectures": ["AukConditionModel"],
        "auk_config": args.config,
        "auk_checkpoint": args.checkpoint,
        "qwen_path": args.qwen_path,
    }
    # Refuse to overwrite pre-existing user metadata.
    with (args.model / "config.json").open("x") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
