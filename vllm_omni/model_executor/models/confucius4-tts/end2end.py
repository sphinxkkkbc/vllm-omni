import argparse

import torch
import torchaudio

from .pipeline import ConfuciusTTS


def main():
    """CLI entry point for ConfuciusTTS inference."""
    parser = argparse.ArgumentParser(description="ConfuciusTTS zero-shot inference")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--lang", type=str, default="zh")
    parser.add_argument("--prompt_wav", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/inference_config.yaml")
    parser.add_argument("--t2s_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cross_fade_duration", type=float, default=0.3)
    parser.add_argument("--edge_fade_duration", type=float, default=0.1)
    parser.add_argument("--edge_pad_duration", type=float, default=0.1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model = ConfuciusTTS(
        config_path=args.config,
        t2s_checkpoint=args.t2s_checkpoint,
        device=args.device,
    )
    audio = model.generate(
        args.text,
        args.lang,
        args.prompt_wav,
        cross_fade_duration=args.cross_fade_duration,
        edge_fade_duration=args.edge_fade_duration,
        edge_pad_duration=args.edge_pad_duration,
        verbose=args.verbose,
    )
    torchaudio.save(args.output, audio.cpu(), model.sample_rate)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
