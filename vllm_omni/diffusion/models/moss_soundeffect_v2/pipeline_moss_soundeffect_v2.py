import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from vllm_omni.diffusion.data import OmniDiffusionConfig

from .wan_audio_pipeline import WanAudioPipeline


def get_moss_soundeffect_post_process_func(od_config: OmniDiffusionConfig):
    def post_process_func(
        audio: torch.Tensor,
        output_type: str = "np",
    ):
        if output_type == "latent":
            return audio
        if output_type == "pt":
            return audio
        # Convert to numpy
        audio_np = audio.cpu().float().numpy()
        return audio_np

    return post_process_func


@dataclass
class MossSoundEffectPipelineOutput:
    """Container returned by :meth:`MossSoundEffectPipeline.__call__`."""

    audios: torch.Tensor
    sample_rate: int
    prompts: list[str]


class MossSoundEffectPipeline(torch.nn.Module):
    r"""Text-to-audio diffusion pipeline (diffusers-style API).

    Wraps :class:`WanAudioPipeline` (DiT + DAC VAE + Qwen3 text encoder +
    flow-match scheduler) and exposes a ``seconds``-oriented call signature
    plus a standard ``from_pretrained`` workflow reading ``model_index.json``.
    """

    def __init__(
        self,
        engine: WanAudioPipeline,
        sample_rate: int = 48000,
        max_inference_seconds: int = 30,
    ):
        super().__init__()
        self.engine = engine
        self.sample_rate = int(sample_rate)
        self.max_inference_seconds = int(max_inference_seconds)

    @staticmethod
    def _normalize_device(device: str | torch.device) -> str:
        requested = str(device)
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            print(
                f"[Warning] Requested device '{requested}' but CUDA is unavailable. Falling back to CPU.",
                flush=True,
            )
            return "cpu"
        return requested

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
        **kwargs,
    ) -> "MossSoundEffectPipeline":
        """Load a pipeline from a local diffusers-style dir or a HF hub repo id."""
        model_dir = Path(cls._resolve_local_dir(pretrained_model_name_or_path, kwargs))
        resolved_device = cls._normalize_device(device)

        engine = WanAudioPipeline.from_pretrained(
            str(model_dir),
            device=resolved_device,
            torch_dtype=torch_dtype,
        )

        sample_rate = 48000
        max_inference_seconds = 30
        index_path = model_dir / "model_index.json"
        if index_path.is_file():
            with open(index_path) as f:
                index = json.load(f)
            sample_rate = int(index.get("sample_rate", sample_rate))
            max_inference_seconds = int(index.get("max_inference_seconds", max_inference_seconds))

        return cls(
            engine=engine,
            sample_rate=sample_rate,
            max_inference_seconds=max_inference_seconds,
        )

    @staticmethod
    def _resolve_local_dir(
        pretrained_model_name_or_path: str | os.PathLike,
        kwargs: dict,
    ) -> str:
        path = str(pretrained_model_name_or_path)
        if os.path.isdir(path):
            return path
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=path,
            cache_dir=kwargs.get("cache_dir"),
            revision=kwargs.get("revision"),
            token=kwargs.get("token"),
            local_files_only=kwargs.get("local_files_only", False),
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        seconds: float = 10.0,
        num_inference_steps: int = 100,
        cfg_scale: float = 4.0,
        sigma_shift: float = 5.0,
        seed: int = 0,
        negative_prompt: str = "",
        append_duration_suffix: bool = True,
        num_channels: int = 1,
        max_inference_seconds: int | None = None,
        return_dict: bool = False,
        progress_bar_cmd=tqdm,
    ) -> torch.Tensor | MossSoundEffectPipelineOutput:
        """Run denoising and return a waveform of shape ``(B, C, T)``.

        Args:
            prompt: A single prompt or a batch of prompts.
            seconds: Output duration. The pipeline always denoises a fixed-size
                latent (``max_inference_seconds`` seconds) and the returned
                tensor is cropped to ``seconds`` worth of samples.
            num_inference_steps: Number of diffusion solver steps.
            cfg_scale: Classifier-free guidance weight.
            sigma_shift: Flow-match shift override applied to the scheduler.
            seed: RNG seed for the noise initializer.
            negative_prompt: CFG negative prompt.
            append_duration_suffix: If True, append ``" duration: <X>s"`` to
                each prompt (matches training-time convention).
            num_channels: Output channels (DAC is mono → 1).
            max_inference_seconds: Override of the configured upper bound.
            return_dict: If True, return :class:`MossSoundEffectPipelineOutput`.
        """
        seconds = round(float(seconds), 1)
        if seconds <= 0:
            raise ValueError(f"seconds must be > 0, got {seconds}")
        full_seconds = int(max_inference_seconds or self.max_inference_seconds)
        if seconds > full_seconds:
            raise ValueError(f"seconds={seconds} exceeds max_inference_seconds={full_seconds}")

        def _format(p: str) -> str:
            p = p.strip()
            return f"{p} duration: {seconds:.1f}s" if append_duration_suffix else p

        if isinstance(prompt, (list, tuple)):
            prompts = [_format(p) for p in prompt]
        else:
            prompts = [_format(prompt)]

        num_samples_full = self.sample_rate * full_seconds
        device_type = self.device.type
        with torch.autocast(device_type, dtype=torch.bfloat16):
            audio = self.engine(
                prompt=prompts if len(prompts) > 1 else prompts[0],
                negative_prompt=negative_prompt,
                seed=int(seed),
                cfg_scale=float(cfg_scale),
                sigma_shift=float(sigma_shift),
                num_inference_steps=int(num_inference_steps),
                num_samples=num_samples_full,
                num_channels=int(num_channels),
                progress_bar_cmd=progress_bar_cmd,
            )

        output_samples = int(self.sample_rate * seconds)
        audio = audio[:, :, :output_samples]

        if return_dict:
            return MossSoundEffectPipelineOutput(
                audios=audio,
                sample_rate=self.sample_rate,
                prompts=prompts,
            )
        return audio

    def save_audio(
        self,
        audio: torch.Tensor,
        output_path: str | Path,
        sample_rate: int | None = None,
    ) -> str:
        import torchaudio

        sr = int(sample_rate or self.sample_rate)
        wav = audio.detach().cpu()
        if wav.ndim == 3:
            wav = wav[0]
        elif wav.ndim == 1:
            wav = wav.unsqueeze(0)
        wav = wav.to(torch.float32)
        output_path = str(Path(output_path).expanduser().resolve())
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(output_path, wav, sr)
        return output_path
