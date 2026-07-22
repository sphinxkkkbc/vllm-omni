import json
import os
from collections.abc import Iterable
from typing import ClassVar

import torch
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .wan_audio_pipeline import WanAudioPipeline

logger = init_logger(__name__)

_DIT_PREFIX = "engine.dit."
_HF_DIT_BLOCK_RENAME = {
    "attn1.norm_k.weight": "self_attn.norm_k.weight",
    "attn1.norm_q.weight": "self_attn.norm_q.weight",
    "attn1.to_k.bias": "self_attn.k.bias",
    "attn1.to_k.weight": "self_attn.k.weight",
    "attn1.to_out.0.bias": "self_attn.o.bias",
    "attn1.to_out.0.weight": "self_attn.o.weight",
    "attn1.to_q.bias": "self_attn.q.bias",
    "attn1.to_q.weight": "self_attn.q.weight",
    "attn1.to_v.bias": "self_attn.v.bias",
    "attn1.to_v.weight": "self_attn.v.weight",
    "attn2.norm_k.weight": "cross_attn.norm_k.weight",
    "attn2.norm_q.weight": "cross_attn.norm_q.weight",
    "attn2.to_k.bias": "cross_attn.k.bias",
    "attn2.to_k.weight": "cross_attn.k.weight",
    "attn2.to_out.0.bias": "cross_attn.o.bias",
    "attn2.to_out.0.weight": "cross_attn.o.weight",
    "attn2.to_q.bias": "cross_attn.q.bias",
    "attn2.to_q.weight": "cross_attn.q.weight",
    "attn2.to_v.bias": "cross_attn.v.bias",
    "attn2.to_v.weight": "cross_attn.v.weight",
    "ffn.net.0.proj.bias": "ffn.0.bias",
    "ffn.net.0.proj.weight": "ffn.0.weight",
    "ffn.net.2.bias": "ffn.2.bias",
    "ffn.net.2.weight": "ffn.2.weight",
    "norm2.bias": "norm3.bias",
    "norm2.weight": "norm3.weight",
    "scale_shift_table": "modulation",
}


_HF_DIT_GLOBAL_RENAME = {
    "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
    "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
    "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
    "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
    "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
    "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
    "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
    "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
    "condition_embedder.time_proj.bias": "time_projection.1.bias",
    "condition_embedder.time_proj.weight": "time_projection.1.weight",
    "scale_shift_table": "head.modulation",
    "proj_out.bias": "head.head.bias",
    "proj_out.weight": "head.head.weight",
    "patch_embedding.bias": "patch_embedding.bias",
    "patch_embedding.weight": "patch_embedding.weight",
}


def _rename_dit_weight(name: str) -> str:
    relative_name = name.removeprefix(_DIT_PREFIX)
    if relative_name in _HF_DIT_GLOBAL_RENAME:
        return f"{_DIT_PREFIX}{_HF_DIT_GLOBAL_RENAME[relative_name]}"

    for source_suffix, target_suffix in _HF_DIT_BLOCK_RENAME.items():
        if relative_name.endswith(source_suffix):
            prefix = relative_name[: -len(source_suffix)]
            return f"{_DIT_PREFIX}{prefix}{target_suffix}"
    return name


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


class MossSoundEffectPipeline(torch.nn.Module):
    r"""Text-to-audio diffusion pipeline (diffusers-style API).

    Wraps :class:`WanAudioPipeline` (DiT + DAC VAE + Qwen3 text encoder +
    flow-match scheduler) and exposes a ``seconds``-oriented call signature
    plus a standard ``from_pretrained`` workflow reading ``model_index.json``.
    """

    support_audio_output: ClassVar[bool] = True
    audio_sample_rate: ClassVar[int] = 48000

    def __init__(
        self,
        od_config: OmniDiffusionConfig | None = None,
        sample_rate: int = 48000,
        max_inference_seconds: int = 30,
    ):
        super().__init__()
        self.od_config = od_config
        model = od_config.model
        if os.path.isdir(model):
            model_root = model
        else:
            from huggingface_hub import snapshot_download

            model_root = snapshot_download(
                repo_id=model,
                revision=od_config.revision,
            )
        self.device = get_local_device()
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="engine.dit.",
                fall_back_to_pt=True,
            ),
        ]
        scheduler_path = os.path.join(model_root, "scheduler/scheduler_config.json")
        with open(scheduler_path) as f:
            sched_cfg = json.load(f)
        self.engine = WanAudioPipeline(model_dir=model_root, flow_shift=sched_cfg.get("shift", 5.0))
        index_path = os.path.join(model_root, "model_index.json")
        with open(index_path) as f:
            index = json.load(f)
        self.sample_rate = int(index.get("sample_rate", sample_rate))
        self.max_inference_seconds = int(index.get("max_inference_seconds", max_inference_seconds))

    @torch.no_grad()
    def forward(
        self,
        prompt: str | list[str],
        seconds: float = 10.0,
        num_inference_steps: int = 10,
        cfg_scale: float = 4.0,
        sigma_shift: float = 5.0,
        seed: int = 0,
        negative_prompt: str = "",
        append_duration_suffix: bool = True,
        num_channels: int = 1,
        max_inference_seconds: int | None = None,
    ) -> DiffusionOutput:
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
            return_dict: If True, return :class:`DiffusionOutput`.
        """
        seconds = round(float(seconds), 1)
        if seconds <= 0:
            raise ValueError(f"seconds must be > 0, got {seconds}")
        full_seconds = int(max_inference_seconds or self.max_inference_seconds)
        if seconds > full_seconds:
            raise ValueError(f"seconds={seconds} exceeds max_inference_seconds={full_seconds}")

        def _format(p: DiffusionRequestBatch) -> str:
            # logger.info(f"p: {p}")
            for _, request in enumerate(p.requests):
                prompt = request.prompt["prompt"]
                p = prompt.strip()
            return f"{p} duration: {seconds:.1f}s" if append_duration_suffix else p

        if isinstance(prompt, (list, tuple)):
            prompts = [_format(p) for p in prompt]
        else:
            prompts = [_format(prompt)]

        num_samples_full = self.sample_rate * full_seconds
        device_type = str(self.device)
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
            )

        output_samples = int(self.sample_rate * seconds)
        audio = audio[:, :, :output_samples]
        return DiffusionOutput(output=audio)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.engine.dit.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            full_param_name = _rename_dit_weight(name)
            param_name = full_param_name.removeprefix(_DIT_PREFIX)
            if param_name not in params_dict:
                continue
            param = params_dict[param_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(full_param_name)
        return loaded_params
