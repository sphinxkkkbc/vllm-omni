import json
import math
import os

import torch
from diffusers import AutoencoderOobleck
from safetensors.torch import load_file
from tqdm import tqdm
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM as Qwen3TextEncoder

from ..utils import BasePipeline, PipelineUnit, PipelineUnitRunner
from .dac import DAC
from .modeling_wan_audio import WanAudioModel, WanPrompter, sinusoidal_embedding_1d

# Maps diffusers-style keys in the exported HF DiT checkpoint back to the
# native WanAudioModel keys. Paired with the forward direction in
# moss_soundeffect_v2/hf_export.py.
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


def _convert_hf_dit_state_dict(state_dict: dict) -> dict:
    out = {}
    for key, param in state_dict.items():
        if key in _HF_DIT_GLOBAL_RENAME:
            out[_HF_DIT_GLOBAL_RENAME[key]] = param
        elif key.startswith("blocks."):
            parts = key.split(".", 2)
            block_idx, suffix = parts[1], parts[2]
            if suffix in _HF_DIT_BLOCK_RENAME:
                out[f"blocks.{block_idx}.{_HF_DIT_BLOCK_RENAME[suffix]}"] = param
            else:
                out[key] = param
        else:
            out[key] = param
    return out


class FlowMatchScheduler:
    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None, dynamic_shift_len=None
    ):
        if shift is not None:
            self.shift = shift
        # sigma is the noise strength at each step.
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            # Linear schedule from high noise to low noise.
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            mu = self.calculate_shift(dynamic_shift_len) if dynamic_shift_len is not None else self.exponential_shift_mu
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            # Classic flow-match shift formula.
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            # BSMNTW (bell-shaped mid-noise training weighting): give larger
            # weight to mid-range timesteps where the denoising objective is
            # most informative, and smaller weight to the extremes.
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            # Normalize so the weights sum to num_inference_steps (avg ~= 1).
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            # Last step: jump straight to the boundary.
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def return_to_timestep(self, timestep, sample, sample_stabilized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stabilized) / sigma
        return model_output

    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        # [B, len_timesteps] distance matrix.
        dists = (self.timesteps[None, :] - timestep[:, None]).abs()
        # [B] nearest timestep id per sample.
        timestep_ids = dists.argmin(dim=1)
        # [B] noise strength per sample.
        sigmas = self.sigmas[timestep_ids].to(original_samples.device)
        # Reshape for broadcasting to [B, C, T].
        sigmas = sigmas.view(-1, 1, 1)

        # x_t = (1 - sigma) * x_0 + sigma * eps
        sample = (1 - sigmas) * original_samples + sigmas * noise
        return sample

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target

    def training_weight(self, timestep):
        timestep = timestep.to(self.timesteps.device)

        # [B, N] distance matrix.
        dists = (self.timesteps[None, :] - timestep[:, None]).abs()
        # [B] nearest timestep id per sample.
        timestep_ids = dists.argmin(dim=1)

        # [B] per-sample weight.
        weights = self.linear_timesteps_weights[timestep_ids]
        return weights

    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu


class WanAudioPipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None, flow_shift=5.0):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,
            width_division_factor=16,
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler(shift=flow_shift, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder = None
        self.image_encoder = None
        self.dit: WanAudioModel = None
        self.vae: AutoencoderOobleck | DAC = None
        self.motion_controller = None
        self.vace = None
        self.in_iteration_models = ("dit", "motion_controller", "vace")
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanAudioUnit_ShapeChecker(),
            WanAudioUnit_NoiseInitializer(),
            WanAudioUnit_InputAudioEmbedder(),
            WanVideoUnit_PromptEmbedder(),
        ]
        self.model_fn = model_fn_wan_video

    def check_resize_num_channels_num_samples(self, num_channels, num_samples):
        # Shape check
        if num_samples % self.num_samples_division_factor != 0:
            num_samples = num_samples // self.num_samples_division_factor * self.num_samples_division_factor
            # print(f"num_samples % {self.num_samples_division_factor} != 0. We round it down to {num_samples}.")
        return num_channels, num_samples

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "WanAudioPipeline":
        """Load a WanAudioPipeline from a HuggingFace-format directory.

        Expected layout (a diffusers-style HF model directory):
            model_dir/
                model_index.json
                scheduler/scheduler_config.json
                transformer/config.json
                transformer/diffusion_pytorch_model.safetensors
                text_encoder/...        (Qwen3)
                tokenizer/...
                vae/vae_128d_48k.pth    (or diffusion_pytorch_model.safetensors)
        """
        with open(os.path.join(model_dir, "model_index.json")) as f:
            index = json.load(f)
        print(f"Loading from: {model_dir}")
        print(f"  Pipeline: {index['_class_name']}, dit_variant: {index.get('dit_variant')}")

        with open(os.path.join(model_dir, "scheduler", "scheduler_config.json")) as f:
            sched_cfg = json.load(f)
        with open(os.path.join(model_dir, "transformer", "config.json")) as f:
            dit_cfg = json.load(f)

        te_path = os.path.join(model_dir, "text_encoder")
        print(f"  Loading text_encoder from {te_path} ...")
        text_encoder = Qwen3TextEncoder(te_path, torch_dtype=torch_dtype)
        text_encoder = text_encoder.to(device)
        print(f"  text_encoder: dim={text_encoder.dim}")

        tok_path = os.path.join(model_dir, "tokenizer")
        print(f"  Loading tokenizer from {tok_path} ...")
        prompter = WanPrompter(tokenizer_path=tok_path)
        prompter.fetch_models(text_encoder)

        vae_dir = os.path.join(model_dir, "vae")
        vae_pth = os.path.join(vae_dir, "vae_128d_48k.pth")
        vae_safetensors = os.path.join(vae_dir, "diffusion_pytorch_model.safetensors")
        if os.path.exists(vae_pth):
            print(f"  Loading DAC VAE from {vae_pth} ...")
            vae = DAC.load(vae_pth)
        elif os.path.exists(vae_safetensors):
            print(f"  Loading DAC VAE from {vae_safetensors} ...")
            vae = DAC.load(vae_safetensors)
        else:
            raise FileNotFoundError(f"No VAE found in {vae_dir}")

        dit_weights_path = os.path.join(model_dir, "transformer", "diffusion_pytorch_model.safetensors")
        print(f"  Loading DiT from {dit_weights_path} ...")
        diffusers_sd = load_file(dit_weights_path)
        custom_sd = _convert_hf_dit_state_dict(diffusers_sd)

        dit = WanAudioModel(
            in_dim=dit_cfg["in_dim"],
            out_dim=dit_cfg["out_dim"],
            text_dim=dit_cfg["text_dim"],
            freq_dim=dit_cfg["freq_dim"],
            eps=dit_cfg["eps"],
            patch_size=tuple(dit_cfg["patch_size"]),
            has_image_input=dit_cfg["has_image_input"],
            dim=dit_cfg["dim"],
            ffn_dim=dit_cfg["ffn_dim"],
            num_heads=dit_cfg["num_heads"],
            num_layers=dit_cfg["num_layers"],
            vae_type=dit_cfg.get("vae_type", "dac"),
        )
        load_result = dit.load_state_dict(custom_sd)
        print(f"  DiT loaded: missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}")

        pipe = cls(
            device=device,
            torch_dtype=torch_dtype,
            flow_shift=sched_cfg.get("shift", 5.0),
        )
        pipe.text_encoder = text_encoder
        pipe.prompter = prompter
        pipe.vae = vae
        pipe.dit = dit
        pipe.audio_latent_dim = dit_cfg["in_dim"]
        pipe.num_samples_division_factor = vae.hop_length
        pipe.dit_variant = index.get("dit_variant")
        pipe.to(device)
        print(f"  Pipeline assembled on {device}")
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = "",
        denoising_strength: float | None = 1.0,
        seed: int | None = None,
        rand_device: str | None = "cpu",
        num_samples=44100 * 10,
        num_channels=2,
        cfg_scale: float | None = 5.0,
        cfg_merge: bool | None = False,
        switch_dit_boundary: float | None = 0.875,
        num_inference_steps: int | None = 50,
        sigma_shift: float | None = 5.0,
        tea_cache_l1_thresh: float | None = None,
        tea_cache_model_id: str | None = "",
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        # print(f"{self.scheduler.timesteps = }")

        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        # Infer batch size; prompt may be a list[str].
        computed_batch_size = len(prompt) if isinstance(prompt, (list, tuple)) else 1
        # For batched input, broadcast a single negative_prompt to the same length.
        if computed_batch_size > 1 and not isinstance(negative_prompt, (list, tuple)):
            inputs_nega["negative_prompt"] = [negative_prompt] * computed_batch_size

        inputs_shared = {
            "num_samples": num_samples,
            "num_channels": num_channels,
            "denoising_strength": denoising_strength,
            "seed": seed,
            "rand_device": rand_device,
            "cfg_scale": cfg_scale,
            "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "batch_size": computed_batch_size,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if (
                timestep.item() < switch_dit_boundary * self.scheduler.num_train_timesteps
                and self.dit2 is not None
                and models["dit"] is not self.dit2
            ):
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2

            # Timestep
            timestep = timestep.unsqueeze(0).to(device=self.device)

            # Inference
            torch.compiler.cudagraph_mark_step_begin()
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                noise_pred_posi = noise_pred_posi.clone()
                noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred_posi = noise_pred_posi.float()
                noise_pred_nega = noise_pred_nega.float()
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(
                noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"]
            )
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

        # Decode
        self.load_models_to_device(["vae"])
        latents = inputs_shared["latents"]
        max_decode_bs = 8
        audio_chunks = []
        for start in range(0, latents.size(0), max_decode_bs):
            end = min(start + max_decode_bs, latents.size(0))
            with torch.autocast("cuda", dtype=torch.float32):
                if isinstance(self.vae, DAC):
                    audio_chunk = self.vae.decode(latents[start:end])
                else:
                    audio_chunk = self.vae.decode(latents[start:end]).sample
            audio_chunks.append(audio_chunk)
        audio = torch.cat(audio_chunks, dim=0)
        # video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        return audio


class WanAudioUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("num_channels", "num_samples"))

    def process(self, pipe: WanAudioPipeline, num_channels, num_samples):
        num_channels, num_samples = pipe.check_resize_num_channels_num_samples(num_channels, num_samples)
        return {"num_channels": num_channels, "num_samples": num_samples}


class WanAudioUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("input_audio", "num_samples", "seed", "rand_device", "batch_size"))

    def process(self, pipe: WanAudioPipeline, input_audio, num_samples, seed, rand_device, batch_size):
        if input_audio is not None:
            bsz = input_audio.size(0) if input_audio.ndim == 3 else 1
        else:
            bsz = batch_size if batch_size is not None else 1
        shape = (bsz, pipe.audio_latent_dim, num_samples // pipe.num_samples_division_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}


class WanAudioUnit_InputAudioEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_audio",
                "audio_latent",
                "noise",
                "tiled",
                "tile_size",
                "tile_stride",
                "vace_reference_image",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanAudioPipeline,
        input_audio,
        audio_latent,
        noise,
        tiled,
        tile_size,
        tile_stride,
        vace_reference_image,
    ):
        # Pass-through branch when an audio_latent is provided directly.
        if audio_latent is not None:
            latents = audio_latent
            if latents.ndim == 2:
                latents = latents.unsqueeze(0)
            latents = latents.to(dtype=pipe.torch_dtype, device=pipe.device)
            if pipe.scheduler.training:
                return {"latents": noise, "input_latents": latents}
            else:
                latents = pipe.scheduler.add_noise(latents, noise, timestep=pipe.scheduler.timesteps[0])
                return {"latents": latents}

        if input_audio is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        if input_audio.ndim == 2:
            # add batch dim
            input_audio = input_audio.unsqueeze(0)
        # print(f"{input_audio.shape = }")
        # from time import perf_counter
        # start_time = perf_counter()
        with torch.autocast("cuda", dtype=torch.float32):
            if isinstance(pipe.vae, DAC):
                input_latents = pipe.vae.encode(input_audio)[0].mode()
            else:
                input_latents = pipe.vae.encode(input_audio).latent_dist.mode()
        input_latents = input_latents.to(device=pipe.device)
        # print(f"{input_latents.mean() = }, {input_latents.std() = }")
        # end_time = perf_counter()
        # print(f"vae.encode time taken: {end_time - start_time} seconds")
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}


class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            separate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",),
        )

    def process(self, pipe: WanAudioPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        # from time import perf_counter
        # start_time = perf_counter()
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        # end_time = perf_counter()
        # print(f"prompter.encode_prompt time taken: {end_time - start_time} seconds")
        return {"context": prompt_emb}


@torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def model_fn_wan_video(
    dit: WanAudioModel,
    vace=None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
    reference_latents=None,
    vace_context=None,
    vace_scale=1.0,
    control_camera_latents_input=None,
):
    with torch.autocast("cuda", dtype=torch.float32):
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        # print(f"{t.shape = }")
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
        # print(f"{t_mod.shape = }")

    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embedding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embedding, context], dim=1)

    # Add camera control
    x, (f,) = dit.patchify(x, control_camera_latents_input)
    # print(f"{f = }")

    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1

    # freqs is now a registered buffer (moves with model.to(device)). Do not
    # write Python attributes here so torch.compile can trace through.
    audio_freqs = dit.freqs
    freqs = torch.cat(
        [
            audio_freqs[0][:f].view(f, -1).expand(f, -1),
            audio_freqs[1][:f].view(f, -1).expand(f, -1),
            audio_freqs[2][:f].view(f, -1).expand(f, -1),
        ],
        dim=-1,
    ).reshape(f, 1, -1)

    if vace_context is not None:
        vace_hints = vace(x, vace_context, context, t_mod, freqs)

    for block_id, block in enumerate(dit.blocks):
        x = block(x, context, t_mod, freqs)
        if vace_context is not None and block_id in vace.vace_layers_mapping:
            current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
            x = x + current_vace_hint * vace_scale

    x = dit.head(x, t)

    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1] :]
        f -= 1
    x = dit.unpatchify(x, (f,))
    return x
