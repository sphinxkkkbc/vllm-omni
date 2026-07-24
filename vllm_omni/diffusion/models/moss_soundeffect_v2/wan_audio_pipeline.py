import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin

from .dac import DAC
from .modeling_wan_audio import WanAudioModel, WanPrompter, sinusoidal_embedding_1d
from .utils import PipelineUnit, PipelineUnitRunner


class Qwen3TextEncoder(nn.Module):
    """Wraps Qwen3 (decoder-only) as a text encoder for Wan audio pipeline.

    Loads the full Qwen3 model and extracts last-layer hidden states
    as text embeddings. Interface matches WanTextEncoder.forward(ids, mask).
    """

    def __init__(self, model_path, dtype=torch.bfloat16):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            output_hidden_states=True,
        )
        self.model.eval()
        self.dim = self.model.config.hidden_size  # 2048 for Qwen3-1.7B

    @torch.no_grad()
    def forward(self, ids, mask=None):
        """
        Args:
            ids:  [batch, seq_len] token ids
            mask: [batch, seq_len] attention mask (1=valid, 0=pad)
        Returns:
            hidden_states: [batch, seq_len, dim] last-layer hidden states
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=ids,
                attention_mask=mask,
                output_hidden_states=True,
                use_cache=False,
            )
        return outputs.hidden_states[-1]


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

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, shift=None, dynamic_shift_len=None):
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


class WanAudioPipeline(nn.Module, ProgressBarMixin):
    def __init__(self, model_dir, device="cuda", dtype=torch.bfloat16, flow_shift=5.0):
        super().__init__()
        self.device = torch.device(device)
        self.torch_dtype = dtype
        self.scheduler = FlowMatchScheduler(shift=flow_shift, sigma_min=0.0, extra_one_step=True)

        te_path = os.path.join(model_dir, "text_encoder")
        self.text_encoder = Qwen3TextEncoder(te_path, dtype=dtype)
        self.text_encoder.to(device)
        tok_path = os.path.join(model_dir, "tokenizer")
        self.prompter = WanPrompter(tokenizer_path=tok_path, text_encoder=self.text_encoder)

        with open(os.path.join(model_dir, "transformer", "config.json")) as f:
            dit_cfg = json.load(f)
        self.dit = WanAudioModel(
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

        self.vae = DAC.load(os.path.join(model_dir, "vae/vae_128d_48k.pth")).to(device)
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanAudioUnit_ShapeChecker(),
            WanAudioUnit_NoiseInitializer(),
            WanAudioUnit_InputAudioEmbedder(),
            WanVideoUnit_PromptEmbedder(),
        ]
        self.audio_latent_dim = dit_cfg["in_dim"]
        self.num_samples_division_factor = self.vae.hop_length

    def to(self, *args, **kwargs):
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        return super().to(*args, **kwargs)

    def generate_noise(
        self,
        shape,
        seed=None,
        rand_device="cpu",
        rand_torch_dtype=torch.float32,
        device=None,
        dtype=None,
    ):
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)
        return noise.to(dtype=dtype or self.torch_dtype, device=device or self.device)

    def check_resize_num_channels_num_samples(self, num_channels, num_samples):
        # Shape check
        self.num_samples_division_factor = np.prod(self.vae.encoder_rates)
        if num_samples % self.num_samples_division_factor != 0:
            num_samples = num_samples // self.num_samples_division_factor * self.num_samples_division_factor
        return num_channels, num_samples

    @torch.no_grad()
    def forward(
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
        num_inference_steps: int | None = 10,
        sigma_shift: float | None = 5.0,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
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

        with self.progress_bar(total=len(self.scheduler.timesteps)) as pbar:
            # Denoise
            for i, timestep in enumerate(self.scheduler.timesteps):
                # Timestep
                timestep = timestep.unsqueeze(0).to(device=self.device)

                # Inference
                torch.compiler.cudagraph_mark_step_begin()
                noise_pred_posi = self._forward(**inputs_shared, **inputs_posi, timestep=timestep)
                if cfg_scale != 1.0:
                    noise_pred_posi = noise_pred_posi.clone()
                    noise_pred_nega = self._forward(**inputs_shared, **inputs_nega, timestep=timestep)
                    noise_pred_posi = noise_pred_posi.float()
                    noise_pred_nega = noise_pred_nega.float()
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                else:
                    noise_pred = noise_pred_posi

                # Scheduler
                inputs_shared["latents"] = self.scheduler.step(
                    noise_pred, self.scheduler.timesteps[i], inputs_shared["latents"]
                )
                if "first_frame_latents" in inputs_shared:
                    inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
                pbar.update()

        # Decode
        latents = inputs_shared["latents"]
        max_decode_bs = 8
        audio_chunks = []
        for start in range(0, latents.size(0), max_decode_bs):
            end = min(start + max_decode_bs, latents.size(0))
            with torch.autocast("cuda", dtype=torch.float32):
                audio_chunk = self.vae.decode(latents[start:end])
            audio_chunks.append(audio_chunk)
        audio = torch.cat(audio_chunks, dim=0)

        return audio

    @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
    def _forward(
        self,
        latents: torch.Tensor = None,
        timestep: torch.Tensor = None,
        context: torch.Tensor = None,
        clip_feature: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        reference_latents=None,
        control_camera_latents_input=None,
        **kwargs,
    ):
        dit = self.dit
        with torch.autocast("cuda", dtype=torch.float32):
            t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
            t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

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

        for block in dit.blocks:
            x = block(x, context, t_mod, freqs)

        x = dit.head(x, t)

        # Remove reference latents
        if reference_latents is not None:
            x = x[:, reference_latents.shape[1] :]
            f -= 1
        x = dit.unpatchify(x, (f,))
        return x


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
            latents = pipe.scheduler.add_noise(latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}

        if input_audio is None:
            return {"latents": noise}
        if input_audio.ndim == 2:
            # add batch dim
            input_audio = input_audio.unsqueeze(0)

        with torch.autocast("cuda", dtype=torch.float32):
            input_latents = pipe.vae.encode(input_audio)[0].mode()
        input_latents = input_latents.to(device=pipe.device)

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
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        return {"context": prompt_emb}
