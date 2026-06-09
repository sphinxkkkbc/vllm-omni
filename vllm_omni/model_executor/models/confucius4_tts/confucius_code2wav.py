import torch
import torch.nn as nn
import torchaudio
import yaml

from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import mel_spectrogram

from .decoder.bigvgan import BigVGAN
from .decoder.campplus import CAMPPlus
from .decoder.flow import MaskedDiffWithXvec, MaskedDiffWithXvecConfig
from .utils import cross_fade_concat


class Conficius4TTSCode2Wav(nn.Module):
    def __init__(
        self, config_path: str = "config/inference_config.yaml", t2s_checkpoint: str | None = None, device: str = "cuda"
    ):
        super().__init__()
        self.device = torch.device(device)

        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        paths = self.cfg["paths"]
        if t2s_checkpoint is not None:
            paths["t2s_checkpoint"] = t2s_checkpoint
        paths.setdefault("t2s_checkpoint", "checkpoints/model.safetensors")
        s2a_config = MaskedDiffWithXvecConfig(**self.cfg["s2a_model"])
        self.s2a_model = MaskedDiffWithXvec(s2a_config)
        self.s2a_model.load_state_dict(torch.load(paths["s2a_checkpoint"], map_location="cpu", weights_only=False))
        self.s2a_model.eval().to(self.device)

        self.bigvgan = BigVGAN.from_pretrained(paths["vocoder_path"], use_cuda_kernel=False)
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval().to(self.device)

        spk_cfg = paths["style_encoder"]
        self.style_encoder = CAMPPlus(**spk_cfg.get("init_args", {}))
        spk_state = torch.load(spk_cfg["checkpoint"], map_location="cpu")
        if isinstance(spk_state, dict) and "state_dict" in spk_state:
            spk_state = spk_state["state_dict"]
        self.style_encoder.load_state_dict(spk_state, strict=False)
        self.style_encoder.eval().to(self.device)

        self.sample_rate = self.cfg["audio"]["target_sample_rate"]
        self.n_mels = self.cfg["audio"]["n_mels"]
        self.n_fft = self.cfg["audio"]["n_fft"]
        self.hop_length = self.cfg["audio"]["hop_length"]
        self.win_length = self.cfg["audio"]["win_length"]
        self.fmin = self.cfg["audio"]["fmin"]
        self.fmax = self.cfg["audio"]["fmax"]

    def _extract_style(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """Extract speaker style embedding using CAMPPlus encoder.

        Args:
            wav_16k: Waveform at 16kHz, shape (1, T)

        Returns:
            Style embedding, shape (1, D_style)
        """
        fbank = torchaudio.compliance.kaldi.fbank(wav_16k, num_mel_bins=80, sample_frequency=16000, dither=0.0)
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
        return self.style_encoder(fbank.unsqueeze(0).to(self.device))

    def _ref_mel(self, wav_tgt: torch.Tensor) -> torch.Tensor:
        """Extract mel-spectrogram from reference audio for S2A conditioning.

        Args:
            wav_tgt: Waveform at target sample rate, shape (C, T)

        Returns:
            Mel-spectrogram with shape (1, T_mel, n_mels)
        """
        mel = mel_spectrogram(
            wav_tgt.to(self.device).float(),
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        return mel.transpose(1, 2).contiguous()

    def forward(self, tokens, cross_fade_duration=0.1):
        chunks = []
        for i, token in enumerate(tokens):
            audio = self.generate(token)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            chunks.append(audio)

        # Merge segments with cross-fade
        merged = cross_fade_concat(chunks, self.sample_rate, silence_duration=cross_fade_duration)

        return merged

    def generate(self, prompt_wav, t2s_out, n_timesteps=256, inference_cfg_rate=0.5):
        wav_16k, wav_tgt = self._load_prompt(prompt_wav)
        # Extract conditioning from reference audio
        style_embedding = self._extract_style(wav_16k)
        reference_mel = self._ref_mel(wav_tgt)
        semantic_codes = t2s_out["semantic_codes"]  # (B, T_semantic)
        lm_latent = t2s_out["latent"]  # (B, T_semantic, D_hidden)

        # Predict target mel length (heuristic: 1.72x semantic length)
        T = semantic_codes.shape[1]
        target_lengths = torch.tensor([int(T * 1.72)], device=self.device)

        # S2A: Generate mel-spectrogram from semantic tokens
        mel = self.s2a_model.inference(
            semantic_token=semantic_codes,
            lm_latent=lm_latent,
            prompt_feat=reference_mel,
            embedding=style_embedding,
            target_feat_len=target_lengths,
            n_timesteps=n_timesteps,
            inference_cfg_rate=inference_cfg_rate,
        )
        # Vocoder: Mel → Waveform
        return self.bigvgan(mel.float().to(self.device)).squeeze(1)
