import logging
from collections.abc import Iterable

import torch
import torch.nn as nn
import torchaudio
from vllm.config import VllmConfig
from vllm.model_executor.model_loader.weight_utils import maybe_prefix
from vllm.model_executor.models.utils import init_vllm_registered_model

from .step_audio_decoder import StepAudioCode2wav

logger = logging.getLogger(__name__)


# output sample rate : 24000, need to check this
class StepAudioEditxPipeline(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model_stage = vllm_config.model_config.model_stage

        if self.model_stage == "ar_stage":
            ar_vllm_config = vllm_config.with_hf_config(
                vllm_config.model_config.hf_config.ar_config,
                architectures=["StepAudioAR"],
            )
            self.ar_stage = init_vllm_registered_model(
                vllm_config=ar_vllm_config,
                prefix=maybe_prefix(prefix, "ar"),
                hf_config=ar_vllm_config.model_config.hf_config,
                architectures=["StepAudioAR"],
            )
            self.model = self.ar_stage

        elif self.model_stage == "code2wav":
            self.decoder = StepAudioCode2wav(vllm_config=vllm_config, prefix=prefix)
            self.model = self.decoder

    def _load_preprocess_audio(self, prompt_wav: str | torch.Tensor):
        if isinstance(prompt_wav, str):
            prompt_wav, prompt_wav_sr = torchaudio.load(prompt_wav)
        if prompt_wav.shape[0] > 1:
            prompt_wav = prompt_wav.mean(dim=0, keepdim=True)

        # volume-normalize avoid clipping
        norm = torch.max(torch.abs(prompt_wav), dim=1, keepdim=True)[0]
        if norm > 0.6:
            prompt_wav = prompt_wav / norm * 0.6

        return prompt_wav, prompt_wav_sr

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        ar_weights, decoder_weights = [], []
        for name, tensor in weights:
            if name.startswith("decoder."):
                decoder_weights.append((name, tensor))
            else:
                ar_weights.append((name, tensor))

        if self.model_stage == "ar_stage":
            return self.ar_stage.load_weights(ar_weights)
        elif self.model_stage == "decoder":
            return self.decoder.load_weights(decoder_weights)
