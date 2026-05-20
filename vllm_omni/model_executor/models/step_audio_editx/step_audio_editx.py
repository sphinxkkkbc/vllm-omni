import logging
import torch
import torch.nn as nn
from typing import Tuple, Optional
import torchaudio
from .step_audio_decoder import CosyVoice
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.config import VllmConfig
from vllm.model_executor.models.utils import init_vllm_registered_model
from vllm.model_executor.model_loader.weight_utils import maybe_prefix

logger = logging.getLogger(__name__)

class StepAudioEditxPipeline(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()   
        self.model_stage = vllm_config.model_config.model_stage

        if self.model_stage == "ar_codec":
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
            self.decoder = CosyVoice(vllm_config=vllm_config, prefix=prefix)
            self.model = self.decoder

    def clone(
        self,
        prompt_wav_path: str,
        prompt_text: str,
        target_text: str
    ) -> Tuple[torch.Tensor, int]:
        try:
            logger.debug(f"Starting voice cloning: {prompt_wav_path}")
            vq0206_codes, audio_tokens, speech_feat, _, speech_embedding = (
                self.preprocess_prompt_wav(prompt_wav_path)
            )
            # prompt_speaker = self.generate_clone_voice_id(prompt_text, prompt_wav)
            prompt_speaker = "debug"
            token_ids = self._encode_audio_edit_clone_prompt(
                target_text,
                prompt_text,
                prompt_speaker,
                audio_tokens,
            )

            output_ids = self._generate(token_ids, max_tokens=8192 - len(token_ids))
            logger.debug("Voice cloning generation completed")
            vq0206_codes_vocoder = torch.tensor([vq0206_codes], dtype=torch.long) - 65536
            return (
                self.cosy_model.token2wav_nonstream(
                    output_ids - 65536,
                    vq0206_codes_vocoder,
                    speech_feat.to(torch.bfloat16),
                    speech_embedding.to(torch.bfloat16),
                ),
                24000,
            )
        except Exception as e:
            logger.error(f"Clone failed: {e}")
            raise

    def edit(
        self,
        prompt_wav_path: str,
        prompt_text: str,
        edit_type: str,
        edit_info: Optional[str] = None,
        target_text: Optional[str] = None
    ) -> Tuple[torch.Tensor, int]:
        try:
            logger.debug(f"Starting audio editing: {edit_type} - {edit_info}")
            vq0206_codes, audio_tokens, speech_feat, _, speech_embedding = (
                self.preprocess_prompt_wav(prompt_wav_path)
            )
            instruct_prefix = self._build_audio_edit_instruction(prompt_text, edit_type, edit_info, target_text)

            prompt_tokens = self._encode_audio_edit_prompt(
                self.edit_sys_prompt,
                instruct_prefix, 
                audio_tokens
            )

            logger.debug(f"Edit instruction: {instruct_prefix}")
            logger.debug(f"Encoded prompt length: {len(prompt_tokens)}")

            output_ids = self._generate(prompt_tokens, max_tokens=8192 - len(prompt_tokens))
            vq0206_codes_vocoder = torch.tensor([vq0206_codes], dtype=torch.long) - 65536
            logger.debug("Audio editing generation completed")
            return (
                self.cosy_model.token2wav_nonstream(
                    output_ids - 65536,
                    vq0206_codes_vocoder,
                    speech_feat.to(torch.bfloat16),
                    speech_embedding.to(torch.bfloat16),
                ),
                24000,
            )
        except Exception as e:
            logger.error(f"Edit failed: {e}")
            raise

    def preprocess_prompt_wav(self, prompt_wav_path: str):
        prompt_wav, prompt_wav_sr = torchaudio.load(prompt_wav_path)
        if prompt_wav.shape[0] > 1:
            prompt_wav = prompt_wav.mean(dim=0, keepdim=True)

        # volume-normalize avoid clipping
        norm = torch.max(torch.abs(prompt_wav), dim=1, keepdim=True)[0]
        if norm > 0.6:
            prompt_wav = prompt_wav / norm * 0.6

        speech_feat = self.cosy_model.frontend.extract_speech_feat(
            prompt_wav, prompt_wav_sr
        )
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.long)
        speech_embedding = self.cosy_model.frontend.extract_spk_embedding(
            prompt_wav, prompt_wav_sr
        )
        vq0206_codes, vq02_codes_ori, vq06_codes_ori = self.audio_tokenizer.wav2token(prompt_wav, prompt_wav_sr)
        audio_tokens = self.audio_tokenizer.merge_vq0206_to_token_str(vq02_codes_ori, vq06_codes_ori)
        return (
            vq0206_codes,
            audio_tokens,
            speech_feat,
            speech_feat_len,
            speech_embedding,
        )
