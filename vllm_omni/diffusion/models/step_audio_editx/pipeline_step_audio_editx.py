import hashlib
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Iterable, Tuple, Optional
import torchaudio
from vllm_omni.diffusion.models.step_audio_editx.utils import AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL, AUDIO_EDIT_SYSTEM_PROMPT
from vllm import SamplingParams
from vllm_omni.diffusion.models.step_audio_editx.decoder.step_audio_decoder import CosyVoice
from vllm_omni.diffusion.models.step_audio_editx.tokenizer.step_audio_editx_audio_tokenizer import StepAudioTokenizer
from vllm_omni.diffusion.models.step_audio_editx.ar.step_audio_vllm import Step1ForCausalLM
from transformers import AutoTokenizer
from vllm.model_executor.models.utils import AutoWeightsLoader

logger = logging.getLogger(__name__)

class StepAudioEditxPipeline(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()   
        self.audio_tokenizer = StepAudioTokenizer(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(**kwargs)
        self.llm = Step1ForCausalLM(**kwargs)
        self.cosy_model = CosyVoice(**kwargs)
        self.edit_clone_sys_prompt_tpl = AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL
        self.edit_sys_prompt = AUDIO_EDIT_SYSTEM_PROMPT

    def clone(
        self,
        prompt_wav_path: str,
        prompt_text: str,
        target_text: str
    ) -> Tuple[torch.Tensor, int]:
        try:
            logger.debug(f"Starting voice cloning: {prompt_wav_path}")
            vq0206_codes, vq02_codes_ori, vq06_codes_ori, speech_feat, _, speech_embedding = (
                self.preprocess_prompt_wav(prompt_wav_path)
            )
            # prompt_speaker = self.generate_clone_voice_id(prompt_text, prompt_wav)
            prompt_speaker = "debug"
            prompt_wav_tokens = self.audio_tokenizer.merge_vq0206_to_token_str(
                vq02_codes_ori, vq06_codes_ori
            )
            token_ids = self._encode_audio_edit_clone_prompt(
                target_text,
                prompt_text,
                prompt_speaker,
                prompt_wav_tokens,
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
            vq0206_codes, vq02_codes_ori, vq06_codes_ori, speech_feat, _, speech_embedding = (
                self.preprocess_prompt_wav(prompt_wav_path)
            )
            audio_tokens = self.audio_tokenizer.merge_vq0206_to_token_str(
                vq02_codes_ori, vq06_codes_ori
            )
            instruct_prefix = self._build_audio_edit_instruction(prompt_text, edit_type, edit_info, target_text)

            prompt_tokens = self._encode_audio_edit_prompt(
                self.edit_sys_prompt, instruct_prefix, audio_tokens
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

    def _build_audio_edit_instruction(
        self,
        audio_text: str,
        edit_type: str,
        edit_info: Optional[str] = None,
        text: Optional[str] = None
    ) -> str:
        """Build audio editing instruction based on request"""
        audio_text = audio_text.strip() if audio_text else ""
        if edit_type in {"emotion", "speed"}:
            if edit_info == "remove":
                instruct_prefix = f"Remove any emotion in the following audio and the reference text is: {audio_text}\n"
            else:
                instruct_prefix = f"Make the following audio more {edit_info}. The text corresponding to the audio is: {audio_text}\n"
        elif edit_type == "style":
            if edit_info == "remove":
                instruct_prefix = f"Remove any speaking styles in the following audio and the reference text is: {audio_text}\n"
            else:
                instruct_prefix = f"Make the following audio more {edit_info} style. The text corresponding to the audio is: {audio_text}\n"
        elif edit_type == "denoise":
            instruct_prefix = f"Remove any noise from the given audio while preserving the voice content clearly. Ensure that the speech quality remains intact with minimal distortion, and eliminate all noise from the audio.\n"
        elif edit_type == "vad":
            instruct_prefix = f"Remove any silent portions from the given audio while preserving the voice content clearly. Ensure that the speech quality remains intact with minimal distortion, and eliminate all silence from the audio.\n"
        elif edit_type == "paralinguistic":
            instruct_prefix = f"Add some non-verbal sounds to make the audio more natural, the new text is : {text}\n  The text corresponding to the audio is: {audio_text}\n"
        else:
            logger.error(f"Unsupported audio editing type: {edit_type}")
        return instruct_prefix

    def _generate(self, token_ids: list[int], max_tokens: int = 4096, temperature: float = 0.7) -> torch.Tensor:
        audio_in = sum(1 for t in token_ids if 65536 <= t < 67584)
        text_in = sum(1 for t in token_ids if t < 65536)
        other_in = sum(1 for t in token_ids if t >= 67584)
        logger.info(f"INPUT tokens: total={len(token_ids)}, audio(65536-67583)={audio_in}, text(<65536)={text_in}, other(>=67584)={other_in}")
        if token_ids:
            logger.info(f"INPUT range: min={min(token_ids)}, max={max(token_ids)}")
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            skip_special_tokens=False,
        )
        
        # Use prompt_token_ids directly instead of decoding to text
        # This preserves audio tokens (65536+) which would be corrupted by decode
        prompt = {"prompt_token_ids": token_ids}
        outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)

        # Extract output token IDs (vLLM only returns generated tokens, not input)
        output_token_ids = list(outputs[0].outputs[0].token_ids)
        
        # Debug: analyze token distribution
        if output_token_ids:
            min_tok = min(output_token_ids)
            max_tok = max(output_token_ids)
            audio_count = sum(1 for t in output_token_ids if 65536 <= t < 67584)
            text_count = sum(1 for t in output_token_ids if t < 65536)
            other_count = sum(1 for t in output_token_ids if t >= 67584)
            logger.info(f"Generated {len(output_token_ids)} tokens: min={min_tok}, max={max_tok}, "
                       f"audio(65536-67583)={audio_count}, text(<65536)={text_count}, other(>=67584)={other_count}")
        
        # Remove eos token if present
        if len(output_token_ids) > 0 and output_token_ids[-1] == 3: # <|EOT|>
            output_token_ids = output_token_ids[:-1]
        
        output_ids = torch.tensor(output_token_ids, dtype=torch.long)

        return output_ids

    def _encode_audio_edit_prompt(
        self, sys_prompt: str, instruct_prefix: str, audio_token_str: str
    ) -> list[int]:
        """Encode audio edit prompt to token sequence"""
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"{instruct_prefix}\n{audio_token_str}\n"}
        ]

        return self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)


    def _encode_audio_edit_clone_prompt(
        self, text: str, prompt_text: str, prompt_speaker: str, prompt_wav_tokens: str
    ):
        
        sys_prompt = self.edit_clone_sys_prompt_tpl.format(
            speaker=prompt_speaker,
            prompt_text=prompt_text,
            prompt_wav_tokens=prompt_wav_tokens
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"{text}"}
        ]

        return self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    def preprocess_prompt_wav(self, prompt_wav_path: str):
        prompt_wav, prompt_wav_sr = torchaudio.load(prompt_wav_path)
        if prompt_wav.shape[0] > 1:
            prompt_wav = prompt_wav.mean(dim=0, keepdim=True)

        # volume-normalize avoid clipping
        norm = torch.max(torch.abs(prompt_wav), dim=1, keepdim=True)[0]
        if norm > 0.6:
            prompt_wav = prompt_wav / norm * 0.6

        speech_feat, speech_feat_len = self.cosy_model.frontend.extract_speech_feat(
            prompt_wav, prompt_wav_sr
        )
        speech_embedding = self.cosy_model.frontend.extract_spk_embedding(
            prompt_wav, prompt_wav_sr
        )
        vq0206_codes, vq02_codes_ori, vq06_codes_ori = self.audio_tokenizer.wav2token(prompt_wav, prompt_wav_sr)
        return (
            vq0206_codes,
            vq02_codes_ori,
            vq06_codes_ori,
            speech_feat,
            speech_feat_len,
            speech_embedding,
        )

    def generate_clone_voice_id(self, prompt_text, prompt_wav):
        hasher = hashlib.sha256()
        hasher.update(prompt_text.encode('utf-8'))
        wav_data = prompt_wav.cpu().numpy()
        if wav_data.size > 2000:
            audio_sample = np.concatenate([wav_data.flatten()[:1000], wav_data.flatten()[-1000:]])
        else:
            audio_sample = wav_data.flatten()
        hasher.update(audio_sample.tobytes())
        voice_hash = hasher.hexdigest()[:16]
        return f"clone_{voice_hash}"
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)