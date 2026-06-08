"""Stage input processor for Qwen3-TTS: Talker -> Code2Wav."""

from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import (
    CodesStruct,
    MetaStruct,
    OmniPayloadStruct,
    to_dict,
)

logger = init_logger(__name__)


def ar2decoder(source_outputs: list[Any], prompt: Any = None, _requires_multimodal_data: bool = False):
    from vllm_omni.inputs.data import OmniTokensPrompt

    talker_outputs = source_outputs
    logger.info(f"talker_outputs: {talker_outputs}")
    additional_information = prompt.get("additional_information") or {}
    if additional_information:
        if additional_information.get("ref_audio"):
            ref_audio = additional_information["ref_audio"]
        else:
            ref_audio = None
    code2wav_inputs: list[OmniTokensPrompt] = []
    for i, talker_output in enumerate(talker_outputs):
        if not talker_output.finished:
            # Non-async decode should only run once, after talker has
            # accumulated the final code sequence.
            continue
        output = talker_output.outputs[0]
        mm = output.multimodal_output
        token_ids = output.token_ids
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()

        codec_codes = [int(tid) - 65536 for tid in token_ids if int(tid) >= 65536]
        mm_codes = mm.get("codes", {})
        ref_code = mm_codes.get("ref")
        if isinstance(ref_code, list):
            ref_code = [i - 65536 for i in ref_code]
        else:
            ref_code = ref_code - 65536
        ref_code_len = mm.get("meta", {}).get("ref_code_len")
        if isinstance(ref_code_len, torch.Tensor):
            ref_code_len = int(ref_code_len.reshape(-1)[-1].item()) if ref_code_len.numel() > 0 else 0
        elif ref_code_len is None:
            ref_code_len = 0
        else:
            ref_code_len = int(ref_code_len)
        if isinstance(ref_code, list):
            ref_code = ref_code[0] if ref_code else None
        if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0:
            ref_code = ref_code.to(torch.long).cpu().contiguous()
        additional_information = to_dict(
            OmniPayloadStruct(
                meta=MetaStruct(left_context_size=ref_code_len) if ref_code_len > 0 else None,
                codes=CodesStruct(ref=ref_code, audio=ref_audio),
            )
        )
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=additional_information if additional_information else None,
            )
        )
    return code2wav_inputs


def talker2code2wav_async_chunk(payload: OmniPayloadStruct):
    pass


def _setup_cache(
    self,
    token: torch.Tensor,
    mel: torch.Tensor,
    spk: torch.Tensor,
    session_id: str,
):
    with self.setup_lock:
        cache = self.flow.setup_cache(
            token.to(self.device),
            mel.to(self.device, self.dtype),
            spk.to(self.device, self.dtype),
            self.n_timesteps,
        )
        cache = {k: (v.clone().detach() if isinstance(v, torch.Tensor) else v) for k, v in cache.items()}
        self.chunk_cache_dict[session_id] = cache
        self.estimator_prompt_length_dict[session_id] = mel.shape[1]
        self.b_first_chunk_dict[session_id] = True
        self.spk_embedding_cache_dict[session_id] = spk.to(self.device, self.dtype).clone()
        self.hift_cache_dict[session_id] = dict(
            mel=torch.zeros(1, mel.shape[2], 0, device=self.device, dtype=self.dtype),
            source=torch.zeros(1, 1, 0, device=self.device, dtype=self.dtype),
            speech=torch.zeros(1, 0, device=self.device, dtype=self.dtype),
        )


def clean_up(self, session_id: str):
    self.speech_token_dict.pop(session_id, None)
    self.chunk_size_dict.pop(session_id, None)
    self.b_first_chunk_dict.pop(session_id, None)
    self.hift_cache_dict.pop(session_id, None)
    self.chunk_cache_dict.pop(session_id, None)
    self.estimator_prompt_length_dict.pop(session_id, None)
    self.spk_embedding_cache_dict.pop(session_id, None)


def fade_in_out(fade_in_mel: torch.Tensor, fade_out_mel: torch.Tensor, window: torch.Tensor):
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel = fade_in_mel.clone()
    fade_in_mel[..., :mel_overlap_len] = (
        fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len]
        + fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    )
    return fade_in_mel


def chunk_decode_streaming(
    self,
    token: list[int],
    prompt_token: torch.Tensor,
    prompt_feat: torch.Tensor,
    embedding: torch.Tensor,
    session_id: str,
    last_chunk: bool,
) -> torch.Tensor | None:
    def _mixed_len(length: int):
        return (length // 3) * 5

    if session_id not in self.chunk_size_dict:
        self.chunk_size_dict[session_id] = deepcopy(self.chunk_size_list)
    self.speech_token_dict[session_id].extend(token)
    mix_token_lookahead_len = _mixed_len(self.token_lookahead)
    if session_id not in self.chunk_cache_dict:
        if len(self.speech_token_dict[session_id]) >= mix_token_lookahead_len:
            lookahead_token = self._reshape(self.speech_token_dict[session_id][:mix_token_lookahead_len]).unsqueeze(0)
            prompt_token = self._reshape(prompt_token.squeeze().tolist()).unsqueeze(0)
            prompt_feat = F.interpolate(
                prompt_feat.transpose(1, 2), size=prompt_token.shape[1] * 2, mode="nearest"
            ).transpose(1, 2)
            self._setup_cache(
                torch.cat([prompt_token, lookahead_token], dim=1),
                prompt_feat,
                embedding,
                session_id,
            )
        return None

    if last_chunk:
        this_token = self.speech_token_dict[session_id]
    else:
        this_token = None
        mix_token_chunk_len = _mixed_len(self.chunk_size_dict[session_id][0])
        if len(self.speech_token_dict[session_id]) >= (mix_token_chunk_len + mix_token_lookahead_len):
            this_token = self.speech_token_dict[session_id][: (mix_token_chunk_len + mix_token_lookahead_len)]
            self.speech_token_dict[session_id] = self.speech_token_dict[session_id][mix_token_chunk_len:]
    if this_token is not None:
        this_token = self._reshape(this_token).unsqueeze(0)
        this_speech = self._token2wav_stream(
            this_token,
            session_id,
            last_chunk,
        )
        if len(self.chunk_size_dict[session_id]) > 1:
            self.chunk_size_dict[session_id].pop(0)
    else:
        this_speech = None
    if last_chunk:
        self.clean_up(session_id)
    return this_speech


def _token2wav_stream(
    self,
    token: torch.Tensor,
    session_id: str,
    last_chunk: bool,
):
    assert session_id in self.chunk_cache_dict, "call setup_cache first to obtain cache"
    cache = self.chunk_cache_dict[session_id]
    embedding = self.spk_embedding_cache_dict[session_id]
    mel, new_cache = self.flow.inference_chunk(
        token.to(self.device),
        embedding,
        cache,
        last_chunk,
        self.n_timesteps,
    )
    left_context_length = int(2 * 48)
    estimator_att_cache = new_cache["estimator_att_cache"]
    prompt_length = self.estimator_prompt_length_dict[session_id]
    if estimator_att_cache.shape[4] > (prompt_length + left_context_length):
        new_cache["estimator_att_cache"] = torch.cat(
            [
                estimator_att_cache[:, :, :, :, :left_context_length],
                estimator_att_cache[:, :, :, :, -prompt_length:],
            ],
            dim=4,
        )

    self.chunk_cache_dict[session_id] = {k: v.clone().detach() for k, v in new_cache.items()}
    hift_cache_mel = self.hift_cache_dict[session_id]["mel"]
    hift_cache_source = self.hift_cache_dict[session_id]["source"]
    hift_cache_speech = self.hift_cache_dict[session_id]["speech"]
    mel = torch.concat([hift_cache_mel, mel], dim=2)
    speech, source = self.hift.inference(mel, hift_cache_source)
    if hift_cache_speech.shape[-1] > 0:
        speech = fade_in_out(speech, hift_cache_speech, self.speech_window)
    self.hift_cache_dict[session_id] = dict(
        mel=mel[..., -self.mel_cache_len :].clone().detach(),
        source=source[:, :, -self.source_cache_len :].clone().detach(),
        speech=speech[:, -self.source_cache_len :].clone().detach(),
    )
    if not last_chunk:
        speech = speech[:, : -self.source_cache_len]
    return speech.cpu().to(torch.float32)
