"""Stage input processor for Qwen3-TTS: Talker -> Code2Wav."""

from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import (
    CodesStruct,
    MetaStruct,
    OmniPayload,
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


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    pooling_output: OmniPayload | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())
    request_payload = getattr(transfer_manager, "request_payload", None)
    if request_payload is None:
        request_payload = {}
        transfer_manager.request_payload = request_payload

    if isinstance(pooling_output, dict):
        frame = _extract_last_frame(pooling_output)
        if frame is not None:
            codec_codes = frame.cpu().tolist()
            transfer_manager.code_prompt_token_ids[request_id].append(codec_codes)
        ref_code = pooling_output.get("codes", {}).get("ref")
        if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0 and request_payload.get(request_id) is None:
            request_payload[request_id] = ref_code.to(torch.long).cpu().contiguous()
    elif not finished:
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))
    configured_initial_chunk_size = int(cfg.get("initial_codec_chunk_frames") or 0)
    ref_code_context_frames = int(cfg.get("ref_code_context_frames") or left_context_size_config)

    # Per-request override takes priority over dynamic IC.
    fixed_initial_chunk_size = configured_initial_chunk_size > 0
    initial_chunk_size = configured_initial_chunk_size
    additional_information = getattr(request, "additional_information", None)

    if (
        additional_information is not None
        and hasattr(additional_information, "entries")
        and "initial_codec_chunk_frames" in additional_information.entries
    ):
        entry = additional_information.entries["initial_codec_chunk_frames"]
        if entry.list_data is not None and len(entry.list_data) == 1:
            initial_chunk_size = int(entry.list_data[0])
            fixed_initial_chunk_size = True

    # Dynamic IC: cache per request so boundaries stay stable for its lifetime.
    if not fixed_initial_chunk_size:
        _ic_cache = getattr(transfer_manager, "_cached_ic", None)
        if _ic_cache is None:
            _ic_cache = {}
            transfer_manager._cached_ic = _ic_cache
        if request_id not in _ic_cache:
            max_ic = max_ic_for_chunk_size(chunk_size)
            active = sum(1 for v in transfer_manager.code_prompt_token_ids.values() if len(v) > 0)
            capacity = getattr(transfer_manager, "scheduler_max_num_seqs", 1)
            _ic_cache[request_id] = compute_dynamic_initial_chunk_size(active, capacity, max_ic)
        initial_chunk_size = _ic_cache[request_id]

    if (
        chunk_size <= 0
        or left_context_size_config < 0
        or configured_initial_chunk_size < 0
        or initial_chunk_size < 0
        or ref_code_context_frames < 0
    ):
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size_config}, "
            f"initial_codec_chunk_frames={initial_chunk_size}, "
            f"ref_code_context_frames={ref_code_context_frames}"
        )

    if initial_chunk_size > chunk_size:
        logger.warning(
            "initial_codec_chunk_frames=%d > codec_chunk_frames=%d, clamping to codec_chunk_frames.",
            initial_chunk_size,
            chunk_size,
        )
        initial_chunk_size = chunk_size
    length = len(transfer_manager.code_prompt_token_ids[request_id])

    if length <= 0:
        if finished:
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
            )
        return None

    use_first_chunk = initial_chunk_size > 0 and initial_chunk_size < chunk_size

    if use_first_chunk and length <= initial_chunk_size:
        if not finished and length < initial_chunk_size:
            return None
        context_length = length if finished and length < initial_chunk_size else initial_chunk_size
    else:
        # The initial chunk is only for TTFA. After that, return to the normal
        # codec chunk size so Code2Wav is not flooded by repeated tiny windows.
        initial_coverage = initial_chunk_size if use_first_chunk else 0
        adjusted = length - initial_coverage
        if not finished and adjusted % chunk_size != 0:
            return None
        chunk_length = adjusted % chunk_size
        context_length = chunk_length if chunk_length != 0 else chunk_size

    end_index = min(length, left_context_size_config + context_length)
    left_context_size = max(0, end_index - context_length)
    window_frames = transfer_manager.code_prompt_token_ids[request_id][-end_index:]

    # Prepend a bounded ref_code tail as decoder context for every chunk so the
    # vocoder keeps voice-clone speaker identity without making Stage1 shapes
    # depend on full reference-audio length. The decoder is causal with sliding
    # attention, so frames older than this context window cannot affect the
    # emitted chunk. Use `.get()` (not `.pop()`) to keep ref_code for later chunks.
    ref_code = request_payload.get(request_id)
    if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0:
        ref_context = ref_code
        if ref_code_context_frames > 0 and int(ref_context.shape[0]) > ref_code_context_frames:
            logger.info_once(
                "Qwen3-TTS async chunk uses the last %d/%d ref_code frames as bounded Code2Wav context.",
                ref_code_context_frames,
                int(ref_context.shape[0]),
            )
            ref_context = ref_context[-ref_code_context_frames:]
        ref_frames = ref_context.tolist()
        window_frames = ref_frames + window_frames
        left_context_size += len(ref_frames)

    num_quantizers = len(window_frames[0])
    num_frames = len(window_frames)
    code_predictor_codes = torch.tensor(
        [window_frames[f][q] for q in range(num_quantizers) for f in range(num_frames)],
        dtype=torch.long,
    )

    return OmniPayloadStruct(
        codes=CodesStruct(audio=code_predictor_codes),
        meta=MetaStruct(
            left_context_size=left_context_size,
            finished=torch.tensor(finished, dtype=torch.bool),
        ),
    )


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
