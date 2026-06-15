"""Stage input processor for Step-Audio-EditX: AR -> Code2Wav."""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import CodesStruct, IdsStruct, MetaStruct, OmniPayload, OmniPayloadStruct, to_dict

logger = init_logger(__name__)


def _payload_get(payload: Any, key: str, default: Any = None) -> Any:
    if payload is None:
        return default
    getter = getattr(payload, "get", None)
    if callable(getter):
        return getter(key, default)
    return default


def _payload_keys(payload: Any) -> list[str]:
    if payload is None:
        return []
    keys = getattr(payload, "keys", None)
    if callable(keys):
        try:
            return [str(k) for k in keys()]
        except TypeError:
            return []
    return []


def _extract_ref_audio(additional_information: Any) -> Any:
    if isinstance(additional_information, dict):
        return additional_information.get("ref_audio")
    entries = getattr(additional_information, "entries", None)
    if not isinstance(entries, dict):
        return None
    entry = entries.get("ref_audio")
    if entry is None:
        return None
    list_data = getattr(entry, "list_data", None)
    if list_data is not None:
        return list_data
    scalar_data = getattr(entry, "scalar_data", None)
    if scalar_data is not None:
        return scalar_data
    return None


def _extract_ref_payload(mm: Any) -> tuple[torch.Tensor, int]:
    mm_codes = _payload_get(mm, "codes", {})
    mm_meta = _payload_get(mm, "meta", {})
    ref_code = _payload_get(mm_codes, "ref")
    if ref_code is None:
        ref_code = _payload_get(mm, "codes.ref")
    ref_code_len = _payload_get(mm_meta, "ref_code_len")
    if ref_code_len is None:
        ref_code_len = _payload_get(mm, "meta.ref_code_len")

    if isinstance(ref_code, list):
        ref_code = ref_code[0] if ref_code else None
    if not isinstance(ref_code, torch.Tensor) or ref_code.numel() == 0:
        raise RuntimeError(
            "StepAudio AR output is missing reference codec tokens "
            f"(codes.ref); multimodal keys={_payload_keys(mm)}, codes keys={_payload_keys(mm_codes)}"
        )

    ref_code = ref_code.to(torch.long).cpu().contiguous()
    if int(ref_code.min().item()) >= 65536:
        ref_code = ref_code - 65536

    if isinstance(ref_code_len, torch.Tensor):
        ref_code_len = int(ref_code_len.reshape(-1)[-1].item()) if ref_code_len.numel() > 0 else 0
    elif ref_code_len is None:
        ref_code_len = int(ref_code.numel())
    else:
        ref_code_len = int(ref_code_len)

    return ref_code, ref_code_len


def _build_code2wav_additional_information(
    ref_code: torch.Tensor,
    ref_code_len: int,
    ref_audio: Any,
) -> dict[str, Any]:
    additional_information = to_dict(
        OmniPayloadStruct(
            meta=MetaStruct(left_context_size=ref_code_len) if ref_code_len > 0 else None,
            codes=CodesStruct(ref=ref_code),
        )
    )
    if ref_audio is not None:
        additional_information["latent"] = ref_audio
    return additional_information


def ar2decoder(source_outputs: list[Any], prompt: Any = None, _requires_multimodal_data: bool = False):
    from vllm_omni.inputs.data import OmniTokensPrompt

    talker_outputs = source_outputs
    additional_information = prompt.get("additional_information") or {}
    ref_audio = _extract_ref_audio(additional_information)
    code2wav_inputs: list[OmniTokensPrompt] = []
    for i, talker_output in enumerate(talker_outputs):
        if not talker_output.finished:
            # Non-async decode should only run once, after talker has
            # accumulated the final code sequence.
            continue
        output = talker_output.outputs[0]
        mm = getattr(output, "multimodal_output", None) or getattr(talker_output, "multimodal_output", None)
        token_ids = output.token_ids

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()

        codec_codes = [int(tid) - 65536 for tid in token_ids if int(tid) >= 65536]
        if len(codec_codes) < 5:
            raise RuntimeError(
                f"StepAudio AR generated too few codec tokens: {len(codec_codes)}; tail={token_ids[-30:]}"
            )
        ref_code, ref_code_len = _extract_ref_payload(mm)

        additional_information = _build_code2wav_additional_information(ref_code, ref_code_len, ref_audio)
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=additional_information if additional_information else None,
            )
        )
    return code2wav_inputs


def talker2code2wav_token_only(
    source_outputs: list,
    prompt=None,
    _requires_multimodal_data: bool = False,
) -> list:
    """Sync-side placeholder for the non-async-chunk Stage-1 (code2wav) input.
    Actual codec ids are delivered via the worker connector payload built
    by `talker2code2wav_full_payload`.
    """
    from vllm_omni.inputs.data import OmniTokensPrompt

    code2wav_inputs: list = []
    for i, talker_output in enumerate(source_outputs):
        if not talker_output.finished:
            continue
        output = talker_output.outputs[0]
        mm = output.multimodal_output if hasattr(output, "multimodal_output") else None
        audio_codes = output.token_ids if output.token_ids is not None else []
        if isinstance(audio_codes, torch.Tensor):
            audio_codes = audio_codes.detach().cpu().tolist()

        audio_codes = torch.tensor([int(t) - 65536 for t in audio_codes if int(t) >= 65536])
        if isinstance(audio_codes, list):
            audio_codes = torch.tensor(audio_codes, dtype=torch.long)
        mm = mm if isinstance(mm, dict) else {}
        mm_codes = mm.get("codes", {}) if isinstance(mm, dict) else {}
        ref_code = mm_codes.get("ref", None) if isinstance(mm_codes, dict) else None
        if isinstance(ref_code, torch.Tensor):
            ref_code = ref_code.to(torch.long)
            ref_code = ref_code - 65536
        ref_code_len = mm["meta"].get("ref_code_len")[0] if isinstance(mm, dict) else 0
        audio = prompt["additional_information"].get("ref_audio") if isinstance(prompt, dict) else None

        additional_information = to_dict(
            OmniPayloadStruct(
                meta=MetaStruct(left_context_size=ref_code_len) if ref_code_len > 0 else None,
                codes=CodesStruct(ref=ref_code),
                latent=audio,
            )
        )
        code2wav_inputs.append(
            OmniTokensPrompt(
                # prompt_token_ids=[0] * len(audio_codes),
                prompt_token_ids=audio_codes.tolist(),
                additional_information=additional_information if additional_information else None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return code2wav_inputs


def talker2code2wav_full_payload(
    transfer_manager,
    pooling_output,
    request,
):
    """Producer-side payload builder.

    Reads accumulated codec from `pooling_output["codes.audio"]` (CONCAT
    across steps via flatten_payload), latest `pooling_output["codes.ref"]`
    (prefill-emitted), and latest `pooling_output["meta.ref_code_len"]`.
    Replicates the orchestrator-path body of `talker2code2wav` (filter,
    crop to seq_len, prepend ref, codebook-major flatten).
    """
    del transfer_manager
    rid = getattr(request, "request_id", "?")
    if not isinstance(pooling_output, dict):
        logger.warning(
            "qwen3_tts.talker2code2wav_full_payload: pooling_output not a dict "
            "(type=%s) for req=%s; consumer wait gate may hang.",
            type(pooling_output).__name__,
            rid,
        )
        return None

    # codes.audio — try flat dotted first (flatten_payload), then nested fallback.
    audio = pooling_output.get("codes.audio")
    if audio is None:
        codes_nested = pooling_output.get("codes")
        if isinstance(codes_nested, dict):
            audio = codes_nested.get("audio")
    if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
        logger.warning(
            "qwen3_tts.talker2code2wav_full_payload: missing/empty codes.audio "
            "(keys=%s) for req=%s; consumer wait gate may hang.",
            list(pooling_output.keys()),
            rid,
        )
        return None
    audio = audio.to(torch.long).reshape(-1)
    audio = audio[audio >= 65536] - 65536
    if audio.numel() == 0:
        logger.warning(
            "qwen3_tts.talker2code2wav_full_payload: audio empty after codec "
            "filter (negative/all-zero/out-of-range rows dropped) for req=%s.",
            rid,
        )
        return None

    output_token_ids = list(getattr(request, "output_token_ids", None) or [])
    seq_len = max(len(output_token_ids) - 1, 0)
    if seq_len > 0 and audio.ndim == 2 and int(audio.shape[0]) > seq_len:
        audio = audio[-seq_len:]

    # meta.ref_code_len — flat dotted then nested fallback.
    ref_code_len_raw = pooling_output.get("meta.ref_code_len")
    if ref_code_len_raw is None:
        meta_nested = pooling_output.get("meta")
        if isinstance(meta_nested, dict):
            ref_code_len_raw = meta_nested.get("ref_code_len")

    # codes.ref — flat dotted then nested fallback.
    ref_code_raw = pooling_output.get("codes.ref")
    ref_code_raw = ref_code_raw.to(torch.long) - 65536
    if ref_code_raw is None:
        codes_nested = pooling_output.get("codes")
        if isinstance(codes_nested, dict):
            ref_code_raw = codes_nested.get("ref")

    codec_codes = audio.transpose(0, 1).to(device="cpu", dtype=torch.long).reshape(-1).contiguous()
    return {
        "codes": {"audio": codec_codes, "ref": ref_code_raw},
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
    }


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    pooling_output: OmniPayload | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    additional_information = getattr(request, "additional_information", None)
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())
    request_payload = getattr(transfer_manager, "request_payload", None)

    if request_payload is None:
        request_payload = {}
        transfer_manager.request_payload = request_payload

    ref_audio = additional_information.entries.get("ref_audio").list_data if additional_information else None
    ref_code, ref_code_len = _extract_ref_payload(pooling_output)
    state = transfer_manager.request_payload.setdefault(request_id, {})
    seen_len = int(state.get("seen_len", 0))

    output_token_ids = list(getattr(request, "output_token_ids", []) or [])
    new_tokens = output_token_ids[seen_len:]
    state["seen_len"] = len(output_token_ids)

    for tok in new_tokens:
        tok = int(tok)
        if tok >= 65536:
            transfer_manager.code_prompt_token_ids[request_id].append([tok - 65536])

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 15))
    lookahead_size = int(cfg.get("codec_left_context_frames", 0))
    initial_chunk_size = int(cfg.get("initial_codec_chunk_frames") or 0)

    sent_audio_len = int(state.get("sent_audio_len", 0))
    audio_tokens = transfer_manager.code_prompt_token_ids[request_id]

    available = len(audio_tokens) - sent_audio_len
    need = chunk_size + lookahead_size

    if not finished and available < need:
        return None

    take = available if finished else need
    chunk_frames = audio_tokens[sent_audio_len : sent_audio_len + take]

    advance = available if finished else chunk_size
    state["sent_audio_len"] = sent_audio_len + advance

    if (
        additional_information is not None
        and hasattr(additional_information, "entries")
        and "initial_codec_chunk_frames" in additional_information.entries
    ):
        entry = additional_information.entries["initial_codec_chunk_frames"]
        if entry.list_data is not None and len(entry.list_data) == 1:
            initial_chunk_size = int(entry.list_data[0])

    if chunk_size <= 0 or initial_chunk_size < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"initial_codec_chunk_frames={initial_chunk_size}, "
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

    code_predictor_codes = torch.tensor(
        [frame[0] for frame in chunk_frames],
        dtype=torch.long,
    )
    ref_prompt_ids = None
    if isinstance(ref_code, torch.Tensor):
        ref_prompt = ref_code.detach().cpu().to(torch.long)
        if ref_prompt.ndim > 1:
            ref_prompt = ref_prompt[0]
        ref_prompt_ids = ref_prompt.reshape(-1).tolist()

    return OmniPayloadStruct(
        codes=CodesStruct(audio=code_predictor_codes, ref=ref_code),
        ids=IdsStruct(prompt=ref_prompt_ids),
        meta=MetaStruct(
            finished=torch.tensor(finished, dtype=torch.bool),
            stream_finished=finished,
            ref_code_len=ref_code_len,
            req_id=request_id,
        ),
        latent=ref_audio,
    )
