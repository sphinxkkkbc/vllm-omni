"""Stage input processor for Qwen3-TTS: Talker -> Code2Wav."""

from typing import Any

import torch
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
        if len(codec_codes) < 5:
            raise RuntimeError(
                f"StepAudio AR generated too few codec tokens: {len(codec_codes)}; tail={token_ids[-30:]}"
            )
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
                codes=CodesStruct(ref=ref_code),
                latent=ref_audio,
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
    logger.info(f"pooling_output: {pooling_output}")
    additional_information = getattr(request, "additional_information", None)
    logger.info(f"additional_information: {additional_information}")
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())
    request_payload = getattr(transfer_manager, "request_payload", None)

    if request_payload is None:
        request_payload = {}
        transfer_manager.request_payload = request_payload

    ref_audio = additional_information.entries.get("ref_audio").list_data if additional_information else None
    ref_code = pooling_output.get("codes", {}).get("ref")

    if isinstance(ref_code, list):
        ref_code = [i - 65536 for i in ref_code]
    else:
        ref_code = ref_code - 65536

    ref_code_len = int(ref_code.numel())
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
    # logger.info(f"cfg: {cfg}")
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
    return OmniPayloadStruct(
        codes=CodesStruct(audio=code_predictor_codes, ref=ref_code),
        meta=MetaStruct(
            finished=torch.tensor(finished, dtype=torch.bool),
            stream_finished=finished,
            ref_code_len=ref_code_len,
            req_id=request_id,
        ),
        latent=ref_audio,
    )
