"""Stage input processor for Qwen3-TTS: Talker -> Code2Wav."""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import (
    CodesStruct,
    MetaStruct,
    OmniPayloadStruct,
    to_dict,
)

logger = init_logger(__name__)

CODE_OFFSET = 65536
BOS_ID, EOS_ID, PAD_ID = 1, 2, 0


def extract_codec_codes(token_ids: list[int]) -> list[int]:
    out: list[int] = []
    for t in token_ids:
        if t == EOS_ID:
            break

        if t >= CODE_OFFSET:
            c = t - CODE_OFFSET
            out.append(c)
    return out


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
        if isinstance(output.token_ids, list):
            codec_codes = extract_codec_codes(output.token_ids)
        else:
            codec_codes = output.token_ids - 65536
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
