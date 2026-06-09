from typing import Any
import logging
import torch

from vllm_omni.data_entry_keys import MetaStruct, OmniPayloadStruct, to_dict

logger = logging.getLogger(__name__)

def talker2code2wav(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[Any]:
    """Non-async: collect all talker codes, then pass to code2wav at once."""
    from vllm_omni.inputs.data import OmniTokensPrompt

    talker_outputs = source_outputs
    code2wav_inputs: list[OmniTokensPrompt] = []
    for i, talker_output in enumerate(talker_outputs):
        if not talker_output.finished:
            # Non-async decode should only run once, after talker has
            # accumulated the final code sequence.
            continue
        output = talker_output.outputs[0]
        mm = output.multimodal_output
        mm_codes = mm.get("codes", {})

        # audio_codes shape: [num_frames, Q] where Q=num_quantizers (16)
        audio_codes = mm_codes["audio"].to(torch.long)
        token_ids = output.cumulative_token_ids

        # token_ids provides an upper bound on the newly generated codec span.
        # audio_codes may still contain zero-padded / invalid rows, so trim only
        # after filtering valid frames instead of trying to align EOS indices.
        seq_len = max(len(token_ids) - 1, 0)
        # Filter invalid frames: zero-padded (EOS), out-of-range values (e.g.
        # stop_token_id=2150 exceeds codebook_size=2048), and negative
        # sentinels (e.g. -1 padding).
        _CODEBOOK_SIZE = 2048
        valid_mask = (
            (audio_codes >= 0).all(dim=1) & audio_codes.any(dim=1) & (audio_codes.max(dim=1).values < _CODEBOOK_SIZE)
        )
        audio_codes = audio_codes[valid_mask]
        if seq_len > 0 and audio_codes.ndim == 2 and int(audio_codes.shape[0]) > seq_len:
            audio_codes = audio_codes[-seq_len:]
        ref_code = mm_codes.get("ref")
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
            if ref_code.ndim == 1:
                num_quantizers = int(audio_codes.shape[1]) if audio_codes.ndim == 2 and audio_codes.shape[1] > 0 else 16
                if ref_code.numel() % num_quantizers != 0:
                    logger.warning(
                        "Ignoring malformed ref_code with %d elements not divisible by num_quantizers=%d",
                        ref_code.numel(),
                        num_quantizers,
                    )
                    ref_code = None
                else:
                    ref_code = ref_code.reshape(-1, num_quantizers)
            elif ref_code.ndim != 2:
                logger.warning("Ignoring malformed ref_code shape %s", tuple(ref_code.shape))
                ref_code = None
            if isinstance(ref_code, torch.Tensor) and ref_code_len > 0 and int(ref_code.shape[0]) > ref_code_len:
                logger.warning(
                    "Trimming ref_code from %d frames to ref_code_len=%d before Code2Wav.",
                    int(ref_code.shape[0]),
                    ref_code_len,
                )
                ref_code = ref_code[:ref_code_len]
            if not isinstance(ref_code, torch.Tensor):
                ref_code_len = 0
            else:
                ref_code_len = int(ref_code.shape[0])
                audio_codes = torch.cat([ref_code.to(audio_codes.device), audio_codes], dim=0)
        else:
            ref_code_len = 0
        # Code2Wav expects codebook-major flat: [Q*num_frames]
        codec_codes = audio_codes.transpose(0, 1).cpu().reshape(-1).tolist()
        additional_information = to_dict(
            OmniPayloadStruct(
                meta=MetaStruct(left_context_size=ref_code_len) if ref_code_len > 0 else None,
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
