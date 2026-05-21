"""Stage input processor for Qwen3-TTS: Talker -> Code2Wav."""

from typing import Any

from vllm.logger import init_logger

from vllm_omni.data_entry_keys import (
    OmniPayloadStruct,
)

logger = init_logger(__name__)


def ar2decoder(source_outputs: list[Any], _prompt: Any = None, _requires_multimodal_data: bool = False):
    from vllm_omni.inputs.data import OmniTokensPrompt

    code2wav_inputs: list[OmniTokensPrompt] = []

    for out in source_outputs:
        if not out.finished:
            continue

        o = out.outputs[0]
        mm = o.multimodal_output or {}

        codec_codes = mm["codec_codes"]
        codec_codes = codec_codes - 65536
        vq0206_codes = (out.intermediate_tensors or {}).get("vq0206_codes")
        vq0206_codes_vocoder = vq0206_codes - 65536

        additional_information = None
        if vq0206_codes_vocoder is not None:
            additional_information = {"vq0206_codes": vq0206_codes_vocoder}

        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=additional_information,
            )
        )


def talker2code2wav_async_chunk(payload: OmniPayloadStruct):
    pass
