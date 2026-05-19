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
from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import (
    compute_dynamic_initial_chunk_size,
    max_ic_for_chunk_size,
)
from vllm_omni.model_executor.stage_input_processors.tts_utils import (
    extract_language_from_prompt,
    extract_language_from_request,
    extract_speaker_from_prompt,
    extract_speaker_from_request,
)

logger = init_logger(__name__)


def tokenizer2ar():
    pass

def tokenizer2decoder():
    pass

def ar2decoder():
    pass
