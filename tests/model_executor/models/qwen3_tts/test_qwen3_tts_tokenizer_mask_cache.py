# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers.cache_utils import DynamicCache

from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderTransformerModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_config(attention_implementation: str) -> Qwen3TTSTokenizerV2DecoderConfig:
    config = Qwen3TTSTokenizerV2DecoderConfig(
        hidden_size=32,
        latent_dim=16,
        max_position_embeddings=512,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=64,
        num_hidden_layers=1,
        num_quantizers=2,
        decoder_dim=32,
        upsample_rates=(2,),
        upsampling_ratios=(2,),
    )
    config._attn_implementation = attention_implementation
    return config


def _make_model(attention_implementation: str) -> Qwen3TTSTokenizerV2DecoderTransformerModel:
    model = Qwen3TTSTokenizerV2DecoderTransformerModel._from_config(_make_config(attention_implementation))
    return model.eval()


def test_kv_cache_supports_incremental_decode():
    model = _make_model("sdpa")
    prefix = torch.randn(1, 11, model.config.latent_dim)
    suffix = torch.randn(1, 6, model.config.latent_dim)

    prefix_output = model(inputs_embeds=prefix, use_cache=True)

    assert isinstance(prefix_output.past_key_values, DynamicCache)
    assert prefix_output.past_key_values.get_seq_length() == prefix.shape[1]

    suffix_output = model(
        inputs_embeds=suffix,
        past_key_values=prefix_output.past_key_values,
        use_cache=True,
    )

    assert suffix_output.last_hidden_state.shape[:2] == suffix.shape[:2]
    assert suffix_output.past_key_values.get_seq_length() == prefix.shape[1] + suffix.shape[1]
