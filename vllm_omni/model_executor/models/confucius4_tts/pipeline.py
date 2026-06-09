# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Confucius4-TTS pipeline: Talker (text → RVQ codec) → Code2Wav (codec → audio).

Chunked vs end-to-end mode is dispatched from ``deploy.async_chunk``.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.confucius4_tts"

CONFUCIUS4_TTS_PIPELINE = PipelineConfig(
    model_type="confucius4_tts",
    # Pipeline-level default; the code2wav stage overrides per-stage below.
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="ar_codec",
            model_arch="Confucius4TTS_AR",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=f"{_PROC}.talker2code2wav_async_chunk",
            custom_process_next_stage_input_func=f"{_PROC}.ar2decoder_full_payload",
            sampling_constraints={
                "detokenize": False,
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="code2wav",
            model_arch="Confucius4TTSCode2Wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            # ``sync_process_input_func`` is the only input-proc override for
            # this stage in sync (non-async-chunk) mode: a length-only
            # ``_token_only`` placeholder.  The bulk codec payload itself
            # ships via the worker connector from stage 0's
            # ``talker2code2wav_full_payload`` producer.  Under async_chunk
            # mode no pre-stage processing is needed -- chunks deliver
            # directly to the consumer.
            custom_process_input_func=f"{_PROC}.talker2code2wav",
            sync_process_input_func=f"{_PROC}.talker2code2wav_token_only",
            sampling_constraints={"detokenize": True},
            extras={"tts_args": {"max_instructions_length": 500}},
        ),
    ),
)
