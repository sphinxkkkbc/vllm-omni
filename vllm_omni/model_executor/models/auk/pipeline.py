# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AuK pipeline configuration for vLLM-Omni."""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

_PROC = "vllm_omni.model_executor.stage_input_processors.auk"

AUK_PIPELINE = PipelineConfig(
    model_type="auk",
    model_arch="AukConditionModel",
    default_deploy_config_name="auk.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="auk_condition",
            execution_type=StageExecutionType.LLM_AR,
            owns_tokenizer=True,  # Qwen tokenizer
            engine_output_type="latent",  # Output condition, not tokens
            model_arch="AukConditionModel",
            custom_process_next_stage_input_func=f"{_PROC}.serialize_condition_payload",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="auk_flow",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="AukFlowModel",
            requires_full_payload_input=True,
            sync_process_input_func=f"{_PROC}.build_stage1_inputs",
            extras={"skip_tokenizer_init": True},
            sampling_constraints={"detokenize": False},
        ),
    ),
)
