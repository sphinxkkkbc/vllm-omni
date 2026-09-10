# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AuK stage input processors."""

import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.models.auk.modeling_auk import AukConditionOutput

logger = init_logger(__name__)

_AUK_PAYLOAD_KEYS = frozenset(
    {
        "semantic_embeddings",
        "semantic_mask",
        "reference_latents",
        "reference_mask",
        "target_length",
        "seed",
    }
)
_FULL_PAYLOAD_REPLACE_KEYS = frozenset(
    {
        "semantic_embeddings",
        "semantic_mask",
        "reference_latents",
        "reference_mask",
        "target_length",
        "seed",
    }
)


def _extract_auk_payload(pooling_output):
    """Extract and normalize one request's AuK payload from runner output."""
    if isinstance(pooling_output, dict) and isinstance(pooling_output.get("multimodal_outputs"), dict):
        pooling_output = pooling_output["multimodal_outputs"]
    if not isinstance(pooling_output, dict) or not _AUK_PAYLOAD_KEYS.issubset(pooling_output):
        return None

    payload = {}
    for key in _AUK_PAYLOAD_KEYS:
        value = pooling_output[key]
        if isinstance(value, (list, tuple)):
            if len(value) != 1 or not isinstance(value[0], torch.Tensor):
                return None
            value = value[0]
        if not isinstance(value, torch.Tensor):
            return None
        payload[key] = value
    return payload


def build_stage1_inputs(source_outputs, prompt=None, requires_multimodal_data=False):
    """Allocate Stage1 token slots; condition data arrives via connector.

    The orchestrator ``RequestOutput`` is the control plane and is not expected
    to carry the AuK condition. ``requires_full_payload_input=True`` gates the
    Stage1 request until the connector installs the serialized condition in
    ``model_intermediate_buffer``.
    """
    from vllm_omni.inputs.data import OmniTokensPrompt

    del prompt, requires_multimodal_data
    if not isinstance(source_outputs, list):
        raise TypeError(f"AuK Stage1 input expects source_outputs list, got {type(source_outputs).__name__}")
    return [
        OmniTokensPrompt(
            prompt_token_ids=[0],
            additional_information=None,
            multi_modal_data=None,
            mm_processor_kwargs=None,
        )
        for source_output in source_outputs
        if source_output.finished
    ]


def serialize_condition_payload(transfer_manager, pooling_output, request, **kwargs):
    """Normalize AuK's one-shot condition payload."""
    del transfer_manager
    request_id = getattr(request, "external_req_id", None) or getattr(request, "request_id", None)
    is_finished = bool(kwargs.get("is_finished", False))
    payload = _extract_auk_payload(pooling_output)
    if payload is None:
        if is_finished:
            logger.info("AuK handoff skipped redundant terminal output: request=%s", request_id)
            return None
        raise RuntimeError("AuK terminal full payload is missing condition fields")
    result = AukConditionOutput.from_payload(payload).to_payload()
    result["meta"] = {"finished": torch.tensor(True, dtype=torch.bool)}
    logger.info(
        "AuK handoff built: request=%s keys=%s finished=True",
        request_id,
        sorted(result),
    )
    return result
