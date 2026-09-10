# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AuK-Flash speech endpoint; uses the standard multi-stage serving path."""

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest
from vllm_omni.model_executor.models.auk.prompt_utils import build_auk_prompt, target_frames


@register_tts_adapter
class AukAdapter(ARTTSAdapter):
    name = "auk"
    stage_keys = frozenset({"auk_condition"})
    model_archs = frozenset({"AukConditionModel"})

    def _load_supported_speakers(self):
        return {"default"}

    def validate(self, request):
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        if request.duration_seconds is None:
            return "AuK requires duration_seconds"
        try:
            if target_frames(request.duration_seconds) > 65536:
                return "AuK duration exceeds 65536 latent frames"
        except ValueError as exc:
            return str(exc)
        if request.speed != 1.0:
            return "AuK uses duration_seconds; speed must be 1.0"
        if request.voice not in (None, "default"):
            return "AuK supports voice='default'; use ref_audio for voice conditioning"
        if request.ref_audio is not None:
            return self.ctx.server._validate_ref_audio_format(request.ref_audio)
        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        reference = None
        if request.ref_audio is not None:
            resolved = await self.ctx.server._resolve_ref_audio(request.ref_audio)
            reference = (resolved[0], resolved[1])
        prompt = build_auk_prompt(
            request.input,
            request.duration_seconds,
            instructions=request.instructions,
            reference_audio=reference,
            seed=request.seed,
        )
        return PreparedRequest(prompt=prompt, tts_params={}, model_type=self.name)
