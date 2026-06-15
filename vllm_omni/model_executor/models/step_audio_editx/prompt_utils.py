from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def estimate_step_audio_editx_prompt_len(
    additional_information: dict[str, Any],
    model_path: str,
    tokenizer_path: str,
    _cache: dict[str, Any] = {},
) -> int:
    """Estimate placeholder prompt length for Step-Audio-EditX."""
    try:
        from vllm_omni.model_executor.models.step_audio_editx.step_audio_tokenizer import StepAudioTokenizer

        cache_key = (model_path, tokenizer_path)
        speech_tok = _cache.get(cache_key)
        if speech_tok is None:
            speech_tok = StepAudioTokenizer(tokenizer_path=tokenizer_path, config_path=model_path)
            _cache[cache_key] = speech_tok

        def _first(x, default=None):
            if isinstance(x, list):
                return x[0] if x else default
            return x if x is not None else default

        ref_audio = _first(additional_information.get("ref_audio"), None)
        ref_text = _first(additional_information.get("ref_text"), "")
        text = _first(additional_information.get("text"), "")
        sr = _first(additional_information.get("sr"), 16000)
        edit_type = _first(additional_information.get("edit_type", "clone"))
        if edit_type == "clone":
            prompt = (ref_text, text)
        else:
            edit_type = _first(additional_information.get("edit_type", None))
            edit_info = _first(additional_information.get("edit_info", None))
            prompt = (ref_text, edit_type, edit_info, text)
        prompt_token, _ = speech_tok.encode(edit_type, audio=ref_audio, prompt=prompt, sr=sr)
        return max(2, len(prompt_token.input_ids))
    except Exception as exc:
        logger.warning("Failed to estimate prompt length, using fallback 2048: %s", exc)
        return 2048
