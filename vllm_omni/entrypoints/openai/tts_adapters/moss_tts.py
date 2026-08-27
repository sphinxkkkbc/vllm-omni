# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MOSS-TTS serving adapters (Nano + full family).

Both variants share the same build/validate flow (``_build_moss_tts_params``
handles each); they are registered under distinct model-type names.
"""

from typing import TYPE_CHECKING, Any

from vllm.inputs import tokens_input

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    PreparedRequest,
    apply_max_new_tokens,
    conditioning_cache_salt,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


class _MossTTSAdapterBase(ARTTSAdapter):
    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._moss_variant = None if self.name == "moss_tts_nano" else self._detect_moss_variant()
        self._moss_processor_cache = None

    @property
    def engine_client(self):
        return self.ctx.engine_client

    @property
    def uploaded_speakers(self):
        return self.ctx.server.uploaded_speakers

    @property
    def _speaker_cache(self):
        return self.ctx.server._speaker_cache

    async def _resolve_ref_audio(self, ref_audio: str):
        return await self.ctx.server._resolve_ref_audio(ref_audio)

    def _voice_created_at(self, voice: str) -> int:
        return self.ctx.server._voice_created_at(voice)

    def _detect_moss_variant(self) -> str:
        """Sub-classify a ``moss_tts``-stage server into the actual MOSS-TTS
        variant family (tts, ttsd, sound_effect, voice_generator, realtime).

        Detection key is the HF repo path / model_name; matches
        ``_try_resolve_omni_model_type`` in entrypoints/utils.py so users get
        consistent behaviour no matter how they launched the server (--model
        OpenMOSS-Team/MOSS-TTSD-v1.0 vs --deploy-config moss_ttsd.yaml).
        """
        try:
            name = (self.engine_client.model_config.model or "").lower().replace("-", "").replace("_", "")
        except Exception:
            name = ""
        if "realtime" in name:
            return "realtime"
        if "local" in name:
            return "local"
        if "ttsd" in name:
            return "ttsd"
        if "soundeffect" in name:
            return "sound_effect"
        if "voicegenerator" in name:
            return "voice_generator"
        return "tts"

    def _get_moss_processor(self):
        """Lazily load the upstream MOSS-TTS processor once per server.

        Cached on ``self._moss_processor_cache``. The processor owns its own
        audio_tokenizer (~1.6 B params); we keep it on CPU so it doesn't
        compete with the talker (~8 GiB) and codec (~7 GiB) for our 96 GiB
        GPU — per-request ref-audio encoding is fast enough on CPU.
        """
        cached = getattr(self, "_moss_processor_cache", None)
        if cached is not None:
            return cached
        from transformers import AutoProcessor

        model_id = self.engine_client.model_config.model
        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        if hasattr(proc, "audio_tokenizer"):
            proc.audio_tokenizer = proc.audio_tokenizer.to("cpu").eval()
        self._moss_processor_cache = proc
        return proc

    async def _build_moss_tts_params(
        self,
        request: "OpenAICreateSpeechRequest",
        *,
        has_inline_ref_audio: bool = False,
    ) -> dict[str, Any]:
        """Build the talker prompt + ``additional_information`` payload for any
        MOSS-TTS-family request (nano + 5 full variants).

        For the legacy ``moss_tts_nano`` model_type, keeps the original nano
        contract (``{text, mode=voice_clone, prompt_audio_array}``); the
        caller still uses a ``[1]`` placeholder prompt.

        For the full MOSS-TTS family (``MossTTSDelay*`` / ``MossTTSRealtime*``)
        we **call the upstream processor server-side** to produce the unified
        ``(text_ids, audio_codes)`` shape the talker actually consumes — same
        flow as ``examples/.../moss_tts/end2end.py:_build_unified_codes``.
        Returns ``{prompt_token_ids: list[int], codes.ref: torch.LongTensor,
        max_new_frames, ...}``. The caller treats ``prompt_token_ids`` as the
        prompt and forwards the rest as ``additional_information``.
        """
        import torch  # local to avoid pulling torch at module import time

        v = self._moss_variant

        # ---- Legacy nano path (unchanged) ----
        if v is None:  # moss_tts_nano
            params: dict[str, Any] = {
                "text": [request.input or ""],
                "mode": ["voice_clone"],
            }
            if request.max_new_tokens is not None:
                params["max_new_frames"] = [request.max_new_tokens]
            wav_list, sr, cache_key = await self._resolve_ref_audio(request.ref_audio)
            params["prompt_audio_array"] = [[wav_list, sr]]
            params["ref_audio_cache_key"] = cache_key
            return params

        # ---- MOSS-TTS-Realtime: keep the old prompt_audio_array path ----
        # ``AutoProcessor.from_pretrained`` doesn't auto-discover
        # ``MossTTSRealtimeProcessor`` (no ``processor_config.json`` in the
        # snapshot), and Realtime's prompt format diverges from MossTTSDelay
        # (16-channel grid, separate per-step text feed). The
        # ``prompt_audio_array`` shape lines up well enough with what the
        # talker reads for short prompts; full Realtime support needs a
        # separate processor.from_module path which we don't wire here.
        if v == "realtime":
            params: dict[str, Any] = {
                "text": [request.input or ""],
                "mode": ["voice_clone"],
            }
            if request.max_new_tokens is not None:
                params["max_new_frames"] = [request.max_new_tokens]
            wav_list, sr, cache_key = await self._resolve_ref_audio(request.ref_audio)
            params["prompt_audio_array"] = [[wav_list, sr]]
            params["ref_audio_cache_key"] = cache_key
            return params

        # ---- MossTTSDelay family (tts/ttsd/sound_effect/voice_generator)
        # and MOSS-TTS-Local-Transformer-v1.5: call the upstream processor
        # server-side to produce unified codes. Local-v1.5 ships its own
        # AutoProcessor (processor_config.json + processing_moss_tts.py) and
        # reuses this exact build_user_message/encode_audios_from_wav path in
        # the offline example (examples/.../moss_tts/end2end.py:
        # _build_unified_codes) -- it is NOT in the same boat as Realtime
        # (no processor_config.json there), so it must not fall back to the
        # prompt_audio_array path above (which the talker's preprocess()
        # never reads -- info_dict["codes"]["ref"] is the only thing it
        # consumes, so skipping this path silently drops all voice-clone
        # conditioning and produces unconditioned/garbage audio online). ----
        proc = self._get_moss_processor()
        n_vq = int(getattr(proc.model_config, "n_vq", 32))
        # Local-v1.5 encodes reference audio at a fixed 24 kHz working rate
        # regardless of its 48 kHz stereo *output* codec -- mirrors the
        # offline example's hardcoded encode_audios_from_wav(sampling_rate=24000)
        # for this variant; proc.model_config.sampling_rate there is the
        # output rate (48000), the wrong value to resample the reference into.
        sr_target = 24000 if v == "local" else int(getattr(proc.model_config, "sampling_rate", 24000))

        # Reference-audio encoding + speaker caching lives in the model package
        # (moss_tts.reference_encoder), mirroring Fish Speech / CosyVoice3 /
        # Qwen3-TTS which keep reference handling with the model rather than in
        # this shared serving file. Imported lazily so the API-server process
        # only pulls it on the delay-family path (alongside the upstream proc).
        from vllm_omni.model_executor.models.moss_tts.reference_encoder import encode_reference_codes

        # Named-voice speaker cache is only valid for uploaded speakers
        # without an inline ref_audio. ``request.voice`` plus a file/URL
        # otherwise keys on (name, created_at=0) and skips the content-aware
        # resolve — the adapter already applies this guard when tagging
        # ``voice_name`` on tts_params.
        _raw_voice = getattr(request, "voice", None)
        _raw_voice = _raw_voice.strip() if isinstance(_raw_voice, str) else ""
        _voice_lower = _raw_voice.lower() if _raw_voice else ""
        _use_named_voice = bool(_voice_lower) and _voice_lower in self.uploaded_speakers and not has_inline_ref_audio
        _voice = _voice_lower if _use_named_voice else ""
        _voice_created = self._voice_created_at(_voice) if _voice else 0
        _resolve_keys: list[str] = []

        async def _tracked_resolve(ref_str: str) -> tuple[list, int, str]:
            wav_list, sr, cache_key = await self._resolve_ref_audio(ref_str)
            _resolve_keys.append(cache_key)
            return wav_list, sr, cache_key

        async def _encode_ref(ref_str: str, *, named_voice: bool) -> torch.Tensor:
            # Named-voice cache is (voice_name, created_at). TTSD speaker 2 is a
            # different clip, so it must stay anonymous or it reuses speaker 1.
            use_named = named_voice and bool(_voice)
            return await encode_reference_codes(
                ref_str,
                processor=proc,
                resolve_ref_audio=_tracked_resolve,
                speaker_cache=self._speaker_cache,
                variant=v,
                n_vq=n_vq,
                sr_target=sr_target,
                voice_name=_voice if use_named else None,
                voice_created_at=_voice_created if use_named else 0,
            )

        user_kwargs: dict[str, Any] = {"text": request.input or ""}
        if v in ("tts", "local"):
            user_kwargs["reference"] = [await _encode_ref(request.ref_audio, named_voice=True)]
        elif v == "ttsd":
            refs = [await _encode_ref(request.ref_audio, named_voice=True)]
            if request.ref_audio_2:
                refs.append(await _encode_ref(request.ref_audio_2, named_voice=False))
            user_kwargs["reference"] = refs
        elif v == "sound_effect":
            user_kwargs["text"] = request.input or ""  # may be empty
            user_kwargs["ambient_sound"] = request.ambient_sound or ""
            if request.duration_seconds is not None:
                user_kwargs["tokens"] = max(1, int(float(request.duration_seconds) * 12.5))
            elif request.max_new_tokens is not None:
                user_kwargs["tokens"] = int(request.max_new_tokens)
        elif v == "voice_generator":
            user_kwargs["instruction"] = request.instructions or ""

        # Optional language tag for the spoken-text variants. MOSS-TTS-v1.5's
        # headline improvement is multilingual synthesis when the language is
        # given (build_user_message(..., language=...)); 1.0 ignores it
        # gracefully. Sound-effect output is non-verbal, so skip it there.
        if v in ("tts", "local", "ttsd", "voice_generator") and getattr(request, "language", None):
            user_kwargs["language"] = request.language

        # Build the unified-codes prompt: (L, 1+n_vq) where col 0 is text/special
        # tokens and cols 1..n_vq are the delay-pattern audio code grid (mostly
        # audio_pad_code outside the reference block).
        user_msg = proc.build_user_message(**user_kwargs)
        batch = proc(conversations=[[user_msg]], mode="generation")
        unified = batch["input_ids"][0]  # torch.LongTensor (L, 1+n_vq)
        text_ids: list[int] = unified[:, 0].tolist()
        audio_codes: torch.Tensor = unified[:, 1:].contiguous().to(torch.int64)

        params: dict[str, Any] = {
            "prompt_token_ids": text_ids,
            "codes": {"ref": audio_codes},
        }
        if request.max_new_tokens is not None:
            params["max_new_frames"] = [request.max_new_tokens]
        if _resolve_keys:
            params["ref_audio_cache_key"] = _resolve_keys[0]
            if len(_resolve_keys) > 1:
                params["ref_audio_2_cache_key"] = _resolve_keys[1]
        return params

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        """Validate any MOSS-TTS-family request (nano + 5 full variants).

        Dispatches by ``self._moss_variant``:
          - ``tts``/``realtime``: require ``ref_audio`` (voice cloning).
          - ``ttsd``: require ``ref_audio`` (speaker 1); ``ref_audio_2``
            optional (defaults to the same ref for both speakers).
          - ``sound_effect``: require ``ambient_sound`` (no ref_audio).
          - ``voice_generator``: require ``instructions`` (no ref_audio).
          - For the legacy moss_tts_nano model_type the variant is None and
            we fall through to the original nano contract (ref_audio only).
        """
        server = self.ctx.server
        err = server._apply_uploaded_speaker(request)
        if err:
            return err

        if not request.input or not request.input.strip():
            # SoundEffect can legitimately have empty input (just ambient_sound).
            if self._moss_variant != "sound_effect":
                return "Input text cannot be empty"

        v = self._moss_variant
        if v in (None, "tts", "realtime", "local"):
            if request.ref_audio is None:
                label = (
                    "MOSS-TTS-Nano"
                    if v is None
                    else (
                        "MOSS-TTS-Realtime"
                        if v == "realtime"
                        else ("MOSS-TTS-Local-Transformer" if v == "local" else "MOSS-TTS")
                    )
                )
                return f"{label} requires 'ref_audio' (reference audio for voice cloning)."
            return server._validate_ref_audio_format(request.ref_audio)

        if v == "ttsd":
            if request.ref_audio is None:
                return "MOSS-TTSD requires 'ref_audio' (speaker 1 reference)."
            fmt_err = server._validate_ref_audio_format(request.ref_audio)
            if fmt_err:
                return fmt_err
            if request.ref_audio_2 is not None:
                return server._validate_ref_audio_format(request.ref_audio_2)
            return None

        if v == "sound_effect":
            if not request.ambient_sound or not request.ambient_sound.strip():
                return (
                    "MOSS-SoundEffect requires 'ambient_sound' (natural language "
                    "description of the sound effect to synthesise)."
                )
            return None

        if v == "voice_generator":
            if not request.instructions or not request.instructions.strip():
                return (
                    "MOSS-VoiceGenerator requires 'instructions' (natural language "
                    "voice description, e.g. 'a warm female voice with an American accent')."
                )
            return None

        return None  # unreachable

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        server = self.ctx.server
        tts_params = await self._build_moss_tts_params(request, has_inline_ref_audio=has_inline_ref_audio)
        if request.voice:
            voice_lower = request.voice.lower()
            if voice_lower in server.uploaded_speakers and not has_inline_ref_audio:
                tts_params["voice_name"] = [voice_lower]
                tts_params["voice_created_at"] = [server._voice_created_at(voice_lower)]
        # MOSS reads the resolved seed at build time (it samples internally).
        if sampling_params_list and getattr(sampling_params_list[0], "seed", None) is not None:
            tts_params["seed"] = [sampling_params_list[0].seed]
        if isinstance(tts_params.get("prompt_token_ids"), list):
            prompt_token_ids = tts_params.pop("prompt_token_ids")
            prompt = tokens_input(prompt_token_ids=prompt_token_ids)
        else:
            prompt = tokens_input(prompt_token_ids=[1])
        prompt["additional_information"] = tts_params
        prompt["cache_salt"] = conditioning_cache_salt(request, tts_params)
        return PreparedRequest(prompt=prompt, tts_params=tts_params, model_type=self.name)

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict | None = None,
        request_id: str | None = None,
    ) -> list:
        return apply_max_new_tokens(sampling_params_list, request)


@register_tts_adapter
class MossTTSNanoAdapter(_MossTTSAdapterBase):
    stage_keys = frozenset({"moss_tts_nano"})
    name = "moss_tts_nano"


@register_tts_adapter
class MossTTSAdapter(_MossTTSAdapterBase):
    stage_keys = frozenset({"moss_tts", "moss_tts_codec", "moss_tts_local", "moss_tts_local_codec"})
    name = "moss_tts"
