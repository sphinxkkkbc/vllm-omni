import logging
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.models.step1 import Step1ForCausalLM

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.model_executor.models.output_templates import OmniOutput

from .step_audio_tokenizer import StepAudioTokenizer

logger = logging.getLogger(__name__)


class StepAudioAR(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        extra = getattr(vllm_config.model_config, "additional_kwargs", None) or {}
        self.tokenizer_path = extra.get("audio_tokenizer")
        self.model = Step1ForCausalLM(vllm_config=vllm_config, prefix=prefix)
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = False
        self.tokenizer = None

    def embed_multimodal(self, **kwargs) -> torch.Tensor:
        return self._encode_ref_audio_to_code(**kwargs)

    def _ensure_audio_tokenizer_loaded(self):
        if self.tokenizer is not None:
            return

        self.tokenizer = StepAudioTokenizer(
            tokenizer_path=self.tokenizer_path,
            config_path=self.model_path,
        )

    @staticmethod
    def estimate_prompt_len_from_additional_information(
        additional_information: dict[str, Any] | None,
        *,
        task_type: str,
        tokenize_prompt: Callable[[str], list[int]],
        estimate_ref_code_len: Callable[[object], int | None] | None = None,
    ) -> int:
        """Compute Stage-0 placeholder prompt length (length-only mirror of `_build_prompt_embeds()`).
        It must match the model-side `inputs_embeds` length to avoid extra padding and quality drop."""

        def _first(x: object, default: object) -> object:
            if isinstance(x, list):
                return x[0] if x else default
            return x if x is not None else default

        info: dict[str, Any] = additional_information or {}
        text = _first(info.get("text"), "")
        # Official defaults: CustomVoice/VoiceDesign -> non_streaming_mode=True; Base -> False.
        non_streaming_mode = task_type in ("edit")

        if not isinstance(text, str):
            text = ""
        # ---- text conditioning portion (matches _build_prompt_embeds) ----
        assistant_text = StepAudioAR._build_assistant_text(text)
        assistant_len = len(tokenize_prompt(assistant_text))
        if assistant_len < 8:
            raise ValueError(f"Unexpected assistant prompt length: {assistant_len}")

        if task_type == "clone":
            voice_clone_prompt = _first(info.get("voice_clone_prompt"), None)

            ref_code = None
            if isinstance(voice_clone_prompt, dict):
                ref_code = _first(voice_clone_prompt.get("ref_code"), None)

            ref_code_len: int | None = None
            if isinstance(ref_code, list):
                if ref_code and isinstance(ref_code[0], list):
                    ref_code_len = len(ref_code)
                elif ref_code:
                    ref_code_len = len(ref_code)
            elif hasattr(ref_code, "shape"):
                try:
                    shape = getattr(ref_code, "shape")
                    if shape and len(shape) >= 1:
                        ref_code_len = int(shape[0])
                except Exception:
                    ref_code_len = None

            if ref_code_len is None and estimate_ref_code_len is not None:
                ref_code_len = estimate_ref_code_len(info.get("ref_audio"))
            if ref_code_len is None:
                raise ValueError(
                    "Base in-context voice cloning requires either `voice_clone_prompt.ref_code` "
                    "or a readable `ref_audio` that can be mapped to a codec frame length."
                )

            codec_lens = 1 + int(ref_code_len)  # codec_bos + ref_code
            if non_streaming_mode:
                prompt_len = 0
                # _generate_icl_prompt(non_streaming_mode=True):
                # text_embed = ref_ids + text_ids + eos.
                ref_ids = _first(info.get("ref_ids"), None)
                if isinstance(voice_clone_prompt, dict) and ref_ids is None:
                    ref_ids = _first(voice_clone_prompt.get("ref_ids") or voice_clone_prompt.get("ref_id"), None)

                if ref_ids is None:
                    ref_text = _first(info.get("ref_text"), "")
                    if not isinstance(ref_text, str) or not ref_text.strip():
                        raise ValueError("Base in-context non-streaming requires `ref_text` or tokenized `ref_ids`.")
                    ref_text_ids = tokenize_prompt(StepAudioAR._build_ref_text(ref_text))
                    ref_ids_len = len(ref_text_ids)
                elif hasattr(ref_ids, "shape"):
                    shape = getattr(ref_ids, "shape", None)
                    ref_ids_len = int(shape[-1]) if shape else 0
                elif isinstance(ref_ids, list):
                    ref_ids_len = len(ref_ids)
                else:
                    ref_ids_len = 0

                # model uses ref_ids[:, 3:-2] (strip 5 tokens) and text_id=input_ids[:, 3:-5] (strip 8).
                ref_id_len = max(0, int(ref_ids_len) - 5)
                text_id_len = max(0, int(assistant_len) - 8)
                text_embed_len = ref_id_len + text_id_len + 1  # + eos
                prompt_len += text_embed_len + codec_lens
            else:
                # _generate_icl_prompt(non_streaming_mode=False): aligned to codec_lens.
                prompt_len += codec_lens

        return max(2, int(prompt_len))

    def _build_prompt_embeds(
        self,
        *,
        task_type: str,
        info_dict: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int | None, torch.Tensor | None]:
        logger.info(f"Building prompt embeds for task_type: {task_type}, info_dict: {info_dict}")
        ref_audio = info_dict.get("ref_audio")
        ref_text = info_dict.get("ref_text")
        text = info_dict.get("text")
        sr = info_dict.get("sr") or 16000
        prompt_token, codec_token = self.tokenizer.encode(task_type, audio=ref_audio, prompt=(ref_text, text), sr=sr)
        logger.info(f"prompt_token: {prompt_token}, codec_token: {codec_token}")
        input_ids = torch.tensor(prompt_token.input_ids)
        input_ids = input_ids.to(next(self.model.parameters()).device)
        logger.info(f"input_ids shape: {input_ids.shape}, codec_token shape: {codec_token.shape}")
        input_ids = self.embed_input_ids(input_ids)
        tts_pad_id = self.tokenizer.text_tokenizer.pad_token_id
        tts_pad_embed = self.embed_input_ids(torch.tensor([tts_pad_id]).to(input_ids.device))
        return input_ids, codec_token.shape[1], codec_token, tts_pad_embed

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        hidden = model_outputs
        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information") or []
        if "runtime_additional_information" in kwargs and "model_intermediate_buffer" not in kwargs:
            logger.warning_once("runtime_additional_information is deprecated, use model_intermediate_buffer")
        audio_codes_list: list[torch.Tensor] = []
        ref_code_len_list: list[torch.Tensor] = []
        ref_code_tensor: torch.Tensor | None = None
        codec_streaming_list: list[torch.Tensor] = []
        for info in info_dicts:
            if not isinstance(info, dict):
                continue
            codes = info.get("codes", {})
            meta = info.get("meta", {})
            ac = codes.get("audio")
            if isinstance(ac, torch.Tensor):
                audio_codes_list.append(ac)
                cs = meta.get("codec_streaming")
                if isinstance(cs, bool):
                    codec_streaming_list.append(
                        torch.full((int(ac.shape[0]),), int(cs), dtype=torch.int8, device=ac.device)
                    )
            ref_code = codes.get("ref")
            if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0:
                ref_code_tensor = ref_code
            ref_len = meta.get("ref_code_len")
            if ref_len is None:
                continue
            if isinstance(ref_len, torch.Tensor):
                if ref_len.numel() == 0:
                    raise ValueError("ref_code_len is an empty tensor")
                ref_len_val = int(ref_len.reshape(-1)[-1].item())
            elif isinstance(ref_len, list):
                if len(ref_len) != 1:
                    raise ValueError(f"ref_code_len must be scalar or 1-element list, got len={len(ref_len)}")
                ref_len_val = int(ref_len[0])
            else:
                ref_len_val = int(ref_len)
            if isinstance(ac, torch.Tensor):
                # Emit ref_code_len per-token span for runner slicing (consumer takes the last value).
                ref_code_len_list.append(
                    torch.full((int(ac.shape[0]),), ref_len_val, dtype=torch.int32, device=ac.device)
                )

        if not audio_codes_list:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

        audio_codes = torch.cat(audio_codes_list, dim=0)
        span_len = int(audio_codes.shape[0])
        # logger.info(f"span_len: {span_len}, hidden.shape: {hidden.shape}")
        # hidden = hidden[:span_len]
        mm: OmniPayload = {"codes": {"audio": audio_codes}}
        if ref_code_len_list:
            mm.setdefault("meta", {})["ref_code_len"] = torch.cat(ref_code_len_list, dim=0)[:span_len]
        if ref_code_tensor is not None:
            mm.setdefault("codes", {})["ref"] = [ref_code_tensor]
        if codec_streaming_list:
            mm.setdefault("meta", {})["codec_streaming"] = torch.cat(codec_streaming_list, dim=0)[:span_len]
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs=mm)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        self._ensure_audio_tokenizer_loaded()
        additional_information = info_dict.get("additional_information")
        if isinstance(additional_information, dict):
            merged: dict[str, Any] = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in additional_information.items():
                merged.setdefault(k, v)
            info_dict = merged

        payload: OmniPayload = info_dict
        embed = payload.get("embed", {})
        meta = payload.get("meta", {})

        text_list = text_list = info_dict.get("text")

        span_len = int(input_ids.shape[0])
        if span_len <= 0:
            return input_ids, input_embeds if input_embeds is not None else self.embed_input_ids(input_ids), {}

        text_list = info_dict.get("text")
        if not isinstance(text_list, list) or not text_list or not text_list[0]:
            raise ValueError("Missing additional_information.text for Qwen3-TTS AR talker.")

        task_type = info_dict.get("task_type") or ["clone"]
        codec_streaming = task_type == "clone"

        prompt_embeds_cpu = embed.get("prefill")
        tts_pad_embed_cpu = embed.get("tts_pad")
        tts_pad_embed = None
        if isinstance(tts_pad_embed_cpu, torch.Tensor) and tts_pad_embed_cpu.numel() > 0:
            tts_pad_embed = tts_pad_embed_cpu.to(device=input_ids.device, dtype=torch.bfloat16).reshape(1, -1)

        # Subsequent prefill rounds (multi-chunk): prompt_embeds_cpu is a Tensor stored by the first round.
        is_first_prefill = not isinstance(prompt_embeds_cpu, torch.Tensor) or prompt_embeds_cpu.ndim != 2
        if is_first_prefill:
            full_prompt_embeds, ref_code_len, ref_code, tts_pad_embed = self._build_prompt_embeds(
                task_type=task_type, info_dict=info_dict
            )
            # Store full prompt embeddings on CPU (large, prefill-only).
            # tailing_text_hidden and tts_pad_embed stay on GPU (gpu_resident_buffer_keys).
            prompt_embeds_cpu = full_prompt_embeds.detach().to("cpu").contiguous()
            info_update: OmniPayload = {
                "embed": {
                    "prefill": prompt_embeds_cpu,
                    "tts_pad": tts_pad_embed.detach(),
                },
                "meta": {
                    "talker_prefill_offset": 0,
                    "talker_text_offset": 0,
                    "codec_streaming": codec_streaming,
                },
            }
            if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0:
                info_update.setdefault("codes", {})["ref"] = ref_code.detach().to("cpu").contiguous()
            if ref_code_len is not None:
                info_update["meta"]["ref_code_len"] = int(ref_code_len)
            # Always return a span_len slice; if the scheduled placeholder is longer, pad with tts_pad_embed.
            # This preserves placeholder/embedding alignment.
            offset = 0
            s = 0
            e = span_len
            take = prompt_embeds_cpu[s:e]
            prompt_embeds = take.to(device=input_ids.device, dtype=torch.bfloat16)
            info_update["meta"]["talker_prefill_offset"] = int(offset + span_len)
        else:
            offset = int(meta.get("talker_prefill_offset", 0) or 0)
            if offset < 0:
                offset = 0
            s = max(0, min(offset, int(prompt_embeds_cpu.shape[0])))
            e = max(0, min(offset + span_len, int(prompt_embeds_cpu.shape[0])))
            take = prompt_embeds_cpu[s:e]
            if int(take.shape[0]) < span_len:
                pad_n = int(span_len - int(take.shape[0]))
                pad_rows = tts_pad_embed.reshape(1, -1).to("cpu").expand(pad_n, -1)
                take = torch.cat([take, pad_rows], dim=0)
            # Subsequent prefill chunk: slice from stored embeddings at running offset.
            prompt_embeds = take.to(device=input_ids.device, dtype=torch.bfloat16)
            info_update = {
                "meta": {
                    "talker_prefill_offset": int(offset + span_len),
                    "codec_streaming": codec_streaming,
                }
            }
        # When inputs_embeds is set, token ids are ignored by the model but must stay in-vocab for vLLM bookkeeping.
        input_ids_out = input_ids.clone()
        input_ids_out[:] = 0

        zeros = torch.zeros(
            (prompt_embeds.shape[0], 1),
            device=input_ids.device,
            dtype=torch.long,
        )
        info_update.setdefault("codes", {})["audio"] = zeros
        return input_ids_out, prompt_embeds, info_update

    def _encode_ref_audio_to_code(self, wav: np.ndarray, sr: int) -> torch.Tensor:
        try:
            self._ensure_audio_tokenizer_loaded()
            audio_prompt, codec_token = self.tokenizer._audio_tokenize(wav, sr=int(sr), return_dict=True)
            return audio_prompt, codec_token
        except Exception as e:
            logger.error("Failed to tokenize audio prompt", exc_info=e)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        return self.model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        inner_loaded = self.model.load_weights(weights)  # set[str]
        fixed = set()

        suffixes = (".input_layernorm.weight", ".post_attention_layernorm.weight", ".norm.weight")

        for name in inner_loaded:
            if name.startswith("model.") and name.endswith(suffixes):
                # model.layers... -> model.model.layers...
                fixed.add("model." + name)
            else:
                fixed.add(name)

        return fixed
