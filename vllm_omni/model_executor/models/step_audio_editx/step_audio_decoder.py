import logging
import os
import threading
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from functools import cached_property, reduce
from typing import Any
from types import SimpleNamespace
import numpy as np
import onnxruntime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml
from vllm.config import VllmConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import SupportsPP

from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import CausalConditionalCFM
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import HiFTGenerator
from vllm_omni.model_executor.models.cosyvoice3.utils import mel_spectrogram
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.step_audio_editx.decoder.flow import (
    CausalMaskedDiffWithXvec,
    DiT,
    DualCodebookEmbedding,
    UpsampleConformerEncoderV2,
)
from vllm_omni.model_executor.models.step_audio_editx.decoder.hift import StepAudioCausalConvRNNF0Predictor

logger = logging.getLogger(__name__)


def fade_in_out(fade_in_mel: torch.Tensor, fade_out_mel: torch.Tensor, window: torch.Tensor):
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel = fade_in_mel.clone()
    fade_in_mel[..., :mel_overlap_len] = (
        fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len]
        + fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    )
    return fade_in_mel


class CosyVoiceFrontEnd:
    def __init__(
        self,
        mel_conf: dict,
        campplus_model: str,
        onnx_provider: str = "CUDAExecutionProvider",
    ):
        super().__init__()
        assert onnx_provider in ["CUDAExecutionProvider", "CPUExecutionProvider"], "invalid onnx provider"
        self.mel_conf = mel_conf
        self.sample_rate = mel_conf["sampling_rate"]
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(
            campplus_model, sess_options=option, providers=["CPUExecutionProvider"]
        )

    def extract_speech_feat(self, audio: torch.Tensor, audio_sr: int):
        if audio_sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=audio_sr, new_freq=self.sample_rate)
        speech_feat = mel_spectrogram(y=audio, **self.mel_conf).transpose(1, 2)
        return speech_feat

    def extract_spk_embedding(self, audio: torch.Tensor, audio_sr: int):
        if audio_sr != 16000:
            audio = torchaudio.functional.resample(audio, orig_freq=audio_sr, new_freq=16000)
        feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        onnx_in = {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()}
        embedding = self.campplus_session.run(None, onnx_in)[0].flatten().tolist()
        return torch.tensor([embedding])


class CosyVoice(nn.Module):
    def __init__(
        self,
        model_dir: str,
        chunk_size_list: list = [15, 24, 48],
        mel_cache_len: int = 8,
        n_timesteps: int = 10,
        dtype=torch.float32,
        yaml_path=None,
    ):
        super().__init__()
        self.model_dir = model_dir
        self._target_dtype = dtype
        config_path = yaml_path or f"{model_dir}/cosyvoice.yaml"
        with open(config_path) as f:
            configs = yaml.safe_load(f)

        flow_cfg = configs["flow"]
        decoder_cfg = flow_cfg["decoder"]

        input_embedding = DualCodebookEmbedding(**flow_cfg["input_embedding"])
        encoder = UpsampleConformerEncoderV2(**flow_cfg["encoder"])

        estimator = DiT(**decoder_cfg["estimator"])
        cfm_params = SimpleNamespace(**decoder_cfg["cfm_params"])
        decoder = CausalConditionalCFM(
            in_channels=decoder_cfg["in_channels"],
            n_spks=decoder_cfg["n_spks"],
            spk_emb_dim=decoder_cfg["spk_emb_dim"],
            cfm_params=cfm_params,
            estimator=estimator,
        )

        self.flow = CausalMaskedDiffWithXvec(
            input_embedding=input_embedding,
            encoder=encoder,
            decoder=decoder,
            input_size=flow_cfg["input_size"],
            output_size=flow_cfg["output_size"],
            spk_embed_dim=flow_cfg["spk_embed_dim"],
            output_type=flow_cfg["output_type"],
            vocab_size=flow_cfg["vocab_size"],
        )

        hift_cfg = configs["hift"]
        f0_predictor = StepAudioCausalConvRNNF0Predictor(**hift_cfg["f0_predictor"])
        self.hift = HiFTGenerator(
            **{k: v for k, v in hift_cfg.items() if k != "f0_predictor"},
            f0_predictor=f0_predictor,
        )

        self.frontend = CosyVoiceFrontEnd(
            configs["mel_conf"],
            campplus_model=f"{model_dir}/CosyVoice-300M-25Hz/campplus.onnx",
        )
        self.n_timesteps = n_timesteps
        self.token_lookahead = self.flow.pre_lookahead_len
        self.mel_cache_len = mel_cache_len
        self.source_cache_len = int(mel_cache_len * 480)
        self.register_buffer("speech_window", torch.from_numpy(np.hamming(2 * self.source_cache_len)), persistent=False)
        self.speech_token_dict = defaultdict(list)
        self.chunk_size_list = chunk_size_list
        self.chunk_size_dict = {}
        self.b_first_chunk_dict = {}
        self.hift_cache_dict = {}
        self.chunk_cache_dict = {}
        self.estimator_prompt_length_dict = {}
        self.spk_embedding_cache_dict = {}
        self.setup_lock = threading.Lock()

    def forward(self, token: torch.Tensor, prompt_token: torch.Tensor, input_wav: torch.Tensor, sample_rate: int):
        speech_feat, speech_embedding = self._feature_extract(input_wav, sample_rate)
        speech_feat = speech_feat.to(torch.bfloat16)
        speech_embedding = speech_embedding.to(torch.bfloat16)

        def _make_len(ts: torch.Tensor):
            return torch.tensor([ts.shape[1]], dtype=torch.long, device=ts.device)

        token = self._reshape(token.squeeze().tolist()).unsqueeze(0)
        prompt_token = self._reshape(prompt_token.squeeze().tolist()).unsqueeze(0)
        speech_feat = F.interpolate(
            speech_feat.transpose(1, 2), size=prompt_token.shape[1] * 2, mode="nearest"
        ).transpose(1, 2)

        token, prompt_token, speech_feat, speech_embedding = map(
            lambda ts: ts.to(self.device),
            (token, prompt_token, speech_feat, speech_embedding),
        )
        mel = self.flow.inference(
            token,
            _make_len(token),
            prompt_token,
            _make_len(prompt_token),
            speech_feat.to(self.dtype),
            _make_len(speech_feat),
            speech_embedding.to(self.dtype),
            self.n_timesteps,
        )
        speech, _ = self.hift.inference(mel)
        return speech.cpu().to(torch.float32)

    def chunk_decode_streaming(
        self,
        token: list[int],
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        session_id: str,
        last_chunk: bool,
    ) -> torch.Tensor | None:
        def _mixed_len(length: int):
            return (length // 3) * 5

        if session_id not in self.chunk_size_dict:
            self.chunk_size_dict[session_id] = deepcopy(self.chunk_size_list)
        self.speech_token_dict[session_id].extend(token)
        mix_token_lookahead_len = _mixed_len(self.token_lookahead)
        if session_id not in self.chunk_cache_dict:
            if len(self.speech_token_dict[session_id]) >= mix_token_lookahead_len:
                lookahead_token = self._reshape(self.speech_token_dict[session_id][:mix_token_lookahead_len]).unsqueeze(
                    0
                )
                prompt_token = self._reshape(prompt_token.squeeze().tolist()).unsqueeze(0)
                prompt_feat = F.interpolate(
                    prompt_feat.transpose(1, 2), size=prompt_token.shape[1] * 2, mode="nearest"
                ).transpose(1, 2)
                self._setup_cache(
                    torch.cat([prompt_token, lookahead_token], dim=1),
                    prompt_feat,
                    embedding,
                    session_id,
                )
            return None

        if last_chunk:
            this_token = self.speech_token_dict[session_id]
        else:
            this_token = None
            mix_token_chunk_len = _mixed_len(self.chunk_size_dict[session_id][0])
            if len(self.speech_token_dict[session_id]) >= (mix_token_chunk_len + mix_token_lookahead_len):
                this_token = self.speech_token_dict[session_id][: (mix_token_chunk_len + mix_token_lookahead_len)]
                self.speech_token_dict[session_id] = self.speech_token_dict[session_id][mix_token_chunk_len:]
        if this_token is not None:
            this_token = self._reshape(this_token).unsqueeze(0)
            this_speech = self._token2wav_stream(
                this_token,
                session_id,
                last_chunk,
            )
            if len(self.chunk_size_dict[session_id]) > 1:
                self.chunk_size_dict[session_id].pop(0)
        else:
            this_speech = None
        if last_chunk:
            self.clean_up(session_id)
        return this_speech

    def _token2wav_stream(
        self,
        token: torch.Tensor,
        session_id: str,
        last_chunk: bool,
    ):
        assert session_id in self.chunk_cache_dict, "call setup_cache first to obtain cache"
        cache = self.chunk_cache_dict[session_id]
        embedding = self.spk_embedding_cache_dict[session_id]
        mel, new_cache = self.flow.inference_chunk(
            token.to(self.device),
            embedding,
            cache,
            last_chunk,
            self.n_timesteps,
        )
        left_context_length = int(2 * 48)
        estimator_att_cache = new_cache["estimator_att_cache"]
        prompt_length = self.estimator_prompt_length_dict[session_id]
        if estimator_att_cache.shape[4] > (prompt_length + left_context_length):
            new_cache["estimator_att_cache"] = torch.cat(
                [
                    estimator_att_cache[:, :, :, :, :left_context_length],
                    estimator_att_cache[:, :, :, :, -prompt_length:],
                ],
                dim=4,
            )

        self.chunk_cache_dict[session_id] = {k: v.clone().detach() for k, v in new_cache.items()}
        hift_cache_mel = self.hift_cache_dict[session_id]["mel"]
        hift_cache_source = self.hift_cache_dict[session_id]["source"]
        hift_cache_speech = self.hift_cache_dict[session_id]["speech"]
        mel = torch.concat([hift_cache_mel, mel], dim=2)
        speech, source = self.hift.inference(mel, hift_cache_source)
        if hift_cache_speech.shape[-1] > 0:
            speech = fade_in_out(speech, hift_cache_speech, self.speech_window)
        self.hift_cache_dict[session_id] = dict(
            mel=mel[..., -self.mel_cache_len :].clone().detach(),
            source=source[:, :, -self.source_cache_len :].clone().detach(),
            speech=speech[:, -self.source_cache_len :].clone().detach(),
        )
        if not last_chunk:
            speech = speech[:, : -self.source_cache_len]
        return speech.cpu().to(torch.float32)

    def _feature_extract(self, input_wav: torch.Tensor, sr: int):
        if input_wav.shape[0] > 1:
            input_wav = input_wav.mean(dim=0, keepdim=True)
        norm = torch.max(torch.abs(input_wav), dim=1, keepdim=True)[0]
        if torch.any(norm > 0.6):
            input_wav = input_wav / norm.clamp_min(1e-6) * 0.6

        speech_feat = self.frontend.extract_speech_feat(input_wav, sr)
        speech_embedding = self.frontend.extract_spk_embedding(input_wav, sr)
        return speech_feat, speech_embedding

    def _setup_cache(
        self,
        token: torch.Tensor,
        mel: torch.Tensor,
        spk: torch.Tensor,
        session_id: str,
    ):
        with self.setup_lock:
            cache = self.flow.setup_cache(
                token.to(self.device),
                mel.to(self.device, self.dtype),
                spk.to(self.device, self.dtype),
                self.n_timesteps,
            )
            cache = {k: (v.clone().detach() if isinstance(v, torch.Tensor) else v) for k, v in cache.items()}
            self.chunk_cache_dict[session_id] = cache
            self.estimator_prompt_length_dict[session_id] = mel.shape[1]
            self.b_first_chunk_dict[session_id] = True
            self.spk_embedding_cache_dict[session_id] = spk.to(self.device, self.dtype).clone()
            self.hift_cache_dict[session_id] = dict(
                mel=torch.zeros(1, mel.shape[2], 0, device=self.device, dtype=self.dtype),
                source=torch.zeros(1, 1, 0, device=self.device, dtype=self.dtype),
                speech=torch.zeros(1, 0, device=self.device, dtype=self.dtype),
            )

    @cached_property
    def device(self):
        return next(self.hift.parameters()).device

    @cached_property
    def dtype(self):
        return next(self.hift.parameters()).dtype

    @staticmethod
    def _reshape(mix_seq: list[int]) -> torch.Tensor:
        if len(mix_seq) % 5 > 0:
            pad_len = 5 - (len(mix_seq) % 5)
            mix_seq += [0, 0, 0, 1024, 1024, 1024][-pad_len:]

        num_groups = len(mix_seq) // 5
        vq02 = reduce(lambda x, y: x + y, [mix_seq[i * 5 : i * 5 + 2] + [1024] for i in range(num_groups)])
        vq06 = reduce(lambda x, y: x + y, [mix_seq[i * 5 + 2 : i * 5 + 5] for i in range(num_groups)])
        vq0206 = torch.stack(
            [
                torch.tensor(vq02, dtype=torch.long),
                torch.tensor(vq06, dtype=torch.long) - 1024 + 1025,
            ],
            dim=1,
        )
        return vq0206

    def clean_up(self, session_id: str):
        self.speech_token_dict.pop(session_id, None)
        self.chunk_size_dict.pop(session_id, None)
        self.b_first_chunk_dict.pop(session_id, None)
        self.hift_cache_dict.pop(session_id, None)
        self.chunk_cache_dict.pop(session_id, None)
        self.estimator_prompt_length_dict.pop(session_id, None)
        self.spk_embedding_cache_dict.pop(session_id, None)


class StepAudioCode2wav(nn.Module, SupportsPP):
    DEBUG_MARKER = "STEP_AUDIO_CODE2WAV_RUNNER_MODEL"
    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        self.core = CosyVoice(
            model_dir=self.model_path,
            yaml_path="/root/workspace/vllm-omni/vllm_omni/model_executor/models/step_audio_editx/decoder/cosyvoice.yaml",
        )

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states, sampling_metadata: Any = None):
        return None

    @staticmethod
    def _extract_runtime_inputs(
        runtime_additional_information: list[dict[str, Any]] | None,
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor | None, int | None]:
        if runtime_additional_information:
            info = runtime_additional_information[0] or {}
            meta = info.get("meta", {}) if isinstance(info, dict) else {}
            input_wav = meta.get("input_wav") or info.get("input_wav")
            sample_rate = meta.get("sample_rate") or info.get("sample_rate")
            if input_wav is not None:
                return input_wav, sample_rate
        return kwargs.get("input_wav"), kwargs.get("sample_rate")

    @staticmethod
    def _extract_prompt_token(intermediate_tensors: Any, kwargs: dict[str, Any]) -> torch.Tensor | None:
        if intermediate_tensors is not None:
            if hasattr(intermediate_tensors, "get"):
                prompt_token = intermediate_tensors.get("vq0206_codes")
                if prompt_token is not None:
                    return prompt_token
            if hasattr(intermediate_tensors, "vq0206_codes"):
                return intermediate_tensors.vq0206_codes
        return kwargs.get("prompt_token")

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        if input_ids is None:
            raise ValueError("StepAudioCode2wav requires input_ids from the previous stage.")
        logger.info(f"intermediate_tensors: {intermediate_tensors}")
        token = input_ids.reshape(-1)
        prompt_token = self._extract_prompt_token(intermediate_tensors, kwargs)
        input_wav, sample_rate = self._extract_runtime_inputs(runtime_additional_information, kwargs)

        if prompt_token is None:
            raise ValueError("StepAudioCode2wav requires prompt_token (vq0206_codes) in intermediate_tensors.")
        if input_wav is None or sample_rate is None:
            raise ValueError(
                "StepAudioCode2wav requires input_wav and sample_rate in runtime_additional_information or kwargs."
            )

        audio = self.core.forward(token, prompt_token, input_wav, sample_rate)
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "audio": audio,
            },
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_params = set()
        params_dict = dict(self.core.flow.named_parameters())
        model_path = os.path.join(self.model_path, "CosyVoice-300M-25Hz")
        flow_weights = torch.load(f"{model_path}/flow.pt", map_location=self.core.device)
        hift_weights = torch.load(f"{model_path}/hift.pt", map_location=self.core.device)
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("attn.qkv_proj", "attn.to_q", "q"),
            ("attn.qkv_proj", "attn.to_k", "k"),
            ("attn.qkv_proj", "attn.to_v", "v"),
        ]
        for name, loaded_weight in flow_weights.items():
            mapped_name = f"core.flow.{name}"
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                try:
                    weight_loader(param, loaded_weight)
                    # print(f"loaded weight from {name} to {name}")
                except AssertionError as err:
                    raise AssertionError(f"Failed to load weight {name!r} as {name!r}") from err
                loaded_params.add(mapped_name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                stacked_name = name.replace(weight_name, param_name)
                if stacked_name not in params_dict:
                    continue
                param = params_dict[stacked_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                # print(f"loaded weight from {stacked_name} to {name}")
                loaded_params.add(stacked_name)
                break

        params_dict = dict(self.core.hift.named_parameters())
        for name, loaded_weight in hift_weights.items():
            mapped_name = f"core.hift.{name}"
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                try:
                    weight_loader(param, loaded_weight)
                    # print(f"loaded weight from {name} to {name}")
                except AssertionError as err:
                    raise AssertionError(f"Failed to load weight {name!r} as {name!r}") from err
                loaded_params.add(mapped_name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                stacked_name = name.replace(weight_name, param_name)
                if stacked_name not in params_dict:
                    continue
                param = params_dict[stacked_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                # print(f"loaded weight from {stacked_name} to {name}")
                loaded_params.add(stacked_name)
                break
        return loaded_params
