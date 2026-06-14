import logging
import os
import threading
from collections import defaultdict
from collections.abc import Iterable
from functools import cached_property, reduce
from types import SimpleNamespace
from typing import Any

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

# from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import CausalConditionalCFM
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import HiFTGenerator

# from vllm_omni.model_executor.models.cosyvoice3.utils import mel_spectrogram
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.step_audio_editx.decoder.flow import (
    CausalMaskedDiffWithXvec,
    DiT,
    DualCodebookEmbedding,
    UpsampleConformerEncoderV2,
)
from vllm_omni.model_executor.models.step_audio_editx.decoder.hift import StepAudioCausalConvRNNF0Predictor

from .decoder.cfm import StepCausalConditionalCFM as CausalConditionalCFM
from .decoder.mel import mel_spectrogram
from .step_audio_tokenizer import StepAudioTokenizer

torch.backends.cudnn.enabled = False

logger = logging.getLogger(__name__)


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
        cfm_params = SimpleNamespace(**decoder_cfg["cfm_params"])
        self.flow = CausalMaskedDiffWithXvec(
            input_embedding=DualCodebookEmbedding(**flow_cfg["input_embedding"]),
            encoder=UpsampleConformerEncoderV2(**flow_cfg["encoder"]),
            decoder=CausalConditionalCFM(
                in_channels=decoder_cfg["in_channels"],
                n_spks=decoder_cfg["n_spks"],
                spk_emb_dim=decoder_cfg["spk_emb_dim"],
                cfm_params=cfm_params,
                estimator=DiT(**decoder_cfg["estimator"]),
            ),
            # decoder=CausalConditionalCFM(estimator=DiT(**decoder_cfg["estimator"])),
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
        self.flow = self.flow.to(device=self.device, dtype=torch.float32)
        self.hift = self.hift.to(device=self.device, dtype=torch.float32)
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
        self.emitted_tail_dict = {}
        self.chunk_debug_idx = {}

    def _feature_extract(self, input_wav: torch.Tensor, sr: int):
        if input_wav.shape[0] > 1:
            input_wav = input_wav.mean(dim=0, keepdim=True)
        norm = torch.max(torch.abs(input_wav), dim=1, keepdim=True)[0]
        if torch.any(norm > 0.6):
            input_wav = input_wav / norm.clamp_min(1e-6) * 0.6

        speech_feat = self.frontend.extract_speech_feat(input_wav, sr)
        speech_embedding = self.frontend.extract_spk_embedding(input_wav, sr)
        return speech_feat, speech_embedding

    @staticmethod
    def fade_in_out(fade_in_mel: torch.Tensor, fade_out_mel: torch.Tensor, window: torch.Tensor):
        mel_overlap_len = int(window.shape[0] / 2)
        fade_in_mel = fade_in_mel.clone()
        fade_in_mel[..., :mel_overlap_len] = (
            fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len]
            + fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
        )
        return fade_in_mel

    def forward(
        self, token: torch.Tensor, prompt_token: torch.Tensor, speech_feat: torch.Tensor, speech_embedding: torch.Tensor
    ):
        flow_device = next(self.flow.parameters()).device
        flow_dtype = next(self.flow.parameters()).dtype

        token = token.to(flow_device)
        prompt_token = prompt_token.to(flow_device)
        speech_feat = speech_feat.to(flow_device, flow_dtype)
        speech_embedding = speech_embedding.to(flow_device, flow_dtype)

        logger.error(
            "FLOW_DTYPE_DEBUG flow=%s token=%s prompt=%s feat=%s emb=%s",
            flow_dtype,
            token.dtype,
            prompt_token.dtype,
            speech_feat.dtype,
            speech_embedding.dtype,
        )

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
            speech_feat.to(flow_dtype),
            _make_len(speech_feat),
            speech_embedding.to(flow_dtype),
            self.n_timesteps,
        )
        logger.info(f"mel: {mel}")
        # hift_dtype = next(self.hift.parameters()).dtype
        mel = mel.to(self.device, torch.float32)
        speech, _ = self.hift.inference(mel)
        # mel = mel.to(self.device, hift_dtype)
        # speech, _ = self.hift.inference(mel)
        logger.info(f"speech: {speech}")

        return speech.cpu().to(torch.float32)

    def forward_chunk(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        prompt_embedding: torch.Tensor,
        session_id: str,
        last_chunk: bool,
    ):
        logger.error(
            "FORWARD_CHUNK token=%s prompt=%s feat=%s emb=%s session=%s last=%s",
            tuple(token.shape),
            tuple(prompt_token.shape),
            tuple(prompt_feat.shape),
            tuple(prompt_embedding.shape),
            session_id,
            last_chunk,
        )
        from copy import deepcopy

        def _mixed_len(length: int):
            return (length // 3) * 5

        if session_id not in self.chunk_size_dict:
            self.chunk_size_dict[session_id] = deepcopy(self.chunk_size_list)
        token = token.reshape(-1).detach().cpu().tolist()
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
                    prompt_embedding,
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
        hift_dtype = next(self.hift.parameters()).dtype
        hift_device = next(self.hift.parameters()).device

        mel = mel.to(device=hift_device, dtype=hift_dtype)
        hift_cache_mel = self.hift_cache_dict[session_id]["mel"].to(device=hift_device, dtype=hift_dtype)
        hift_cache_source = self.hift_cache_dict[session_id]["source"].to(device=hift_device)
        hift_cache_speech = self.hift_cache_dict[session_id]["speech"].to(device=hift_device)

        mel = torch.concat([hift_cache_mel, mel], dim=2)

        mel_f = mel.float()
        logger.info(
            "[ASYNC_MEL_DEBUG] session=%s last=%s mel_shape=%s mel_min=%.6f mel_max=%.6f "
            "mel_mean=%.6f mel_std=%.6f mel_abs_max=%.6f",
            session_id,
            last_chunk,
            tuple(mel.shape),
            float(mel_f.min().item()),
            float(mel_f.max().item()),
            float(mel_f.mean().item()),
            float(mel_f.std().item()),
            float(mel_f.abs().max().item()),
        )

        speech, source = self.hift.inference(mel, hift_cache_source)

        raw_speech = speech
        raw_diff = raw_speech.float().diff(dim=-1).abs()
        logger.info(
            "[ASYNC_RAW_SPEECH] session=%s last=%s raw_samples=%d raw_max_abs=%.6f "
            "raw_max_delta=%.6f raw_max_abs_idx=%d raw_max_delta_idx=%d "
            "mel_shape=%s source_shape=%s source_max_abs=%.6f",
            session_id,
            last_chunk,
            int(raw_speech.shape[-1]),
            float(raw_speech.abs().max().item()),
            float(raw_diff.max().item()) if raw_diff.numel() > 0 else 0.0,
            int(raw_speech.abs().reshape(-1).argmax().item()) if raw_speech.numel() > 0 else -1,
            int(raw_diff.reshape(-1).argmax().item()) if raw_diff.numel() > 0 else -1,
            tuple(mel.shape),
            tuple(source.shape),
            float(source.abs().max().item()) if source.numel() > 0 else 0.0,
        )

        if raw_diff.numel() > 0 and float(raw_diff.max().item()) > 1.5:
            torch.save(
                {
                    "mel": mel.detach().cpu(),
                    "speech": raw_speech.detach().cpu(),
                    "source": source.detach().cpu(),
                    "token": token.detach().cpu(),
                    "last_chunk": last_chunk,
                },
                f"/tmp/step_audio_spike_{session_id}_{last_chunk}.pt",
            )

        speech = speech.to(self.device)
        hift_cache_speech = hift_cache_speech.to(self.device)
        self.speech_window = self.speech_window.to(self.device)
        if hift_cache_speech.shape[-1] > 0:
            speech = self.fade_in_out(speech, hift_cache_speech, self.speech_window)
        self.hift_cache_dict[session_id] = dict(
            mel=mel[..., -self.mel_cache_len :].clone().detach(),
            source=source[:, :, -self.source_cache_len :].clone().detach(),
            speech=speech[:, -self.source_cache_len :].clone().detach(),
        )
        if not last_chunk:
            speech = speech[:, : -self.source_cache_len]

        post_diff = speech.float().diff(dim=-1).abs()
        logger.info(
            "[ASYNC_POST_FADE] session=%s last=%s post_samples=%d post_max_abs=%.6f "
            "post_max_delta=%.6f post_max_abs_idx=%d post_max_delta_idx=%d",
            session_id,
            last_chunk,
            int(speech.shape[-1]),
            float(speech.abs().max().item()),
            float(post_diff.max().item()) if post_diff.numel() > 0 else 0.0,
            int(speech.abs().reshape(-1).argmax().item()) if speech.numel() > 0 else -1,
            int(post_diff.reshape(-1).argmax().item()) if post_diff.numel() > 0 else -1,
        )

        chunk_idx = int(self.chunk_debug_idx.get(session_id, 0))
        self.chunk_debug_idx[session_id] = chunk_idx + 1
        prev_tail = self.emitted_tail_dict.get(session_id)
        if speech.numel() > 0:
            head_len = min(400, int(speech.shape[-1]))
            tail_len = min(400, int(speech.shape[-1]))
            boundary_jump = None
            if isinstance(prev_tail, torch.Tensor) and prev_tail.numel() > 0:
                boundary_jump = float((speech[:, 0] - prev_tail[:, -1]).abs().max().item())
            logger.info(
                "[ASYNC_AUDIO_BOUNDARY] session=%s chunk=%d last=%s samples=%d "
                "max_abs=%.6f head_rms=%.6f tail_rms=%.6f boundary_jump=%s",
                session_id,
                chunk_idx,
                last_chunk,
                int(speech.shape[-1]),
                float(speech.abs().max().item()),
                float(speech[:, :head_len].float().pow(2).mean().sqrt().item()),
                float(speech[:, -tail_len:].float().pow(2).mean().sqrt().item()),
                "None" if boundary_jump is None else f"{boundary_jump:.6f}",
            )
            self.emitted_tail_dict[session_id] = speech[:, -tail_len:].detach().clone()

        return speech.cpu().to(torch.float32)

    def _setup_cache(
        self,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        session_id: str,
    ):
        with self.setup_lock:
            cache = self.flow.setup_cache(
                prompt_token.to(self.device),
                prompt_feat.to(self.device, self.dtype),
                embedding.to(self.device, self.dtype),
                self.n_timesteps,
            )
            cache = {k: (v.clone().detach() if isinstance(v, torch.Tensor) else v) for k, v in cache.items()}
            self.chunk_cache_dict[session_id] = cache
            self.estimator_prompt_length_dict[session_id] = prompt_feat.shape[1]
            self.b_first_chunk_dict[session_id] = True
            self.spk_embedding_cache_dict[session_id] = embedding.to(self.device, self.dtype).clone()
            hift_dtype = next(self.hift.parameters()).dtype
            hift_device = next(self.hift.parameters()).device

            self.hift_cache_dict[session_id] = dict(
                mel=torch.zeros(1, prompt_feat.shape[2], 0, device=hift_device, dtype=hift_dtype),
                source=torch.zeros(1, 1, 0, device=hift_device, dtype=hift_dtype),
                speech=torch.zeros(1, 0, device=hift_device, dtype=hift_dtype),
            )

    def clean_up(self, session_id: str):
        self.speech_token_dict.pop(session_id, None)
        self.chunk_size_dict.pop(session_id, None)
        self.b_first_chunk_dict.pop(session_id, None)
        self.hift_cache_dict.pop(session_id, None)
        self.chunk_cache_dict.pop(session_id, None)
        self.estimator_prompt_length_dict.pop(session_id, None)
        self.spk_embedding_cache_dict.pop(session_id, None)
        self.emitted_tail_dict.pop(session_id, None)
        self.chunk_debug_idx.pop(session_id, None)

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


class StepAudioCode2wav(nn.Module):
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
            yaml_path="/root/autodl-tmp/vllm-omni/vllm_omni/model_executor/models/step_audio_editx/decoder/cosyvoice.yaml",
        )

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states, sampling_metadata: Any = None):
        return None

    def preprocess_wav(self, audio, sample_rate):
        audio, sample_rate = StepAudioTokenizer._load_audio(audio, sample_rate)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # volume-normalize avoid clipping
        norm = torch.max(torch.abs(audio), dim=1, keepdim=True)[0]
        if norm.item() > 0.6:
            audio = audio / norm * 0.6
        return audio, sample_rate

    @staticmethod
    def _extract_runtime_inputs(
        runtime_additional_information: list[dict[str, Any]] | None,
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor | None, int | None]:
        if (
            isinstance(runtime_additional_information, list)
            and runtime_additional_information
            and isinstance(runtime_additional_information[0], dict)
        ):
            ref = runtime_additional_information[0].get("latent", {})
            # logger.info(f"ref: {ref}")
            loaded = StepAudioTokenizer._load_audio(ref, kwargs.get("sample_rate"))
            if isinstance(loaded, list):
                if len(loaded) != 1:
                    raise ValueError(
                        "StepAudioCode2wav currently expects one reference audio per forward; "
                        f"got {len(loaded)}. Batched audio needs per-request decoding."
                    )
                ref_audio, sr = loaded[0]
            else:
                ref_audio, sr = loaded
            return ref_audio, sr
        return kwargs.get("input_wav"), kwargs.get("sample_rate")

    @staticmethod
    def _extract_prompt_token(intermediate_tensors: Any, kwargs: dict[str, Any]) -> torch.Tensor | None:
        if (
            isinstance(intermediate_tensors, list)
            and intermediate_tensors
            and isinstance(intermediate_tensors[0], dict)
        ):
            info = intermediate_tensors[0]

            ref = info.get("codes", {}).get("ref")
            if isinstance(ref, list):
                ref = torch.tensor(ref, dtype=torch.long)
            if isinstance(ref, torch.Tensor):
                ref = ref.to(torch.long)
                if ref.ndim > 1:
                    ref = ref[0]
                return ref.reshape(-1)

            ref = info.get("ids", {}).get("prompt")
            if isinstance(ref, list):
                ref = torch.tensor(ref, dtype=torch.long)
            if isinstance(ref, torch.Tensor):
                ref = ref.to(torch.long)
                if ref.ndim > 1:
                    ref = ref[0]
                return ref.reshape(-1)

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
        # logger.info(f"input_ids: {input_ids}")
        # logger.info(f"runtime_additional_information: {runtime_additional_information}, kwargs: {kwargs}")
        if input_ids is None:
            raise ValueError("StepAudioCode2wav requires input_ids from the previous stage.")
        token = input_ids.reshape(-1)
        prompt_token = self._extract_prompt_token(runtime_additional_information, kwargs)
        # logger.info(f"prompt_token: {prompt_token}")
        input_wav, sample_rate = self._extract_runtime_inputs(runtime_additional_information, kwargs)
        # logger.info(f"input_wav:{input_wav}, sample_rate:{sample_rate}")
        if input_wav is not None:
            input_wav, sample_rate = self.preprocess_wav(input_wav, sample_rate)
        if prompt_token is None:
            prompt_token = torch.zeros((5,), dtype=torch.long, device=token.device)

        if input_wav is None or sample_rate is None:
            sample_rate = 16000
            input_wav = torch.zeros((1, sample_rate), dtype=torch.float32, device=token.device)

        speech_feat, speech_embedding = self.core._feature_extract(input_wav, sample_rate)
        flow_dtype = next(self.core.flow.parameters()).dtype
        flow_device = next(self.core.flow.parameters()).device

        speech_feat = speech_feat.to(flow_device, flow_dtype)
        speech_embedding = speech_embedding.to(flow_device, flow_dtype)

        async_chunk = self.vllm_config.model_config.async_chunk
        if async_chunk:
            if runtime_additional_information is not None:
                meta = runtime_additional_information[0].get("meta", {})
                session_id = meta.get("req_id")
                last_chunk = meta.get("stream_finished", False)
                if hasattr(last_chunk, "item"):
                    last_chunk = bool(last_chunk.item())
                else:
                    last_chunk = bool(last_chunk)
                logger.info(f"session_id: {session_id}, last_chunk: {last_chunk}")
            else:
                session_id = None
                last_chunk = False
            logger.error(
                "ASYNC_CODE2WAV input_ids_shape=%s token_shape=%s token_numel=%d "
                "prompt_token_shape=%s prompt_token_numel=%d session_id=%s last_chunk=%s",
                tuple(input_ids.shape) if input_ids is not None else None,
                tuple(token.shape),
                token.numel(),
                tuple(prompt_token.shape) if isinstance(prompt_token, torch.Tensor) else None,
                prompt_token.numel() if isinstance(prompt_token, torch.Tensor) else -1,
                session_id,
                last_chunk,
            )
            logger.error(
                "ASYNC_CODE2WAV token min=%s max=%s prompt min=%s max=%s",
                int(token.min().item()) if token.numel() else None,
                int(token.max().item()) if token.numel() else None,
                int(prompt_token.min().item())
                if isinstance(prompt_token, torch.Tensor) and prompt_token.numel()
                else None,
                int(prompt_token.max().item())
                if isinstance(prompt_token, torch.Tensor) and prompt_token.numel()
                else None,
            )
            audio = self.core.forward_chunk(token, prompt_token, speech_feat, speech_embedding, session_id, last_chunk)
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "audio": audio,
                },
            )
        logger.error(
            "CODE2WAV_SYNC_CODES token_len=%d head=%s tail=%s prompt_len=%d ref_len=%s",
            int(token.numel()),
            token[:20].detach().cpu().tolist(),
            token[-20:].detach().cpu().tolist(),
            int(prompt_token.numel()) if isinstance(prompt_token, torch.Tensor) else -1,
            ...,
        )
        # logger.info(f"speech_feat: {speech_feat}, speech_embedding: {speech_embedding}")
        audio = self.core.forward(token, prompt_token, speech_feat, speech_embedding)
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
        for name, loaded_weight in flow_weights.items():
            mapped_name = f"core.flow.{name}"
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                try:
                    weight_loader(param, loaded_weight)
                except AssertionError as err:
                    raise AssertionError(f"Failed to load weight {name!r} as {name!r}") from err
                loaded_params.add(mapped_name)
                continue

        params_dict = dict(self.core.hift.named_parameters())
        for name, loaded_weight in hift_weights.items():
            mapped_name = f"core.hift.{name}"
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                try:
                    weight_loader(param, loaded_weight)
                except AssertionError as err:
                    raise AssertionError(f"Failed to load weight {name!r} as {name!r}") from err
                loaded_params.add(mapped_name)
                continue
        return loaded_params
