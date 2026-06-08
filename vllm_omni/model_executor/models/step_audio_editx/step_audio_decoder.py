import logging
import os
import threading
from collections import defaultdict
from collections.abc import Iterable
from functools import cached_property, reduce
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

from .decoder.cfm import CausalConditionalCFM
from .decoder.mel import mel_spectrogram
from .step_audio_tokenizer import StepAudioTokenizer

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

        self.flow = CausalMaskedDiffWithXvec(
            input_embedding=DualCodebookEmbedding(**flow_cfg["input_embedding"]),
            encoder=UpsampleConformerEncoderV2(**flow_cfg["encoder"]),
            decoder=CausalConditionalCFM(estimator=DiT(**decoder_cfg["estimator"])),
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
        flow_dtype = next(self.flow.parameters()).dtype
        flow_device = next(self.flow.parameters()).device

        speech_feat = speech_feat.to(flow_device, flow_dtype)
        speech_embedding = speech_embedding.to(flow_device, flow_dtype)

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
        hift_dtype = next(self.hift.parameters()).dtype
        mel = mel.to(self.device, hift_dtype)
        speech, _ = self.hift.inference(mel)
        return speech

    def _feature_extract(self, input_wav: torch.Tensor, sr: int):
        if input_wav.shape[0] > 1:
            input_wav = input_wav.mean(dim=0, keepdim=True)
        norm = torch.max(torch.abs(input_wav), dim=1, keepdim=True)[0]
        if torch.any(norm > 0.6):
            input_wav = input_wav / norm.clamp_min(1e-6) * 0.6

        speech_feat = self.frontend.extract_speech_feat(input_wav, sr)
        speech_embedding = self.frontend.extract_spk_embedding(input_wav, sr)
        return speech_feat, speech_embedding

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
            ref = runtime_additional_information[0].get("codes", {}).get("audio")
            ref_audio, sr = StepAudioTokenizer._load_audio(ref, kwargs.get("sample_rate"))
            return ref_audio, sr
        return kwargs.get("input_wav"), kwargs.get("sample_rate")

    @staticmethod
    def _extract_prompt_token(intermediate_tensors: Any, kwargs: dict[str, Any]) -> torch.Tensor | None:
        if (
            isinstance(intermediate_tensors, list)
            and intermediate_tensors
            and isinstance(intermediate_tensors[0], dict)
        ):
            ref = intermediate_tensors[0].get("codes", {}).get("ref")
            if isinstance(ref, torch.Tensor):
                return ref[0]
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
        token = input_ids.reshape(-1)
        prompt_token = self._extract_prompt_token(runtime_additional_information, kwargs)
        input_wav, sample_rate = self._extract_runtime_inputs(runtime_additional_information, kwargs)
        logger.info(f"input_wav:{input_wav}, sample_rate:{sample_rate}")
        if input_wav is not None:
            input_wav, sample_rate = self.preprocess_wav(input_wav, sample_rate)
        if prompt_token is None:
            prompt_token = torch.zeros((5,), dtype=torch.long, device=token.device)

        if input_wav is None or sample_rate is None:
            sample_rate = 16000
            input_wav = torch.zeros((1, sample_rate), dtype=torch.float32, device=token.device)

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
