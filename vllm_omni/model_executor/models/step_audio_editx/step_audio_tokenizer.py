import threading
import time
import os
import numpy as np
import torch
import onnxruntime
import whisper
import logging
import os.path
import yaml
from .utils import prepare_data_iterator, resample_audio, energy_norm_fn, trim_silence
from .audio_tokenizer.paraformer import ParaformerStreaming
from .audio_tokenizer.frontend import WavFrontendOnline
from typing import Iterable
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class FunASRModel:
    def __init__(self, model_path, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            kwargs = yaml.safe_load(f)
        assert "model" in kwargs
        kwargs["init_param"] = os.path.join(model_path, "model.pt")
        kwargs["frontend_conf"]["cmvn_file"] = os.path.join(model_path, "am.mvn")

        device = kwargs.get("device", "cuda")
        if not torch.cuda.is_available() or kwargs.get("ngpu", 1) == 0:
            device = "cpu"
            kwargs["batch_size"] = 1
        kwargs["device"] = device

        if kwargs.get("ncpu", None):
            torch.set_num_threads(kwargs.get("ncpu"))
        vocab_size = -1
        self.frontend = WavFrontendOnline(**kwargs["frontend_conf"])
        kwargs["frontend"] = self.frontend
        kwargs["input_size"] = self.frontend.output_size()
        self.model = ParaformerStreaming(**kwargs["model_conf"], vocab_size=vocab_size, encoder_conf=kwargs["encoder_conf"], input_size=kwargs["input_size"])
        # self.model.load_weight(kwargs["init_param"])
        self.model.to(device).eval()
        init_param = kwargs.get("init_param", None)
        if init_param is None:
            raise ValueError("init_param is required but was not provided or is None")
        self.kwargs = kwargs

    @torch.inference_mode()
    def infer_encoder(
        self, input, input_len=None, kwargs=None, key=None, **cfg
    ):
        kwargs = self.kwargs if kwargs is None else kwargs
        kwargs.update(cfg)
        batch_size = kwargs.get("batch_size", 1)
        key_list, data_list = prepare_data_iterator(
            input, data_type=kwargs.get("data_type", None), key=key
        )
        asr_result_list = []
        num_samples = len(data_list)
        for beg_idx in range(0, num_samples, batch_size):
            end_idx = min(num_samples, beg_idx + batch_size)
            data_batch = data_list[beg_idx:end_idx]
            key_batch = key_list[beg_idx:end_idx]
            batch = {"data_in": data_batch, "key": key_batch}
            if (end_idx - beg_idx) == 1 and kwargs.get(
                "data_type", None
            ) == "fbank":  # fbank
                batch["data_in"] = data_batch[0]
                batch["data_lengths"] = input_len

            results, meta_data, cache = self.model.infer_encoder(**batch, **kwargs, frontend=self.frontend)
            asr_result_list.extend(results)

        torch.cuda.empty_cache()
        return asr_result_list, cache
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return self.model.load_weights(weights)

class StepAudioTokenizer:
    def __init__(
        self,
        tokenizer_path,
        config_path,
        funasr_model_id="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online",
    ):
        self.text_tokenizer = AutoTokenizer.from_pretrained(config_path)
        self.funasr_model = FunASRModel(model_path=os.path.join(tokenizer_path, funasr_model_id), config_path=config_path)
        kms_path = os.path.join(tokenizer_path, "linguistic_tokenizer.npy")
        cosy_tokenizer_path = os.path.join(tokenizer_path, "speech_tokenizer_v1.onnx")
        self.kms = torch.tensor(np.load(kms_path))

        providers = ["CUDAExecutionProvider"]
        session_option = onnxruntime.SessionOptions()
        session_option.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_option.intra_op_num_threads = 1
        self.ort_session = onnxruntime.InferenceSession(
            cosy_tokenizer_path, sess_options=session_option, providers=providers
        )
        self.chunk_size = [0, 4, 5]
        self.encoder_chunk_look_back = 4
        self.decoder_chunk_look_back = 1

        self.vq02_sessions = {}
        self.vq02_lock = threading.Lock()
        self.vq06_lock = threading.Lock()

    def forward(self, task_type, audio, prompt, sr):
        audio_tokens, vq0206_codes = self._audio_tokenize(audio, sr)
        token_ids = self._text_tokenize(task_type, audio_tokens, prompt)
        return token_ids, vq0206_codes

    def _audio_tokenize(self, audio, sr):
        vq0206_codes, vq02_codes_ori, vq06_codes_ori = self.wav2token(audio, sr)
        vq0206_codes = torch.tensor(vq0206_codes, dtype=torch.long) - 65536
        audio_tokens = self.merge_vq0206_to_token_str(vq02_codes_ori, vq06_codes_ori)
        return audio_tokens, vq0206_codes
    
    def _text_tokenize(self, task_type, audio_tokens, prompt):
        if task_type == "edit":  
            prompt_text, edit_type, edit_info, target_text = prompt
            instruct_prefix = self._build_audio_edit_instruction(prompt_text, edit_type, edit_info, target_text)

            token_ids = self._encode_edit_prompt(
                self.edit_sys_prompt,
                instruct_prefix, 
                audio_tokens
            )

        if task_type == "clone":
            prompt_text, target_text = prompt
            prompt_speaker = "debug"
            token_ids = self._encode_clone_prompt(
                target_text,
                prompt_text,
                prompt_speaker,
                audio_tokens,
            )

        return token_ids   
    
    def preprocess_wav(self, audio, sample_rate, enable_trim=True, energy_norm=True):
        audio = resample_audio(audio, sample_rate, 16000)

        if audio.dim() == 2:
            audio = audio.squeeze(0)
        audio = audio.cpu().to(torch.float32)
            
        if energy_norm:
            audio = energy_norm_fn(audio)

        if enable_trim:
            audio_np = audio.numpy()
            audio_np = trim_silence(audio_np, 16000)
            audio = torch.from_numpy(audio_np).to(torch.float32)
            audio = audio.unsqueeze(0)
        return audio

    def wav2token(self, audio, sample_rate, enable_trim=True, energy_norm=True):
        audio = self.preprocess_wav(
            audio, sample_rate, enable_trim=enable_trim, energy_norm=energy_norm
        )

        vq02_ori = self.get_vq02_code(audio)
        vq02 = [int(x) + 65536 for x in vq02_ori]
        vq06_ori = self.get_vq06_code(audio)
        vq06 = [int(x) + 65536 + 1024 for x in vq06_ori]

        chunk = 1
        chunk_nums = min(len(vq06) // (3 * chunk), len(vq02) // (2 * chunk))
        speech_tokens = []
        for idx in range(chunk_nums):
            speech_tokens += vq02[idx * chunk * 2 : (idx + 1) * chunk * 2]
            speech_tokens += vq06[idx * chunk * 3 : (idx + 1) * chunk * 3]
        return speech_tokens, vq02_ori, vq06_ori

    def get_vq02_code(self, audio, session_id=None, is_final=True):
        if audio.dim() == 2:
            audio_in = audio.squeeze(0).cpu()
        else:
            audio_in = audio.cpu()

        with self.vq02_lock:
            cache = {}
            if session_id in self.vq02_sessions:
                cache = self.vq02_sessions[session_id].get("cache", {})

            res, new_cache = self.funasr_model.infer_encoder(
                input=[audio_in],
                chunk_size=self.chunk_size,
                encoder_chunk_look_back=self.encoder_chunk_look_back,
                decoder_chunk_look_back=self.decoder_chunk_look_back,
                device=0,
                is_final=is_final,
                cache=cache,
            )
            c_list = []
            for j, res_ in enumerate(res):
                feat = res_["enc_out"]
                if len(feat) > 0:
                    c_list = self.dump_label([feat], self.kms)[0]

            if is_final:
                if session_id in self.vq02_sessions:
                    self.vq02_sessions.pop(session_id)
            else:
                if isinstance(session_id, str) and len(session_id) > 0:
                    self.vq02_sessions[session_id] = {
                        "cache": new_cache,
                        "update_time": time.time(),
                    }

            return c_list

    def get_vq06_code(self, audio):
        def split_audio(audio, chunk_duration=480000):
            start = 0
            chunks = []
            while start < len(audio):
                end = min(start + chunk_duration, len(audio))
                chunk = audio[start:end]
                if len(chunk) < 480:
                    pass
                else:
                    chunks.append(chunk)
                start = end
            return chunks

        with self.vq06_lock:
            audio = audio.squeeze(0)
            chunk_audios = split_audio(audio, chunk_duration=30 * 16000)  # Maximum support 30s
            speech_tokens = []
            for chunk in chunk_audios:
                duration = round(chunk.shape[0] / 16000, 2)
                feat = whisper.log_mel_spectrogram(chunk, n_mels=128)
                feat = feat.unsqueeze(0)
                feat_len = np.array([feat.shape[2]], dtype=np.int32)
                chunk_token = (
                    self.ort_session.run(
                        None,
                        {
                            self.ort_session.get_inputs()[0]
                            .name: feat.detach()
                            .cpu()
                            .numpy(),
                            self.ort_session.get_inputs()[1].name: feat_len,
                        },
                    )[0]
                    .flatten()
                    .tolist()
                )
                assert abs(len(chunk_token) - duration * 25) <= 2
                speech_tokens += chunk_token

            return speech_tokens

    def kmean_cluster(self, samples, means):
        dists = torch.cdist(samples, means)
        indices = dists.argmin(dim=1).cpu().numpy()
        return indices.tolist()

    def dump_label(self, samples, mean):
        dims = samples[0].shape[-1]
        x_lens = [x.shape[1] for x in samples]
        total_len = sum(x_lens)
        x_sel = torch.FloatTensor(1, total_len, dims)
        start_len = 0
        for sample in samples:
            sample_len = sample.shape[1]
            end_len = start_len + sample_len
            x_sel[:, start_len:end_len] = sample
            start_len = end_len
        dense_x = x_sel.squeeze(0)
        indices = self.kmean_cluster(dense_x, mean)
        indices_list = []
        start_len = 0
        for x_len in x_lens:
            end_len = start_len + end_len
            indices_list.append(indices[start_len:end_len])
        return indices_list

    def merge_vq0206_to_token_str(self, vq02, vq06):
        _vq06 = [1024 + x for x in vq06]
        result = []
        i = 0
        j = 0
        while i < len(vq02) - 1 and j < len(_vq06) - 2:
            sublist = vq02[i : i + 2] + _vq06[j : j + 3]
            result.extend(sublist)
            i += 2
            j += 3
        return "".join([f"<audio_{x}>" for x in result])

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return self.funasr_model.load_weights(weights)

    def _encode_edit_prompt(
        self, sys_prompt: str, instruct_prefix: str, audio_token_str: str
    ) -> list[int]:
        """Encode audio edit prompt to token sequence"""
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"{instruct_prefix}\n{audio_token_str}\n"}
        ]

        return self.text_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)


    def _encode_clone_prompt(
        self, text: str, prompt_text: str, prompt_speaker: str, prompt_wav_tokens: str
    ):
        
        sys_prompt = self.edit_clone_sys_prompt_tpl.format(
            speaker=prompt_speaker,
            prompt_text=prompt_text,
            prompt_wav_tokens=prompt_wav_tokens
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"{text}"}
        ]

        return self.text_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    def _build_audio_edit_instruction(
        self,
        audio_text: str,
        edit_type: str,
        edit_info: Optional[str] = None,
        text: Optional[str] = None
    ) -> str:
        """Build audio editing instruction based on request"""
        audio_text = audio_text.strip() if audio_text else ""
        if edit_type in {"emotion", "speed"}:
            if edit_info == "remove":
                instruct_prefix = f"Remove any emotion in the following audio and the reference text is: {audio_text}\n"
            else:
                instruct_prefix = f"Make the following audio more {edit_info}. The text corresponding to the audio is: {audio_text}\n"
        elif edit_type == "style":
            if edit_info == "remove":
                instruct_prefix = f"Remove any speaking styles in the following audio and the reference text is: {audio_text}\n"
            else:
                instruct_prefix = f"Make the following audio more {edit_info} style. The text corresponding to the audio is: {audio_text}\n"
        elif edit_type == "denoise":
            instruct_prefix = f"Remove any noise from the given audio while preserving the voice content clearly. Ensure that the speech quality remains intact with minimal distortion, and eliminate all noise from the audio.\n"
        elif edit_type == "vad":
            instruct_prefix = f"Remove any silent portions from the given audio while preserving the voice content clearly. Ensure that the speech quality remains intact with minimal distortion, and eliminate all silence from the audio.\n"
        elif edit_type == "paralinguistic":
            instruct_prefix = f"Add some non-verbal sounds to make the audio more natural, the new text is : {text}\n  The text corresponding to the audio is: {audio_text}\n"
        else:
            logger.error(f"Unsupported audio editing type: {edit_type}")
        return instruct_prefix