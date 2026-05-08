from io import BytesIO
import threading
import time
import numpy as np
import torch
import soundfile as sf
import onnxruntime
import whisper
import logging
import os.path
from .utils import resample_audio, energy_norm_fn, trim_silence
from tqdm import tqdm
from .model_loader import ModelSource, UnifiedModelLoader, prepare_data_iterator, AutoModel

logger = logging.getLogger(__name__)

class AudioProcessor(AutoModel):
    def __init__(self, model_path, model_source=ModelSource.HUGGINGFACE, **kwargs):
        super().__init__(model_path=model_path, model_source=model_source, **kwargs)
        self.kwargs = kwargs

    def inference(
        self, input, input_len=None, model=None, kwargs=None, key=None, **cfg
    ):
        kwargs = self.kwargs if kwargs is None else kwargs
        kwargs.update(cfg)
        model = self.model if model is None else model
        model = model.cuda()
        model.eval()

        batch_size = kwargs.get("batch_size", 1)

        key_list, data_list = prepare_data_iterator(
            input, input_len=input_len, data_type=kwargs.get("data_type", None), key=key
        )

        speed_stats = {}
        asr_result_list = []
        num_samples = len(data_list)
        disable_pbar = kwargs.get("disable_pbar", False)
        pbar = (
            tqdm(colour="blue", total=num_samples, dynamic_ncols=True)
            if not disable_pbar
            else None
        )
        time_speech_total = 0.0
        time_escape_total = 0.0
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

            time1 = time.perf_counter()
            with torch.no_grad():
                results, meta_data = model.inference(**batch, **kwargs)
            time2 = time.perf_counter()

            asr_result_list.extend(results)

            # batch_data_time = time_per_frame_s * data_batch_i["speech_lengths"].sum().item()
            batch_data_time = meta_data.get("batch_data_time", -1)
            time_escape = time2 - time1
            speed_stats["load_data"] = meta_data.get("load_data", 0.0)
            speed_stats["extract_feat"] = meta_data.get("extract_feat", 0.0)
            speed_stats["forward"] = f"{time_escape:0.3f}"
            speed_stats["batch_size"] = f"{len(results)}"
            speed_stats["time_cost"] = f"{(time_escape)}"
            speed_stats["rtf"] = f"{(time_escape) / batch_data_time:0.3f}"
            description = f"{speed_stats}, "
            if pbar:
                pbar.update(1)
                pbar.set_description(description)
            time_speech_total += batch_data_time
            time_escape_total += time_escape

        if pbar:
            # pbar.update(1)
            pbar.set_description(f"rtf_avg: {time_escape_total/time_speech_total:0.3f}")
        torch.cuda.empty_cache()
        return asr_result_list

    def infer_encoder(
        self, input, input_len=None, model=None, kwargs=None, key=None, **cfg
    ):
        kwargs = self.kwargs if kwargs is None else kwargs
        kwargs.update(cfg)
        model = self.model if model is None else model
        model = model.cuda()
        model.eval()

        batch_size = kwargs.get("batch_size", 1)

        key_list, data_list = prepare_data_iterator(
            input, input_len=input_len, data_type=kwargs.get("data_type", None), key=key
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

            with torch.no_grad():
                results, meta_data, cache = model.infer_encoder(**batch, **kwargs)
            asr_result_list.extend(results)

        torch.cuda.empty_cache()
        return asr_result_list, cache

class StepAudioTokenizer:
    def __init__(
        self,
        encoder_path,
        model_source=ModelSource.HUGGINGFACE,
        funasr_model_id="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online"
    ):
        """
        Initialize StepAudioTokenizer

        Args:
            encoder_path: Encoder path
            model_source: Model source (auto/local/modelscope/huggingface)
            funasr_model_id: FunASR model ID or path
        """
        model_loader = UnifiedModelLoader()
        funasr_model_path = os.path.join(encoder_path, funasr_model_id)
        # Load FunASR model - use unified loader to handle all modes
        try:
            self.funasr_model = model_loader.load_funasr_model(
                encoder_path,
                funasr_model_path,
                source=model_source,
                model_revision="main"
            )
        except Exception as e:
            print(f"Failed to load FunASR model from {model_source}: {e}")
            # Fallback to default method
            self.funasr_model = AutoModel(model=funasr_model_path, model_revision="main")

        # Load other resource files (these are usually local files)
        kms_path = os.path.join(self.funasr_model.repo_path, "linguistic_tokenizer.npy")
        cosy_tokenizer_path = os.path.join(self.funasr_model.repo_path, "speech_tokenizer_v1.onnx")

        if not os.path.exists(kms_path):
            raise FileNotFoundError(f"KMS file not found: {kms_path}")
        if not os.path.exists(cosy_tokenizer_path):
            raise FileNotFoundError(f"Cosy tokenizer file not found: {cosy_tokenizer_path}")

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

    def __call__(self, audio, sr):
        _, vq02, vq06 = self.wav2token(audio, sr, False)
        text = self.merge_vq0206_to_token_str(vq02, vq06)
        return text

    def preprocess_wav(self, audio, sample_rate, enable_trim=True, energy_norm=True):
        audio = resample_audio(audio, sample_rate, 16000)
        if energy_norm:
            audio = energy_norm_fn(audio)

        if enable_trim:
            audio = audio.cpu().numpy().squeeze(0)
            audio = trim_silence(audio, 16000)
            audio = torch.from_numpy(audio)
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
        _tmp_wav = BytesIO()
        # Use soundfile instead of torchaudio.save (TorchCodec doesn't support BytesIO)
        audio_np = audio.squeeze(0).cpu().numpy() if audio.dim() > 1 else audio.cpu().numpy()
        sf.write(_tmp_wav, audio_np, 16000, format='WAV')
        _tmp_wav.seek(0)

        with self.vq02_lock:
            cache = {}
            if session_id in self.vq02_sessions:
                cache = self.vq02_sessions[session_id].get("cache", {})

            res, new_cache = self.funasr_model.infer_encoder(
                input=[_tmp_wav],
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