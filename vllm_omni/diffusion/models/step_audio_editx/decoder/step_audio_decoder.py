from functools import cached_property, reduce
from typing import List, Optional, Union, Iterable
from copy import deepcopy
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
from vllm_omni.diffusion.models.step_audio_editx.decoder.flow import CausalMaskedDiffWithXvec
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import HiFTGenerator
from vllm_omni.model_executor.models.cosyvoice3.utils import mel_spectrogram
import threading
import onnxruntime
import torchaudio
from typing import Dict
import torchaudio.compliance.kaldi as kaldi
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

"""perform fade_in_out in tensor style
"""
def fade_in_out(fade_in_mel:torch.Tensor, fade_out_mel:torch.Tensor, window:torch.Tensor):
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel = fade_in_mel.clone()
    fade_in_mel[..., :mel_overlap_len] = \
        fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + \
        fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    return fade_in_mel


class CosyVoiceFrontEnd(object):
    def __init__(self, 
                 mel_conf:Dict,
                 campplus_model:str,
                 onnx_provider:str='CUDAExecutionProvider',
                 ):
        super().__init__()
        assert onnx_provider in ['CUDAExecutionProvider', 'CPUExecutionProvider'], 'invalid onnx provider'
        self.mel_conf = mel_conf
        self.sample_rate = mel_conf['sampling_rate']
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(
            campplus_model, sess_options=option,
            providers=["CPUExecutionProvider"]
        )
    
    def extract_speech_feat(self, audio:torch.Tensor, audio_sr:int):
        if audio_sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=audio_sr, new_freq=self.sample_rate)
            audio_sr = self.sample_rate
        speech_feat = mel_spectrogram(y=audio, **self.mel_conf).transpose(1, 2) # (b=1, t, num_mels)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.long)
        return speech_feat, speech_feat_len

    def extract_spk_embedding(self, audio:torch.Tensor, audio_sr:int):
        if audio_sr != 16000:
            audio = torchaudio.functional.resample(audio, orig_freq=audio_sr, new_freq=16000)
            audio_sr = 16000
        feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        onnx_in = {
            self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()
        }
        embedding = self.campplus_session.run(None, onnx_in)[0].flatten().tolist()
        embedding = torch.tensor([embedding])
        return embedding


"""
A wrapper for managing stream caches. 
"""
class CosyVoice_stream_impl_(torch.nn.Module):
    def __init__(self, 
                 flow: CausalMaskedDiffWithXvec,
                 hift: Union[HiFTGenerator], #  hift: Union[HiFTGenerator, BigVGAN],
                 chunk_size_list: List = [15, 24, 48],  # (0.6s, 0.96s, 1.92s) 
                 mel_cache_len: int = 8,
                 n_timesteps: int = 10, # for both stream/non-stream
                 ):
        super().__init__()
        self.flow = flow
        self.hift = hift
        self.n_timesteps = n_timesteps
        # hard coded!
        # self.sample_rate = hift.sampling_rate
        self.token_lookahead = flow.pre_lookahead_len
        # stream conf
        self.mel_cache_len = mel_cache_len

        self.source_cache_len = int(mel_cache_len * 480)   # 50hz mel -> 24k wave

        self.register_buffer('speech_window', torch.from_numpy(np.hamming(2 * self.source_cache_len)), persistent=False)
        # session management
        self.speech_token_dict = defaultdict(list)
        self.chunk_size_list = chunk_size_list
        self.chunk_size_dict = {}
        self.b_first_chunk_dict = {}  # indicate if it's the first chunk of this session
        # hifigan cache
        self.hift_cache_dict = {}
        # model att/cnn cache
        self.chunk_cache_dict = {}
        self.estimator_prompt_length_dict = {}
        # speaker embedding cache
        self.spk_embedding_cache_dict = {}
        # setup lock
        self.setup_lock = threading.Lock()

    @cached_property
    def device(self):
        return next(self.hift.parameters()).device
    
    @cached_property
    def dtype(self):
        return next(self.hift.parameters()).dtype
    
    """NOTE Non-stream interface.
    """
    def token2wav_nonstream(self,
                            token: torch.Tensor,
                            prompt_token: torch.Tensor,
                            prompt_feat: torch.Tensor,
                            embedding: torch.Tensor,
                            ):
        def _make_len(ts:torch.Tensor):
            return torch.tensor([ts.shape[1]], dtype=torch.long, device=ts.device)
        # [02, 02, 06, 06, 06] -> [[02, 02, PAD], [06, 06, 06]]

        token = self._reshape(
            token.squeeze().tolist()
        ).unsqueeze(0)
        prompt_token = self._reshape(
            prompt_token.squeeze().tolist()
        ).unsqueeze(0)
        # align prompt mel
        prompt_feat = F.interpolate(
            prompt_feat.transpose(1, 2), 
            size=prompt_token.shape[1]*2, 
            mode='nearest'
        ).transpose(1, 2)
        
        token, prompt_token, prompt_feat, embedding = map(
            lambda ts: ts.to(self.device),
            (token, prompt_token, prompt_feat, embedding),
        )
        # inference flow
        mel = self.flow.inference(
            token, 
            _make_len(token),
            prompt_token,
            _make_len(prompt_token),
            prompt_feat.to(self.dtype),
            _make_len(prompt_feat),
            embedding.to(self.dtype),
            self.n_timesteps,
        )
        speech, _ = self.hift.inference(mel)
        speech = speech.cpu().to(torch.float32)
        return speech
    
    def _setup_cache(self,
                     token: torch.Tensor,
                     mel: torch.Tensor,
                     spk: torch.Tensor,
                     session_id: str,
                     ):
        # att/cnn-cache
        with self.setup_lock:
            cache = self.flow.setup_cache(
                token.to(self.device), 
                mel.to(self.device, self.dtype),
                spk.to(self.device, self.dtype),
                self.n_timesteps,
            )
            # 对 cache dict 里的每个 tensor 做 clone().detach()
            cache = {k: (v.clone().detach() if isinstance(v, torch.Tensor) else v) for k, v in cache.items()}
            self.chunk_cache_dict[session_id] = cache
            self.estimator_prompt_length_dict[session_id] = mel.shape[1]
            self.b_first_chunk_dict[session_id] = True
            # spk embedding
            self.spk_embedding_cache_dict[session_id] = spk.to(self.device, self.dtype).clone()
            # hift cache
            self.hift_cache_dict[session_id] = dict(
                mel = torch.zeros(1, mel.shape[2], 0, device=self.device, dtype=self.dtype), 
                source = torch.zeros(1, 1, 0, device=self.device, dtype=self.dtype),
                speech = torch.zeros(1, 0, device=self.device, dtype=self.dtype),
            )
            return 

    def _token2wav_stream(self,
                          token: torch.Tensor,
                          session_id: str,
                          last_chunk: bool,
                          ):
        
        assert session_id in self.chunk_cache_dict, 'call setup_cache first to obtain cache'
        # fetch cache & speaker embedding
        cache = self.chunk_cache_dict[session_id]
        embedding = self.spk_embedding_cache_dict[session_id]
        # inference this chunk
        mel, new_cache = self.flow.inference_chunk(
            token.to(self.device), # int64
            embedding,
            cache,
            last_chunk,
            self.n_timesteps,
        )
        # NOTE(sfy) truncate attention cache (prompt_length + 2s left context)
        left_context_length = int(2 * 48)
        estimator_att_cache = new_cache['estimator_att_cache']
        prompt_length = self.estimator_prompt_length_dict[session_id]
        if estimator_att_cache.shape[4] > (prompt_length + left_context_length):
            new_cache['estimator_att_cache'] = torch.cat([
                estimator_att_cache[:, :, :, :, :left_context_length],
                estimator_att_cache[:, :, :, :, -prompt_length:],
            ], dim=4)

        self.chunk_cache_dict[session_id] = {k: v.clone().detach() for k, v in new_cache.items()}
        # vocoder cache
        hift_cache_mel = self.hift_cache_dict[session_id]['mel']
        hift_cache_source = self.hift_cache_dict[session_id]['source']
        hift_cache_speech = self.hift_cache_dict[session_id]['speech']
        mel = torch.concat([hift_cache_mel, mel], dim=2)
        # inference vocoder
        speech, source = self.hift.inference(mel, hift_cache_source)
        # overlap speech smooth
        if hift_cache_speech.shape[-1] > 0:
            speech = fade_in_out(speech, hift_cache_speech, self.speech_window)
        # update vocoder cache
        self.hift_cache_dict[session_id] = dict(
            mel = mel[..., -self.mel_cache_len:].clone().detach(),
            source = source[:, :, -self.source_cache_len:].clone().detach(),
            speech = speech[:, -self.source_cache_len:].clone().detach(),
        )
        if not last_chunk:
            speech = speech[:, :-self.source_cache_len]
        return speech.cpu().to(torch.float32)

    @staticmethod
    def _reshape(mix_seq: List[int])->torch.Tensor:
        # assert len(mix_seq)%5 == 0, len(mix_seq)
        # NOTE add padding to avoid assert error 
        # (don't care the final speech as it's wrong anyway)
        if len(mix_seq)%5 > 0:
            pad_len = 5-(len(mix_seq)%5)
            mix_seq += [0, 0, 0, 1024, 1024, 1024][-pad_len:]

        num_groups = len(mix_seq) // 5
        vq02 = reduce(
            lambda x, y: x+y, 
            [mix_seq[i*5: i*5+2] + [1024] for i in range(num_groups)]
        )
        vq06 = reduce(
            lambda x, y: x+y, 
            [mix_seq[i*5+2: i*5+5] for i in range(num_groups)]
        )
        vq0206 = torch.stack([
            torch.tensor(vq02, dtype=torch.long),
            torch.tensor(vq06, dtype=torch.long)-1024+1025,
        ], dim=1)
        return vq0206

    """NOTE Stream interface. Called whenever one token is generated.
    NOTE(sfy) not need to transfer device or dtype

    This is a specialized version for vq0206, we change the mixed sequence to time-aligned sequence.
    eg.: [02, 02, 06, 06, 06] -> [[02, 02, PAD], [06, 06, 06]]
    """
    def token2wav_stream(self,
                         token: List[int], # vq0206 mixed seq tokens
                         prompt_token: torch.Tensor,
                         prompt_feat: torch.Tensor,
                         embedding: torch.Tensor,
                         session_id: str,
                         last_chunk: bool,
                         )->Optional[torch.Tensor]:
        # FIXME hard coded
        def _mixed_len(l:int):
            return (l // 3) * 5

        # init chunk size tracking
        if session_id not in self.chunk_size_dict:
            self.chunk_size_dict[session_id] = deepcopy(self.chunk_size_list)
        # add token
        self.speech_token_dict[session_id].extend(token)
        # waiting to setup cache
        mix_token_lookahead_len = _mixed_len(self.token_lookahead)
        if session_id not in self.chunk_cache_dict:
            if len(self.speech_token_dict[session_id]) >= mix_token_lookahead_len:
                # [02, 02, 06, 06, 06] -> [[02, 02, PAD], [06, 06, 06]]
                lookahead_token = self._reshape(
                    self.speech_token_dict[session_id][:mix_token_lookahead_len]
                ).unsqueeze(0)   # (1, t, 2)
                prompt_token = self._reshape(
                    prompt_token.squeeze().tolist()
                ).unsqueeze(0)
                # align prompt mel
                prompt_feat = F.interpolate(
                    prompt_feat.transpose(1, 2), 
                    size=prompt_token.shape[1]*2, 
                    mode='nearest'
                ).transpose(1, 2)
                self._setup_cache(
                    torch.cat([prompt_token, lookahead_token], dim=1),
                    prompt_feat,
                    embedding,
                    session_id,
                )
            return None
        
        # deal with remaining tokens
        if last_chunk:
            this_token = self.speech_token_dict[session_id]
        else:
        # cut to one chunk
            this_token = None
            mix_token_chunk_len = _mixed_len(self.chunk_size_dict[session_id][0])
            if len(self.speech_token_dict[session_id]) >= (mix_token_chunk_len+mix_token_lookahead_len):
                this_token = self.speech_token_dict[session_id][:(mix_token_chunk_len+mix_token_lookahead_len)]            
                self.speech_token_dict[session_id] = self.speech_token_dict[session_id][mix_token_chunk_len:]
        # go synthesis
        if this_token is not None:
            # [02, 02, 06, 06, 06] -> [[02, 02, PAD], [06, 06, 06]]
            this_token = self._reshape(this_token).unsqueeze(0)
            this_speech = self._token2wav_stream(
                this_token,
                session_id,
                last_chunk,
            )
            # update chunk size
            if len(self.chunk_size_dict[session_id]) > 1:
                self.chunk_size_dict[session_id].pop(0)
        else:
            this_speech = None
        # clear all caches
        if last_chunk:
            self.clean_up(session_id)
        return this_speech

    def clean_up(self, session_id: str):
        self.chunk_size_dict.pop(session_id, None)
        self.hift_cache_dict.pop(session_id, None)
        self.chunk_cache_dict.pop(session_id, None)
        self.estimator_prompt_length_dict.pop(session_id, None)
        self.spk_embedding_cache_dict.pop(session_id, None)
        self.speech_token_dict.pop(session_id, None)
        torch.cuda.empty_cache()


"""Keep compatible with cosyvoice1
"""
class CosyVoice:
    def __init__(self, 
                 model_dir:str, 
                 chunk_size_list: List = [15, 24, 48],  # (0.6s, 0.96s, 1.92s) 
                 mel_cache_len: int = 8,
                 n_timesteps: int = 10,
                 enable_cuda_graph: bool = False,
                 dtype=torch.float32,
                 yaml_path=None,
                 ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        # initiate streaming wrapper
        self.model_dir = model_dir
        config_path = yaml_path or f"{model_dir}/cosyvoice.yaml"
        with open(config_path, "r") as f:
            configs = load_hyperpyyaml(f)
        self.flow = CausalMaskedDiffWithXvec(configs["flow"])
        self.hift = HiFTGenerator(configs["hift"])
        # self.flow.load_state_dict(torch.load(f"{model_dir}/flow.pt", map_location='cpu'))
        # flow = flow.eval()
        # self.hift.load_state_dict(torch.load(f"{model_dir}/hift.pt", map_location='cpu'))
        # hift = hift.eval()
        cosy_impl = CosyVoice_stream_impl_(self.flow, self.hift, chunk_size_list, mel_cache_len, n_timesteps)
        self.cosy_impl = cosy_impl.to(self.device, self.dtype)
        if enable_cuda_graph:
            self.cosy_impl.flow.scatter_cuda_graph(enable_cuda_graph)
            self.cosy_impl.hift._init_cuda_graph()
        # feature frontend
        self.frontend = CosyVoiceFrontEnd(
            configs["mel_conf"],
            campplus_model='{}/campplus.onnx'.format(model_dir),
        )
    
    # Just proxy
    def token2wav_nonstream(self,
                            token: torch.Tensor,    # vq0206 mixed seq
                            prompt_token: torch.Tensor,
                            prompt_feat: torch.Tensor,
                            embedding: torch.Tensor,
                            )->torch.Tensor:
        return self.cosy_impl.token2wav_nonstream(
            token,
            prompt_token,
            prompt_feat,
            embedding,
        )
    
    # Just proxy
    def token2wav_stream(self,
                         token: List[int], # vq0206 mixed seq tokens
                         prompt_token: torch.Tensor,
                         prompt_feat: torch.Tensor,
                         embedding: torch.Tensor,
                         session_id: str,
                         last_chunk: bool,
                         )->Optional[torch.Tensor]:
        return self.cosy_impl.token2wav_stream(
            token,
            prompt_token,
            prompt_feat,
            embedding,
            session_id,
            last_chunk,
        )

    def clean_up(self, session_id: str):
        self.cosy_impl.clean_up(session_id)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        # self.flow.load_state_dict(torch.load(weights, map_location='cpu'))
        # self.hift.load_state_dict(torch.load(weights, map_location='cpu'))
        loaded_params = set()
        for name, loaded_weight in weights:
            if name not in params_dict:
                print(f"Miss key in ckpt: {name}")
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
