import logging
import os
import tempfile

import soundfile as sf
import torch
from safetensors.torch import load_file

from .step_audio_decoder import StepAudioCode2wav
from .step_audio_tokenizer import StepAudioTokenizer

logger = logging.getLogger(__name__)


def test_tokenizer():
    tokenizer = StepAudioTokenizer(
        tokenizer_path="/root/workspace/Model/hub/models--stepfun-ai--Step-Audio-Tokenizer/snapshots/af7e5a3ec06175a7facae9d4100073d6e4dbb36c",
        funasr_model_id="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online",
    )
    wav_path = "/root/lanyun-tmp/vllm-omni/Rajat_sharma_hin_25s.wav"
    speech_np, sample_rate = sf.read(wav_path)
    if speech_np.ndim == 1:
        wav = torch.from_numpy(speech_np).float().unsqueeze(0)  # (samples,) -> (1, samples)
    else:
        wav = torch.from_numpy(speech_np).float().T
    print(f"Loaded wav file: {wav_path}, sample_rate: {sample_rate}, wav shape: {wav.shape}")
    vq0206_codes, vq02_codes_ori, vq06_codes_ori = tokenizer.wav2token(audio=wav, sample_rate=24000)

    # print(f"vq0206_codes: {vq0206_codes}, vq02_codes_ori: {vq02_codes_ori}, vq06_codes_ori: {vq06_codes_ori}")


def test_decoder():
    directory = "Model/hub/models--stepfun-ai--Step-Audio-EditX/snapshots/5fe2f8a05c2353301ad47d3c1747b262115da138/"
    path = "/CosyVoice-300M-25Hz"
    model_path = os.path.join(directory, path)
    cosyvoice1 = StepAudioCode2wav(
        model_dir=model_path,
        yaml_path="/root/lanyun-tmp/vllm-omni/vllm_omni/diffusion/models/step_audio_editx/decoder/cosyvoice.yaml",
    )
    print("Initialized CosyVoice Successfully!")
    device = cosyvoice1.device if hasattr(cosyvoice1, "device") else torch.device("cuda")
    dtype = cosyvoice1.dtype if hasattr(cosyvoice1, "dtype") else torch.float32

    token = torch.randint(low=0, high=4096, size=(1, 20), device=device, dtype=torch.long)
    prompt_token = torch.randint(low=0, high=4096, size=(1, 10), device=device, dtype=torch.long)
    prompt_feat = torch.randn(1, 20, 80, device=device, dtype=dtype)
    embedding = torch.randn(1, 192, device=device, dtype=dtype)

    with torch.inference_mode():
        wav = cosyvoice1.token2wav_nonstream(
            token=token,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
        )
    print(wav.shape)


def test_ar_vllm():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    from vllm_omni.model_executor.models.step_audio_editx.step_audio_ar import Step1CausalLMConfig, Step1ForCausalLM

    with set_current_vllm_config(VllmConfig()):
        tmp_file = tempfile.mkstemp()[1]
        backend = "nccl" if torch.cuda.is_available() else "gloo"

        if not torch.distributed.is_initialized():
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{tmp_file}",
                local_rank=0,
                backend=backend,
            )

        if not model_parallel_is_initialized():
            initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
            )

        config = Step1CausalLMConfig()
        # ar_model = Step1ForCausalLM(config=config).cuda()
        ar_model = Step1ForCausalLM(config=config).cuda()
        weights = load_file(
            "/root/lanyun-tmp/Model/hub/models--stepfun-ai--Step-Audio-EditX/snapshots/5fe2f8a05c2353301ad47d3c1747b262115da138/model-00001.safetensors",
            device="cuda",
        )
        ar_model.load_weights(weights)
        x = torch.tensor([10, 8, 1, 12, 9], device="cuda")
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # x = torch.ones(1,1,3072).cuda()
        output = ar_model(x)
        print(output)


def test_tokenizer_weight_load():
    tokenizer = StepAudioTokenizer(
        tokenizer_path="/root/workspace/Model/hub/models--stepfun-ai--Step-Audio-Tokenizer/snapshots/af7e5a3ec06175a7facae9d4100073d6e4dbb36c",
        funasr_model_id="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online",
        config_path="/root/workspace/vllm-omni/vllm_omni/diffusion/models/step_audio_editx/tokenizer/tokenizer.yaml",
    )
    wav_path = "/root/workspace/out_seed42_1.wav"
    speech_np, sample_rate = sf.read(wav_path)
    if speech_np.ndim == 1:
        wav = torch.from_numpy(speech_np).float().unsqueeze(0)  # (samples,) -> (1, samples)
    else:
        wav = torch.from_numpy(speech_np).float().T
    print(f"Loaded wav file: {wav_path}, sample_rate: {sample_rate}, wav shape: {wav.shape}")
    vq0206_codes, vq02_codes_ori, vq06_codes_ori = tokenizer.wav2token(audio=wav, sample_rate=24000)


if __name__ == "__main__":
    # main()
    # test_tokenizer()
    # test_ar_vllm()
    test_decoder()
