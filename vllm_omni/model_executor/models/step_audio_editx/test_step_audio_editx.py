# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import logging
import os
from typing import Any

import soundfile as sf
from vllm import SamplingParams

from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults
from vllm_omni.entrypoints.omni import Omni

logger = logging.getLogger(__name__)


def _estimate_prompt_len(
    additional_information: dict[str, Any],
    model_path,
    tokenizer_path,
    _cache: dict[str, Any] = {},
) -> int:
    """Estimate prompt_token_ids placeholder length for the Talker stage.

    The AR Talker replaces all input embeddings via ``preprocess``, so the
    placeholder values are irrelevant but the **length** must match the
    embeddings that ``preprocess`` will produce.
    """
    try:
        from vllm_omni.model_executor.models.step_audio_editx.step_audio_tokenizer import StepAudioTokenizer

        cache_key = (model_path, tokenizer_path)
        speech_tok = _cache.get(cache_key)
        if speech_tok is None:
            speech_tok = StepAudioTokenizer(
                tokenizer_path=tokenizer_path,
                config_path=model_path,
            )
            _cache[cache_key] = speech_tok

        def _first(x, default=None):
            if isinstance(x, list):
                return x[0] if x else default
            return x if x is not None else default

        task_type = _first(additional_information.get("task_type"), "clone")
        ref_audio = _first(additional_information.get("ref_audio"), None)
        ref_text = _first(additional_information.get("ref_text"), "")
        text = _first(additional_information.get("text"), "")
        sr = _first(additional_information.get("sr"), 16000)
        logger.info(f"task_type: {task_type}, ref_audio: {ref_audio}, ref_text: {ref_text}, text: {text}, sr: {sr}")
        prompt_token, _ = speech_tok.encode(
            task_type,
            audio=ref_audio,
            prompt=(ref_text, text),
            sr=sr,
        )

        return max(2, len(prompt_token.input_ids))

    except Exception as exc:
        logger.warning("Failed to estimate prompt length, using fallback 2048: %s", exc)
        return 2048


def get_base_query(args):
    """Build Base (voice clone) sample inputs.
    Returns:
        QueryResult with Omni inputs and the Base model path.
    """
    task_type = "clone"
    ref_audio_path_1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
    ref_audio_single = ref_audio_path_1
    ref_text_single = (
        "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
    )
    syn_text_single = "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye."
    additional_information = {
        "task_type": task_type,
        "ref_audio": [ref_audio_single],
        "ref_text": [ref_text_single],
        "text": [syn_text_single],
    }
    input_length = _estimate_prompt_len(additional_information, args.model, args.audio_tokenizer)
    inputs = {
        "prompt_token_ids": [0] * input_length,
        "additional_information": additional_information,
    }
    return inputs


def _build_inputs(args) -> tuple[str, list]:
    """Resolve model name and inputs list from CLI args."""
    inputs = get_base_query(args)
    inputs = inputs if isinstance(inputs, list) else [inputs]

    return inputs


def run_e2e():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to StepAudioEditx (e.g., stepfun-ai/Step-Audio-EditX).",
    )
    parser.add_argument(
        "--audio-tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer directory (e.g., stepfun-ai/Step-Audio-Tokenizer).",
    )
    parser.add_argument(
        "--deploy-config",
        type=str,
        default="vllm-omni/vllm_omni/deploy/step_audio_editx.yaml",
        help="Override the deploy config path. If unset, auto-loads "
        "vllm_omni/deploy/step_audio_editx.yaml based on the HF model_type.",
    )
    parser.add_argument("--text", type=str, default="Hello, this is a test of the StepAudioEditx system capability.")
    parser.add_argument(
        "--prompt-text",
        type=str,
        default="You are a helpful assistant.<|endofprompt|>希望你以后，能够做的比我还好呦!",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Path to reference audio for voice cloning. "
        "If unset, downloads the upstream CosyVoice3 zero-shot prompt audio clip",
    )
    parser.add_argument("--edit-type", type=str, default=None, help="Type of edit to perform. ")
    parser.add_argument("--edit-info", type=str, default=None, help="Additional information for the edit. ")
    nullify_stage_engine_defaults(parser)
    args = parser.parse_args()
    # Ensure tokenizer directory exists
    if not os.path.exists(args.audio_tokenizer):
        raise FileNotFoundError(f"{args.audio_tokenizer} does not exist!")

    if args.deploy_config is not None and not os.path.exists(args.deploy_config):
        raise FileNotFoundError(f"{args.deploy_config} does not exist!")

    print(f"Initializing StepAudioEditx E2E with model={args.model}")
    print(f"Deploy config: {args.deploy_config}")
    os.environ["STEP_AUDIO_TOKENIZER_PATH"] = args.audio_tokenizer
    omni = Omni(
        model=args.model,
        deploy_config=args.deploy_config,
        log_stats=True,
        trust_remote_code=True,
    )

    inputs = _build_inputs(args)

    print(f"Generating for prompt: {args.text}")

    # Start profiling (requires VLLM_TORCH_PROFILER_DIR env var)
    if os.environ.get("VLLM_TORCH_PROFILER_DIR"):
        print("Starting profiler...")
        omni.start_profile()
    prompt_token_ids = inputs[0].get("prompt_token_ids", [])
    print(f"Prompt length: {len(prompt_token_ids)}")
    prompt_len = len(prompt_token_ids)
    max_tokens = 8192 - prompt_len

    gpt_sampling = SamplingParams(temperature=0.7, max_tokens=max_tokens, skip_special_tokens=False)
    s2mel_sampling = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=2.0,
        max_tokens=256,
        detokenize=False,
    )
    sampling_params_list = [gpt_sampling, s2mel_sampling]
    outputs = list(omni.generate(inputs, sampling_params_list=sampling_params_list))

    if os.environ.get("VLLM_TORCH_PROFILER_DIR"):
        print("Stopping profiler...")
        profile_results = omni.stop_profile()
        print(f"Profile traces saved to: {profile_results}")

    print(outputs)
    # Verify outputs
    print(f"Received {len(outputs)} outputs.")
    for i, output in enumerate(outputs):
        try:
            ro = output.request_output
            if ro is None:
                print("No request_output found.")
                continue

            # Multimodal output may be attached to RequestOutput or CompletionOutput.
            mm = getattr(ro, "multimodal_output", None)
            if not mm and ro.outputs:
                mm = getattr(ro.outputs[0], "multimodal_output", None)

            if mm:
                print(f"Multimodal output keys: {mm.keys()}")
                if "audio" in mm:
                    audio_out = mm["audio"]
                    print(f"Generated Audio Shape: {audio_out.shape}")
                    out_path = f"output_{i}.wav"
                    sf.write(out_path, audio_out.cpu().numpy().squeeze(), 22050)
                    print(f"Saved audio to {out_path}")
            else:
                print("No multimodal output found.")
        except Exception as e:
            print(f"Error inspecting output: {e}")
    omni.close()


if __name__ == "__main__":
    run_e2e()
