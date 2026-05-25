# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import os
from typing import Any
import numpy as np
import soundfile as sf

from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults
from vllm_omni.entrypoints.omni import Omni

import logging
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
        from vllm_omni.model_executor.models.step_audio_editx.step_audio_ar import StepAudioAR
        from transformers import AutoTokenizer
        from vllm_omni.model_executor.models.step_audio_editx.step_audio_tokenizer import StepAudioTokenizer
        speech_tok = StepAudioTokenizer(
            tokenizer_path=tokenizer_path,
            config_path=model_path,
        )
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
        task_type = (additional_information.get("task_type") or ["clone"])[0]

        def _estimate_ref_code_len(ref_audio: object) -> int | None:
            """Encode ref_audio with the actual codec to get exact frame count."""
            if not isinstance(ref_audio, (str, list)):
                return None
            audio_path = ref_audio[0] if isinstance(ref_audio, list) else ref_audio
            if not isinstance(audio_path, str) or not audio_path.strip():
                return None
            try:
                from urllib.parse import urlparse

                import numpy as np

                def _is_url(path: str) -> bool:
                    try:
                        parsed = urlparse(path)
                        if parsed.scheme in ("http", "https"):
                            return bool(parsed.netloc)
                        return parsed.scheme in ("file", "data")
                    except Exception:
                        return False

                if _is_url(audio_path):
                    from vllm.multimodal.media import MediaConnector

                    connector = MediaConnector(allowed_local_media_path="/")
                    audio, sr = connector.fetch_audio(audio_path)
                else:
                    from vllm.multimodal.media.audio import load_audio

                    audio, sr = load_audio(audio_path, sr=None, mono=True)

                wav_np = np.asarray(audio, dtype=np.float32)

                if speech_tok is not None:
                    enc = speech_tok.encode(wav_np, sr=int(sr), return_dict=True)
                    ref_code = getattr(enc, "token_ids", None)
                    if isinstance(ref_code, list):
                        ref_code = ref_code[0] if ref_code else None
                    if ref_code is not None and hasattr(ref_code, "shape"):
                        shape = ref_code.shape
                        return int(shape[0]) if len(shape) == 2 else int(shape[1]) if len(shape) == 3 else None

                codec_hz = 24
                return int(len(audio) / sr * codec_hz)
            except Exception:
                return None

        return StepAudioAR.estimate_prompt_len_from_additional_information(
            additional_information=additional_information,
            task_type=task_type,
            tokenize_prompt=lambda t: tok(t, padding=False)["input_ids"],
            estimate_ref_code_len=_estimate_ref_code_len,
        )
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
        "max_new_tokens": [2048],
    }
    inputs = {
        "prompt_token_ids": [0] * _estimate_prompt_len(additional_information, args.model, args.audio_tokenizer),
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
    )

    inputs = _build_inputs(args)

    print(f"Generating for prompt: {args.text}")

    # Start profiling (requires VLLM_TORCH_PROFILER_DIR env var)
    if os.environ.get("VLLM_TORCH_PROFILER_DIR"):
        print("Starting profiler...")
        omni.start_profile()

    outputs = list(omni.generate(inputs))

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
