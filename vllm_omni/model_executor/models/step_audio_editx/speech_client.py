"""OpenAI-compatible client for Step-Audio-Editx via /v1/audio/speech endpoint.

This script demonstrates how to use the OpenAI-compatible speech API
to generate audio from text using Step-Audio-Editx models.

Examples:
    # VoiceClone Task
    python openai_speech_client.py \
        --ref-audio /path/to/ref_audio.wav \
        --ref-text "This is the reference transcript" \
        --text "What Are you talking about"

    # VoiceEdit Task With Edit Info
    python openai_speech_client.py
        --task-type edit \
        --ref-audio /path/to/ref_audio.wav \
        --ref-text "This is the reference transcript" \
        --text "What Are you talking about" \
        --edit-type "emotion" \
        --edit-info "angry" \

    # VoiceEdit Task With Edit Info
    python openai_speech_client.py
        --task-type edit \
        --ref-audio /path/to/ref_audio.wav \
        --ref-text "This is the reference transcript" \
        --text "[cough]What Are you talking about" \
        --edit-type "paralinguistic" \
"""

import argparse
import base64
import os

import httpx

# Default server configuration
DEFAULT_API_BASE = "http://localhost:8091"
DEFAULT_API_KEY = "EMPTY"


def encode_audio_to_base64(audio_path: str) -> str:
    """Encode a local audio file to base64 data URL."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Detect MIME type from extension
    audio_path_lower = audio_path.lower()
    if audio_path_lower.endswith(".wav"):
        mime_type = "audio/wav"
    elif audio_path_lower.endswith((".mp3", ".mpeg")):
        mime_type = "audio/mpeg"
    elif audio_path_lower.endswith(".flac"):
        mime_type = "audio/flac"
    elif audio_path_lower.endswith(".ogg"):
        mime_type = "audio/ogg"
    else:
        mime_type = "audio/wav"  # Default

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


def run_tts_generation(args) -> None:
    """Run TTS generation via OpenAI-compatible /v1/audio/speech API."""

    # Build request payload
    payload = {
        "model": args.model,
        "input": args.text,
        "response_format": args.response_format,
    }

    # Add optional parameters
    if args.task_type:
        payload["task_type"] = args.task_type

    # Voice clone parameters (Base task)
    if args.ref_audio:
        if args.ref_audio.startswith(("http://", "https://")):
            payload["ref_audio"] = args.ref_audio
        elif args.ref_audio.startswith("data:"):
            payload["ref_audio"] = args.ref_audio
        else:
            payload["ref_audio"] = encode_audio_to_base64(args.ref_audio)
    if args.ref_text:
        payload["ref_text"] = args.ref_text

    if args.task_type == "edit":
        assert args.edit_type is not None, "edit_type and edit_info must be provided for edit task."
        payload["edit_type"] = args.edit_type
        if args.edit_info:
            payload["edit_info"] = args.edit_info

    print(f"Model: {args.model}")
    print(f"Task type: {args.task_type}")
    print(f"Text: {args.text}")
    print("Generating audio...")

    # Make the API call
    api_url = f"{args.api_base}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    with httpx.Client(timeout=300.0) as client:
        response = client.post(api_url, json=payload, headers=headers)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    # Check for JSON error response (only if content is valid UTF-8 text)
    try:
        text = response.content.decode("utf-8")
        if text.startswith('{"error"'):
            print(f"Error: {text}")
            return
    except UnicodeDecodeError:
        pass  # Binary audio data, not an error

    # Save audio response
    output_path = args.output or "tts_output.wav"
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Audio saved to: {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible client for Step-Audio-Editx via /v1/audio/speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Server configuration
    parser.add_argument(
        "--api-base",
        type=str,
        default=DEFAULT_API_BASE,
        help=f"API base URL (default: {DEFAULT_API_BASE})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_API_KEY,
        help="API key (default: EMPTY)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="stepfun-ai/Step-Audio-Editx",
        help="Model name/path",
    )
    parser.add_argument(
        "--audio-tokenizer",
        type=str,
        default="stepfun-ai/Step-Audio-Tokenizer",
        help="Model name/path",
    )

    # Task configuration
    parser.add_argument(
        "--task-type",
        choices=("clone", "edit"),
        default=None,
        help="Task type: clone or edit.",
    )

    # Input text
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize",
    )
    # Base (voice clone) parameters
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Reference audio file path, URL, or base64 for voice cloning (Base task)",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Reference audio transcript for voice cloning (Base task)",
    )

    parser.add_argument("--edit-type", type=str, default=None, help="Type of edit to perform.")
    parser.add_argument("--edit-info", type=str, default=None, help="Additional information for the edit. ")
    # Generation parameters
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum new tokens to generate",
    )

    # Output
    parser.add_argument(
        "--response-format",
        type=str,
        default="wav",
        choices=["wav", "mp3", "flac", "pcm", "aac", "opus"],
        help="Audio output format (default: wav)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output audio file path (default: tts_output.wav)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tts_generation(args)
