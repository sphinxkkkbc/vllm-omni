#!/bin/bash
# Launch vLLM-Omni server for Qwen3-TTS models
#
# Usage:
#   ./run_server.sh

set -e
MODEL="stepfun-ai/Step-Audio-EditX"
echo "Starting StepAudioEditx server with model: $MODEL"

vllm-omni serve "$MODEL" \
    --deploy-config vllm_omni/deploy/step_audio_editx.yaml \
    --host 0.0.0.0 \
    --port 8091 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --omni
